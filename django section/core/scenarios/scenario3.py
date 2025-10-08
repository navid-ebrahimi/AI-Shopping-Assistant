from core.models import *
from core.serializers import *
from openai import OpenAI
import re
import numpy as np
import json
import logging
import os
from core.faiss_index import get_faiss_index

logger = logging.getLogger(__name__)

def validate_and_calculate_average(llm_output):
    try:
        data = json.loads(llm_output)
        
        if 'prices' not in data:
            return None
        
        if not isinstance(data['prices'], list):
            return None
        
        prices = data['prices']
        if not all(isinstance(price, (int, float)) for price in prices):
            return None
        
        if len(prices) == 0:
            return None
        
        average_price = round(sum(prices) / len(prices), 2)
        
        return average_price
    
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None

def get_member_descriptions(best_key):
    try:
        # Get the base product using the random_key
        base_product = BaseProduct.objects.get(random_key=best_key)
        
        # Get all members for that base product with related shop and city data
        members = Member.objects.filter(base_product=base_product).select_related(
            'shop', 'shop__city'
        )
        
        descriptions = []
        for member in members:
            # Build the description string for each member
            warranty_status = "گارانتی دارد" if member.shop.has_warranty else "گارانتی ندارد"
            city_title = member.shop.city.title if member.shop.city else "شهر مشخص نیست"
            
            description = (
                f"کلید تصادفی: {member.random_key}، "
                f"قیمت: {member.price}، "
                f"گارانتی: {warranty_status}، "
                f"امتیاز: {member.shop.score}،"
                f"شهر: {city_title}"
            )
            descriptions.append(description)
        
        return descriptions
    
    except BaseProduct.DoesNotExist:
        return [f"No product found with key: {best_key}"]
    except Exception as e:
        return [f"Error retrieving data: {str(e)}"]


def find_property_of_shops(last_message):
    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=last_message
    )

    faiss_dict = get_faiss_index()

    query_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

    k = 1 
    distances, indices = faiss_dict['index_product'].search(query_vec, k)

    best_idx = indices[0][0]

    if best_idx >= len(faiss_dict['product_keys']):
        return None, None

    logger.info(f"[find_property_of_shop] best_key: {faiss_dict['product_keys'][best_idx]}")

    member_descriptions = get_member_descriptions(faiss_dict['product_keys'][best_idx])

    prompt = f"""
شما یک دستیار هوش مصنوعی هستید. 
با توجه به اطلاعات زیر، فقط به سوال کاربر پاسخ دهید. 
هیچ توضیح اضافی ندهید و فقط پاسخ مستقیم به سوال بدهید. شما باید به سوال کاربر در مورد فروشگاه های ارائه دهنده یک محصول پاسخ دهید. به شما اطلاعات فروشگاه های ارائه دهنده آن محصول داده میشود. شما ابتدا باید ببینید با توجه به ورودی کاربر و اطلاعات فروشگاه، کدام یک از فروشگاهها مورد هدف کاربر هستند (برای مثال فروشگاههایی که در شهر تهران هستند یا آنها که  گارانتی دارند یا ترکیبی از آن)
و پس از شناسایی فروشگاههای مورد هدف، به سوال اصلی کاربر جواب دهید. این سوال اصلی میتواند مواردی مثل حداقل قیمت، حداکثر قیمت، میانگین قیمت، تعداد فروشگاه های هدف، کلید تصادفی فروشگاه هدف و ... باشد.
در صورت وجود عدد در خروجی، آن را تا دو رقم اعشار نمایش دهید.

برای مثال اگر کاربر حداقل قیمت محصول در فروشگاههای شهر تهران که گارانتی دارند را درخواست کرد، ابتدا این فروشگاهها جدا شده و قیمت آنها با یکدگیر مقایسه شده و حداقل قیمت با حداکثر دو رقم اعشار نمایش داده میشود.

در صورتی که فروشگاه هدف پیدا نشد یا آن محصول توسط فروشگاهی ارائه نمیشد، به معنی موجود نبودن آن است.

نکته: در صورتی که جواب سوال کاربر از نوع عددی است (سوالاتی مثل حداقل، حداکثر قیمت یا تعداد فروشگاهها)، تنها عدد مورد نظر را خروجی دهید به طوری که هیچ حرف الفبایی در آن نباشد.

نکته: تنها در صورتی که کاربر، میانگین قیمت محصول را درخواست کرد، ابتدا فروشگاههای هدف را پیدا کنید و در خروجی یک قالب JSON دهید که به شکل زیر است:
{{
    "prices": [p1,p2,p3,...]
}}
که مقادیر p1,p2,p3,... قیمت محصول در فروشگاههای هدف هستند.
توجه کنید که در این حالت، خروجی شما باید به صورت JSON باشد بدون هیچ خروجی اضافه.
سوال کاربر:
{last_message}

فروشگاه های ارائه دهنده محصول: 
[

"""
    for ind, mem_des in enumerate(member_descriptions):
        prompt += f"{ind+1}- {mem_des}\n"

    prompt += "]\n\nفقط پاسخ نهایی به سوال کاربر را بده."

    logger.info(f"[find_property_of_shop] prompt length: {len(prompt)}")
    logger.info(f"[find_property_of_shop] prompt:\n{prompt}")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )

    result_message = response.choices[0].message.content.strip()
    logger.info(f"[find_property_of_shop] LLM response: {result_message}")

    average_price = validate_and_calculate_average(result_message)

    if average_price is not None:
        return {
            "message": str(average_price),
            "base_random_keys": None,
            "member_random_keys": None
        }
    else:
        return {
            "message": result_message,
            "base_random_keys": None,
            "member_random_keys": None
        }
