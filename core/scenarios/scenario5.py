from core.models import *
from core.serializers import *
from openai import OpenAI
import re
import numpy as np
import json
import logging
from core.faiss_index import get_faiss_index
import os

logger = logging.getLogger("product_logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



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

        
def compare_bases_for_user_query(last_message):
    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    # logger.info("Generating product list from LLM...")
    prompt = f"""
متنی به تو داده می‌شود که شامل چند محصول است. لطفاً همه محصول‌ها را شناسایی کن و هر محصول را در یک خط جداگانه بنویس. هیچ توضیح اضافی، نقطه، کاما یا متن اضافی اضافه نکن. محصولات باید دقیقاً همانطور که در متن آمده‌اند، حفظ شوند.


متن: "{last_message}"


"""

    # logger.info(f"first prompt: {prompt}")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4096,
    )
    llm_text = response.choices[0].message.content
    # logger.info(f"llm_text: {llm_text}")
    products_list = [line.strip() for line in llm_text.splitlines() if line.strip()]
    # logger.info(f"Extracted {len(products_list)} products")

    # -------------------------------
    # مرحله 2: ایجاد embedding برای محصولات
    # -------------------------------
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=products_list
    )
    embeddings_list = [item.embedding for item in response.data]

    # -------------------------------
    # مرحله 3: پیدا کردن نزدیک‌ترین random_key از FAISS
    # -------------------------------
    # logger.info("Searching for closest random_key in FAISS index...")
    closest_keys = []

    faiss_dict = get_faiss_index()

    for i, emb in enumerate(embeddings_list, 1):
        query_vec = np.array(emb, dtype=np.float32).reshape(1, -1)
        k = 1
        distances, indices = faiss_dict['index_product'].search(query_vec, k)
        best_idx = indices[0][0]

        if best_idx >= len(faiss_dict['product_keys']):
            closest_keys.append(None)
            # logger.debug(f"[{i}] No matching key found")
        else:
            closest_keys.append(faiss_dict['product_keys'][best_idx])
            # logger.debug(f"[{i}] Closest key: {faiss_dict['product_keys'][best_idx]}")


    # -------------------------------
    # مرحله 4: گرفتن رکوردها از دیتابیس
    # -------------------------------
    valid_keys = [k for k in closest_keys if k is not None]
    products_qs = BaseProduct.objects.filter(random_key__in=valid_keys)
    products_list = list(products_qs.values())

    all_member_descriptions = {}

    for key in valid_keys:
        member_desc = get_member_descriptions(key)
        all_member_descriptions[key] = member_desc


    # -------------------------------
    # مرحله 5: آماده‌سازی پرامپت نهایی برای پاسخ‌دهی به کاربر
    # -------------------------------
    records_text = ""
    for i, prod in enumerate(products_list, 1):
        random_key = prod.get("random_key", "")
        persian_name = prod.get("persian_name", "")
        extra = prod.get("extra_features", {})

        # تبدیل extra_features به دیکشنری
        if isinstance(extra, str):
            try:
                extra_dict = json.loads(extra)
            except json.JSONDecodeError:
                extra_dict = {"raw": extra}
        elif isinstance(extra, dict):
            extra_dict = extra
        else:
            extra_dict = {"raw": str(extra)}

        extra_str = ", ".join(f"{k}: {v}" for k, v in extra_dict.items())

        # توضیحات اعضای محصول
        members_info = "\n    ".join(all_member_descriptions.get(random_key, []))

        records_text += (
            f'  {{\n'
            f'    "random_key": "{random_key}",\n'
            f'    "name": "{persian_name}",\n'
            f'    "features": "{extra_str}",\n'
            f'    "members": [\n"{members_info}"\n]\n'
            f'  }},\n'
        )

    prompt = f"""
شما یک دستیار هوشمند هستید که باید به سوالات کاربران درباره محصولات پاسخ دهید.

اطلاعات محصولات به صورت رکوردهای JSON در اختیار شماست. هر رکورد شامل:
- random_key 
- نام فارسی محصول
- ویژگی‌های اضافی (extra_features) در قالب JSON

دستورالعمل‌ها:
1. فقط بر اساس اطلاعات موجود در رکوردها به سوال پاسخ بده.
2. پاسخ باید دقیق، روشن و مختصر باشد.
3. خروجی نهایی فقط به صورت JSON معتبر باشد و هیچ متن اضافه‌ای خارج از JSON تولید نشود.
4. در خروجی دو کلید باید وجود داشته باشد:
   - "answer": متن پاسخ نهایی برای کاربر به زبان فارسی روان همراه با دلایل مشخص بر اساس اطلاعات محصول که چرا محصول انتخاب شده بهتر است
   - "random_key ": random_key  محصول انتخاب‌شده (یا null اگر هیچ محصولی مناسب نبود)

سوال کاربر:
"{last_message}"


محصولات موردنظر:

{records_text}

خروجی مورد انتظار (فقط JSON معتبر):
{{
  "answer": "اینجا متن پاسخ",  
  "random_key": "random_key محصول موردنظر"  
}}
"""

    # -------------------------------
    # مرحله 6: دریافت پاسخ نهایی از LLM
    # -------------------------------
    # logger.info("Sending final prompt to LLM...")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4096,
    )
    result_message = response.choices[0].message.content.strip()

    data = json.loads(result_message)
    answer = data.get("answer", "").strip()
    random_key = data.get("random_key")


    # -------------------------------
    # مرحله 7: خروجی نهایی
    # -------------------------------
    return {
        "message": answer,
        "base_random_keys": [random_key],
        "member_random_keys": None
    }
