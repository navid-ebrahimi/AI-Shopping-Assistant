from core.models import *
from core.serializers import *
from openai import OpenAI
import re
import numpy as np
import json
import logging
from django.db.models import Q
import os
from core.faiss_index import get_faiss_index

logger = logging.getLogger(__name__)

def find_best_product(customer_data, extra_features_dict, chat):
    logger.info(f"شروع find_best_product با داده‌های اصلی: {customer_data}")
    
    client = OpenAI(api_key="trb-2c31d1370852408ae5-85e4-464e-9d1b-27a793c97e14", base_url="https://turbo.torob.com/v1")

    prompt=f"""
اسم محصول را بدون توجه به جزئیات آن استخراح کن و خروجی بده

متن:
{chat.messages[0]}
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4096,
    )

    result_message = response.choices[0].message.content.strip()
    logger.info(f"result_message for product persian name: {result_message}")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=result_message
    )

    faiss_dict = get_faiss_index()

    query_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

    k = 10000

    distances, indices = faiss_dict['index_product'].search(query_vec, k)
    logger.info(f"indices: {distances[0][0]}")
    logger.info(f"indices: {distances[0][1]}")

    threshold = 0.8

    # mask = distances >= threshold  
    mask = distances <= threshold  

    filtered_indices = [idx[mask_row] for idx, mask_row in zip(indices, mask)]
    filtered_distances = [dist[mask_row] for dist, mask_row in zip(distances, mask)]

    best_keys = []
    for idx in filtered_indices[0]:
        if idx < len(faiss_dict['product_keys']):
            best_keys.append(faiss_dict['product_keys'][idx])
        else:
            best_keys.append(None) 


    members_qs = Member.objects.filter(base_product__random_key__in=best_keys)
    
    if customer_data.get("city") not in [None, "", "none", "None"]:
        members_qs = members_qs.filter(shop__city__title=customer_data["city"])
        logger.info(f"فیلتر بر اساس شهر: {customer_data.get('city')} → {members_qs.count()} رکورد")
    if customer_data.get("score") not in [None, "", "none", "None"]:
        try:
            desired_score = float(customer_data["score"])
            members_qs = members_qs.filter(shop__score__gte=desired_score)
            logger.info(f"فیلتر بر اساس score ≥ {customer_data.get('score')} → {members_qs.count()} رکورد")
        except:
            logger.info(f"could not cast the score to a float number. The score: {customer_data.get('score')}")
    if customer_data.get("has_warranty") not in [None, "", "none", "None"]:
        if customer_data.get("has_warranty") in [True, False, "true", "True", "false", "False"]:
            has_warr = True if customer_data.get("has_warranty") in ["True", "true"] else False
            members_qs = members_qs.filter(shop__has_warranty=has_warr)
            logger.info(f"فیلتر بر اساس has_warranty={customer_data.get('has_warranty')} → {members_qs.count()} رکورد")
    if customer_data.get("price") not in [None, "", "none", "None"]:
        try:
            desired_price = float(customer_data.get('price'))
            lower_bound = desired_price * 0.95  # 5% less
            upper_bound = desired_price * 1.05  # 5% more
            members_qs = members_qs.filter(price__gte=lower_bound, price__lte=upper_bound)
            logger.info(f"فیلتر بر اساس محدوده قیمت: {lower_bound:.0f} تا {upper_bound:.0f} → {members_qs.count()} رکورد")
        except:
            logger.info(f"could not cast the price to a float number. The price: {customer_data.get('price')}")
    # NEW FILTER: Filter by extra_features keys matching (Static)
    if extra_features_dict:
        # Create a list of conditions for each key that should exist in extra_features
        conditions = Q()
        for key in extra_features_dict.keys():
            if extra_features_dict[key] not in [None, "", "none", "None"]:
                # Check if the key exists in the extra_features JSON field
                conditions &= Q(base_product__extra_features__has_key=key)
        
        if conditions:
            members_qs = members_qs.filter(conditions)
            logger.info(f"فیلتر بر اساس کلیدهای extra_features: {list(extra_features_dict.keys())} → {members_qs.count()} رکورد")
    
    members_qs = members_qs.distinct()
    logger.info(f"تعداد رکوردهای یکتا بعد از فیلتر: {members_qs.count()}")
    # Get all member random_keys as a list
    # Get both values in one query to ensure perfect alignment
    member_data = list(members_qs.values_list('random_key', 'base_product__random_key'))

    mem_to_bp = {}
    bp_to_mem = {}

    for member_key, bp_key in member_data:
        mem_to_bp[member_key] = bp_key
        if bp_key not in bp_to_mem:
            bp_to_mem[bp_key] = [member_key]
        else:
            bp_to_mem[bp_key].append(member_key)

    logger.info(f"Created mappings: {len(mem_to_bp)} members -> {len(bp_to_mem)} base products")
    logger.info(f"mem_to_bp: {mem_to_bp}")

    if len(bp_to_mem) == 0:
        logger.warning("هیچ محصولی پیدا نشد")
        return None  

    if  members_qs.count() == 1:
        best_member = members_qs.first()
        logger.info(f"یافته شد: {best_member.random_key}")
        return best_member

    logger.info("استفاده از extra_features برای جستجوی نزدیک‌ترین محصول")

    client = OpenAI(
        api_key=os.getenv("TOROB_API_KEY"),
        base_url="https://turbo.torob.com/v1"
    )
    
    filtered_dict = {k: v for k, v in extra_features_dict.items() if v not in [None, "", "none", "None"]}
    query_str = json.dumps(filtered_dict, ensure_ascii=False)
    logger.info(f"query_str: {query_str}")
    query_embedding = np.array(
        client.embeddings.create(model="text-embedding-3-small", input=query_str).data[0].embedding,
        dtype=np.float32
    ).reshape(1, -1)
    logger.info(f"embedding query ساخته شد")

    
    # Search for more results initially to ensure we find matches within our filtered set
    search_k = 2000 
    distances, indices = faiss_dict['index_extra_features'].search(query_embedding, search_k)

    # Step 5: Filter results to only include base products that match customer criteria
    valid_results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(faiss_dict['product_keys']): 
            bp_key = faiss_dict['product_keys'][idx]
            if bp_key in bp_to_mem: 
                valid_results.append({
                    'bp_key': bp_key,
                    'distance': distances[0][i],
                    'index': idx
                })
    
    best_members_key = []
    if valid_results:
        valid_results.sort(key=lambda x: x['distance'])
        best_matches = valid_results[:15]
        for item in best_matches:
            bpk = item['bp_key']
            for mk in bp_to_mem[bpk]:
                best_members_key.append(mk)
    
        logger.info(f"Final Members for top 15 base products are: {best_members_key}")

    logger.info(f"len(chat.messages): {len(chat.messages)}")
    if len(chat.messages) >= 2:
        candidate_members = Member.objects.filter(
            random_key__in=best_members_key
        ).select_related(
            "base_product", "shop", "shop__city"
        ).values(
            "random_key",
            "price",
            "shop__score",
            "shop__has_warranty",
            "shop__city__title",   
            "base_product__persian_name",
            "base_product__extra_features",
        )

        prompt = f"""
    شما یک دستیار خبره هستید که باید بهترین محصول را برای کاربر انتخاب کنید.

    نیازهای کاربر:
    - محصول ورودی: {result_message}
    - اطلاعات کاربر: {json.dumps(customer_data, ensure_ascii=False)}
    - ویژگی‌های اضافه: {json.dumps(extra_features_dict, ensure_ascii=False)}

    محصولات کاندید:
    """

        for m in candidate_members:
            prompt += f"""
        - member_random_key: {m['random_key']}
        نام محصول: {m['base_product__persian_name']}
        قیمت: {m['price']}
        شهر فروشگاه: {m['shop__city__title']}
        امتیاز فروشگاه: {m['shop__score']}
        گارانتی: {m['shop__has_warranty']}
        ویژگی‌های محصول: {json.dumps(m['base_product__extra_features'], ensure_ascii=False)}
        """

        prompt += """
        قضاوت کنید کدام محصول بیشترین تطابق را با نیازهای کاربر دارد.
        فقط و فقط مقدار `member_random_key` آن محصول را خروجی دهید.
        هیچ متن دیگری ننویسید.
        """

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )

        best_member_key = response.choices[0].message.content.strip()
        logger.info(f"LLM انتخاب کرد: {best_member_key}")

    return None


def find_product_after_chat_with_user(message, chat_id):
    logger.info(f"شروع find_product_after_chat_with_user با chat_id={chat_id}")
    
    chat, created = Chat.objects.get_or_create(chat_id=chat_id)

    client = OpenAI(
        api_key=os.getenv("TOROB_API_KEY"),
        base_url="https://turbo.torob.com/v1"
    )

    if created:
        logger.info("Chat جدید ایجاد شد، بخش فیلدهای اصلی اجرا می‌شود")
        customer_info = {'price': None, 'city': None, 'score': None, 'has_warranty': None}

        prompt = f"""
اگر کاربر به هرکدام از فیلدهایی که در فایل json تعریف شده‌اند جواب داده است، آن مقدار را در جای مناسب قرار بده و فقط و فقط به فرمت json خواسته‌شده خروجی بده.
در صورتی که کاربر به آن جوابی نداده است، مقدار none قرار بده.

نکته: تمام مقادیر کلیدها صرف نظر از اینکه عدد هستند یا مقادیر true و false و none، باید به صورت رشته باشند یعنی بین دو تا علامت " قرار بگیرند. 

نکته: منظور از فیلد price همان قیمت حدودی مدنظر کاربر است.
نکته: فیلدهای price و score در صورت داشتن مقدار، باید حتما مقداری عددی اما بین دو علامت " باشند.
کوئری کاربر: 
{message}

فرمت خروجی باید دقیقاً به شکل زیر باشد:

{{
    "price": "",
    "city": "",
    "score": "",
    "has_warranty": ""
}}
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
        )

        result_message = response.choices[0].message.content.strip()
        data = json.loads(result_message)
        logger.info(f"data از LLM: {data}")

        x = bool(data['price'])
        logger.info(f"x: {x}")

        prompt = f"""
با توجه به json زیر، برای فیلدی که مقادیر خالی دارد (مقادیر خالی شامل "none" و "None" و "" میباشند) سوال آماده کن و خروجی بده.

سوال برای هر فیلد:
1. حدود قیمتی مورد نظر شما برای کالا چقدر است؟
2. در کدام شهر مایل هستید فروشگاه باشد؟
3. حداقل امتیاز فروشگاه از نظر شما چند باشد؟
4. آیا فروشگاه باید ضمانت یا گارانتی داشته باشد؟

مقادیر json:
{data}
"""
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
        )

        result_message = response.choices[0].message.content.strip()
        chat.add_interaction(message=message, response=data)
        logger.info(f"سوال‌های آماده شده برای کاربر: {result_message}")

        return {"message": result_message, "base_random_keys": None, "member_random_keys": None}

    else:
        logger.info("Chat موجود، بخش مدیریت extra_features اجرا می‌شود")
        try:
            previous_data = chat.responses[-1] if chat.responses else {}
        except Exception:
            previous_data = {}
        logger.info(f"previous_data: {previous_data}")

        prompt_update = f"""
شما یک دستیار هوش مصنوعی هستید. وظیفه شما بروزرسانی یک شیء JSON براساس پیام جدید کاربر است. قوانین زیر را رعایت کنید:

1. شما دو ورودی دریافت می‌کنید:
   - شیء JSON قبلی به صورت {previous_data}
   - پیام جدید کاربر به صورت {message}

2. خروجی شما باید یک شیء JSON کامل باشد که:
   - تمام کلیدها و مقادیر، همیشه رشته باشند (یعنی بین " قرار بگیرند).
   - همه اعداد باید به صورت انگلیسی در فیلد موردنظر جایگذاری شوند
   - ممکن است کاربر نام فیلد را مستقیماً نگوید؛ شما باید از متن پیام و نام فیلدهایی که مقدار ندارند متوجه شوید چه فیلدی باید بروزرسانی شود.
   - فیلدهایی که در پیام جدید کاربر ذکر نشده‌اند باید مقدار قبلی خود را حفظ کنند.
   - فیلدهای `price` و `score` در صورت بروزرسانی باید فقط عدد انگلیسی باشند (ولی همچنان داخل " قرار بگیرند).
   - فیلد `has_warranty` فقط می‌تواند یکی از این مقادیر باشد: `"true"`, `"false"`, یا `"none"`.
   - در فیلد `stock_status` باید به صورت انگلیسی نوشته شوند.
   - سایر فیلدها دقیقاً همان‌طور که کاربر وارد می‌کند درج شوند.

3. نکات:
   - `price` به معنی حدود قیمتی مورد نظر کاربر است و باید فقط یک عدد باشد.
   - خروجی شما باید یک JSON معتبر و قابل parse باشد.
   - فقط JSON خروجی را برگردانید، هیچ توضیح اضافه ننویسید.

---

JSON قبلی:
{previous_data}

پیام جدید کاربر:
{message}

خروجی:
یک شیء JSON کامل و معتبر.

"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt_update}],
            temperature=0.1,
            max_tokens=4096
        )

        logger.info(f"پاسخ LLM: {response.choices[0].message.content.strip()}")

        

        try:
            previous_data = json.loads(response.choices[0].message.content.strip())
            logger.info(f"JSON به‌روزرسانی شده با پاسخ کاربر: {previous_data}")
        except Exception as e:
            logger.error(f"خطا در تبدیل پاسخ LLM به JSON: {e}")
            previous_data = previous_data


        if "extra_features" not in previous_data:
            logger.info("ساخت extra_features برای اولین بار")

            # ایجاد embedding از پیام کاربر
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chat.messages[0]
            )
            query_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            logger.info("embedding پیام کاربر ساخته شد")

            # بارگذاری FAISS index
            faiss_dict = get_faiss_index()

            logger.info(f"FAISS index بارگذاری شد ({len(faiss_dict['product_keys'])} کلید)")

            # جستجوی بهترین محصولات مشابه
            k = 5
            distances, indices = faiss_dict['index_product'].search(query_vec, k)
            best_keys = [faiss_dict['product_keys'][idx] if idx < len(faiss_dict['product_keys']) else None for idx in indices[0]]
            logger.info(f"best_keys پیدا شد: {best_keys}")

            # دریافت extra_features محصولات
            products = BaseProduct.objects.filter(random_key__in=best_keys)
            extra_features_list = list(products.values_list('extra_features', flat=True))
            logger.info(f"تعداد extra_features برای پردازش: {len(extra_features_list)}")

            # استخراج همه کلیدهای unique از extra_features
            all_keys = set()
            for ef in extra_features_list:
                try:
                    ef_json = json.loads(ef) if isinstance(ef, str) else ef
                    if isinstance(ef_json, dict):
                        all_keys.update(ef_json.keys())
                except Exception:
                    continue

            final_json = {k: None for k in all_keys}
            previous_data["extra_features"] = final_json

            logger.info(f"JSON نهایی ساخته شد: {previous_data}")

        find_best_product(previous_data, previous_data["extra_features"], chat)

        # ساخت سوال برای کلیدهای خالی
        empty_main_keys = [k for k, v in previous_data.items() if k != "extra_features" and v in [None, ""]]
        empty_extra_keys = []
        if "extra_features" in previous_data and isinstance(previous_data["extra_features"], dict):
            empty_extra_keys = [k for k, v in previous_data["extra_features"].items() if v in [None, ""]]

        if not empty_main_keys and not empty_extra_keys:
            logger.info("همه فیلدها پر شده‌اند")
            return {"message": "همه فیلدها پر شده‌اند ✅",
                    "base_random_keys": None, "member_random_keys": None}

        # ساخت سوالات برای کلیدهای خالی
        mapping_questions = {
            "price": "محدوده قیمتی مورد نظر شما برای کالا چقدر است؟",
            "city": "در کدام شهر مایل هستید فروشگاه باشد؟",
            "score": "حداقل امتیاز فروشگاه از نظر شما چند باشد؟",
            "has_warranty": "آیا فروشگاه باید ضمانت یا گارانتی داشته باشد؟"
        }

        questions_to_translate = ""
        q_index = 1
        for key in empty_main_keys:
            if key in mapping_questions:
                questions_to_translate += f"{q_index}. {mapping_questions[key]} (کلید: {key})\n"
            else:
                questions_to_translate += f"{q_index}. لطفاً مقدار مناسب برای فیلد {key} را وارد کنید.\n"
            q_index += 1

        for key in empty_extra_keys:
            questions_to_translate += f"{q_index}. لطفاً مقدار مناسب برای ویژگی {key} را مشخص کنید.\n"
            q_index += 1

        logger.info(f"سوالات قبل از ترجمه: {questions_to_translate}")

        # ارسال به OpenAI برای فارسی‌سازی طبیعی سوالات
        translation_prompt = f"""
لطفاً متن زیر را به فارسی روان و طبیعی ترجمه کن، با حفظ ساختار هر سوال، به گونه‌ای که:
1. هر سوال در یک خط جدا نوشته شود.
2. معادل فارسی کلمات انگلیسی استفاده شود و از نوشتن کلمات انگلیسی خودداری شود.
3. متن نهایی برای کاربر کاملاً واضح و قابل فهم باشد.

مثال نادرست:
 لطفاً مقدار مناسب برای ویژگی «material» را مشخص کنید. (این مثال به شدت خشک نوشته شده و لحن گرم و صمیمی ندارد. از طرفی در آن از کلمه انگلیسی هم استفاده شده است)

سوالات:
{questions_to_translate}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": translation_prompt}],
            temperature=0.3,
            max_tokens=4096
        )

        translated_questions = response.choices[0].message.content.strip()
        logger.info(f"سوالات ترجمه شده به فارسی: {translated_questions}")

        chat.add_interaction(message=message, response=previous_data)
        return {
            "message": translated_questions,
            "base_random_keys": None,
            "member_random_keys": None
        }