from core.models import *
from core.serializers import *
from openai import OpenAI
import re
import base64
import requests
import numpy as np
import json
import logging
from core.faiss_index import get_faiss_index
import os

logger = logging.getLogger(__name__)

def get_image_base64_data_url(image_url):
    response = requests.get(image_url)
    response.raise_for_status()  # Check for errors
    image_data = response.content
    mime_type = response.headers.get('content-type', 'image/jpeg')
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:{mime_type};base64,{base64_encoded}"
    return data_url


def extract_object(response):
    logger.info(f"{response}")
    # Try common attributes
    if hasattr(response, 'output_text'):
        result_message = response.output_text.strip()
    elif hasattr(response, 'text'):
        result_message = response.text.strip()
    elif hasattr(response, 'content'):
        result_message = response.content.strip()
    elif hasattr(response, 'choices'):
        result_message = response.choices[0].message.content.strip()
    else:
        try:
            import json
            result_message = json.dumps(response, default=str)
        except:
            result_message = str(response)

    logger.info(f"[find_object_in_image] LLM raw response: {result_message}")

    raw_text = result_message.strip()

    return raw_text

def find_object_in_image(message, image_url):
    base64_image = (image_url)
    prompt = f"""
همه اشیای محصول واقعی و قابل تشخیص در تصویر را در لیست قرار بده. 
اشیای انسانی، ابزارهای موقتی، یا مواد غذایی را نادیده بگیر.

اگر دو یا چند محصول به گونه‌ای با هم ترکیب شده باشند که از نظر کاربردی و ماهیتی یک واحد واقعی و مستقل بسازند، آن‌ها را به عنوان یک آیتم مشترک گزارش کن (به صورت "object1 با object2"). 
منظور از ترکیب:
- اجزاء ذاتاً یک واحد طبیعی می‌سازند.
- یکی از اجزاء بدون دیگری ناقص یا بی‌کاربرد است.
- صرفاً کنار هم بودن کافی نیست.

تمام اشیای دیگر، حتی اگر با هم دیده شوند، اگر ماهیت مستقل دارند، باید جداگانه فهرست شوند.

خروجی باید یک لیست مرتب باشد، به طوری که بیشترین بخش تصویر را اشغال‌کننده در جایگاه اول باشد و بقیه بر اساس میزان اشغال تصویر ادامه پیدا کنند.
هر محصول یا شیء را فقط با **نام کلی محصول** بنویس و از ذکر جزئیات یا ویژگی‌های فرعی خودداری کن.

نوع خروجی: [object1 , object2 , object3 با object4 , ...]  

"""

    logger.info(f"[find_object_in_image] prompt length: {len(prompt)}")
    logger.info(f"[find_object_in_image] prompt:\n{prompt}")

    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    response = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": base64_image},
            ],
        }],
        temperature=0.5,
    )

    raw_text = extract_object(response)
    raw_text = raw_text.strip("[]")
    object_list = [x.strip() for x in re.split(r"[,\u060C]", raw_text) if x.strip()]

    logger.info(f"[find_object_in_image] Parsed objects: {object_list}")

    faiss_dict = get_faiss_index()
    parent_title = None

    THRESHOLD = 0.5

    for obj in object_list:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=obj
        )
        query_vec = np.array(emb.data[0].embedding, dtype=np.float32).reshape(1, -1)

        k = 3 
        distances, indices = faiss_dict['index_product'].search(query_vec, k)

        logger.info(f"[find_object_in_image] obj={obj}, indices={indices[0]}, dists={distances[0]}")

        if distances[0][0] > THRESHOLD:
            logger.warning(f"[find_object_in_image] Weak match for {obj} (dist={distances[0][0]}) → skipping")
            continue

        result_message = obj
        break

    if not result_message:
        logger.warning("[find_object_in_image] No valid product/category found after checking all objects")

    return {
        "message": result_message,
        "base_random_keys": None,
        "member_random_keys": None
    }
