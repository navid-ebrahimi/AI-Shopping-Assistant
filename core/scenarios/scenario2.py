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

def find_property_of_good(last_message):
    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=last_message
    )

    faiss_dict = get_faiss_index()

    query_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

    k = 1
    distances, indices = faiss_dict['index_product'].search(query_vec, k)

    best_keys = []
    for idx in indices[0]:
        if idx < len(faiss_dict['product_keys']):
            best_keys.append(faiss_dict['product_keys'][idx])
        else:
            best_keys.append(None) 

    # logger.info(f"[find_property_of_good] best_keys: {best_keys}")

    products = BaseProduct.objects.filter(random_key__in=best_keys)
    products_ordered = sorted(
        products,
        key=lambda p: best_keys.index(p.random_key)
    )

    for i, prod in enumerate(products_ordered, 1):
        logger.info(f"[find_property_of_good] محصول {i}: {prod.random_key} - {prod.persian_name}")

    prompt = f"""
شما یک دستیار هوش مصنوعی هستید. 
با توجه به نمونه‌های زیر، فقط به سوال کاربر پاسخ دهید. 
هیچ توضیح اضافی ندهید و فقط پاسخ مستقیم به سوال بدهید.

سوال کاربر:
{last_message}

نمونه‌های محصول (۵ رکورد نزدیک‌ترین):
[
"""

    for i, prod in enumerate(products_ordered, 1):
        persian_name = getattr(prod, "persian_name", "")
        extra = getattr(prod, "extra_features", "")
        
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
        
        prompt += f'  {{ "id": {i}, "name": "{persian_name}", "features": "{extra_str}" }},\n'

    prompt += "]\n\nفقط پاسخ نهایی به سوال کاربر را بده."

    logger.info(f"[find_property_of_good] prompt length: {len(prompt)}")
    logger.info(f"[find_property_of_good] prompt:\n{prompt}")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=4096,
    )

    result_message = response.choices[0].message.content.strip()
    logger.info(f"[find_property_of_good] LLM response: {result_message}")

    return {
        "message": result_message,
        "base_random_keys": None,
        "member_random_keys": None
    }