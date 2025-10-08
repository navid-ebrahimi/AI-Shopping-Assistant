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
    # Download the image
    response = requests.get(image_url)
    response.raise_for_status()  # Check for errors
    
    # Get the image content and MIME type
    image_data = response.content
    mime_type = response.headers.get('content-type', 'image/jpeg')
    
    # Encode to base64
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    
    # Create data URL
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

def find_object_in_image_and_products(message, image_url):
    base64_image = (image_url)
    if base64_image.startswith("data:"):
        base64_image_2 = base64_image.split(",")[1]   
    API_URL = "https://model-api.darkube.app/embed_image"

    payload = {"base64_images": [base64_image_2]}
    response = requests.post(API_URL, json=payload)

    # ---------- نتیجه ----------
    if response.status_code == 200:
        data = response.json()
        print("Embedding dims:", data["dims"])
        print("First vector (truncated):", data["embeddings"][0][:10]) 
    else:
        print("Error:", response.status_code, response.text)

    faiss_dict = get_faiss_index()

    
    query_vec = np.array(data["embeddings"][0], dtype=np.float32).reshape(1, -1)

    k = 1000
    distances, indices = faiss_dict['index_images'].search(query_vec, k)

    best_idx = indices[0][0]

    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    for i in range(k):
        index = indices[0][i]
        base_random_key = faiss_dict['images_keys'][str(index)]

        product = BaseProduct.objects.get(random_key=base_random_key)
        product_base64_image = get_image_base64_data_url(product.image_url)

        prompt = """
آیا دو تصویر یک محصول را نشان می‌دهند؟
اگر بله True خروجی بده، اگر خیر False.
هیچ توضیح اضافه‌ای نده
"""
        response = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": base64_image},
                    {"type": "input_image", "image_url": product_base64_image},
                ],
            }],
            temperature=0.5,
        )

        result_message = extract_object(response)

        if (bool(result_message)):
            final_result = base_random_key
            logger.info("final_result:", final_result)
            break


    return {
        "message": None,
        "base_random_keys": [str(final_result)],
        "member_random_keys": None
    }
