from core.models import *
from core.serializers import *
from openai import OpenAI
import re
import numpy as np
import json
import os
import logging
from core.faiss_index import get_faiss_index

logger = logging.getLogger(__name__)

def find_product_based_name(last_message):
    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=last_message
    )

    faiss_dict = get_faiss_index()
    query_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)  # 2D array

    k = 1
    distances, indices = faiss_dict['index_product'].search(query_vec, k)
    best_idx = indices[0][0]

    if best_idx >= len(faiss_dict['product_keys']):
        return None, None

    return {
        "message": None,
        "base_random_keys": [faiss_dict['product_keys'][best_idx]],
        "member_random_keys": None
    }