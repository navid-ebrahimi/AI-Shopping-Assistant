from core.models import *
from core.serializers import *
from openai import OpenAI
import re
import numpy as np
import json
import logging
from core.faiss_index import get_faiss_index

logger = logging.getLogger(__name__)

def extract_special_case(message: str):
    """
    اگر پیام شامل یکی از حالت‌های خاص (ping, base key, member key) بود،
    خروجی مناسب رو برگردونه. در غیر این صورت None برمی‌گردونه.
    """
    if re.search(r"\bping\b", message, re.IGNORECASE):
        return {"message": "pong", "base_random_keys": None, "member_random_keys": None}

    match_base = re.search(
        r"return\s+base\s+random\s+key:\s*([0-9a-fA-F-]{36})",
        message,
        re.IGNORECASE
    )
    if match_base:
        return {"message": None, "base_random_keys": [match_base.group(1)], "member_random_keys": None}

    match_member = re.search(
        r"return\s+member\s+random\s+key:\s*([\w-]+)",
        message,
        re.IGNORECASE
    )
    if match_member:
        return {"message": None, "base_random_keys": None, "member_random_keys": [match_member.group(1)]}

    return None