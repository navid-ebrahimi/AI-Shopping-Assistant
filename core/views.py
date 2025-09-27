from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import viewsets
from .models import *
from .serializers import *
from openai import OpenAI
import re
import os
import numpy as np
import json
import logging
from .faiss_index import get_faiss_index

from core.scenarios.scenario0 import extract_special_case
from core.scenarios.scenario1 import find_product_based_name
from core.scenarios.scenario2 import find_property_of_good
from core.scenarios.scenario3 import find_property_of_shops
from core.scenarios.scenario4 import find_product_after_chat_with_user
from core.scenarios.scenario5 import compare_bases_for_user_query
from core.scenarios.scenario6 import find_object_in_image
from core.scenarios.scenario7 import find_object_in_image_and_products


logger = logging.getLogger(__name__)

class SearchViewSet(viewsets.ModelViewSet):
    queryset = Search.objects.all()
    serializer_class = SearchSerializer


class BaseViewViewSet(viewsets.ModelViewSet):
    queryset = BaseView.objects.all()
    serializer_class = BaseViewSerializer


class FinalClickViewSet(viewsets.ModelViewSet):
    queryset = FinalClick.objects.all()
    serializer_class = FinalClickSerializer


class BaseProductViewSet(viewsets.ModelViewSet):
    queryset = BaseProduct.objects.all()
    serializer_class = BaseProductSerializer


class MemberViewSet(viewsets.ModelViewSet):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer


class ShopViewSet(viewsets.ModelViewSet):
    queryset = Shop.objects.all()
    serializer_class = ShopSerializer


class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer


class BrandViewSet(viewsets.ModelViewSet):
    queryset = Brand.objects.all()
    serializer_class = BrandSerializer


class CityViewSet(viewsets.ModelViewSet):
    queryset = City.objects.all()
    serializer_class = CitySerializer


def detect_scenario_with_llm(message: str, last_message_type) -> str:
    """
    پیام رو به LLM می‌ده و فقط شماره سناریو (۱ تا ۷) رو برمی‌گردونه.
    """

    
    client = OpenAI(api_key=os.getenv("TOROB_API_KEY"), base_url="https://turbo.torob.com/v1")

    prompt = f"""
شما یک داور هوش مصنوعی هستید که باید تشخیص دهید ورودی کاربر به کدام یک از 7 سناریوی از پیش تعریف‌شده تعلق دارد.  

message_type = {last_message_type}

سناریوها و مثال‌ها:

1. کاربر دنبال یک محصول خاص است (base مشخص)
   هدف: کاربر یک محصول مشخص از ترب را می‌خواهد و کوئری به‌طور دقیق به یک محصول (base) مپ می‌شود.
   مثال: "لطفاً دراور چهار کشو (کد D14) را برای من تهیه کنید."
   فقط در صورتی که message_type = Text باشد این مورد میتواند انتخاب شود

2. کاربر دنبال یک ویژگی از یک محصول خاص است (بدون اشاره به عضو، فروشگاه، اعضا یا چند فروشگاه)
هدف: کاربر می‌خواهد اطلاعات یا ویژگی‌های مشخص یک محصول خاص را بداند. این ویژگی‌ها می‌توانند شامل ابعاد و اندازه، رنگ و طرح، ظرفیت یا حجم، جنس یا متریال، وضعیت موجودی (نو/دست دوم، تعداد در انبار), انرژی یا رتبه مصرف انرژی, گارانتی و شرایط ویژه محصول, سایر ویژگی‌های فنی یا کاربردی محصول باشند.
ویژگی مهم: کوئری کاربر به‌طور دقیق به یک محصول مشخص (base) مپ می‌شود، اما هدف اصلی، دانستن اطلاعات یا ویژگی‌های آن محصول است، نه خرید یا انتخاب محصول.
فقط در صورتی که message_type = Text باشد این مورد میتواند انتخاب شود

3. کاربر درباره فروشنده یک محصول خاص سوال می‌کند (با اشاره به عضو، فروشگاه، اعضا یا چند فروشگاه)
   هدف: کاربر اطلاعات مربوط به فروشنده‌ها یک محصول مشخص را می‌خواهد. کوئری دقیق به یک base مپ می‌شود.
   مثال: "کمترین قیمت در این پایه برای گیاه طبیعی بلک گلد بنسای نارگل کد ۰۱۰۸ چقدر است؟"
    فقط در صورتی که message_type = Text باشد این مورد میتواند انتخاب شود

4. کاربر دنبال محصول خاصی است ولی کوئری مستقیم به محصول مپ نمی‌شود (ورودی کاربر نباید اشاره‌ای به تصویر در ورودی کرده باشد)
   هدف: کاربر دنبال محصول است اما پرسش اولیه قابل نگاشت مستقیم به یک base نیست. دستیار باید با پرسش و پاسخ راهنمایی کند تا محصول مناسب پیدا شود.
   مثال: "می‌خواهم یه میز تحریر برای کارهای روزمره و نوشتن پیدا کنم. می‌تونی یه فروشنده خوب معرفی کنی؟"
    فقط در صورتی که message_type = Text باشد این مورد میتواند انتخاب شود

5. کاربر می‌خواهد دو یا چند محصول را مقایسه کند
   هدف: کاربر دو یا چند محصول مشخص را با توجه به یک ویژگی یا کاربرد خاص می‌خواهد مقایسه کند.
   مثال: "کدام یک از این ماگ‌ها، ماگ-لیوان هندوانه فانتزی و کارتونی کد 1375 یا ماگ لته خوری سرامیکی با زیره کد 741، برای کودکان یا نوجوانان مناسب‌تر و سبک کارتونی و فانتزی دارد؟"
    فقط در صورتی که message_type = Text باشد این مورد میتواند انتخاب شود

6. کاربر عکس می‌فرستد و می‌خواهد بداند شیء اصلی چیست 
   هدف: کاربر دنبال شناخت آبجکت اصلی در عکس است، بدون اینکه محصول مشخص باشد.
   مثال: "یک تصویر از محصول آپلود شده است. لطفاً بگو این شیء اصلی چه چیزی است."
    فقط در صورتی که message_type = image باشد این مورد میتواند انتخاب شود

7. کاربر عکس می‌فرستد که به محصول مشخص مپ می‌شود (ورودی کاربر باید اشاره‌ای به تصویر در ورودی کرده باشد)
   هدف: کاربر عکس محصولی را فرستاده و می‌خواهد اطلاعات همان محصول مشخص را ببیند.
   مثال: "این تصویر مربوط به محصول کد B23 است. لطفاً بررسی کن و اطلاعات محصول را نشان بده."
    فقط در صورتی که message_type = image باشد این مورد میتواند انتخاب شود

** تفاوت سناریو 6 و 7 در این است که سناریو 6 صرفا تشخیص نوع و مفهوم شی برایش اهمیت دارد ولی سناریو 7 یک محصول مرتبط با تصویر میخواهد **

قوانین:
- فقط شماره سناریو (1 تا 7) را برگردان.
- خروجی JSON نساز.
- اگر ابهام وجود داشت، محتمل‌ترین سناریو را انتخاب کن.

ورودی کاربر:
{message}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=100,
    )

    return response.choices[0].message.content.strip()


@api_view(["POST"])
def chat(request):
    chat_id = request.data.get("chat_id")
    messages = request.data.get("messages", [])

    if messages:
        message_dict = messages[0] 
        message_content_1 = messages[0].get("content", "")
        message_type_1 = messages[0].get("type", None)

        if len(messages) == 2:
            message_content_2 = messages[1].get("content", "")
            message_type_2 = messages[1].get("type", None)
    else:
        message_content_1 = ""
        message_type_1 = None
        message_content_2 = ""
        message_type_2 = None

    exists = Chat.objects.filter(chat_id=chat_id).exists()

    # if exists:
    #     result = find_product_after_chat_with_user(message_content_1, chat_id)
    #     logger.info(f"[chat] scenario=4 → result={result}")
    #     return Response(result)

    logger.info(f"[chat] chat_id={chat_id}, message_content_1={message_content_1}")

    special_case = extract_special_case(message_content_1)
    if special_case:
        logger.info(f"[chat] special_case detected → {special_case}")
        return Response(special_case)

    if len(messages) == 2:
        scenario_number = detect_scenario_with_llm(message_content_1, message_type_2)
    else:
        scenario_number = detect_scenario_with_llm(message_content_1, message_type_1)
    logger.info(f"[chat] scenario_number={scenario_number}")

    if int(scenario_number) == 1:
        result = find_product_based_name(message_content_1)
        logger.info(f"[chat] scenario=1 → result={result}")
        return Response(result)

    if int(scenario_number) == 2:
        result = find_property_of_good(message_content_1)
        logger.info(f"[chat] scenario=2 → result={result}")
        return Response(result)
    
    if int(scenario_number) == 3:
        result = find_property_of_shops(message_content_1)
        logger.info(f"[chat] scenario=3 → result={result}")
        return Response(result)

    # if int(scenario_number) == 4:
    #     result = find_product_after_chat_with_user(message_content_1, chat_id)
    #     logger.info(f"[chat] scenario=4 → result={result}")
    #     return Response(result)

    if int(scenario_number) == 5:
        result = compare_bases_for_user_query(message_content_1)
        logger.info(f"[chat] scenario=5 → result={result}")
        return Response(result)

    if int(scenario_number) == 6:
        if len(messages) < 2:
            return Response({
                "message": "There is no image in the request",
                "base_random_keys": None,
                "member_random_keys": None
            })
        image_url = messages[1].get("content")
        result = find_object_in_image(message_content_1, image_url)
        logger.info(f"[chat] scenario=6 → result={result}")
        return Response(result)

    if int(scenario_number) == 7:
        if len(messages) < 2:
            return Response({
                "message": "There is no image in the request",
                "base_random_keys": None,
                "member_random_keys": None
            })
        image_url = messages[1].get("content")
        result = find_object_in_image_and_products(message_content_1, image_url)
        logger.info(f"[chat] scenario=7 → result={result}")
        return Response(result)


    response_data = {
        "message": scenario_number,
        "base_random_keys": None,
        "member_random_keys": None,
    }
    logger.info(f"[chat] default response → {response_data}")
    return Response(response_data)