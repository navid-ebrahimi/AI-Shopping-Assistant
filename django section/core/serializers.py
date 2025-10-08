from rest_framework import serializers
from .models import *

class SearchSerializer(serializers.ModelSerializer):
    class Meta:
        model = Search
        fields = "__all__"

class BaseViewSerializer(serializers.ModelSerializer):
    class Meta:
        model = BaseView
        fields = "__all__"

class FinalClickSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinalClick
        fields = "__all__"

class BaseProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = BaseProduct
        fields = "__all__"

class MemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Member
        fields = "__all__"

class ShopSerializer(serializers.ModelSerializer):
    class Meta:
        model = Shop
        fields = "__all__"

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = "__all__"

class BrandSerializer(serializers.ModelSerializer):
    class Meta:
        model = Brand
        fields = "__all__"

class CitySerializer(serializers.ModelSerializer):
    class Meta:
        model = City
        fields = "__all__"
