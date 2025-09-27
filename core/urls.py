from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter()
router.register(r"searches", SearchViewSet)
router.register(r"baseviews", BaseViewViewSet)
router.register(r"finalclicks", FinalClickViewSet)
router.register(r"baseproducts", BaseProductViewSet)
router.register(r"members", MemberViewSet)
router.register(r"shops", ShopViewSet)
router.register(r"categories", CategoryViewSet)
router.register(r"brands", BrandViewSet)
router.register(r"cities", CityViewSet)

urlpatterns = [
    path("api/", include(router.urls)), 
    path("chat", chat, name="chat"), 
]
