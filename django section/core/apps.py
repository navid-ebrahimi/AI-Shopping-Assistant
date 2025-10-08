from django.apps import AppConfig
import sys

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        from .faiss_index import get_faiss_index
        if "runserver" in sys.argv:
            get_faiss_index()
