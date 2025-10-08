import os
import sys
import pickle
import faiss
import gdown
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DIM = 1536
DATA_DIR = "/var/lib/data"
os.makedirs(DATA_DIR, exist_ok=True)

id_to_key_path = os.path.join(DATA_DIR, "id_to_key.pkl")
products_index_path = os.path.join(DATA_DIR, "products.index")
extra_features_index_path = os.path.join(DATA_DIR, "extra_features.index")
categories_id_to_key_path = os.path.join(DATA_DIR, "categories_id_to_key.pkl")
categories_index_path = os.path.join(DATA_DIR, "categories.index")
images_index_path = os.path.join(DATA_DIR, "images.index")
images_id_to_key_path = os.path.join(DATA_DIR, "images_id_to_key.pkl")

URL_ID_TO_KEY = "https://drive.google.com/uc?id=1biSHt_AJeEYO5erpJGN_aBFHZcdBE-Qw"
URL_PRODUCTS = "https://drive.google.com/uc?id=1TtUlCllCnXXyDuvjoMpiwpFWJmWl5feb"
URL_EXTRA_FEATURES = "https://drive.google.com/uc?id=1d255ts3RhFqS0zc1oLluPg1eA__QUU1I"
URL_CATEGORIES_EMBEDDING = "https://drive.google.com/uc?id=1vcXawlNSwDITFfuKJcFZWOJocBsoUgpu"
URL_CATEGORIES_ID_TO_KEY = "https://drive.google.com/uc?id=12ovzTiUrf7YEUwyQ0bQ2W0PZoUtrBqw7"
URL_IMAGES_EMBEDDING = "https://drive.google.com/uc?id=10hIr22ZOMv4Sgfoesly7Y7iSJSL0j6JX"
URL_IMAGES_ID_TO_KEY = "https://drive.google.com/uc?id=1hA_3kNGsHslZRSTC32jeDcJZfv4fK8Pg"


_index_product = None
_index_extra_features = None
_index_categories = None
_index_images = None
_categories_keys = None
_images_keys = None
_keys = None


def _download_if_missing():
    if not os.path.exists(id_to_key_path):
        logging.info("Downloading id_to_key.pkl ...")
        gdown.download(URL_ID_TO_KEY, id_to_key_path, quiet=False)
        logging.info("Downloaded id_to_key.pkl ✅")
    else:
        logging.info("id_to_key.pkl already exists")

    if not os.path.exists(products_index_path):
        logging.info("Downloading products.index ...")
        gdown.download(URL_PRODUCTS, products_index_path, quiet=False)
        logging.info("Downloaded products.index ✅")
    else:
        logging.info("products.index already exists")

    if not os.path.exists(extra_features_index_path):
        logging.info("Downloading extra_features.index ...")
        gdown.download(URL_EXTRA_FEATURES, extra_features_index_path, quiet=False)
        logging.info("Downloaded extra_features.index ✅")
    else:
        logging.info("extra_features.index already exists")

    if not os.path.exists(categories_index_path):
        logging.info("Downloading categories.index ...")
        gdown.download(URL_CATEGORIES_EMBEDDING, categories_index_path, quiet=False)
        logging.info("Downloaded categories.index ✅")
    else:
        logging.info("categories.index already exists")

    if not os.path.exists(categories_id_to_key_path):
        logging.info("Downloading categories_id_to_key.index ...")
        gdown.download(URL_CATEGORIES_ID_TO_KEY, categories_id_to_key_path, quiet=False)
        logging.info("Downloaded categories_id_to_key.index ✅")
    else:
        logging.info("categories_id_to_key.index already exists")

    if not os.path.exists(images_id_to_key_path):
        logging.info("Downloading images_id_to_key.index ...")
        gdown.download(URL_IMAGES_ID_TO_KEY, images_id_to_key_path, quiet=False)
        logging.info("Downloaded images_id_to_key.index ✅")
    else:
        logging.info("images_id_to_key.index already exists")

    if not os.path.exists(images_index_path):
        logging.info("Downloading images.index ...")
        gdown.download(URL_IMAGES_EMBEDDING, images_index_path, quiet=False)
        logging.info("Downloaded images.index ✅")
    else:
        logging.info("images.index already exists")


def get_faiss_index():
    """لود singleton FAISS index و keys"""
    global _index_product, _index_extra_features, _index_categories, _categories_keys, _keys, _images_keys, _index_images

    if "runserver" not in sys.argv:
        logging.info("Not running under runserver → skipping FAISS load")
        return {}

    if _index_product is None or _keys is None or _index_extra_features is None or _index_categories is None or _categories_keys is None or _index_images is None or _images_keys is None:
        logging.info("FAISS index/keys not loaded → loading now")
        _download_if_missing()

        logging.info(f"Loading keys from {id_to_key_path}")
        with open(id_to_key_path, "rb") as f:
            _keys = pickle.load(f)

        with open(categories_id_to_key_path, "rb") as f:
            _categories_keys = pickle.load(f)

        with open(images_id_to_key_path, "r", encoding="utf-8") as f:
            _images_keys = json.load(f)

        logging.info(f"Loaded {len(_keys)} keys ✅")

        logging.info("Loading FAISS index")
        _index_product = faiss.read_index(products_index_path, faiss.IO_FLAG_MMAP)
        _index_extra_features = faiss.read_index(extra_features_index_path, faiss.IO_FLAG_MMAP)
        _index_categories = faiss.read_index(categories_index_path, faiss.IO_FLAG_MMAP)
        _index_images = faiss.read_index(images_index_path, faiss.IO_FLAG_MMAP)
        logging.info("FAISS index loaded ✅")

    ret_dict = {'index_product': _index_product, 'index_extra_features': _index_extra_features, 'index_categories': _index_categories, 'product_keys': _keys, 'category_keys': _categories_keys, 'images_keys': _images_keys, 'index_images': _index_images}
    return ret_dict
