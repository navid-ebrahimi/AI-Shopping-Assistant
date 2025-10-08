from django.core.management.base import BaseCommand
from django.db import transaction
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
from tqdm import tqdm

from core.models import (
    Search, BaseView, FinalClick, BaseProduct,
    Member, Shop, Category, Brand, City
)


class Command(BaseCommand):
    help = "Import all parquet data into Postgres via Django ORM efficiently with progress bar"

    # -------------------- Helpers --------------------
    @staticmethod
    def _to_timestamp(ts):
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    @staticmethod
    def _safe_get(model, **kwargs):
        try:
            return model.objects.get(**kwargs)
        except model.DoesNotExist:
            return None

    @staticmethod
    def _load_parquet_chunks(path, chunksize=1000):
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            for batch in pf.read_row_group(row_group).to_batches(max_chunksize=chunksize):
                yield batch.to_pandas()

    # -------------------- Importers --------------------
    def import_categories(self):
        self.stdout.write(self.style.NOTICE("Starting import of Categories..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("categories.parquet"), desc="Categories"):
                objs = [
                    Category(id=row.id, title=row.title, parent_id=row.parent_id)
                    for row in chunk.itertuples(index=False)
                ]
                Category.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("Categories imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing Categories: {str(e)}"))
            raise

    def import_brands(self):
        self.stdout.write(self.style.NOTICE("Starting import of Brands..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("brands.parquet"), desc="Brands"):
                objs = [Brand(id=row.id, title=row.title) for row in chunk.itertuples(index=False)]
                Brand.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("Brands imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing Brands: {str(e)}"))
            raise

    def import_cities(self):
        self.stdout.write(self.style.NOTICE("Starting import of Cities..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("cities.parquet"), desc="Cities"):
                objs = [City(id=row.id, title=row.name) for row in chunk.itertuples(index=False)]
                City.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("Cities imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing Cities: {str(e)}"))
            raise

    def import_shops(self):
        self.stdout.write(self.style.NOTICE("Starting import of Shops..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("shops.parquet"), desc="Shops"):
                objs = [
                    Shop(
                        id=int(row.id),
                        city=self._safe_get(City, id=row.city_id),
                        score=float(row.score),
                        has_warranty=bool(row.has_warranty),
                    )
                    for row in chunk.itertuples(index=False)
                ]
                Shop.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("Shops imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing Shops: {str(e)}"))
            raise

    def import_baseproducts(self):
        self.stdout.write(self.style.NOTICE("Starting import of BaseProducts..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("base_products_embeddings.parquet"), desc="BaseProducts"):
                objs = []
                for row in chunk.itertuples(index=False):
                    # extra_features -> JSONField
                    extra_features = None
                    if isinstance(row.extra_features, str) and row.extra_features.strip() != "":
                        try:
                            extra_features = json.loads(row.extra_features)
                        except json.JSONDecodeError:
                            self.stdout.write(self.style.WARNING(
                                f"⚠️ Could not parse extra_features for random_key={row.random_key}"
                            ))
                            extra_features = None

                    objs.append(
                        BaseProduct(
                            random_key=row.random_key,
                            persian_name=row.persian_name,
                            english_name=row.english_name,
                            category=None if row.category_id == 0 else self._safe_get(Category, id=row.category_id),
                            brand=None if row.brand_id == -1 else self._safe_get(Brand, id=row.brand_id),
                            extra_features=extra_features,
                            image_url=row.image_url,
                        )
                    )
                if objs:
                    BaseProduct.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)

            self.stdout.write(self.style.SUCCESS("BaseProducts imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing BaseProducts: {str(e)}"))
            raise



    def import_members(self):
        self.stdout.write(self.style.NOTICE("Starting import of Members..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("members.parquet"), desc="Members"):
                objs = [
                    Member(
                        random_key=row.random_key,
                        base_product=self._safe_get(BaseProduct, random_key=row.base_random_key),
                        shop=self._safe_get(Shop, id=row.shop_id),
                        price=row.price,
                    )
                    for row in chunk.itertuples(index=False)
                ]
                Member.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("Members imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing Members: {str(e)}"))
            raise

    def import_searches(self, limit=None):
        self.stdout.write(self.style.NOTICE("Starting import of Searches..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("searches.parquet"), desc="Searches"):
                objs = []
                for idx, row in enumerate(chunk.itertuples(index=False)):
                    category = None if row.category_id == 0 else self._safe_get(Category, id=row.category_id)

                    result_rks = row.result_base_product_rks
                    if isinstance(result_rks, (pd.Series, np.ndarray)):
                        result_rks = result_rks.tolist()

                    category_boosts = getattr(row, "category_brand_boosts", None)
                    if isinstance(category_boosts, (pd.Series, np.ndarray)):
                        category_boosts = category_boosts.tolist()

                    objs.append(
                        Search(
                            id=row.id,
                            uid=row.uid,
                            query=row.query,
                            page=row.page,
                            timestamp=self._to_timestamp(row.timestamp),
                            session_id=row.session_id,
                            result_base_product_rks=result_rks,
                            category=category,
                            category_brand_boosts=category_boosts,
                        )
                    )

                    if limit and idx + 1 >= limit:
                        break

                if objs:
                    Search.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("Searches imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing Searches: {str(e)}"))
            raise


    def import_baseviews(self, limit=None):
        self.stdout.write(self.style.NOTICE("Starting import of BaseViews..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("base_views.parquet"), desc="BaseViews"):
                objs = []
                for idx, row in enumerate(chunk.itertuples(index=False)):
                    objs.append(
                        BaseView(
                            id=row.id,
                            search=self._safe_get(Search, id=row.search_id),
                            base_product_rk=row.base_product_rk,
                            timestamp=self._to_timestamp(row.timestamp),
                        )
                    )
                    if limit and idx + 1 >= limit:
                        break

                if objs:
                    BaseView.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("BaseViews imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing BaseViews: {str(e)}"))
            raise


    def import_finalclicks(self, limit=None):
        self.stdout.write(self.style.NOTICE("Starting import of FinalClicks..."))
        try:
            for chunk in tqdm(self._load_parquet_chunks("final_clicks.parquet"), desc="FinalClicks"):
                objs = []
                for idx, row in enumerate(chunk.itertuples(index=False)):
                    objs.append(
                        FinalClick(
                            id=row.id,
                            base_view=self._safe_get(BaseView, id=row.base_view_id),
                            shop=None if row.shop_id == 0 else self._safe_get(Shop, id=row.shop_id),
                            timestamp=self._to_timestamp(row.timestamp),
                        )
                    )
                    if limit and idx + 1 >= limit:
                        break

                if objs:
                    FinalClick.objects.bulk_create(objs, ignore_conflicts=True, batch_size=1000)
            self.stdout.write(self.style.SUCCESS("FinalClicks imported successfully!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing FinalClicks: {str(e)}"))
            raise


    # -------------------- Main Handle --------------------
    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.NOTICE("Starting data import process..."))
        # self.import_categories()
        # self.import_brands()
        # self.import_cities()
        # self.import_shops()
        # self.import_baseproducts()
        self.import_members()
        # self.import_searches()
        # self.import_baseviews()
        # self.import_finalclicks()
        self.stdout.write(self.style.SUCCESS("✅ All parquet data imported successfully!"))