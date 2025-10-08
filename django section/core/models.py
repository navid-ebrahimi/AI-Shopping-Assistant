from django.db import models

class Category(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    title = models.CharField(max_length=255)
    parent_id = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.title

class Search(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    uid = models.CharField(max_length=100)
    query = models.TextField()
    page = models.IntegerField()
    timestamp = models.DateTimeField()
    session_id = models.CharField(max_length=100)
    result_base_product_rks = models.JSONField()
    category = models.ForeignKey(
        Category, on_delete=models.SET_NULL, null=True, blank=True
    )
    category_brand_boosts = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.id} - {self.query[:30]}"



class BaseView(models.Model):
    id = models.CharField(primary_key=True, max_length=100) 
    search = models.ForeignKey(Search, on_delete=models.CASCADE, null=True, blank=True)
    base_product_rk = models.CharField(max_length=100)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.id} ({self.base_product_rk})"

class City(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)

    def __str__(self):
        return self.title
    

class Shop(models.Model):
    id = models.AutoField(primary_key=True)
    city = models.ForeignKey(City, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField()
    has_warranty = models.BooleanField(default=False)

    def __str__(self):
        return f"Shop {self.id} - score {self.score}"

class FinalClick(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    base_view = models.ForeignKey(BaseView, on_delete=models.CASCADE, null=True, blank=True)
    shop = models.ForeignKey(Shop, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.id} - shop {self.shop_id}"


class Brand(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)

    def __str__(self):
        return self.title


class BaseProduct(models.Model):
    random_key = models.CharField(max_length=1024, primary_key=True)
    persian_name = models.TextField()
    english_name = models.TextField()
    category = models.ForeignKey(
        Category, on_delete=models.SET_NULL, null=True, blank=True
    )
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE, null=True, blank=True)
    extra_features = models.JSONField(null=True, blank=True)
    image_url = models.TextField(null=True)

    def __str__(self):
        return self.persian_name


class Member(models.Model):
    random_key = models.CharField(max_length=100, primary_key=True)
    base_product = models.ForeignKey(BaseProduct, on_delete=models.CASCADE, related_name="members", null=True, blank=True)
    shop = models.ForeignKey(Shop, on_delete=models.CASCADE, null=True, blank=True)
    price = models.DecimalField(max_digits=25, decimal_places=2)

    def __str__(self):
        return f"{self.random_key} - {self.price}"


class Chat(models.Model):
    id = models.AutoField(primary_key=True)
    chat_id = models.TextField(unique=True) 
    messages = models.JSONField(default=list)  # List of user messages
    responses = models.JSONField(default=list)  # List of responses
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Chat {self.chat_id} ({len(self.messages)} messages)"

    def add_interaction(self, message, response):
        """Helper method to add a new message-response pair"""
        self.messages.append(message)
        self.responses.append(response)
        self.save()

    def get_conversation_history(self):
        """Get the entire conversation history as a list of tuples"""
        return list(zip(self.messages, self.responses))