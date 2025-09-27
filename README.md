# AI Shopping Assistant

An intelligent AI-powered shopping assistant capable of handling diverse user intents—product search, Q&A, price comparison, seller recommendation, and image-based queries—via a conversational interface. The system guides users efficiently toward their target product or seller.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [API](#api)
- [Scenarios](#scenarios)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Deployment & CI/CD](#deployment--cicd)
- [Examples](#examples)
- [License](#license)

---

## Problem Statement
The AI assistant interacts with users through chat and helps them achieve their goals. User intents may include:
- Searching for a specific product
- Asking questions about products or sellers
- Comparing products
- Finding products based on images  

The assistant processes structured e-commerce logs, maps queries to relevant products, and returns appropriate recommendations.

---

## Dataset
The competition dataset includes the following tables:

### Searches
Logs of search result pages.
- `id`, `uid`, `query`, `page`, `timestamp`, `session_id`, `result_base_product_rks`, `category_id`, `category_brand_boosts`

### Base Views
Logs of user clicks on base products.
- `id`, `search_id`, `base_product_rk`, `timestamp`

### Final Click
Final product clicks.
- `id`, `base_view_id`, `shop_id`, `timestamp`

### Base Products
- `random_key`, `persian_name`, `english_name`, `category_id`, `brand_id`, `extra_features`, `image_url`, `members`

### Members (Shop Products)
- `random_key`, `base_random_key`, `shop_id`, `price`

### Shops
- `id`, `city_id`, `score`, `has_warranty`

### Categories
- `id`, `title`, `parent_id`

### Brands
- `id`, `title`

### Cities
- `id`, `name`

---

## API

### Endpoint
`POST /chat`

### Request
```json
{
  "chat_id": "string",
  "messages": [
    {
      "type": "text | image",
      "content": "string (text or base64 image)"
    }
  ]
}
```

### Response Format

```json
{
  "message": "string | null",
  "base_random_keys": ["string"] | null,
  "member_random_keys": ["string"] | null
}
```

- **message**: optional textual response
- **base_random_keys**: optional list of base product keys (max 10)
- **member_random_keys**: optional list of shop product keys (max 10)

## Scenarios

- **Scenario 0**: System connection test (ping → pong)
- **Scenario 1**: Exact product search (return base_random_keys)
- **Scenario 2**: Attribute query (return product feature in message)
- **Scenario 3**: Seller-related question (return response in message)
- **Scenario 4**: Seller-focused dialogue (multi-turn Q&A to find product, return member_random_keys)
- **Scenario 5**: Product comparison (return reason and base_random_keys)
- **Scenario 6**: Image object recognition (return main object name in message)
- **Scenario 7**: Image-to-product mapping (return best base_random_keys)

## Tech Stack

- **Backend**: Django REST framework (structured data processing, database access)
- **AI Model Serving**: FastAPI (for conversational agent and image handling)
- **Database**: PostgreSQL
- **Vector Search**: FAISS (for embedding similarity search)
- **Deployment**: Scalable cloud cluster (Hamravesh)
- **CI/CD**: GitHub Actions pipeline for automated testing and deployment

## Architecture

1. **User Query Processing**: Accepts text or image input
2. **Query Understanding**: Maps queries to base products, sellers, or features
3. **Dialogue Management**: Multi-turn interaction for complex queries
4. **Product Retrieval & Ranking**: Uses FAISS embeddings to find similar products
5. **Response Generation**: Constructs user-friendly messages and returns product keys

## Deployment & CI/CD

- Backend services deployed on cloud cluster
- GitLab Actions for:
  - Linting and unit testing
  - Automatic deployment of FastAPI and Django services
  - PostgreSQL migrations

## Examples

### Text Query

**Request:**
```json
{
  "chat_id": "a1b2c3",
  "messages": [{"type": "text", "content": "سلام، یک گوشی آیفون ۱۶ پرومکس میخوام"}]
}
```

**Response:**
```json
{
  "message": "این گوشی رو به شما پیشنهاد میکنم.",
  "base_random_keys": ["awsome-gooshi-rk"],
  "member_random_keys": null
}
```

### Multi-turn Seller Dialogue (Scenario 4)

**Request:**
```json
{
  "chat_id": "q4-abc",
  "messages": [{"type": "text", "content": "میخوام یه میز تحریر برای کارهای روزمره پیدا کنم."}]
}
```

**Response:**
```json
{
  "message": null,
  "base_random_keys": null,
  "member_random_keys": ["xpjtgd"]
}
```

## License

MIT License