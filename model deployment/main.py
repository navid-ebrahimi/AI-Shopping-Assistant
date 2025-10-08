from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
from io import BytesIO

app = FastAPI(title="Image Base64 Embedding API with CLIP")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

class EmbedRequest(BaseModel):
    base64_images: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dims: int

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/embed_image", response_model=EmbedResponse)
def embed_image(req: EmbedRequest):
    images = []
    for b64 in req.base64_images:
        try:
            img_data = base64.b64decode(b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = F.normalize(image_features, p=2, dim=1)

    return {"embeddings": image_features.cpu().tolist(), "dims": image_features.shape[1]}
