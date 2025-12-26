from fastapi import FastAPI, UploadFile, File
from model_utils import load_midas_model
import cv2
import numpy as np
import torch
from PIL import Image
import io

app = FastAPI()

MODEL = load_midas_model()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "healthy"}


@app.post("/depth")
async def depth_estimation(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (384, 384))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        depth = MODEL(img)

    depth = depth.squeeze().cpu().numpy()

    return {
        "min_depth": float(depth.min()),
        "max_depth": float(depth.max())
    }
