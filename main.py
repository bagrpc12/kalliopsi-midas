from fastapi import FastAPI, UploadFile, File
from model_utils import ensure_model, load_midas_model
import torch
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        model_path = ensure_model()
        MODEL = load_midas_model(model_path)
    return MODEL


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/depth")
async def depth_estimation(file: UploadFile = File(...)):
    model = get_model()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    image = cv2.resize(image, (384, 384))
    image = image / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        prediction = model(image)
        depth = prediction.squeeze().cpu().numpy()

    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth = (depth * 255).astype(np.uint8)

    _, buffer = cv2.imencode(".png", depth)
    return buffer.tobytes()

