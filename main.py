from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np

from model_utils import predict_depth

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/depth")
async def depth_estimation(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    depth = predict_depth(image)

    return {
        "depth_shape": depth.shape,
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
    }
