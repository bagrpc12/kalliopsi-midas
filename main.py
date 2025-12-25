from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as T
import os

from model_utils import load_midas_model, ensure_model

app = FastAPI(title="Kalliopsi MiDaS API")

# βεβαιωνόμαστε ότι υπάρχει το μοντέλο τοπικά
MODEL = ensure_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = load_midas_model(MODEL, device)
transform = T.Compose([
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

@app.post("/depth")
async def depth(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = img.size

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = midas(input_tensor)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = pred.cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, buf = cv2.imencode(".png", depth)
    b64 = base64.b64encode(buf).decode("utf-8")
    return JSONResponse({"depth_map": b64})

@app.get("/health")
def health():
    return {"status": "ok"}

