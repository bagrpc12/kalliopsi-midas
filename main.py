from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import numpy as np
import cv2

from model_utils import predict_depth

app = FastAPI(title="MiDaS Depth API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/depth")
async def depth_estimation(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    depth = predict_depth(image)

    # Normalize to 0–255 for visualization
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    _, encoded = cv2.imencode(".png", depth_uint8)
    return Response(content=encoded.tobytes(), media_type="image/png")
