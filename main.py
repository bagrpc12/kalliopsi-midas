from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from model_utils import predict_depth

app = FastAPI(title="Kalliopsi MiDaS API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/depth")
async def depth_estimation(file: UploadFile = File(...)):
    image_bytes = await file.read()
    depth_map = predict_depth(image_bytes)

    return Response(
        content=depth_map.tobytes(),
        media_type="image/png"
    )
