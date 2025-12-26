from fastapi import FastAPI, UploadFile, File
from model_utils import load_midas_model, predict_depth

app = FastAPI()

# 🔥 GLOBAL MODEL (φορτώνεται ΜΙΑ ΦΟΡΑ)
model, transform = None, None

@app.on_event("startup")
def startup_event():
    global model, transform
    model, transform = load_midas_model()

@app.post("/depth")
async def depth_estimation(file: UploadFile = File(...)):
    image_bytes = await file.read()
    depth = predict_depth(image_bytes, model, transform)
    return {"depth": depth}
