import os
import requests
import torch
import cv2
import numpy as np

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# -------- CONFIG --------
MODEL_URL = "https://huggingface.co/intel-isl/MiDaS/resolve/main/dpt_large_384.pt"
MODEL_PATH = "/tmp/dpt_large_384.pt"
DEVICE = "cpu"
# ------------------------


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MiDaS model...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    return MODEL_PATH


def load_midas_model():
    model = DPTDepthModel(
        path=None,
        backbone="vitl16_384",
        non_negative=True,
    )

    path = ensure_model()
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    transform = torch.nn.Sequential()  # placeholder
    return model


def predict_depth(image_bytes: bytes):
    model = load_midas_model()

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    transform = ComposeTransforms()
    sample = transform(image)
    input_tensor = torch.from_numpy(sample).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = model(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype("uint8")
    return depth


class ComposeTransforms:
    def __init__(self):
        self.transforms = [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]

    def __call__(self, img):
        for t in self.transforms:
            img = t({"image": img})["image"]
        return img
