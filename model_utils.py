import os
import torch
import requests

HF_MODEL_URL = "https://huggingface.co/halffried/midas_v3_1_dpt_swin2_large_384/resolve/main/dpt_swin2_large_384.pt"

CACHE_DIR = "/tmp/midas_models"
MODEL_FILENAME = "dpt_swin2_large_384.pt"
MODEL_PATH = os.path.join(CACHE_DIR, MODEL_FILENAME)


def ensure_model():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading MiDaS model...")
        r = requests.get(HF_MODEL_URL, stream=True)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return MODEL_PATH


def load_midas_model(device="cpu"):
    model = torch.hub.load(
        "isl-org/MiDaS",
        "DPT_Swin2_Large_384",
        pretrained=False
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
