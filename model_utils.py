import os
import torch
import requests

HF_MODEL_URL = (
    "https://huggingface.co/halffried/midas_v3_1_dpt_swin2_large_384/"
    "resolve/bff19fa1d6bc502560dc02c7d93d58bf5da12104/"
    "dpt_swin2_large_384.pt"
)

CACHE_DIR = "/tmp/midas"
MODEL_PATH = os.path.join(CACHE_DIR, "dpt_swin2_large_384.pt")


def ensure_model():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading MiDaS weights...")
        r = requests.get(HF_MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    return MODEL_PATH


def load_midas_model():
    device = torch.device("cpu")

    model = torch.hub.load(
        "isl-org/MiDaS",
        "DPT_Swin2_Large_384",
        pretrained=False
    )

    state = torch.load(ensure_model(), map_location=device)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model
