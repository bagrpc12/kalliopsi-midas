import os
import torch
import requests
import zipfile

# URL του μοντέλου από Hugging Face
HF_MODEL_URL =  "https://huggingface.co/intel-isl/MiDaS/resolve/main/dpt_hybrid-d0508457.pt"
MODEL_FILENAME = "dpt_hybrid-d0508457.pt"
CACHE_DIR = "/tmp/midas_models"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MiDaS model...")
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        r = requests.get(HF_MODEL_URL, headers=headers)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    return MODEL_PATH

def load_midas_model(path, device):
    """Φορτώνει το DPT Swin2 Tiny MiDaS μοντέλο."""
    model = torch.hub.load("isl-org/MiDaS", "DPT_Swin2_Tiny", pretrained=False)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model





