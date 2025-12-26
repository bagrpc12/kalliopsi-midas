import os
import torch
import requests

# URL του μοντέλου από Hugging Face
HF_MODEL_URL = "https://huggingface.co/halffried/midas_v3_1_dpt_swin2_large_384/blob/main/dpt_swin2_large_384.pt"
MODEL_FILENAME = "dpt_swin2_large_384.pt"
CACHE_DIR = "/tmp/midas_models"

# Τελική διαδρομή για το αρχείο μοντέλου
MODEL_PATH = os.path.join(CACHE_DIR, MODEL_FILENAME)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MiDaS model...")
        os.makedirs(CACHE_DIR, exist_ok=True)
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

