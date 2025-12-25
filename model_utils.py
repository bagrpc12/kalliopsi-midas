import os
import torch
import requests
import zipfile

# URL του μοντέλου από Hugging Face
HF_MODEL_URL = "URL = "https://huggingface.co/intel-isl/MiDaS/resolve/main/dpt_hybrid-d0508457.pt"
MODEL_FILENAME = "dpt_hybrid-d0508457.pt"
CACHE_DIR = "/tmp/midas_models"

def ensure_model():
    """Κατεβάζει το μοντέλο αν δεν υπάρχει."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_file = os.path.join(CACHE_DIR, MODEL_FILENAME)
    if not os.path.isfile(model_file):
        print("Downloading MiDaS model...")
        with requests.get(HF_MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return model_file

def load_midas_model(path, device):
    """Φορτώνει το DPT Swin2 Tiny MiDaS μοντέλο."""
    model = torch.hub.load("isl-org/MiDaS", "DPT_Swin2_Tiny", pretrained=False)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



