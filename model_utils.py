import torch
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import cv2
import numpy as np
from torchvision.transforms import Compose

MODEL_PATH = "dpt_swin2_large_384.pt"

_device = torch.device("cpu")
_model = None
_transform = None


def load_midas_model():
    global _model, _transform

    if _model is not None:
        return _model, _transform

    # Load model architecture (LOCAL midas/)
    model = DPTDepthModel(
        path=None,
        backbone="swin2_large_384",
        non_negative=True,
    )

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=_device)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    transform = Compose(
        [
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
    )

    _model = model
    _transform = transform
    return _model, _transform


def predict_depth(image_bgr: np.ndarray) -> np.ndarray:
    model, transform = load_midas_model()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform({"image": image_rgb})["image"]
    input_batch = torch.from_numpy(input_batch).unsqueeze(0).to(_device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return depth
