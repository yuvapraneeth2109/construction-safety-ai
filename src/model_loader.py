from ultralytics import YOLO
from src.config import DEVICE
import os

MODEL_PATHS = {
    "harness": "models/ppeharness.pt",
    "lanyard": "models/lanryardrope.pt",
    "edge": "models/edgee.pt",
    "lanyard_v2": "models/YP.pt",
    "hook_primary": "models/hook.pt",
    "hook_secondary": "models/hook2.pt"
}


def load_models():
    models = {}

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        model = YOLO(path)
        model.to(DEVICE)
        model.fuse()

        models[name] = model

    print("All models loaded successfully on:", DEVICE)
    return models
