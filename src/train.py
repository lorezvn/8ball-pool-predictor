import os

import torch
from ultralytics import YOLO

from .config import MERGED_DIR
from .dataset import check_dataset

TRIAL_CFG = dict(model="yolov8n.pt", epochs=3, imgsz=640, batch=16, name="trial")
FULL_CFG = dict(model="yolov8s.pt", epochs=80, imgsz=640, batch=16, name="full")
FULL_1280_CFG = dict(model="yolov8s.pt", epochs=150, imgsz=1280, batch=-1, name="full_1280")


def train_model(full: bool = False) -> None:
    """
    Train the YOLO ball-detection model on the merged dataset.

    Results are saved under models/<name>/weights/best.pt.

    Args:
        full: If True, run the full training. If False, run the
            quick sanity check instead.
    """
    check_dataset(MERGED_DIR)

    cfg = FULL_1280_CFG if full else TRIAL_CFG
    model_name, epochs, imgsz, batch, name = cfg.values()
    data_yaml = os.path.join(MERGED_DIR, "data.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== TRAINING ({'FULL' if full else 'QUICK TEST'}) ===")
    print(f"  device: {device}")
    print(f"  model:  {model_name}")
    print(f"  epochs: {epochs}")
    print(f"  data:   {data_yaml}")

    pretrained_path = os.path.join("models", "pretrained", model_name)
    os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
    model = YOLO(pretrained_path)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,                          # alzate i workers, sul mio pc dava warning
        name=name,
        exist_ok=True,
        project=os.path.abspath("models"),  # absolute path: avoids ultralytic merging it with its own runs_dir
    )