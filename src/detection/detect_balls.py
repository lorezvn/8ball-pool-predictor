import os
from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS

# Loading the weights is expensive; keep one instance per mode alive instead of
# re-reading them from disk on every frame.
_MODELS: dict[str, YOLO] = {}

# Mapping model to imgsz used in training
MODEL_RESOLUTIONS = {
    "trial": 640,
    "full": 640,
    "full_1280": 1280,
}


def _weights_path(model_mode: str) -> str:
    """Path of the weights for a run name, e.g. "full" -> models/full/weights/best.pt."""
    return os.path.join("models", model_mode, "weights", "best.pt")


def _get_model(model_mode: str) -> YOLO:
    """Returns the cached YOLO model for the given mode, loading it on first use."""
    if model_mode not in _MODELS:
        _MODELS[model_mode] = YOLO(_weights_path(model_mode))
    return _MODELS[model_mode]


def detect_balls(frame: np.ndarray, conf: float = 0.2, model_mode: str = "full") -> list[dict]:
    """Runs YOLO inference and extracts ball locations and classified IDs."""

    if not os.path.isfile(_weights_path(model_mode)):
        available = sorted(
            d for d in os.listdir("models")
            if os.path.isfile(_weights_path(d))
        ) if os.path.isdir("models") else []
        raise ValueError(
            f"No weights at {_weights_path(model_mode)}. Available: {available or 'none'}"
        )

    model = _get_model(model_mode)

    infer_imgsz = MODEL_RESOLUTIONS.get(model_mode, 640)

    results = model.predict(
        frame,
        conf=conf,              
        iou=0.45,               # NMS IoU threshold: suppresses overlapping boxes with >45% area overlap
        imgsz=infer_imgsz,      # must match the training resolution
        agnostic_nms=True,      # Avoid multiple box on the same ball
        verbose=False,
    )[0]

    balls = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].int().tolist()
        cls_id = int(b.cls[0])
        label = model.names[cls_id] if model.names else str(cls_id)
        confidence = round(float(b.conf[0]), 2)

        balls.append({
            "box": (x1, y1, x2, y2),
            "center": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32),
            "radius": float((x2 - x1 + y2 - y1) / 4.0),
            "label": label,
            "conf": confidence,
        })

    return balls


def draw_ball_overlay(frame: np.ndarray, balls: list[dict]) -> np.ndarray:
    """Renders detected balls and predicted numbers onto the frame."""
    overlay = frame.copy()
    for ball in balls:
        x1, y1, x2, y2 = ball["box"]
        label = ball["label"]
        confidence = ball["conf"]
        color = (0, 255, 0) if label == "0" else (0, 0, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} ({confidence})",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    return overlay


def save_ball_outputs(
    video_filename: str,
    frame_index: int = 0,
    conf: float = 0.2,
    model_mode: str = "full",
):
    """Loads a frame, executes ball detection, and writes output overlay."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
    overlay = draw_ball_overlay(frame, balls)

    out_dir = os.path.join(OUTPUT_DIR, "table", "balls")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_f{frame_index}"

    cv2.imwrite(os.path.join(out_dir, f"{model_mode}_{conf}_balls_{tag}.png"), overlay)
    print(f"[{video_filename}] Detected {len(balls)} balls. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    for i in range(2, 6):
        save_ball_outputs(f"video{i}.mp4", frame_index=100, model_mode="full_1280")

