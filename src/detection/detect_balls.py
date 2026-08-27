import os
from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS


def detect_balls(frame: np.ndarray, conf: float = 0.2, model_mode: str = "full") -> list[dict]:
    """Runs YOLO inference and extracts ball locations and classified IDs."""
    if model_mode not in ("full", "trial"):
        raise ValueError(f"'{model_mode}' model mode is not valid. Insert 'full' or 'trial'")


    weights_path = os.path.join("models", model_mode, "weights", "best.pt")
    model = YOLO(weights_path)

    results = model.predict(
        frame,
        conf=conf,              
        iou=0.45,               # Soglia NMS
        agnostic_nms=True,      # Evita box multipli sulla stessa identica palla
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
        save_ball_outputs(f"video{i}.mp4", conf=0.35, model_mode="full")

