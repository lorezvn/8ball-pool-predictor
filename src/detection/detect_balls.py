import os
from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS

def _whiteness_score(frame: np.ndarray, box: tuple) -> float:
    """Computes a whiteness metric (high value, low saturation) on the central patch
    of the bounding box to ignore background cloth pixels at the corners.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h_box, w_box = y2 - y1, x2 - x1

    # Sample only the central 50% area of the bounding box
    cx1, cx2 = x1 + int(w_box * 0.25), x2 - int(w_box * 0.25)
    cy1, cy2 = y1 + int(h_box * 0.25), y2 - int(h_box * 0.25)

    roi = frame[max(0, cy1) : max(0, cy2), max(0, cx1) : max(0, cx2)]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0

    return float(np.mean(v * (1.0 - s)))


def enforce_uniqueness(balls: list[dict]) -> list[dict]:
    """Keeps at most one detection per class label (highest confidence wins);
    duplicates are relabeled '?' rather than dropped, so they still count
    as anonymous obstacles for the geometry."""
    best_by_label: dict[str, int] = {}  # label -> index of best conf so far
    for i, ball in enumerate(balls):
        label = ball["label"]
        if label not in best_by_label or ball["conf"] > balls[best_by_label[label]]["conf"]:
            best_by_label[label] = i

    keep_indices = set(best_by_label.values())
    for i, ball in enumerate(balls):
        if i not in keep_indices:
            ball["label"] = "?"
    return balls


def enforce_cue_ball(frame: np.ndarray, balls: list[dict], min_score: float = 0.40) -> list[dict]:
    """Forces exactly one detection to be the cue ball ('0') chosen by whiteness metric.
    Demotes conflicting or duplicate '0' labels to '?'.
    """
    if not balls:
        return balls

    scores = [_whiteness_score(frame, b["box"]) for b in balls]
    best_idx = int(np.argmax(scores))

    for i, ball in enumerate(balls):
        if i == best_idx and scores[best_idx] >= min_score:
            ball["label"] = "0"
        elif ball.get("label") == "0":
            ball["label"] = "?"

    return balls


def detect_balls(frame: np.ndarray, conf: float = 0.2, model_mode: str = "full") -> list[dict]:
    """Runs YOLO inference and extracts ball locations and classified IDs."""
    if model_mode not in ("full", "trial"):
        raise ValueError(f"'{model_mode}' model mode is not valid. Insert 'full' or 'trial'")


    weights_path = os.path.join("models", model_mode, "weights", "best.pt")
    model = YOLO(weights_path)

    results = model.predict(
        frame,
        conf=conf,              
        iou=0.45,               # NMS IoU threshold: suppresses overlapping boxes with >45% area overlap
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

    #balls = enforce_uniqueness(detect_balls(frame, conf=conf, model_mode=model_mode))
    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
    overlay = draw_ball_overlay(frame, balls)

    out_dir = os.path.join(OUTPUT_DIR, "table", "balls")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_f{frame_index}"

    cv2.imwrite(os.path.join(out_dir, f"{model_mode}_{conf}_balls_{tag}.png"), overlay)
    print(f"[{video_filename}] Detected {len(balls)} balls. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    for i in range(2, 6):
        save_ball_outputs(f"video{i}.mp4", frame_index=100, model_mode="full")

