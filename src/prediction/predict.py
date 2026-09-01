"""Full shot prediction on a frame: detect, simulate, draw.

Ties the detection modules to the geometry in trajectory.py. The detections live
in image coordinates, the simulation lives in the rectified top-down view, and
the resulting paths are projected back onto the frame for the overlay.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS
from detection.detect_balls import detect_balls
from detection.detect_cue import detect_cue, find_cue_ball
from detection.detect_table import CANONICAL_W, CANONICAL_H, detect_table
from prediction.trajectory import (
    CAPTURE_FACTOR,
    MAX_BOUNCES,
    MAX_DEPTH,
    POCKETS_TOP,
    cue_deflection,
    simulate_chain,
)

# Standard ball colours for the schematic 2D map (BGR). 9-15 reuse the colour of
# 1-7 with a white ring drawn on top.
BALL_BGR = {
    "0": (255, 255, 255), "1": (0, 215, 255), "2": (200, 0, 0), "3": (0, 0, 220),
    "4": (150, 30, 140), "5": (0, 140, 255), "6": (0, 150, 0), "7": (40, 40, 140),
    "8": (20, 20, 20), "?": (130, 130, 130),
}

CHAIN_PALETTE = [(0, 200, 0), (255, 200, 0), (255, 0, 200), (0, 200, 255)]
CUE_PATH_BGR = (0, 0, 255)
DEFLECTION_BGR = (200, 200, 200)


def base_color(label: str) -> tuple:
    """Colour of a ball, mapping stripes 9-15 onto their solid counterparts."""
    if label in BALL_BGR:
        return BALL_BGR[label]
    number = int(label)
    return BALL_BGR.get(str(number - 8), (130, 130, 130)) if number > 8 else (130, 130, 130)


def outcome_text(pred: dict) -> str:
    """One-line description of the predicted outcome."""
    if pred["outcome"] == "IN":
        return f"IN: ball {pred['potted_label']}"
    return pred["outcome"]


def run_prediction(
    frame: np.ndarray,
    conf: float = 0.3,
    model_mode: str = "full",
    max_bounces: int = MAX_BOUNCES,
    max_depth: int = MAX_DEPTH,
    capture_factor: float = CAPTURE_FACTOR,
    table: dict | None = None,
    balls: list | None = None,
) -> dict | None:
    """Detect the table, balls and cue, then simulate the shot.

    Args:
        table: Result of detect_table, when the caller already has it. The table
            does not move, so a video loop should detect it once and pass it in:
            it saves the per-frame cloth segmentation and avoids the mask
            breaking on frames where a player leans over the cloth.
        balls: Result of detect_balls, to avoid running the model twice.

    Returns:
        A dict with the outcome and everything needed to draw it, or None if the
        cue ball or the cue stick could not be found.
    """
    if table is None:
        table = detect_table(frame)
    matrix = table["transform_matrix"]
    matrix_inv = np.linalg.inv(matrix)

    def to_top(point) -> np.ndarray:
        pt = np.array([[list(point)]], dtype=np.float32)
        return cv2.perspectiveTransform(pt, matrix)[0, 0].astype(np.float64)

    if balls is None:
        balls = detect_balls(frame, conf=conf, model_mode=model_mode)
    cue_ball = find_cue_ball(balls)
    if cue_ball is None:
        return None

    cue = detect_cue(frame, cue_ball["center"], table_mask=table["mask"])
    if cue is None:
        return None

    others = [b for b in balls if b is not cue_ball]
    others_top = [to_top(b["center"]) for b in others]

    start = to_top(cue_ball["center"])
    # A homography does not act linearly on directions, so the direction has to
    # be transformed as the difference between two transformed points.
    direction = to_top(cue_ball["center"] + cue["direction"] * 200.0) - start

    # Radius in the top-down view: the median over every ball is far steadier
    # than trusting the cue ball's own box.
    def radius_top(ball) -> float:
        offset = np.array([ball["radius"], 0.0], dtype=np.float32)
        return float(np.linalg.norm(to_top(ball["center"] + offset) - to_top(ball["center"])))

    radius = float(np.median([radius_top(b) for b in balls]))
    capture = capture_factor * radius

    segments, potted_index = simulate_chain(
        start, direction, radius, others_top, capture,
        max_depth=max_depth, max_bounces=max_bounces,
    )
    deflection, scratches = cue_deflection(
        segments, others_top, radius, capture, max_bounces
    )

    if potted_index is None:
        outcome, potted_label = "OUT", None
    elif potted_index == -1:
        outcome, potted_label = "SCRATCH", None
    else:
        outcome, potted_label = "IN", others[potted_index]["label"]

    return {
        "balls": balls,
        "others": others,
        "cue_ball": cue_ball,
        "cue": cue,
        "segments": segments,
        "chain": [others[i]["label"] for _, i in segments if i is not None],
        "potted_index": potted_index,
        "potted_label": potted_label,
        "outcome": outcome,
        "deflection": deflection,
        "scratches": scratches,
        "matrix_inv": matrix_inv,
        "to_top": to_top,
        "radius_top": radius,
        "capture": capture,
    }


def _draw_path(overlay: np.ndarray, path_top, matrix_inv, color, thickness=3) -> None:
    """Project a top-down path back onto the frame and stroke it."""
    pts = cv2.perspectiveTransform(np.array([path_top], dtype=np.float32), matrix_inv)[0]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(overlay, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)


def draw_chain_paths(overlay: np.ndarray, pred: dict) -> None:
    """Stroke the predicted chain and the cue ball deflection onto the overlay."""
    for path, ball_index in pred["segments"]:
        color = CUE_PATH_BGR if ball_index is None else CHAIN_PALETTE[ball_index % len(CHAIN_PALETTE)]
        _draw_path(overlay, path, pred["matrix_inv"], color, 3)

    if pred["deflection"] is not None:
        _draw_path(overlay, pred["deflection"], pred["matrix_inv"], DEFLECTION_BGR, 2)


def draw_outcome_banner(overlay: np.ndarray, text: str, positive: bool) -> None:
    """Write the outcome in the top-left corner."""
    cv2.putText(overlay, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(overlay, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (0, 200, 0) if positive else (0, 0, 255), 4, cv2.LINE_AA)


def draw_prediction_overlay(frame: np.ndarray, pred: dict) -> np.ndarray:
    """Draw the predicted chain, the cue ball deflection and the outcome."""
    overlay = frame.copy()
    draw_chain_paths(overlay, pred)

    for ball in pred["balls"]:
        x, y = int(ball["center"][0]), int(ball["center"][1])
        color = (255, 255, 255) if ball["label"] == "0" else (0, 255, 255)
        cv2.putText(overlay, ball["label"], (x - 10, y - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, ball["label"], (x - 10, y - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if pred["potted_index"] is not None and pred["potted_index"] >= 0:
        potted = pred["others"][pred["potted_index"]]
        cv2.circle(overlay, tuple(potted["center"].astype(int)),
                   int(potted["radius"]) + 6, (0, 200, 0), 3, cv2.LINE_AA)

    text = outcome_text(pred)
    if pred["scratches"] and pred["outcome"] != "SCRATCH":
        text += " + SCRATCH RISK"
    draw_outcome_banner(overlay, text, pred["outcome"] == "IN")
    return overlay


def draw_2d_map(pred: dict, margin: int = 70) -> np.ndarray:
    """Schematic top-down map: table, pockets, numbered balls and the chain."""
    w, h = int(CANONICAL_W), int(CANONICAL_H)
    canvas = np.full((h + 2 * margin, w + 2 * margin, 3), (70, 70, 70), np.uint8)

    cv2.rectangle(canvas, (margin, margin), (margin + w, margin + h), (150, 110, 40), -1)
    cv2.rectangle(canvas, (margin, margin), (margin + w, margin + h), (40, 40, 40), 4)
    for px, py in POCKETS_TOP:
        cv2.circle(canvas, (int(px) + margin, int(py) + margin), 20, (25, 25, 25), -1)

    def stroke(path, color, thickness):
        pts = [(int(p[0]) + margin, int(p[1]) + margin) for p in path]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)

    for path, ball_index in pred["segments"]:
        color = CUE_PATH_BGR if ball_index is None else CHAIN_PALETTE[ball_index % len(CHAIN_PALETTE)]
        stroke(path, color, 2)
    if pred["deflection"] is not None:
        stroke(pred["deflection"], DEFLECTION_BGR, 2)

    radius = max(10, int(pred["radius_top"]))
    for ball in pred["balls"]:
        centre = pred["to_top"](ball["center"])
        c = (int(centre[0]) + margin, int(centre[1]) + margin)
        cv2.circle(canvas, c, radius, base_color(ball["label"]), -1)
        if ball["label"] not in ("0", "?") and int(ball["label"]) > 8:
            cv2.circle(canvas, c, radius, (255, 255, 255), 2)     # stripe ring
        cv2.circle(canvas, c, radius, (0, 0, 0), 1)
        cv2.putText(canvas, ball["label"], (c[0] - 7, c[1] - radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    text = outcome_text(pred)
    cv2.putText(canvas, text, (margin, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 200, 0) if pred["outcome"] == "IN" else (0, 0, 255), 3, cv2.LINE_AA)
    return canvas


def save_prediction_outputs(
    video_filename: str,
    frame_index: int = 0,
    conf: float = 0.3,
    model_mode: str = "full",
    max_bounces: int = MAX_BOUNCES,
):
    """Predict the shot on one frame and write the overlay and the 2D map."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    pred = run_prediction(frame, conf=conf, model_mode=model_mode, max_bounces=max_bounces)
    if pred is None:
        print(f"[{video_filename} f{frame_index}] cue ball or cue stick not found, skipped.")
        return None

    out_dir = os.path.join(OUTPUT_DIR, "prediction")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_f{frame_index}"

    cv2.imwrite(os.path.join(out_dir, f"predict_{tag}.png"), draw_prediction_overlay(frame, pred))
    cv2.imwrite(os.path.join(out_dir, f"map_{tag}.png"), draw_2d_map(pred))

    chain = " -> ".join(pred["chain"]) if pred["chain"] else "no ball hit"
    scratch = " | scratch risk" if pred["scratches"] else ""
    print(f"[{video_filename} f{frame_index}] {pred['outcome']} | chain: cue -> {chain}"
          f"{scratch} | saved in {out_dir}")
    return pred


if __name__ == "__main__":
    save_prediction_outputs("video8.mp4", frame_index=20)
