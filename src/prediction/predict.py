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
from config import OUTPUT_DIR, VIDEOS, CANONICAL_W, CANONICAL_H, POCKETS_TOP
from detection.detect_balls import detect_balls
from detection.detect_cue import detect_cue, find_cue_ball
from detection.detect_table import detect_table
from prediction.trajectory import (
    CAPTURE_FACTOR,
    MAX_BOUNCES,
    MAX_DEPTH,
    cue_deflection,
    simulate_chain,
)


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