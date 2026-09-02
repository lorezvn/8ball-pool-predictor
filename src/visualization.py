import cv2
import numpy as np
import sys 
from pathlib import Path

# Import color maps and dimensions from your centralized config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    BALL_BGR, CHAIN_PALETTE, CUE_PATH_BGR, DEFLECTION_BGR,
    CANONICAL_W, CANONICAL_H, POCKETS_TOP
)

# --- 1. Base Entity Overlays ---
def draw_table_overlay(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draws the detected bounding polygon and labeled corner vertices."""
    overlay = frame.copy()
    pts = corners.astype(int)

    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
    for (x, y), label in zip(pts, ["TL", "TR", "BR", "BL"]):
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(overlay, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return overlay


def draw_pockets_overlay(frame: np.ndarray, pockets: list[dict]) -> np.ndarray:
    """Renders the 6 detected pocket positions onto the frame."""
    overlay = frame.copy()
    for p in pockets:
        cx, cy = int(p["center"][0]), int(p["center"][1])
        cv2.circle(overlay, (cx, cy), 30, (0, 0, 255), 3) # CAPTURE_R replaced with 30 for visualization
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)
    return overlay


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


def draw_cue_overlay(
    frame: np.ndarray,
    cue_center: np.ndarray,
    cue_data: dict | None,
    arrow_length: float = 140.0,
    color_bgr: tuple = (0, 220, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Renders the cue stick axis and dashed aiming guide with a solid arrowhead."""
    overlay = frame.copy()
    if cue_data is None:
        return overlay

    direction = cue_data["direction"]
    cx, cy = float(cue_center[0]), float(cue_center[1])

    # Cue stick physical segment
    p1, p2 = cue_data["endpoints"]
    cv2.line(overlay, tuple(p1.astype(int)), tuple(p2.astype(int)), color_bgr, 2, cv2.LINE_AA)

    # Dashed aiming guide ray
    start_pt = np.array([cx, cy], dtype=np.float32)
    end_pt = start_pt + direction * arrow_length

    num_dashes = 12
    for i in range(num_dashes):
        t1 = i / num_dashes
        t2 = (i + 0.6) / num_dashes
        pa = tuple((start_pt + t1 * (end_pt - start_pt)).astype(int))
        pb = tuple((start_pt + t2 * (end_pt - start_pt)).astype(int))

        cv2.line(overlay, pa, pb, (20, 20, 20), thickness + 2, cv2.LINE_AA)
        cv2.line(overlay, pa, pb, color_bgr, thickness, cv2.LINE_AA)

    # Solid arrowhead (filled triangle)
    head_len, head_width = 16.0, 10.0
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    base = end_pt - direction * head_len

    triangle = np.array(
        [
            end_pt,
            base + perp * (head_width / 2.0),
            base - perp * (head_width / 2.0),
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(overlay, [triangle], color_bgr, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [triangle], isClosed=True, color=(20, 20, 20), thickness=1, lineType=cv2.LINE_AA)

    # Cue ball contact center indicator
    cv2.circle(overlay, (int(cx), int(cy)), 4, color_bgr, -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (int(cx), int(cy)), 6, (20, 20, 20), 1, lineType=cv2.LINE_AA)

    return overlay


# --- 2. Prediction Overlays ---

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