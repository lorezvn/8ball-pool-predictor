import os
from pathlib import Path
import sys
import cv2
import numpy as np

# Add 'src' to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS
from detection.detect_balls import detect_balls
from detection.detect_table import detect_table

# --- Detection Parameters ---
CUE_ROI_RADIUS = 200             # px: search radius around the cue ball
CANNY_LOW, CANNY_HIGH = 30, 100  # Permissive thresholds to capture thin/dark sticks
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 40       # px: minimum segment length
HOUGH_MAX_LINE_GAP = 15
MAX_ALIGNMENT_DISTANCE = 12.0    # px: max distance from segment axis to cue ball center
ROI_ERODE_MARGIN = 15            # Cloth erosion margin to reject outer cushions


def find_cue_ball(balls: list[dict]) -> dict | None:
    """Returns the most confident '0' (cue ball) detection."""
    candidates = [b for b in balls if b["label"] == "0"]
    return max(candidates, key=lambda b: b["conf"]) if candidates else None


def detect_cue(frame: np.ndarray, cue_center: np.ndarray, table_mask: np.ndarray | None = None) -> dict | None:
    """Detects cue stick orientation by restricting search to a local ROI around the cue ball
    and filtering out cushion edges via alignment distance verification.
    """
    h, w = frame.shape[:2]
    cx, cy = int(cue_center[0]), int(cue_center[1])

    # Define bounded ROI centered at the cue ball
    x1, y1 = max(0, cx - CUE_ROI_RADIUS), max(0, cy - CUE_ROI_RADIUS)
    x2, y2 = min(w, cx + CUE_ROI_RADIUS), min(h, cy + CUE_ROI_RADIUS)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Local edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    # Optional masking to exclude outer table cushions
    if table_mask is not None:
        table_roi = table_mask[y1:y2, x1:x2]
        eroded_roi = cv2.erode(
            table_roi,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_ERODE_MARGIN, ROI_ERODE_MARGIN)),
        )
        edges = cv2.bitwise_and(edges, edges, mask=eroded_roi)

    # Extract Hough segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None:
        return None

    lines = lines.reshape(-1, 4)
    cue_pts = []

    for lx1, ly1, lx2, ly2 in lines:
        # Shift ROI coordinates back to frame coordinates
        p1 = np.array([lx1 + x1, ly1 + y1], dtype=np.float32)
        p2 = np.array([lx2 + x1, ly2 + y1], dtype=np.float32)

        vec = p2 - p1
        seg_len = float(np.linalg.norm(vec))
        if seg_len == 0:
            continue

        # Point-to-line distance from cue ball to the segment line (must point at the ball)
        dist_to_line = abs(np.cross(vec, cue_center - p1)) / seg_len
        if dist_to_line > MAX_ALIGNMENT_DISTANCE:
            continue

        # Sample points along valid segments (longer segments get more weight)
        n_pts = max(2, int(seg_len / 5.0))
        for step in np.linspace(0.0, 1.0, n_pts):
            cue_pts.append(p1 + step * (p2 - p1))

    if not cue_pts:
        return None

    pts = np.array(cue_pts, dtype=np.float32)

    # Robust line fit via least squares
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float32)
    point_on_line = np.array([x0, y0], dtype=np.float32)

    # Orient shot vector: from stick centroid towards the cue ball
    centroid = pts.mean(axis=0)
    shot_vector = cue_center - centroid
    if np.dot(direction, shot_vector) < 0:
        direction = -direction

    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    # Projected visible endpoints
    projections = (pts - point_on_line) @ direction
    endpoint_1 = point_on_line + projections.min() * direction
    endpoint_2 = point_on_line + projections.max() * direction

    return {
        "direction": direction,
        "endpoints": (endpoint_1, endpoint_2),
        "point_on_line": point_on_line,
    }