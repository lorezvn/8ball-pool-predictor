import os
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS, CANONICAL_H, CANONICAL_W

# Color and canonical size configuration
HUE_TOL, SAT_TOL, VAL_TOL = 15, 80, 90


def get_cloth_mask(frame: np.ndarray) -> np.ndarray:
    """Isolates the cloth by sampling the color at the center of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # Sample the central 20% of the table to find the median color
    patch = hsv[int(h * 0.4) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)]
    med = np.median(patch.reshape(-1, 3), axis=0)

    # HSV thresholds with safety limits
    lower = np.clip(med - [HUE_TOL, SAT_TOL, VAL_TOL], [0, 30, 30], [179, 255, 255]).astype(
        np.uint8
    )
    upper = np.clip(med + [HUE_TOL, SAT_TOL, VAL_TOL], [0, 30, 30], [179, 255, 255]).astype(
        np.uint8
    )

    mask = cv2.inRange(hsv, lower, upper)

    # Close holes caused by balls/pockets and remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Orders vertices in clockwise sequence: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    return np.array(
        [
            pts[np.argmin(s)],     # TL
            pts[np.argmin(diff)],  # TR
            pts[np.argmax(s)],     # BR
            pts[np.argmax(diff)],  # BL
        ],
        dtype=np.float32,
    )


def find_table_corners(mask: np.ndarray) -> np.ndarray:
    """Finds the 4 corners of the largest detected contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No table contour found in the cloth mask.")

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    return order_corners(box)


def compute_transform(corners: np.ndarray) -> np.ndarray:
    """Computes the perspective transformation matrix targeting canonical dimensions."""
    dst = np.array(
        [
            [0, 0],
            [CANONICAL_W - 1, 0],
            [CANONICAL_W - 1, CANONICAL_H - 1],
            [0, CANONICAL_H - 1],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners, dst)


def detect_table(frame: np.ndarray) -> dict:
    """Runs the full table detection pipeline."""
    mask = get_cloth_mask(frame)

    # Sanity check: verify cloth area ratio is between 15% and 95% of the frame
    area_ratio = cv2.countNonZero(mask) / mask.size
    if not (0.15 <= area_ratio <= 0.95):
        raise ValueError(f"Invalid cloth area ratio: {area_ratio:.2%}")

    corners = find_table_corners(mask)
    matrix = compute_transform(corners)
    warped = cv2.warpPerspective(frame, matrix, (CANONICAL_W, CANONICAL_H))

    return {
        "mask": mask,
        "corners": corners,
        "transform_matrix": matrix,
        "warped": warped,
    }