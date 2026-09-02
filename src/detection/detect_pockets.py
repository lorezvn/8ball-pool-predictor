import os
from pathlib import Path
import sys
import cv2
import numpy as np

# Add the 'src' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, VIDEOS
from detection.detect_table import detect_table

# Pocket detection thresholds
MIN_POCKET_AREA = 400
MAX_POCKET_AREA = 6000
MATCH_DIST = 140
CLOSE_K = 121
CAPTURE_R = 30


def get_geometric_positions(corners: np.ndarray) -> np.ndarray:
    """Computes default 6 expected positions: 4 corners + 2 long-edge midpoints."""
    tl, tr, br, bl = corners
    return np.array(
        [
            tl,
            tr,
            br,
            bl,
            (tl + tr) / 2.0,  # Top-edge midpoint
            (bl + br) / 2.0,  # Bottom-edge midpoint
        ],
        dtype=np.float32,
    )


def extract_candidate_blobs(mask: np.ndarray) -> list[dict]:
    """Isolates candidate pocket regions via morphological closing and difference analysis."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No valid contours found in the cloth mask.")

    largest_cnt = max(contours, key=cv2.contourArea)

    # Fill cloth contour and extract concave border regions
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [largest_cnt], -1, 255, cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    diff = cv2.subtract(closed, filled)
    diff = cv2.morphologyEx(
        diff, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    blob_contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in blob_contours:
        area = cv2.contourArea(c)
        m = cv2.moments(c)
        if m["m00"] == 0 or not (MIN_POCKET_AREA <= area <= MAX_POCKET_AREA):
            continue
        center = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)
        blobs.append({"center": center, "radius": float(np.sqrt(area / np.pi)), "area": area})

    return blobs


def find_pockets(mask: np.ndarray, corners: np.ndarray) -> list[dict]:
    """Matches detected blobs to the 6 expected pocket anchors with geometric fallback."""
    blobs = extract_candidate_blobs(mask)
    geo_positions = get_geometric_positions(corners)
    pockets = []

    for g in geo_positions:
        best_match = None
        best_dist = MATCH_DIST

        for blob in blobs:
            dist = float(np.linalg.norm(blob["center"] - g))
            if dist < best_dist:
                best_match = blob
                best_dist = dist

        if best_match is not None:
            pockets.append(best_match)
        else:
            # Fallback to pure geometric coordinate when occluded
            pockets.append({"center": g.astype(np.float32), "radius": 25.0, "area": 0.0})

    return pockets