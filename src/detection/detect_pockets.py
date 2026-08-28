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


def draw_pockets_overlay(frame: np.ndarray, pockets: list[dict]) -> np.ndarray:
    """Renders the 6 detected pocket positions onto the frame."""
    overlay = frame.copy()
    for p in pockets:
        cx, cy = int(p["center"][0]), int(p["center"][1])
        cv2.circle(overlay, (cx, cy), CAPTURE_R, (0, 0, 255), 3)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)
    return overlay


def save_pocket_outputs(video_filename: str, frame_index: int = 0):
    """Extracts a frame, detects pockets, and writes both standard and top-down overlays."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    # Run table and pocket detection pipelines
    table_data = detect_table(frame)
    pockets = find_pockets(table_data["mask"], table_data["corners"])

    # Draw standard perspective overlay
    overlay = draw_pockets_overlay(frame, pockets)

    # Project pocket coordinates onto the top-down canonical view
    topdown = table_data["warped"].copy()
    centers = np.array([p["center"] for p in pockets], dtype=np.float32)
    centers_topdown = cv2.perspectiveTransform(
        centers.reshape(-1, 1, 2), table_data["transform_matrix"]
    ).reshape(-1, 2)

    topdown_pockets = [{"center": pt} for pt in centers_topdown]
    topdown_overlay = draw_pockets_overlay(topdown, topdown_pockets)

    # Save output images
    out_dir = os.path.join(OUTPUT_DIR, "table", "pockets")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_f{frame_index}"

    cv2.imwrite(os.path.join(out_dir, f"pockets_overlay_{tag}.png"), overlay)
    cv2.imwrite(os.path.join(out_dir, f"pockets_topdown_{tag}.png"), topdown_overlay)

    print(f"Detected {len(pockets)} pockets. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    save_pocket_outputs("video2.mp4")