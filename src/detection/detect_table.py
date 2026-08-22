import os
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS

# Color and canonical size configuration
HUE_TOL, SAT_TOL, VAL_TOL = 15, 80, 90
CANONICAL_W, CANONICAL_H = 1000, 500


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


def draw_overlay(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draws the detected bounding polygon and labeled corner vertices."""
    overlay = frame.copy()
    pts = corners.astype(int)

    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
    for (x, y), label in zip(pts, ["TL", "TR", "BR", "BL"]):
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(overlay, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return overlay


def save_table_outputs(video_filename: str, frame_index: int = 0):
    """Extracts a frame from a video, detects the table, and saves the output assets."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    res = detect_table(frame)
    overlay = draw_overlay(frame, res["corners"])

    out_dir = os.path.join(OUTPUT_DIR, "table")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_f{frame_index}"

    cv2.imwrite(os.path.join(out_dir, f"table_overlay_{tag}.png"), overlay)
    cv2.imwrite(os.path.join(out_dir, f"table_mask_{tag}.png"), res["mask"])
    cv2.imwrite(os.path.join(out_dir, f"table_topdown_{tag}.png"), res["warped"])

    print(f"Files saved successfully to: {out_dir}")


if __name__ == "__main__":
    save_table_outputs("video2.mp4")