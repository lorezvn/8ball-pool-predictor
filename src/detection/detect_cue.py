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


def save_cue_outputs(video_filename: str, frame_index: int = 0, conf: float = 0.2, model_mode: str = "full_1280"):
    """Loads a single frame, extracts cue stick data, and writes the output overlay image."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    table_data = detect_table(frame)
    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
    cue_ball = find_cue_ball(balls)

    if cue_ball is None:
        raise RuntimeError("Cue ball ('0') not detected in this frame.")

    cue_data = detect_cue(frame, cue_ball["center"], table_mask=table_data["mask"])
    overlay = draw_cue_overlay(frame, cue_ball["center"], cue_data)

    out_dir = os.path.join(OUTPUT_DIR, "table", "cue")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_f{frame_index}"

    out_path = os.path.join(out_dir, f"cue_{tag}.png")
    cv2.imwrite(out_path, overlay)

    if cue_data is None:
        print(f"[{video_filename}] Cue NOT found. Saved: {out_path}")
    else:
        print(f"[{video_filename}] Cue detected successfully. Saved: {out_path}")


if __name__ == "__main__":
    save_cue_outputs("video2.mp4")