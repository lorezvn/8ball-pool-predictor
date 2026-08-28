import os
from pathlib import Path
import sys
import cv2
from tqdm import tqdm

# Add 'src' to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, VIDEOS
from detection.detect_balls import detect_balls, draw_ball_overlay
from detection.detect_pockets import draw_pockets_overlay, find_pockets
from detection.detect_table import detect_table, draw_table_overlay


def make_full_overlay_video(
    video_filename: str,
    conf: float = 0.2,
    model_mode: str = "full",
    every: int = 1,
    draw_balls: bool = True,
    draw_table: bool = True,
    draw_pockets: bool = True,
):
    """Processes a video and optionally renders selected overlays (balls, table, pockets)."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate an informative suffix for the output filename
    # 't' = table, 'p' = pockets, 'b' = balls
    suffix = "".join([
        "t" if draw_table else "",
        "p" if draw_pockets else "",
        f"b_{conf}" if draw_balls else "",
    ]) or "raw"

    out_dir = os.path.join(OUTPUT_DIR, "videos")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_{suffix}"
    out_path = os.path.join(out_dir, f"{tag}.mp4")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    table_corners = None
    pockets = None

    # Compute table and pockets on frame 0 only if at least one is requested
    if draw_table or draw_pockets:
        ok, first_frame = cap.read()
        if ok:
            try:
                table_data = detect_table(first_frame)
                table_corners = table_data["corners"]
                if draw_pockets:
                    pockets = find_pockets(table_data["mask"], table_corners)
            except Exception as e:
                tqdm.write(f"[{video_filename}] Warning: Failed to detect table/pockets on frame 0: {e}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_overlay = None

    with tqdm(total=total, desc=f"Processing {video_filename}", unit="frames") as pbar:
        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every == 0:
                current_frame = frame

                # Table overlay
                if draw_table and table_corners is not None:
                    current_frame = draw_table_overlay(current_frame, table_corners)

                # Pocket overlay
                if draw_pockets and pockets is not None:
                    current_frame = draw_pockets_overlay(current_frame, pockets)

                # Ball detection and overlay
                if draw_balls:
                    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
                    current_frame = draw_ball_overlay(current_frame, balls)

                last_overlay = current_frame

            writer.write(last_overlay if last_overlay is not None else frame)
            pbar.update(1)

    cap.release()
    writer.release()
    tqdm.write(f"  [{video_filename}] Video saved successfully to: {out_path}\n")


if __name__ == "__main__":
    for i in range(2, 6):
        make_full_overlay_video(
            f"video{i}.mp4",
            conf=0.2,
            model_mode="full",
            every=1,
            draw_table=False,
            draw_pockets=False
        )