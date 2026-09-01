import os
from pathlib import Path
import sys
import cv2
from tqdm import tqdm
import numpy as np

# Add 'src' to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, VIDEOS
from detection.detect_balls import detect_balls, draw_ball_overlay
from detection.detect_pockets import draw_pockets_overlay, find_pockets
from detection.detect_cue import detect_cue, draw_cue_overlay, find_cue_ball
from detection.detect_table import detect_table, draw_table_overlay
from prediction.predict import (
    draw_chain_paths,
    draw_outcome_banner,
    outcome_text,
    run_prediction,
)

BALL_MOVEMENT_THRESHOLD = 6.0


def make_full_overlay_video(
    video_filename: str,
    conf: float = 0.3,
    model_mode: str = "full_1280",
    every: int = 1,
    draw_table: bool = True,
    draw_pockets: bool = True,
    draw_cue_stick: bool = True,
    draw_balls: bool = True,
    draw_prediction: bool = False,
):
    """Processes a video and optionally renders selected overlays (balls, table, pockets, cue, prediction)."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate an informative suffix for the output filename
    # 't' = table, 'p' = pockets, 'c' = cue, 'b' = balls, 'x' = prediction
    suffix = "".join([
        "t" if draw_table else "",
        "p" if draw_pockets else "",
        "c" if draw_cue_stick else "",
        "x" if draw_prediction else "",
        f"b_{conf}" if draw_balls else "",
    ]) or "raw"

    out_dir = os.path.join(OUTPUT_DIR, "videos", f"{model_mode}")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{Path(video_filename).stem}_{suffix}"
    out_path = os.path.join(out_dir, f"{tag}.mp4")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    table_data = None
    pockets = None

    # Frame 0 gives the static table and pocket overlays, plus a fallback for
    # frames where the per-frame detection below fails.
    if draw_table or draw_pockets or draw_cue_stick or draw_prediction:
        ok, first_frame = cap.read()
        if ok:
            try:
                table_data = detect_table(first_frame)
                if draw_pockets:
                    pockets = find_pockets(table_data["mask"], table_data["corners"])
            except Exception as e:
                tqdm.write(f"[{video_filename}] Warning: Failed to detect table/pockets on frame 0: {e}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    table_corners = table_data["corners"] if table_data is not None else None

    last_cue_ball_pos = None
    shot_taken = False
    last_pred = None          # most recent successful prediction, kept after the shot
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

                # The table has to be re-detected on the frame we are working on.
                # Reusing the frame-0 estimate looks harmless because the table
                # does not move, but the cloth mask and the fitted corners drift
                # with the players and the lighting: measured 19 px of corner
                # drift on one clip, enough to move the predicted path 30 px at
                # the pocket and flip the verdict. The frame-0 result is only a
                # fallback for frames where segmentation fails outright.
                frame_table = table_data
                if draw_cue_stick or draw_prediction:
                    try:
                        frame_table = detect_table(frame)
                    except Exception:
                        frame_table = table_data

                # Balls, cue stick and prediction all start from one detection pass
                if draw_balls or draw_cue_stick or draw_prediction:
                    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
                    cue_ball = find_cue_ball(balls)

                    if cue_ball is not None:
                        current_pos = cue_ball["center"]
                        if last_cue_ball_pos is not None and not shot_taken:
                            displacement = np.linalg.norm(current_pos - last_cue_ball_pos)
                            if displacement > BALL_MOVEMENT_THRESHOLD:
                                shot_taken = True
                        last_cue_ball_pos = current_pos

                    # Predict only while the player is still aiming
                    if draw_prediction and not shot_taken and frame_table is not None:
                        pred = run_prediction(
                            frame, conf=conf, model_mode=model_mode,
                            table=frame_table, balls=balls,
                        )
                        if pred is not None:
                            last_pred = pred
                            current_frame = current_frame.copy()
                            draw_chain_paths(current_frame, pred)

                    # Cue stick, also only before the shot
                    if draw_cue_stick and not shot_taken and cue_ball is not None:
                        mask = frame_table["mask"] if frame_table is not None else None
                        cue_data = detect_cue(frame, cue_ball["center"], table_mask=mask)
                        current_frame = draw_cue_overlay(current_frame, cue_ball["center"], cue_data)

                    # Ball bounding boxes overlay
                    if draw_balls:
                        current_frame = draw_ball_overlay(current_frame, balls)

                    # The expected outcome stays on screen once it is known
                    if draw_prediction and last_pred is not None:
                        if current_frame is frame:
                            current_frame = frame.copy()
                        text = outcome_text(last_pred)
                        if shot_taken:
                            text = f"predicted: {text}"
                        draw_outcome_banner(current_frame, text, last_pred["outcome"] == "IN")

                last_overlay = current_frame

            writer.write(last_overlay if last_overlay is not None else frame)
            pbar.update(1)

    cap.release()
    writer.release()

    verdict = outcome_text(last_pred) if last_pred is not None else "no prediction"
    tqdm.write(f"  [{video_filename}] {verdict} | saved to: {out_path}\n")


if __name__ == "__main__":
    for i in range(1, 8):
        make_full_overlay_video(
            f"video{i}.mp4",
            conf=0.3,
            model_mode="full_1280",
            every=1,
            draw_table=True,
            draw_pockets=True,
            draw_balls=True,
            draw_cue_stick=True,
            draw_prediction=True,
        )
