import os
from pathlib import Path
import sys
import cv2
from tqdm import tqdm
import numpy as np

# Add 'src' to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, VIDEOS, BALL_MOVEMENT_THRESHOLD
from detection.detect_balls import detect_balls
from detection.detect_pockets import find_pockets
from detection.detect_cue import detect_cue, find_cue_ball
from detection.detect_table import detect_table
from prediction.predict import run_prediction

from visualization import (
    draw_ball_overlay,
    draw_cue_overlay,
    draw_pockets_overlay,
    draw_table_overlay,
    draw_chain_paths,
    draw_outcome_banner,
    outcome_text
)

def process_single_frame(frame, initial_table, pockets, state, conf, model_mode, config_draw):
    """Handles inference, state tracking, and rendering for a single frame."""
    current_frame = frame
    shot_taken = state['shot_taken']

    # Static overlays (Table and Pockets from frame 0)
    if config_draw['table'] and initial_table:
        current_frame = draw_table_overlay(current_frame, initial_table["corners"])
    if config_draw['pockets'] and pockets:
        current_frame = draw_pockets_overlay(current_frame, pockets)

    # If nothing else needs to be drawn, exit early
    if not (config_draw['balls'] or config_draw['cue_stick'] or config_draw['prediction']):
        return current_frame, state

    # Dynamic update of the table mask
    frame_table = initial_table
    if config_draw['cue_stick'] or config_draw['prediction']:
        try:
            frame_table = detect_table(frame)
        except Exception:
            pass # Fallback to initial_table

    # Entity detection
    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
    cue_ball = find_cue_ball(balls)

    # Check for the moment the shot is taken
    if cue_ball is not None:
        current_pos = cue_ball["center"]
        if state['last_cue_pos'] is not None and not shot_taken:
            displacement = np.linalg.norm(current_pos - state['last_cue_pos'])
            if displacement > BALL_MOVEMENT_THRESHOLD:
                shot_taken = True
                state['shot_taken'] = True
        state['last_cue_pos'] = current_pos

    # Predictive inference (only during aiming phase)
    if config_draw['prediction'] and not shot_taken and frame_table is not None:
        pred = run_prediction(frame, conf=conf, model_mode=model_mode, table=frame_table, balls=balls)
        if pred is not None:
            state['last_pred'] = pred
            current_frame = current_frame.copy()
            draw_chain_paths(current_frame, pred)

    # Cue stick overlay (only during aiming phase)
    if config_draw['cue_stick'] and not shot_taken and cue_ball is not None:
        mask = frame_table["mask"] if frame_table is not None else None
        cue_data = detect_cue(frame, cue_ball["center"], table_mask=mask)
        current_frame = draw_cue_overlay(current_frame, cue_ball["center"], cue_data)

    # Balls overlay
    if config_draw['balls']:
        current_frame = draw_ball_overlay(current_frame, balls)

    # Outcome banner rendering
    if config_draw['prediction'] and state['last_pred'] is not None:
        if current_frame is frame:
            current_frame = frame.copy()
        text = outcome_text(state['last_pred'])
        if shot_taken:
            text = f"predicted {text}"
        draw_outcome_banner(current_frame, text, state['last_pred']["outcome"] == "IN")

    return current_frame, state


def make_full_overlay_video(
    video_filename: str, conf: float = 0.3, model_mode: str = "full_1280", every: int = 1,
    draw_table: bool = True, draw_pockets: bool = True, draw_cue_stick: bool = True,
    draw_balls: bool = True, draw_prediction: bool = False,
):
    """Main pipeline: Video I/O and frame iteration."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate an informative suffix for the output filename
    # 't' = table, 'p' = pockets, 'c' = cue,  'x' = prediction, 'b' = balls
    suffix = "".join([
        "t" if draw_table else "",
        "p" if draw_pockets else "",
        "c" if draw_cue_stick else "",
        "x" if draw_prediction else "",
        f"b_{conf}" if draw_balls else "",
    ]) or "raw"

    out_dir = os.path.join(OUTPUT_DIR, "videos", f"{model_mode}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{Path(video_filename).stem}_{suffix}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    config_draw = {
        'table': draw_table, 'pockets': draw_pockets, 'cue_stick': draw_cue_stick,
        'balls': draw_balls, 'prediction': draw_prediction
    }
    
    # Initialize state variables
    state = {'shot_taken': False, 'last_cue_pos': None, 'last_pred': None}
    initial_table, pockets = None, None

    # Acquire static data from Frame 0
    if any(config_draw.values()):
        ok, first_frame = cap.read()
        if ok:
            try:
                initial_table = detect_table(first_frame)
                if draw_pockets:
                    pockets = find_pockets(initial_table["mask"], initial_table["corners"])
            except Exception as e:
                tqdm.write(f"[{video_filename}] Warning: Failed to detect table/pockets on frame 0: {e}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_overlay = None

    with tqdm(total=total, desc=f"Processing {video_filename} ({model_mode})", unit="frames") as pbar:
        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every == 0:
                last_overlay, state = process_single_frame(
                    frame, initial_table, pockets, state, conf, model_mode, config_draw
                )

            writer.write(last_overlay if last_overlay is not None else frame)
            pbar.update(1)

    cap.release()
    writer.release()

    tqdm.write(f"  [{video_filename}] saved to: {out_path}\n")


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
