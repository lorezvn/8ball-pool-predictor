import os
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS
from detection.detect_table import detect_table
from detection.detect_pockets import find_pockets
from detection.detect_balls import detect_balls
from detection.detect_cue import detect_cue, find_cue_ball
from prediction.predict import run_prediction

from visualization import (
    draw_table_overlay, draw_pockets_overlay, 
    draw_ball_overlay, draw_cue_overlay, 
    draw_prediction_overlay, draw_2d_map
)


def inspect_single_frame(video_filename: str, frame_index: int = 0, conf: float = 0.3, model_mode: str = "full_1280"):
    """Runs the entire detection and prediction pipeline on a single frame for debugging."""
    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    out_dir = os.path.join(OUTPUT_DIR, "debug", f"{Path(video_filename).stem}_f{frame_index}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Table and Pockets
    table = detect_table(frame)
    pockets = find_pockets(table["mask"], table["corners"])
    cv2.imwrite(os.path.join(out_dir, "1_table.png"), draw_table_overlay(frame, table["corners"]))
    cv2.imwrite(os.path.join(out_dir, "2_pockets.png"), draw_pockets_overlay(frame, pockets))

    # 2. Balls
    balls = detect_balls(frame, conf=conf, model_mode=model_mode)
    cv2.imwrite(os.path.join(out_dir, "3_balls.png"), draw_ball_overlay(frame, balls))

    # 3. Cue Stick
    cue_ball = find_cue_ball(balls)
    if cue_ball:
        cue = detect_cue(frame, cue_ball["center"], table_mask=table["mask"])
        cv2.imwrite(os.path.join(out_dir, "4_cue.png"), draw_cue_overlay(frame, cue_ball["center"], cue))

    # 4. Physical Prediction
    pred = run_prediction(frame, conf=conf, model_mode=model_mode, table=table, balls=balls)
    if pred:
        cv2.imwrite(os.path.join(out_dir, "5_prediction.png"), draw_prediction_overlay(frame, pred))
        cv2.imwrite(os.path.join(out_dir, "6_map_2d.png"), draw_2d_map(pred))

    print(f"Diagnostics complete. Images saved in: {out_dir}")

if __name__ == "__main__":
    inspect_single_frame("video2.mp4", frame_index=1, model_mode="full_1280")