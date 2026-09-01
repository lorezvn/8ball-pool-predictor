"""Check a prediction against what actually happens later in the clip.

The prediction is made on the last frame before the cue ball starts moving.
Frames after the shot are used only to score it, never to predict:

  1. predict on the shot frame -> IN (which ball) / OUT / SCRATCH;
  2. on a later frame, once the balls are still again, detect what is left;
  3. balls that vanished from open play are the ones actually potted;
  4. compare.

Honest limits: detection and identity are noisy, and a potted ball can stay
visible in the pocket netting, so only balls in OPEN PLAY (away from the
pockets) are counted. This is an approximate ground truth, not a perfect one:
a wrong label on a ball is enough to make a correct prediction look wrong.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR, VIDEOS
from detection.detect_balls import detect_balls
from detection.detect_cue import find_cue_ball
from detection.detect_table import detect_table
from prediction.predict import draw_2d_map, draw_prediction_overlay, run_prediction
from prediction.trajectory import CAPTURE_FACTOR, POCKETS_TOP

SHOT_MOVEMENT_THRESHOLD = 6.0   # px of cue ball travel that means the shot is away
POCKET_EXCLUSION = 40.0         # top-down units: closer than this to a pocket is not open play
STILL_DIFF = 0.5                # mean abs frame difference below which the table is still


def find_shot_frame(video_path: str, model_mode: str = "full", conf: float = 0.3) -> int | None:
    """Index of the last frame before the cue ball starts moving.

    Returns None if the cue ball is never seen, or never moves.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    previous_pos, previous_idx = None, None
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        cue_ball = find_cue_ball(detect_balls(frame, conf=conf, model_mode=model_mode))
        if cue_ball is None:
            continue

        if previous_pos is not None:
            if np.linalg.norm(cue_ball["center"] - previous_pos) > SHOT_MOVEMENT_THRESHOLD:
                cap.release()
                return previous_idx
        previous_pos, previous_idx = cue_ball["center"], idx

    cap.release()
    return None


def last_still_frame(video_path: str, after: float = 0.5) -> int:
    """Last frame in the back half of the clip where nothing is moving."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    previous, best, idx = None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2GRAY)
        if previous is not None and idx > after * total:
            if float(np.mean(cv2.absdiff(small, previous))) < STILL_DIFF:
                best = idx
        previous, idx = small, idx + 1

    cap.release()
    return best if best is not None else total - 1


def read_frame(video_path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} from {video_path}")
    return frame


def open_play_labels(frame: np.ndarray, table: dict, conf: float, model_mode: str) -> set:
    """Labels of the balls still in open play, i.e. away from every pocket."""
    matrix = table["transform_matrix"]
    labels = set()

    for ball in detect_balls(frame, conf=conf, model_mode=model_mode):
        if ball["label"] in ("0", "?"):
            continue
        pt = np.array([[list(ball["center"])]], dtype=np.float32)
        top = cv2.perspectiveTransform(pt, matrix)[0, 0]
        if min(np.linalg.norm(top - np.array(p)) for p in POCKETS_TOP) < POCKET_EXCLUSION:
            continue
        labels.add(ball["label"])

    return labels


def validate_video(
    video_filename: str,
    conf: float = 0.3,
    model_mode: str = "full",
    capture_factor: float = CAPTURE_FACTOR,
    save_images: bool = True,
) -> dict | None:
    """Predict on the shot frame and score it against a later still frame."""
    video_path = os.path.join(VIDEOS, video_filename)

    shot_frame = find_shot_frame(video_path, model_mode=model_mode, conf=conf)
    if shot_frame is None:
        return {"video": video_filename, "status": "no shot detected"}

    frame = read_frame(video_path, shot_frame)
    table = detect_table(frame)
    pred = run_prediction(frame, conf=conf, model_mode=model_mode,
                          capture_factor=capture_factor, table=table)
    if pred is None:
        return {"video": video_filename, "status": "no cue ball or cue stick",
                "shot_frame": shot_frame}

    before = {b["label"] for b in pred["balls"] if b["label"] not in ("0", "?")}

    after_frame = last_still_frame(video_path)
    after = open_play_labels(read_frame(video_path, after_frame), table, conf, model_mode)
    disappeared = before - after

    if pred["outcome"] == "IN":
        correct = pred["potted_label"] in disappeared
    elif pred["outcome"] == "OUT":
        correct = len(disappeared) == 0
    else:                                   # SCRATCH: the cue ball is not tracked here
        correct = None

    if save_images:
        out_dir = os.path.join(OUTPUT_DIR, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{Path(video_filename).stem}_shot{shot_frame}"
        cv2.imwrite(os.path.join(out_dir, f"predict_{tag}.png"),
                    draw_prediction_overlay(frame, pred))
        cv2.imwrite(os.path.join(out_dir, f"map_{tag}.png"), draw_2d_map(pred))

    return {
        "video": video_filename,
        "status": "ok",
        "shot_frame": shot_frame,
        "after_frame": after_frame,
        "predicted": pred["outcome"],
        "potted_label": pred["potted_label"],
        "chain": pred["chain"],
        "disappeared": sorted(disappeared),
        "correct": correct,
    }


def validate_all(video_filenames: list, conf: float = 0.3, model_mode: str = "full") -> list:
    """Run the validation over several clips and print a summary table."""
    results = []
    for name in video_filenames:
        result = validate_video(name, conf=conf, model_mode=model_mode)
        results.append(result)

        if result.get("status") != "ok":
            print(f"  {name:14s} {result.get('status')}")
            continue

        mark = {True: "OK ", False: "NO ", None: "?  "}[result["correct"]]
        predicted = result["predicted"]
        if result["potted_label"]:
            predicted += f" {result['potted_label']}"
        print(f"  {name:14s} shot f{result['shot_frame']:<4d} "
              f"predetto {predicted:<8s} sparite {str(result['disappeared']):<18s} {mark}")

    scored = [r for r in results if r.get("correct") is not None]
    if scored:
        hits = sum(1 for r in scored if r["correct"])
        print(f"\n  coerenti: {hits}/{len(scored)}")
    return results


if __name__ == "__main__":
    validate_all([f"video{i}.mp4" for i in range(1, 10)])
