import os
import cv2
import sys
from pathlib import Path

# Add 'src' directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, VIDEOS

def inspect_video(
    video_filename: str,
    n_samples: int = 5,
) -> dict:
    """Open a video, print its properties and save evenly spaced sample frames.

    Args:
        video_filename: filename of the video.
        n_samples: how many frames to save.
    """

    video_path = os.path.join(VIDEOS, video_filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps else 0.0

    print("\n=== VIDEO INFO ===")
    print(f" path:       {video_path}")
    print(f" resolution: {width}x{height}")
    print(f" fps:        {fps:.2f}")
    print(f" frames:     {n_frames}")
    print(f" duration:   {duration:.1f}s")

    output_dir = os.path.join(OUTPUT_DIR, "frames")

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved = []

    if n_samples > 0 and n_frames > 0:
        step = (n_frames - 1) / max(n_samples - 1, 1)
        indices = sorted({int(round(i * step)) for i in range(n_samples)})

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"WARNING: could not read frame {idx}")
                continue
            out_path = os.path.join(output_dir, f"{video_name}_f{idx}.png")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)

    cap.release()

    print(f"\n Saved {len(saved)} sample frames to: {output_dir}")
    for p in saved:
        print(f"  {p}")

if __name__ == "__main__":
    inspect_video("video2.mp4", n_samples=6)        