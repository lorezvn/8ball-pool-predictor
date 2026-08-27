import os
import cv2
import matplotlib.pyplot as plt

img_dir = os.path.join("output", "table", "balls")  # Adatta al tuo percorso
out_dir = os.path.join(img_dir, "comparisons")
os.makedirs(out_dir, exist_ok=True)

video_ids = range(2, 6)
frame_idx = 0

for v_id in video_ids:
    trial_path = os.path.join(img_dir, f"trial_balls_overlay_video{v_id}_f{frame_idx}.png")
    full_path = os.path.join(img_dir, f"full_balls_overlay_video{v_id}_f{frame_idx}.png")

    img_trial = cv2.cvtColor(cv2.imread(trial_path), cv2.COLOR_BGR2RGB)
    img_full = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].imshow(img_trial)
    axes[0].set_title(f"trial", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_full)
    axes[1].set_title(f"full", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(f"Video {v_id} (Frame {frame_idx})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_filepath = os.path.join(out_dir, f"compare_video{v_id}_f{frame_idx}.png")
    plt.savefig(out_filepath, dpi=300, bbox_inches="tight")
    plt.close(fig) 

    print(f"Saved: {out_filepath}")