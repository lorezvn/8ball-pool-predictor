import os
import cv2
import matplotlib.pyplot as plt


def compare_two_experiments(
    first_exp: tuple[str, float],
    second_exp: tuple[str, float],
    video_ids: list[int] = list(range(2, 6)),
    frame_idx: int = 0,
    img_dir: str = os.path.join("output", "table", "balls"),
    out_dir: str = None,
    dpi: int = 300,
) -> None:
    """Generates side-by-side comparison plots between two experiment runs."""
    out_dir = out_dir or os.path.join(img_dir, "comparisons")
    os.makedirs(out_dir, exist_ok=True)

    tag_first, conf_first = first_exp
    tag_second, conf_second = second_exp

    for v_id in video_ids:
        first_path = os.path.join(img_dir, f"{tag_first}_{conf_first}_balls_video{v_id}_f{frame_idx}.png")
        second_path = os.path.join(img_dir, f"{tag_second}_{conf_second}_balls_video{v_id}_f{frame_idx}.png")

        if not os.path.isfile(first_path) or not os.path.isfile(second_path):
            print(f"Warning: Missing files for Video {v_id}, skipping.")
            continue

        img_first = cv2.cvtColor(cv2.imread(first_path), cv2.COLOR_BGR2RGB)
        img_second = cv2.cvtColor(cv2.imread(second_path), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        axes[0].imshow(img_first)
        axes[0].set_title(f"{tag_first} (conf {conf_first})", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img_second)
        axes[1].set_title(f"{tag_second} (conf {conf_second})", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        plt.suptitle(f"Video {v_id} (Frame {frame_idx})", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_filepath = os.path.join(out_dir, f"compare_video{v_id}_f{frame_idx}.png")
        plt.savefig(out_filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_filepath}")


if __name__ == '__main__':

    first_exp = ("full", 0.15)
    second_exp = ("full", 0.35)
    compare_two_experiments(first_exp, second_exp)