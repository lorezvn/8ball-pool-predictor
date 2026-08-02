import os
import re
import shutil
from tqdm import tqdm
import yaml
from typing import Optional

from .config import V2, V3, MERGED_DIR

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
LABEL_EXTENSIONS = (".txt",)

# Expected prefix in filenames exported by Roboflow: <prefix>-<video number>-...
# Real examples observed:
#   V2: "out-2-596_png.rf.x4ivyb....png"     -> "out-2"
#   V3: "video-3_frame_000045_jpg.rf....jpg" -> "video-3"
# Captures letters + dash + digits at the start of the filename, stopping at
# the first non-digit character after the number (so "out-2-596" -> "out-2").
VIDEO_ID_PATTERN = re.compile(r"^([a-zA-Z]+-\d+)")


def count_files(folder: str, extensions: tuple[str, ...]) -> int:
    """Count files with the given extensions in a folder (case-insensitive).

    Args:
        folder: Path to the folder to scan.
        extensions: Tuple of accepted extensions, e.g. (".jpg", ".png").

    Returns:
        int: Number of matching files. 0 if the folder does not exist.
    """
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    )


def get_source_video(filename: str) -> Optional[str]:
    """Extract the source video ID from a frame filename.

    Example:
        "video-3_frame_000045_jpg.rf.abc123.jpg" -> "video-3"

    Args:
        filename: Name of the image file (not a full path).

    Returns:
        str | None: The video ID, or None if the filename does not match
        the expected pattern.
    """
    match = VIDEO_ID_PATTERN.match(filename)
    return match.group(1) if match else None


def assign_split_by_video(
    by_video: dict[str, list[str]],
    valid_video_ids: set[str],
) -> dict[str, str]:
    """Assign each frame to "train" or "valid" based on its source video.

    The assignment is done per VIDEO, never per individual file, so that a
    whole video always ends up in the same split (no leakage).

    Args:
        by_video: Mapping {video_id: [filenames, ...]}, as returned by
            list_images_by_video or list_all_images_by_video.
        valid_video_ids: Iterable of video IDs to send to "valid"
            (e.g. {"video-3"}).

    Returns:
        dict[str, str]: Mapping {filename: "train" | "valid"}.
    """
    valid_video_ids = set(valid_video_ids)
    assignment = {}

    for video_id, files in by_video.items():
        split = "valid" if video_id in valid_video_ids else "train"
        for fname in files:
            assignment[fname] = split

    return assignment


def list_all_images_by_video(
    dataset_path: str,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Group ALL images of a dataset by source video, across every split.

    We need this to rebuild the split from scratch without losing the
    images originally placed in its own valid/test splits.

    Args:
        dataset_path: Root path of the source dataset (e.g. V2 or V3).

    Returns:
        tuple:
            by_video (dict[str, list[str]]): {video_id: [filenames, ...]}.
            file_split (dict[str, str]): {filename: original_split}, used
                to locate the physical file to copy at
                dataset_path/original_split/images/filename.
    """
    by_video, file_split = {}, {}

    for split in ("train", "valid", "test"):
        images_dir = os.path.join(dataset_path, split, "images")
        if not os.path.isdir(images_dir):
            continue
        for fname in os.listdir(images_dir):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            video_id = get_source_video(fname) or "unknown"
            by_video.setdefault(video_id, []).append(fname)
            file_split[fname] = split

    return by_video, file_split


def write_merged_data_yaml(output_dir: str, source_dataset_path: str = V2) -> str:
    """Write a data.yaml for the merged dataset, reusing the class names
    from a source dataset (V2 and V3 share the same 16 classes).

    Args:
        output_dir: Merged dataset root (e.g. "datasets/merged").
        source_dataset_path: Dataset to copy class names from (V2 by default).

    Returns:
        str: Path to the written data.yaml.
    """
    with open(os.path.join(source_dataset_path, "data.yaml"), "r", encoding="utf-8") as f:
        source_yaml = yaml.safe_load(f)

    merged_yaml = {
        "names": source_yaml["names"],
        "nc": len(source_yaml["names"]),
        "train": "train/images",
        "val": "valid/images",
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_yaml, f, allow_unicode=True)

    return yaml_path


def build_merged_dataset(
    output_dir: str,
    valid_video_ids: set[str],
    dry_run: bool = True,
) -> dict[str, int]:
    """Build the merged dataset (V2 + V3) with a PER-VIDEO train/valid split.
 
    Every video ends up entirely in train or entirely in valid, which
    avoids the leakage caused by Roboflow's original random per-frame
    split (near-duplicate frames from the same video ending up in both
    train and valid).
    No separate test split is created: the real evaluation is running
    predict.py on our own videos.
 
    WARNING: if output_dir already exists and dry_run=False, it is
    DELETED and rebuilt from scratch (shutil.rmtree).
 
    Args:
        output_dir: Destination folder.
        valid_video_ids: Video IDs to send to valid (e.g. {"video-3"}).
        dry_run: If True (default), nothing is written to disk -- only
            counts how many files would end up in train/valid, so the
            numbers can be checked before copying anything for real.
 
    Returns:
        dict[str, int]: {"train": n, "valid": n} -- file counts per split.
    """
    print("\n=== MERGING DATASETS === ")
    if not dry_run and os.path.isdir(output_dir):
        print(f" Removing existing merged dataset at: {output_dir}")
        shutil.rmtree(output_dir)
 
    counts = {"train": 0, "valid": 0}
 
    for dataset_path in (V2, V3):
        by_video, file_split = list_all_images_by_video(dataset_path)
 
        if "unknown" in by_video:
            print(
                f"WARNING: {len(by_video['unknown'])} files in {dataset_path} "
                f"do not match the video filename pattern and will be SKIPPED."
            )
 
        assignment = assign_split_by_video(by_video, valid_video_ids)
 
        dataset_name = os.path.basename(dataset_path)
        iterator = tqdm(assignment.items(), desc=f" Copying {dataset_name}", unit="file") if not dry_run else assignment.items()
 
        for fname, split in iterator:
            video_id = get_source_video(fname)
            if video_id is None:
                continue  # already reported above as "unknown"
 
            orig_split = file_split[fname]
            src_img = os.path.join(dataset_path, orig_split, "images", fname)
            label_name = os.path.splitext(fname)[0] + ".txt"
            src_label = os.path.join(dataset_path, orig_split, "labels", label_name)
 
            dst_img_dir = os.path.join(output_dir, split, "images")
            dst_label_dir = os.path.join(output_dir, split, "labels")
 
            if not dry_run:
                os.makedirs(dst_img_dir, exist_ok=True)
                os.makedirs(dst_label_dir, exist_ok=True)
                shutil.copy2(src_img, os.path.join(dst_img_dir, fname))
                if os.path.isfile(src_label):
                    shutil.copy2(src_label, os.path.join(dst_label_dir, label_name))
                else:
                    print(f"WARNING: missing label for {fname}")
 
            counts[split] += 1
 
    if not dry_run:
        write_merged_data_yaml(output_dir)
 
    mode = "DRY RUN (no files copied)" if dry_run else "REAL COPY"
    print(f"\n [{mode}] Per-video split result:")
    print(f"  train: {counts['train']} images")
    print(f"  valid: {counts['valid']} images")
    print(f"  TOTAL: {counts['train'] + counts['valid']} images")
 
    return counts

def check_dataset(path: str) -> dict:
    """Validate the structure of a YOLO dataset and print a summary.

    Acts as a sanity check before using a dataset (e.g. before running
    build_merged_dataset). Expected structure:

        path/
            data.yaml
            train/images, train/labels
            valid/images, valid/labels
            test/images,  test/labels   (optional)

    Args:
        path: Root path of the dataset to check (e.g. V2 or V3).

    Returns:
        dict: Summary info, so the result can be reused instead of
        re-scanning the dataset. Keys: "path", "num_classes", "names",
        "splits" (per-split image/label counts, or None if missing).

    Raises:
        FileNotFoundError: If the dataset path or data.yaml is missing.
    """
    print("\n=== SANITY CHECK === ")

    info: dict = {"path": path, "splits": {}}

    # 1. the base path must exist
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Check the .env file (is DATASET_DIR the right path?)"
        )

    # 2. data.yaml holds the class names and is the source of truth
    yaml_path = os.path.join(path, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found in: {path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    names = data_yaml.get("names", [])
    info["num_classes"] = len(names)
    info["names"] = names

    # 3. for each split, check that images/ and labels/ exist and that the
    #    number of images and labels match (a mismatch often signals
    #    unlabeled images or orphan label files)
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(path, split)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")

        if not os.path.isdir(split_dir):
            # "test" is optional, train/valid are expected to always exist
            info["splits"][split] = None
            continue

        n_images = count_files(images_dir, IMAGE_EXTENSIONS)
        n_labels = count_files(labels_dir, LABEL_EXTENSIONS)

        info["splits"][split] = {"n_images": n_images, "n_labels": n_labels}

    # 4. human-readable summary
    print(f" Dataset: {path}")
    print(f" Classes ({info['num_classes']}): {names}")
    for split, counts in info["splits"].items():
        if counts is None:
            print(f" {split}: missing")
        else:
            mismatch = " <-- MISMATCH" if counts["n_images"] != counts["n_labels"] else ""
            print(f" {split}: {counts['n_images']} images, {counts['n_labels']} labels{mismatch}")
            info['total_images'] = info.get('total_images', 0) + counts['n_images']
            info['total_labels'] = info.get('total_labels', 0) + counts['n_labels']

    print(f" TOTAL: {info['total_images']} images, {info['total_labels']} labels")
    return info


if __name__ == "__main__":
    # Validate both source datasets
    check_dataset(V2)
    check_dataset(V3)
    check_dataset(MERGED_DIR)