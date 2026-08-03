import os
import sys
import random
import cv2
import shutil
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SEED, MERGED_DIR


def load_class_names(dataset_path: str) -> list[str]:
    """Read the class names from a dataset's data.yaml.

    Args:
        dataset_path: Root path of the dataset (e.g. MERGED_DIR, V2, V3).

    Returns:
        list[str]: Class names, indexed by YOLO class id
        (names[i] is the label for class id i).
    """
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    return data_yaml["names"]


def read_yolo_labels(label_path: str) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO-format label file.

    Each line is: class_id x_center y_center width height, all normalized
    to [0, 1] relative to the image size.

    Args:
        label_path: Path to a single .txt label file.

    Returns:
        list of (class_id, x_center, y_center, width, height).
        Empty list if the file doesn't exist (unlabeled image).
    """
    if not os.path.isfile(label_path):
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def draw_boxes(image, boxes, class_names):
    """Draw YOLO-format bounding boxes and class labels on an image.

    Args:
        image: Image as a numpy array (BGR, as loaded by cv2.imread).
        boxes: List of (class_id, x_center, y_center, width, height),
            normalized as in read_yolo_labels.
        class_names: List of class names, indexed by class_id.

    Returns:
        The image with boxes drawn on it (modified in place, also returned
        for convenience).
    """
    img_h, img_w = image.shape[:2]

    for class_id, x_center, y_center, width, height in boxes:
        # Convert normalized YOLO coords -> pixel corners
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        label = class_names[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    return image


def qa_sample(
    dataset_path: str = MERGED_DIR,
    split: str = "train",
    n_samples: int = 10,
    output_dir: str = "output/qa_labels",
    debug: bool = False,
    delete: bool = False
) -> list[str]:
    """Draw boxes on a random sample of images and save them for visual QA.

    This is a manual sanity check: after building a dataset (especially
    after copying files around, like build_merged_dataset does), the only
    reliable way to catch a label/image mismatch is to actually look at
    the pictures.

    Args:
        dataset_path: Root path of the dataset to check (default: MERGED_DIR).
        split: Which split to sample from ("train", "valid" or "test").
        n_samples: How many images to sample.
        output_dir: Where to save the annotated images.
        debug: Boolean flag used for debugging.
        delete: if it's true the existing files at output_dir will be deleted.

    Returns:
        list[str]: Paths of the saved annotated images.
    """

    print("\n=== VISUAL QA ===")
    class_names = load_class_names(dataset_path)

    images_dir = os.path.join(dataset_path, split, "images")
    labels_dir = os.path.join(dataset_path, split, "labels")

    all_images = os.listdir(images_dir)

    random.seed(SEED)
    sample = random.sample(all_images, min(n_samples, len(all_images)))

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_dir) and delete:
        items = os.listdir(output_dir)
        if items:
            for item in tqdm(items, desc="Deleting old files", unit="file"):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    saved_paths = []

    for fname in sample:
        image_path = os.path.join(images_dir, fname)
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"WARNING: could not read image {image_path}, skipping.")
            continue

        boxes = read_yolo_labels(label_path)
        if not boxes:
            print(f"WARNING: no labels found for {fname} (empty or missing .txt).")

        if debug:
            print(f"\nImage path: {image_path}\n\nLabel path: {label_path}")
            print(f"\nBoxes: {boxes}")

        draw_boxes(image, boxes, class_names)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, image)
        saved_paths.append(out_path)

    print(f"Saved {len(saved_paths)} annotated images to: {output_dir}")
    return saved_paths


if __name__ == "__main__":
    qa_sample(dataset_path=MERGED_DIR, split="train", n_samples=10, delete=True)