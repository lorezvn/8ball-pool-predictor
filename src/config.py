import os
from dotenv import load_dotenv

load_dotenv()

SEED = 42

DATASET_DIR = os.getenv("DATASET_DIR")

ROOT = os.path.join(DATASET_DIR, "Pool Ball Detection V3.yolov8")

TRAIN = os.path.join(ROOT, "train")
VALID = os.path.join(ROOT, "valid")
TEST = os.path.join(ROOT, "test")