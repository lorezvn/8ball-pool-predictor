import os
from dotenv import load_dotenv

load_dotenv(override=True)

SEED = 42

DATASET_DIR = os.getenv("DATASET_DIR")

V2 = os.path.join(DATASET_DIR, "Pool Ball Detection V2.yolov8")
V3 = os.path.join(DATASET_DIR, "Pool Ball Detection V3.yolov8")

MERGED_DIR = os.path.join(DATASET_DIR, "merged")
VIDEOS = os.path.join(DATASET_DIR, "videos")
OUTPUT_DIR = "output"
