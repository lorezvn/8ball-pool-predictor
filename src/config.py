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


# Constants
BALL_MOVEMENT_THRESHOLD = 6.0      # px of cue ball travel that means the shot is away
POCKET_EXCLUSION = 40.0         # top-down units: closer than this to a pocket is not open play
STILL_DIFF = 0.5                # mean abs frame difference below which the table is still
CANONICAL_W, CANONICAL_H = 1000, 500


# Standard ball colours for the schematic 2D map (BGR). 9-15 reuse the colour of
# 1-7 with a white ring drawn on top.
BALL_BGR = {
    "0": (255, 255, 255), "1": (0, 215, 255), "2": (200, 0, 0), "3": (0, 0, 220),
    "4": (150, 30, 140), "5": (0, 140, 255), "6": (0, 150, 0), "7": (40, 40, 140),
    "8": (20, 20, 20), "?": (130, 130, 130),
}
CHAIN_PALETTE = [(0, 200, 0), (255, 200, 0), (255, 0, 200), (0, 200, 255)]
CUE_PATH_BGR = (0, 0, 255)
DEFLECTION_BGR = (200, 200, 200)

# In the canonical view the pockets are exactly the 4 corners plus the midpoints
# of the two long edges, so they need no detection.
POCKETS_TOP = [
    (0.0, 0.0),
    (CANONICAL_W, 0.0),
    (CANONICAL_W, CANONICAL_H),
    (0.0, CANONICAL_H),
    (CANONICAL_W / 2, 0.0),
    (CANONICAL_W / 2, CANONICAL_H),
]

