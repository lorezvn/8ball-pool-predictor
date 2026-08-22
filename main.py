from src.init_folders import create_folders
from src.utils import set_seed
from src.config import *
from src.dataset import build_merged_dataset, check_dataset
from src.train import train_model
from src.config import VIDEOS

from src.tools.qa_labels import qa_sample
from src.tools.inspect_video import inspect_video

from src.detection.detect_table import save_table_outputs
from src.detection.detect_pockets import save_pocket_outputs

demo_video = os.path.join(VIDEOS, "video2.mp4")

def main():
    create_folders()
    set_seed(SEED)

    build_merged_dataset(MERGED_DIR, valid_video_ids={"video-3"}, dry_run=True)
    qa_sample(dataset_path=MERGED_DIR, split="train", n_samples=10, delete=True)
    train_model(full=False) # <- runnatelo con full=True per fare 80 epochs su train.py sta scritto tutto

if __name__ == '__main__':
    #main()
    for i in range (2, 6):
        save_table_outputs(f"video{i}.mp4")