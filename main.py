from src.utils import set_seed, create_folders
from src.config import *
from src.dataset import build_merged_dataset
from src.train import train_model
from src.config import VIDEOS

demo_video = os.path.join(VIDEOS, "video2.mp4")

def main():
    create_folders()
    set_seed(SEED)
    build_merged_dataset(MERGED_DIR, valid_video_ids={"video-3"}, dry_run=True)
    #train_model(full=True)

if __name__ == '__main__':
    main()