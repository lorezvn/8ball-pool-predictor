from src.init_folders import create_folders
from src.utils import set_seed
from src.config import *
from src.build_merged import build_merged_dataset, check_dataset
from src.qa_labels import qa_sample

def main():
    create_folders()
    set_seed(SEED)
    build_merged_dataset(MERGED_DIR, valid_video_ids={"video-3"}, dry_run=False)
    #for dataset in (V2, V3, MERGED_DIR): check_dataset(dataset)
    qa_sample(dataset_path=MERGED_DIR, split="train", n_samples=10, delete=True)

if __name__ == '__main__':
    main()
