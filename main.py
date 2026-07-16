from src.init_folders import create_folders
from src.utils import set_seed
from src.config import SEED

def main():
    create_folders()
    set_seed(SEED)



if __name__ == '__main__':
    main()