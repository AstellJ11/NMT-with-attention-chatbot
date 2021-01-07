import os
import sys
import logging


def init_logging(debug=False):  # Logging function
    format_str = "%(levelname)s %(asctime)s - %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=format_str, level=level)


init_logging()
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))  # Access the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

git_url = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git"
file_dir = "raw_data"
data_dir = os.path.join(current_dir, file_dir)  # Combine the new 'data' directory with the parent directory

if __name__ == "__main__":
    check_exists = os.path.isdir(data_dir)
    if check_exists:
        logger.info("This dataset has already been successfully downloaded!")
    else:
        os.makedirs(file_dir, exist_ok=True)  # Create the 'data' directory to hold the dataset

        logger.info("Downloading the Schema-Guided-Dialogue dataset...")
        os.chdir(data_dir)  # Change the working directory
        os.system(f"git clone {git_url}")  # Execute the system command to clone the repository
        logger.info("Download Complete!")
