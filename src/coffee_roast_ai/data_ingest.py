import kagglehub
import os


def download_and_list_files():
    # 1. Download the dataset (returns the local path)
    path = kagglehub.dataset_download("gpiosenka/coffee-bean-dataset-resized-224-x-224")

    print(f"Dataset downloaded to: {path}")

    # 2. List the folders to confirm what's inside (Train/Test/Valid)
    print("Contents of the dataset directory:")
    print(os.listdir(path))

    return path


if __name__ == "__main__":
    download_and_list_files()
