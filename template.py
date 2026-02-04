import os
from pathlib import Path
import logging

# Standard logging setup to track the creation process
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "coffee_roast_ai"

list_of_files = [
    ".github/workflows/.gitkeep", # For future CI/CD deployment
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/preprocessing.py",
    f"src/{project_name}/model_engine.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/pipeline/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html" # If you use Flask later instead of Streamlit
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")