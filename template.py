import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

project_name = "Human_Brain_mapping"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",
    "requirements.txt",
    "setup.py",
    "research/training_pipeline.py",
    "research/experiment.py",
    "templates/index.html",
] 

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")
    
    if not (os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")