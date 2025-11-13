import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    "dataset/yolo/images/train",
    "dataset/yolo/images/val",
    "dataset/yolo/images/test",
    "dataset/yolo/labels/train",
    "dataset/yolo/labels/val",
    "dataset/outputs"
    "dataset/license_plate.yaml",
    "models",
    "src/finetune_ocr.py",
    "src/train_yolo.py",
    "test/test_yaml.py",
    "test/test_yolo.py",
    "test/test_ocr.py",
    "requirements.txt",
    "setup.py",
    "main.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for file: {filename}")
        
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists and is not empty - skipping creation.")