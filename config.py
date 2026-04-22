"""
Shared configuration for the Galaxy Classifier project.
"""

# Paths — used by prepare_labels.py
RAW_CATALOG_PATH = "data/raw/gz2_hart16.csv.gz"
LABELS_PATH = "data/interim/labels_q1_q2.csv"

# Paths — used by download_images.py
IMAGES_DIR = "data/processed/train"


# Model — used by GalaxyClassifier   Source: https://docs.pytorch.org/vision/stable/models.html
IMG_SIZE = 224
DEVICE = "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]