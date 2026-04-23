"""
Shared configuration for the Galaxy Classifier project.
"""
from torchvision import transforms

# Paths — used by prepare_labels.py
RAW_CATALOG_PATH = "data/raw/gz2_hart16.csv.gz"
LABELS_PATH = "data/interim/labels_q1_q2.csv"

# Paths — used by download_images.py
IMAGES_DIR = "data/processed/train"


# Model — used by GalaxyClassifier   Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
IMG_SIZE = 224
DEVICE = "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# Image transforms — added after deciding to use ResNet18
# Source: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Training split settings
VAL_SPLIT = 0.20
SEED = 42

# Training hyperparameters — used by train function
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
ARTIFACTS_DIR = "data/artifacts"
MODEL_WEIGHTS = "galaxy_classifier.pth"