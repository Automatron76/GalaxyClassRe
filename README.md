# Galaxy Zoo Classifier

A PyTorch project that classifies galaxy morphology using images from the Galaxy Zoo 2 dataset.
The model answers two questions: galaxy shape (smooth / features or disk / star or artifact)
and whether the galaxy is edge-on.

---

## Requirements

- Python 3.11
- A virtual environment is included at `.venv/`

---

## Activate the virtual environment

Open a terminal in the project root (`D:\GalaxyClassRe`) and run:

**Windows (Command Prompt)**
```
.venv\Scripts\activate
```

**Windows (PowerShell)**
```
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear at the start of your prompt.

To deactivate when done:
```
deactivate
```

---

## Install dependencies (first time only)

If the `.venv` is missing or you are setting up on a new machine:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data setup

Before running any scripts, place the raw Galaxy Zoo 2 catalogue at:

```
data/raw/gz2_hart16.csv.gz
```

This file can be downloaded from the Galaxy Zoo 2 data release (Hart et al., 2016).

---

## Running the scripts

Run the scripts in order from the project root with the `.venv` active.

### 1. Prepare labels
Reads the raw catalogue and creates a cleaned CSV with hard class labels.

```
python prepare_labels.py
```

Output: `data/interim/labels_q1_q2.csv`

---

### 2. Download images
Downloads galaxy images from the SDSS Image Cutout API.
Already-downloaded images are skipped automatically.

```
python download_images.py
```

Output: `data/processed/train/<galaxy_id>.jpg`

> This step downloads up to ~240,000 images and may take several hours depending on your connection.

---

### 3. Train the model
Trains the ResNet18-based classifier for 8 epochs (default) and saves the weights.

```
python train.py
```

Optional arguments:
```
python train.py --epochs 10 --batch-size 64
```

Output: `data/artifacts/galaxy_classifier.pth`

---

### 4. Run predictions
Loads the saved model and runs predictions on sample images.

```
python predict.py
```

---

## Project structure

```
GalaxyClassRe/
├── config.py             # Shared settings (paths, transforms, hyperparameters)
├── prepare_labels.py     # Label preparation from raw catalogue
├── download_images.py    # Image download from SDSS API
├── train.py              # Model definition, dataset, and training loop
├── predict.py            # Inference on new images
├── requirements.txt      # Python dependencies
├── index.html            # Project page
└── data/
    ├── raw/              # gz2_hart16.csv.gz (place here)
    ├── interim/          # labels_q1_q2.csv (generated)
    ├── processed/train/  # Downloaded galaxy images (generated)
    └── artifacts/        # Saved model weights (generated)
```
