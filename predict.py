"""
Classify a galaxy image.

Loads the trained model and prints the predicted morphology for both
Galaxy Zoo questions:
  Q1: Galaxy shape — smooth / features or disk / star or artifact
  Q2: Edge-on? — edge-on / not edge-on

"""

import os
import torch
from config import ARTIFACTS_DIR, MODEL_WEIGHTS, DEVICE
from train import GalaxyClassifier

def load_model():
    model = GalaxyClassifier(pretrained=False)
    path = os.path.join(ARTIFACTS_DIR, MODEL_WEIGHTS)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)