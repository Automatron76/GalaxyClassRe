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
from PIL import Image
from config import IMAGES_DIR, Q1_CLASSES, Q2_CLASSES, val_transform


def load_model():
    model = GalaxyClassifier(pretrained=False)
    path = os.path.join(ARTIFACTS_DIR, MODEL_WEIGHTS)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


def predict(image_path):
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits_q1, logits_q2 = model(tensor)
        probs_q1 = torch.softmax(logits_q1, dim=1)[0].cpu().numpy()
        probs_q2 = torch.softmax(logits_q2, dim=1)[0].cpu().numpy()

    q1_label = Q1_CLASSES[int(probs_q1.argmax())]
    q2_label = Q2_CLASSES[int(probs_q2.argmax())]
    q1_probs = {Q1_CLASSES[i]: float(probs_q1[i]) for i in Q1_CLASSES}
    q2_probs = {Q2_CLASSES[i]: float(probs_q2[i]) for i in Q2_CLASSES}

    return q1_label, q1_probs, q2_label, q2_probs