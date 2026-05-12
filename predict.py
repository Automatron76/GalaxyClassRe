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
import matplotlib.pyplot as plt
import argparse

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

    return img, q1_label, q1_probs, q2_label, q2_probs




def show_prediction(img, image_path, q1_label, q1_probs, q2_label, q2_probs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(os.path.basename(image_path), fontsize=9, color="grey")

    ax = axes[1]
    ax.axis("off")
    lines = [f"Q1 – Galaxy shape: {q1_label}", ""]
    for name, p in q1_probs.items():
        lines.append(f"  {name:<22} {p:.3f}")
    lines += ["", f"Q2 – Edge-on?: {q2_label}", ""]
    for name, p in q2_probs.items():
        lines.append(f"  {name:<22} {p:.3f}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace")

    plt.suptitle("Galaxy Classifier — Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None)
    args = parser.parse_args()

    if args.image:
        img_path = args.image
    else:
        files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
        img_path = os.path.join(IMAGES_DIR, files[0])

    img, q1_label, q1_probs, q2_label, q2_probs = predict(img_path)

    print(f"Image : {img_path}\n")
    print(f"Q1 – Galaxy shape: {q1_label}")
    for name, p in q1_probs.items():
        print(f"    {name:20s}: {p:.3f}")

    print(f"\nQ2 – Edge-on?: {q2_label}")
    for name, p in q2_probs.items():
        print(f"    {name:20s}: {p:.3f}")
    
    show_prediction(img, img_path, q1_label, q1_probs, q2_label, q2_probs)