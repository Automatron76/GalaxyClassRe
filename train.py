"""Train the galaxy morphology classifier."""

import os
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from config import (
    LABELS_PATH, IMAGES_DIR, VAL_SPLIT, SEED,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE,
    ARTIFACTS_DIR, MODEL_WEIGHTS,
    train_transform, val_transform,
)


class GalaxyDataset(Dataset):

    def __init__(self, csv_path, images_dir, transform=None):
        self.transform = transform
        df = pd.read_csv(csv_path, dtype={"id": "string"})
        df["img_path"] = df["id"].apply(lambda gid: os.path.join(images_dir, f"{gid}.jpg"))
        self.df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["q1_label"]), int(row["q2_label"])


class GalaxyClassifier(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_q1 = nn.Linear(num_features, 3)
        self.head_q2 = nn.Linear(num_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.head_q1(features), self.head_q2(features)


def make_loaders(batch_size):
    train_ds = GalaxyDataset(LABELS_PATH, IMAGES_DIR, transform=train_transform)
    val_ds   = GalaxyDataset(LABELS_PATH, IMAGES_DIR, transform=val_transform)
    n_total = len(train_ds)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(
        Subset(train_ds, indices[:n_train].tolist()),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_ds, indices[n_train:].tolist()),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, n_train, n_val


def train(epochs, batch_size):
    train_loader, val_loader, n_train, n_val = make_loaders(batch_size)
    print(f"Device: {DEVICE}  |  train: {n_train}  val: {n_val}\n")
    model = GalaxyClassifier(pretrained=True).to(DEVICE)
    criterion_q1 = nn.CrossEntropyLoss()
    criterion_q2 = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, epochs + 1):

        model.train()
        running_loss, total = 0.0, 0
        ok_q1, ok_q2 = 0, 0

        for images, lbl_q1, lbl_q2 in train_loader:
            images = images.to(DEVICE)
            lbl_q1 = lbl_q1.to(DEVICE)
            lbl_q2 = lbl_q2.to(DEVICE)

            optimizer.zero_grad()
            logits_q1, logits_q2 = model(images)

            loss_q1 = criterion_q1(logits_q1, lbl_q1)
            loss_q2 = criterion_q2(logits_q2, lbl_q2)
            loss = loss_q1 + loss_q2

            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            total += bs
            ok_q1 += (logits_q1.argmax(1) == lbl_q1).sum().item()
            ok_q2 += (logits_q2.argmax(1) == lbl_q2).sum().item()
            
            model.eval()
        v_ok_q1, v_ok_q2, v_total = 0, 0, 0

        with torch.inference_mode():
            for images, lbl_q1, lbl_q2 in val_loader:
                images = images.to(DEVICE)
                lbl_q1 = lbl_q1.to(DEVICE)
                lbl_q2 = lbl_q2.to(DEVICE)

                logits_q1, logits_q2 = model(images)
                bs = images.size(0)
                v_total += bs
                v_ok_q1 += (logits_q1.argmax(1) == lbl_q1).sum().item()
                v_ok_q2 += (logits_q2.argmax(1) == lbl_q2).sum().item()

        print(
            f"  Epoch {epoch}/{epochs}  |  "
            f"loss {running_loss / total:.4f}  |  "
            f"train Q1 {ok_q1/total:.3f}  Q2 {ok_q2/total:.3f}  |  "
            f"val Q1 {v_ok_q1/v_total:.3f}  Q2 {v_ok_q2/v_total:.3f}"
        )
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, MODEL_WEIGHTS)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")
        
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)