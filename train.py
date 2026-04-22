"""Train the galaxy morphology classifier.
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
import torch

from config import (
    LABELS_PATH, IMAGES_DIR, VAL_SPLIT, SEED,
    train_transform, val_transform,
)

class GalaxyDataset(Dataset):

    def __init__(self, csv_path, images_dir, transform=None):
        self.transform = transform

        df = pd.read_csv(csv_path, dtype={"id": "string"}) # read labels CSV, ID as string to preserve 18-digit number

        df["img_path"] = df["id"].apply(lambda gid: os.path.join(images_dir, f"{gid}.jpg"))  # build full image path for each galaxy

        self.df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True) # keep only rows where image exists on disk

    def __len__(self):       # returns total number of available galaxy images
     return len(self.df)  

    def __getitem__(self, idx):
        row = self.df.iloc[idx] # get the row at the given index
        img = Image.open(row["img_path"]).convert("RGB")  # open image and force RGB format
        if self.transform:
            img = self.transform(img)    # apply transform pipeline if provided
        return img, int(row["q1_label"]), int(row["q2_label"])    # return image and both labels
    

    class GalaxyClassifier(nn.Module):

     def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        num_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head_q1 = nn.Linear(num_features, 3)  # smooth / features / star
        self.head_q2 = nn.Linear(num_features, 2)  # edge-on / not edge-on

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