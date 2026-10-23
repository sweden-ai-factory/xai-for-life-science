"""Shared globals, data loading, and image transformations for the COVIDx
leakage-study notebooks (experiment-covidx and xai-covidx).

Everything here is copied from experiment-covidx.md so that the XAI notebook
can simply do:

    from utils_covidx import *
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Reproducibility / device
# ------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# NOTE: if you want to force, for example to use SHAP
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Root folders - change these if you already have the raw datasets somewhere
# ------------------------------------------------------------
ROOT = RAW = DATA = Path("data")

# COVIDx CXR-2 raw data locations
covid_repo = DATA / "covidx"
covid_train_txt = covid_repo / "train.txt"
covid_test_txt = covid_repo / "test.txt"
covid_train_images = covid_repo / "train"
covid_test_images = covid_repo / "test"

# ------------------------------------------------------------
# Global model/data config
# ------------------------------------------------------------
IMAGE_SIZE = 224
CLASSES = ["negative", "positive"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BATCH_SIZE = 32  # decreasing because of memory constraints

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def read_txt(image_dir, path):
    """Custom read function since there are spaces in the first column, which pd.read_csv cannot handle"""

    with open(path) as f:
        # rsplit from the right, at most 3 splits -> always 4 columns;
        # any extra spaces stay inside the first column (patient id)
        rows = [line.rstrip("\n").rsplit(" ", maxsplit=3) for line in f]

    rows_with_file = filter(lambda row: (image_dir / row[1]).exists(), rows)

    return pd.DataFrame(rows_with_file, columns=["patient_id", "filename", "class", "data_source"])


def make_train_val_split(train_df):
    """Stratified 80/20 train/val split, stratified by class AND source dataset."""
    class_and_data_source = train_df["class"].str.cat(train_df["data_source"])
    train_df_split, val_df_split = train_test_split(
        train_df, test_size=0.2, random_state=SEED, shuffle=True, stratify=class_and_data_source
    )
    return train_df_split, val_df_split


# ------------------------------------------------------------
# Custom composable transforms (only what torchvision lacks)
# ------------------------------------------------------------
class CropTop:
    """Crop the top `percent` of the image (X-rays often have embedded text/markers there)."""

    def __init__(self, percent=0.08):
        self.percent = percent

    def __call__(self, img):
        top = int(img.height * self.percent)
        return F.crop(img, top, 0, img.height - top, img.width)


class CenterSquareCrop:
    """Center-crop to a square of the smaller side."""

    def __call__(self, img):
        return F.center_crop(img, min(img.size))


class RandomRatioResize:
    """COVID-Net augmentation: with probability `prob`, distort the aspect ratio by
    up to `delta`, then pad back out to a square `target_size` canvas."""

    def __init__(self, prob=0.3, delta=0.1, target_size=480):
        self.prob = prob
        self.delta = delta
        self.target_size = target_size

    def __call__(self, img):
        if np.random.rand() >= self.prob:
            return img

        w, h = img.size
        ratio = h / w
        ratio = np.random.uniform(max(ratio - self.delta, 0.01), ratio + self.delta)

        if ratio <= 1.0:
            new_w, new_h = int(w * ratio), h
        else:
            new_w, new_h = w, int(h / ratio)
        img = F.resize(img, (new_h, new_w))

        pad_l = (self.target_size - img.width) // 2
        pad_t = (self.target_size - img.height) // 2
        pad_r = self.target_size - img.width - pad_l
        pad_b = self.target_size - img.height - pad_t
        return F.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=0)


def build_transforms(input_size=IMAGE_SIZE, top_percent=0.08, train=True):
    """Using torchvision, compose a image transform pipeline."""
    tf = [
        CropTop(top_percent),
        CenterSquareCrop(),
        transforms.Resize((input_size, input_size)),
    ]
    if train:
        tf += [
            RandomRatioResize(prob=0.3, delta=0.1, target_size=480),
            transforms.Resize((input_size, input_size)),  # undo the 480x480 pad canvas
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), fill=0),
        ]
    tf += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(tf)


train_tf = build_transforms(IMAGE_SIZE, train=True)
eval_tf = build_transforms(IMAGE_SIZE, train=False)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class COVIDxDataset(Dataset):
    """Reads (path, label) pairs from a manifest dataframe. `label_col` and
    `class_to_idx` let us reuse this exact class for the disease classifier
    (label_col='class') and for the leakage probe (label_col='data_source')."""

    def __init__(self, df, image_dir, transform, label_col="class", class_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_col = label_col
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = row[self.label_col]

        with Image.open(self.image_dir / filename) as img:
            img.load()

        img = self.transform(img.convert("RGB"))
        idx = self.class_to_idx[label]
        return img, idx


# ------------------------------------------------------------
# Training loop (used by both the disease classifier and the source probe)
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, use_bce=False):
    """One pass over the training loader: forward, backward, optimizer step."""
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        if use_bce:
            # BCEWithLogitsLoss (the numerically stable BCELoss) on the
            # positive-class logit; class indices become float targets
            loss = criterion(out[:, 1], y.float())
        else:
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, use_bce=False):
    """One pass over an eval loader: mean loss and accuracy."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        if use_bce:
            loss = criterion(out[:, 1], y.float())
            preds = (torch.sigmoid(out[:, 1]) >= 0.5).long()
        else:
            loss = criterion(out, y)
            preds = out.argmax(1)
        loss_sum += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader.dataset), correct / total


def train_model(model, optimizer, train_loader, val_loader, history, n_epochs=22, patience=5, epoch_start=0, use_bce=False):
    """Training loop."""
    # Two classes only, so binary cross-entropy is an alternative to
    # cross-entropy. With use_bce=True the positive-class logit is scored
    # against a float target; the dataset still yields integer class indices.
    criterion = nn.BCEWithLogitsLoss() if use_bce else nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epoch_start, n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, use_bce)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, use_bce)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # early stopping: halt when val_loss has not improved for `patience` epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping at epoch {epoch+1}: no val_loss improvement for {patience} epochs")

                # Restore best model
                model.load_state_dict(best_model_state)
                break

    return epoch