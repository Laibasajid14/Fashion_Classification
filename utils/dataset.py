"""
utils/dataset.py
Shared dataset class and dataloader factory for both models.

Key design decisions (from M1 EDA):
  - One row per garment annotation; images may appear multiple times → deduplicate
    by (path, category_id) for classification (one label per crop).
  - Crop to bounding box before resizing to focus the model on the garment region.
  - Filter extreme aspect-ratio bboxes (< 0.2 or > 5.0) as flagged in M1.
  - Stratified 70/15/15 split with fixed seed for reproducibility.
  - Class-weighted loss weights computed from training split only.
"""

import ast
import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

# Allow importing configs from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.dataset_config import (
    IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BBOX_ASPECT_MIN, BBOX_ASPECT_MAX, CLASS_NAMES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_bbox(raw):
    """Parse b_box field '[x1, y1, x2, y2]' to (x1,y1,x2,y2) ints."""
    if isinstance(raw, str):
        coords = ast.literal_eval(raw)
    else:
        coords = list(raw)
    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    return x1, y1, x2, y2


def _bbox_aspect(x1, y1, x2, y2):
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    return w / h


def load_and_clean_df(csv_path, drop_extreme_bbox=True):
    """
    Load a DeepFashion2 dataframe CSV and clean it.
    Returns a DataFrame with columns: path, category_id, x1, y1, x2, y2.
    """
    df = pd.read_csv(csv_path)

    # Keep only columns we need
    needed = ["path", "category_id", "b_box"]
    df = df[needed].dropna(subset=needed)

    # Parse bounding boxes
    df[["x1", "y1", "x2", "y2"]] = df["b_box"].apply(
        lambda v: pd.Series(_parse_bbox(v))
    )
    df = df.drop(columns=["b_box"])

    # Filter invalid boxes (w<=0 or h<=0)
    df = df[(df["x2"] > df["x1"]) & (df["y2"] > df["y1"])].copy()

    # Filter extreme aspect ratios (M1 finding: 93 outliers)
    if drop_extreme_bbox:
        ar = df.apply(lambda r: _bbox_aspect(r.x1, r.y1, r.x2, r.y2), axis=1)
        df = df[(ar >= BBOX_ASPECT_MIN) & (ar <= BBOX_ASPECT_MAX)].copy()

    # Ensure category_id is int and in [1, 13]
    df["category_id"] = df["category_id"].astype(int)
    df = df[df["category_id"].between(1, NUM_CLASSES)].copy()

    # Convert to 0-indexed label
    df["label"] = df["category_id"] - 1

    df = df.reset_index(drop=True)
    return df


def stratified_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                     seed=RANDOM_SEED):
    """
    Stratified 70/15/15 split on (path, label) unique pairs.
    Returns train_df, val_df, test_df.
    """
    # Deduplicate: one entry per (image_path, label)
    unique = df[["path", "label", "x1", "y1", "x2", "y2"]].drop_duplicates(
        subset=["path", "label"]
    ).reset_index(drop=True)

    test_ratio = 1.0 - train_ratio - val_ratio

    train_val, test = train_test_split(
        unique, test_size=test_ratio, stratify=unique["label"], random_state=seed
    )
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size_relative, stratify=train_val["label"],
        random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def compute_class_weights(train_df):
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(train_df["label"].values, minlength=NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1.0, counts)   # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  # normalise
    return torch.tensor(weights, dtype=torch.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class FashionDataset(Dataset):
    """
    Crops each garment bounding box from the full image, then applies transforms.
    One sample = one annotated garment instance.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df:        DataFrame with columns [path, label, x1, y1, x2, y2]
            transform: torchvision transform pipeline
        """
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        # Crop to bounding box (clamped to image bounds)
        w, h = img.size
        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(w, int(row["x2"]))
        y2 = min(h, int(row["y2"]))
        img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        label = int(row["label"])
        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(image_size, split="train", augmentations=None):
    """
    Returns a torchvision transform pipeline.

    Args:
        image_size:    target square size (e.g. 224)
        split:         'train', 'val', or 'test'
        augmentations: list of augmentation names (from config); used only for train
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    if split == "train" and augmentations:
        aug_list = []

        if "random_resized_crop" in augmentations:
            scale = (0.7, 1.0) if "improved" in str(augmentations) else (0.8, 1.0)
            aug_list.append(T.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        else:
            aug_list.append(T.Resize((image_size, image_size)))

        if "random_horizontal_flip" in augmentations:
            aug_list.append(T.RandomHorizontalFlip(p=0.5))

        if "color_jitter" in augmentations:
            aug_list.append(T.ColorJitter(brightness=0.3, contrast=0.3,
                                           saturation=0.3, hue=0.1))

        if "random_rotation" in augmentations:
            degrees = 20 if "random_grayscale" in augmentations else 15
            aug_list.append(T.RandomRotation(degrees=degrees))

        if "random_grayscale" in augmentations:
            aug_list.append(T.RandomGrayscale(p=0.1))

        aug_list += [T.ToTensor(), T.Normalize(mean, std)]
        return T.Compose(aug_list)

    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(train_df, val_df, test_df,
                      image_size, batch_size, num_workers,
                      augmentations=None, use_weighted_sampler=False):
    """
    Build train / val / test DataLoaders.

    Args:
        use_weighted_sampler: if True, over-sample minority classes in training
                              (alternative to weighted loss; not used by default)
    """
    train_tf = get_transforms(image_size, split="train", augmentations=augmentations)
    eval_tf  = get_transforms(image_size, split="val")

    train_ds = FashionDataset(train_df, transform=train_tf)
    val_ds   = FashionDataset(val_df,   transform=eval_tf)
    test_ds  = FashionDataset(test_df,  transform=eval_tf)

    if use_weighted_sampler:
        labels  = train_df["label"].values
        counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        weights = 1.0 / np.where(counts == 0, 1.0, counts)
        sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def save_splits(train_df, val_df, test_df, output_dir):
    """Save split DataFrames as CSV for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "split_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "split_val.csv"),   index=False)
    test_df.to_csv(os.path.join(output_dir, "split_test.csv"), index=False)
    split_info = {
        "train_samples": len(train_df),
        "val_samples":   len(val_df),
        "test_samples":  len(test_df),
        "random_seed":   RANDOM_SEED,
        "train_ratio":   TRAIN_RATIO,
        "val_ratio":     VAL_RATIO,
        "test_ratio":    TEST_RATIO,
    }
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[Dataset] Splits saved to {output_dir}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
