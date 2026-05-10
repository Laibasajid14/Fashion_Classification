"""
utils/metrics.py
All evaluation metrics, confusion matrix, and qualitative analysis helpers.
Shared by both baseline and improved model evaluation scripts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from configs.dataset_config import CLASS_NAMES, NUM_CLASSES


# ── Core metrics ───────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all required metrics for image classification.

    Args:
        y_true: array-like of ground truth labels (0-indexed)
        y_pred: array-like of predicted labels (0-indexed)
        y_prob: array-like of shape (N, C) softmax probabilities (optional, for AUC)

    Returns:
        dict with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro",  zero_division=0)
    r_macro = recall_score(   y_true, y_pred, average="macro",  zero_division=0)
    f1_macro= f1_score(       y_true, y_pred, average="macro",  zero_division=0)
    f1_w    = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    # Per-class
    p_per  = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    r_per  = recall_score(   y_true, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    f1_per = f1_score(       y_true, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

    # Top-5 accuracy: requires probabilities
    top5_acc = None
    if y_prob is not None:
        y_prob = np.array(y_prob)
        top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
        top5_acc = np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))])

    # ROC-AUC (macro OvR)
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                    average="macro", labels=list(range(NUM_CLASSES)))
        except ValueError:
            roc_auc = None   # may fail if some classes absent in small splits

    metrics = {
        "accuracy":       float(acc),
        "precision_macro": float(p_macro),
        "recall_macro":    float(r_macro),
        "f1_macro":        float(f1_macro),
        "f1_weighted":     float(f1_w),
        "top5_accuracy":   float(top5_acc) if top5_acc is not None else None,
        "roc_auc_macro":   float(roc_auc)  if roc_auc  is not None else None,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p_per[i]),
                "recall":    float(r_per[i]),
                "f1":        float(f1_per[i]),
            }
            for i in range(NUM_CLASSES)
        },
    }
    return metrics


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))


def top_confused_pairs(cm, top_k=3):
    """Return top_k most confused (non-diagonal) class pairs."""
    cm_copy = cm.copy().astype(float)
    np.fill_diagonal(cm_copy, 0)
    pairs = []
    for _ in range(top_k):
        idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        pairs.append({
            "true_class":  CLASS_NAMES[idx[0]],
            "pred_class":  CLASS_NAMES[idx[1]],
            "count":       int(cm_copy[idx]),
        })
        cm_copy[idx] = 0
    return pairs


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Confusion matrix saved → {save_path}")


def plot_per_class_f1(metrics_dict, save_path, title="Per-class F1 Score"):
    classes = CLASS_NAMES
    f1s = [metrics_dict["per_class"][c]["f1"] for c in classes]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(classes, f1s, color="steelblue")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    for bar, v in zip(bars, f1s):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Per-class F1 plot saved → {save_path}")


def plot_training_curves(history, save_path, title="Training Curves"):
    """
    Plot loss and metric curves across epochs.
    history: dict with keys 'train_loss', 'val_loss', 'train_f1', 'val_f1'
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train loss", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val loss",   marker="o", ms=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Macro F1
    axes[1].plot(epochs, history["train_f1"], label="Train F1", marker="o", ms=3)
    axes[1].plot(epochs, history["val_f1"],   label="Val F1",   marker="o", ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"{title} — Macro F1"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Training curves saved → {save_path}")


def plot_lr_curve(lrs, losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lrs, losses)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title("LR Finder")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] LR finder plot saved → {save_path}")


# ── Qualitative analysis helpers ───────────────────────────────────────────────

def collect_predictions(model, loader, device):
    """
    Run model on a DataLoader and collect all predictions with metadata.

    Returns:
        dict with keys: indices, y_true, y_pred, y_prob, correct
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_true.extend(labels.numpy())
            all_pred.extend(preds)
            all_prob.extend(probs)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)
    correct = (y_true == y_pred)

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "correct": correct}


def save_qualitative_examples(dataset_df, preds_dict, save_dir, n_tp=5, n_fp=5,
                               n_fn=5, n_hard=3, image_size=224):
    """
    Save grids of True Positives, False Positives, False Negatives, and Hard Cases.

    Args:
        dataset_df:  DataFrame used to build the dataset (with 'path', 'label')
        preds_dict:  output of collect_predictions()
        save_dir:    directory to save PNG grids
    """
    os.makedirs(save_dir, exist_ok=True)
    from PIL import Image as PILImage

    y_true   = preds_dict["y_true"]
    y_pred   = preds_dict["y_pred"]
    y_prob   = preds_dict["y_prob"]
    correct  = preds_dict["correct"]
    paths    = dataset_df["path"].values
    labels   = dataset_df["label"].values

    # Confidence of predicted class
    confidence = y_prob[np.arange(len(y_prob)), y_pred]

    def _grid(indices, title, filename, max_n):
        indices = indices[:max_n]
        if len(indices) == 0:
            return
        cols = min(len(indices), 5)
        rows = (len(indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        axes = np.array(axes).flatten()
        for ax in axes:
            ax.axis("off")
        for i, idx in enumerate(indices):
            try:
                img = PILImage.open(paths[idx]).convert("RGB")
                x1, y1, x2, y2 = (dataset_df.iloc[idx][c] for c in ["x1","y1","x2","y2"])
                img = img.crop((max(0,x1), max(0,y1), x2, y2))
                img = img.resize((image_size, image_size))
            except Exception:
                img = PILImage.new("RGB", (image_size, image_size), (200, 200, 200))
            axes[i].imshow(img)
            t = CLASS_NAMES[y_true[idx]]
            p = CLASS_NAMES[y_pred[idx]]
            c = confidence[idx]
            axes[i].set_title(f"GT: {t}\nPred: {p}\nConf: {c:.2f}", fontsize=7)
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=120)
        plt.close()
        print(f"[Qualitative] {filename} saved")

    # True Positives (correct, high confidence)
    tp_idx = np.where(correct)[0]
    tp_idx = tp_idx[np.argsort(confidence[tp_idx])[::-1]]
    _grid(tp_idx, "True Positives (correct & confident)", "true_positives.png", n_tp)

    # False Positives (wrong, high confidence)
    fp_idx = np.where(~correct)[0]
    fp_idx = fp_idx[np.argsort(confidence[fp_idx])[::-1]]
    _grid(fp_idx, "False Positives (wrong & confident)", "false_positives.png", n_fp)

    # False Negatives — per class: model missed the true class
    fn_idx = np.where(~correct)[0]
    fn_idx = fn_idx[np.argsort(confidence[fn_idx])]   # lowest confidence = most missed
    _grid(fn_idx, "False Negatives (missed predictions)", "false_negatives.png", n_fn)

    # Hard Cases — borderline confidence (closest to 1/N)
    margin = np.abs(confidence - (1.0 / NUM_CLASSES))
    hard_idx = np.argsort(margin)  # smallest margin = most uncertain
    _grid(hard_idx, "Hard Cases (most uncertain)", "hard_cases.png", n_hard)


def save_results(metrics, confused_pairs, history, output_dir, prefix=""):
    """Save metrics dict, confused pairs, and history to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Full metrics JSON
    out_path = os.path.join(output_dir, f"{prefix}metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Results] Metrics saved → {out_path}")

    # Per-class CSV
    rows = []
    for cls, vals in metrics["per_class"].items():
        rows.append({"class": cls, **vals})
    per_class_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{prefix}per_class_metrics.csv")
    per_class_df.to_csv(csv_path, index=False)
    print(f"[Results] Per-class CSV saved → {csv_path}")

    # Summary CSV (one row)
    summary = {k: v for k, v in metrics.items() if k != "per_class"}
    summary_df = pd.DataFrame([summary])
    sum_path = os.path.join(output_dir, f"{prefix}summary_metrics.csv")
    summary_df.to_csv(sum_path, index=False)

    # Confused pairs JSON
    cp_path = os.path.join(output_dir, f"{prefix}confused_pairs.json")
    with open(cp_path, "w") as f:
        json.dump(confused_pairs, f, indent=2)

    # Training history CSV
    if history:
        hist_df = pd.DataFrame(history)
        hist_path = os.path.join(output_dir, f"{prefix}training_history.csv")
        hist_df.to_csv(hist_path, index=False)
        print(f"[Results] Training history saved → {hist_path}")
