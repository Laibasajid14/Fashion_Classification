"""
comparison/compare_models.py
Side-by-side comparison of baseline vs improved model results.
Produces comparison plots and a summary table.

Run after both models have been evaluated:
    python comparison/compare_models.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.dataset_config import CLASS_NAMES

OUTPUT_DIR       = "comparison/outputs"
BASELINE_RESULTS = "baseline_model/outputs/results"
IMPROVED_RESULTS = "improved_model/outputs/results"


def load_metrics(results_dir, prefix):
    path = os.path.join(results_dir, f"{prefix}metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics not found: {path}\nRun the model evaluation first.")
    with open(path) as f:
        return json.load(f)


def load_history(results_dir):
    path = os.path.join(results_dir, "training_history.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def plot_comparison_metrics(base_m, imp_m, save_path):
    """Bar chart comparing top-level metrics side by side."""
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)"]
    base_vals = [base_m.get(k, 0) for k in metric_keys]
    imp_vals  = [imp_m.get(k, 0)  for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, base_vals, width, label="Baseline (ResNet-18)", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, imp_vals,  width, label="Improved (ResNet-50)", color="#DD8452", alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Improved — Key Metrics (Test Set)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Comparison] Metrics comparison saved → {save_path}")


def plot_per_class_f1_comparison(base_m, imp_m, save_path):
    """Horizontal grouped bar chart of per-class F1 for both models."""
    classes = CLASS_NAMES
    base_f1 = [base_m["per_class"][c]["f1"] for c in classes]
    imp_f1  = [imp_m["per_class"][c]["f1"]  for c in classes]

    y = np.arange(len(classes))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(y - height/2, base_f1, height, label="Baseline (ResNet-18)", color="#4C72B0", alpha=0.85)
    ax.barh(y + height/2, imp_f1,  height, label="Improved (ResNet-50)", color="#DD8452", alpha=0.85)

    ax.set_yticks(y); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-class F1 Score — Baseline vs Improved")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Comparison] Per-class F1 comparison saved → {save_path}")


def plot_training_curves_comparison(base_hist, imp_hist, save_path):
    """Overlay training curves from both models on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, ylabel) in enumerate([("val_loss", "Validation Loss"),
                                                ("val_f1",   "Validation Macro F1")]):
        ax = axes[ax_idx]
        if base_hist is not None and metric in base_hist.columns:
            ax.plot(base_hist["epoch"], base_hist[metric],
                    label="Baseline (ResNet-18)", marker="o", ms=3, color="#4C72B0")
        if imp_hist is not None and metric in imp_hist.columns:
            ax.plot(imp_hist["epoch"], imp_hist[metric],
                    label="Improved (ResNet-50)", marker="o", ms=3, color="#DD8452")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — Comparison")
        ax.legend(); ax.grid(alpha=0.3)
        if metric == "val_f1":
            ax.set_ylim(0, 1.05)

    plt.suptitle("Training Curves Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Comparison] Training curves comparison saved → {save_path}")


def make_summary_table(base_m, imp_m, save_path):
    """Create and save a summary comparison CSV and print it."""
    metric_keys = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "top5_accuracy", "roc_auc_macro",
    ]
    rows = []
    for k in metric_keys:
        b = base_m.get(k)
        i = imp_m.get(k)
        delta = (i - b) if (b is not None and i is not None) else None
        rows.append({
            "metric":   k,
            "baseline": round(b, 4) if b is not None else "N/A",
            "improved": round(i, 4) if i is not None else "N/A",
            "delta":    round(delta, 4) if delta is not None else "N/A",
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading results...")
    base_m = load_metrics(BASELINE_RESULTS, "baseline_")
    imp_m  = load_metrics(IMPROVED_RESULTS, "improved_")
    base_hist = load_history(BASELINE_RESULTS)
    imp_hist  = load_history(IMPROVED_RESULTS)

    # Comparison plots
    plot_comparison_metrics(
        base_m, imp_m,
        os.path.join(OUTPUT_DIR, "metrics_comparison.png"),
    )
    plot_per_class_f1_comparison(
        base_m, imp_m,
        os.path.join(OUTPUT_DIR, "per_class_f1_comparison.png"),
    )
    plot_training_curves_comparison(
        base_hist, imp_hist,
        os.path.join(OUTPUT_DIR, "training_curves_comparison.png"),
    )

    # Summary table
    df = make_summary_table(
        base_m, imp_m,
        os.path.join(OUTPUT_DIR, "model_comparison_summary.csv"),
    )

    # Save full merged JSON
    full = {
        "baseline": {k: v for k, v in base_m.items() if k != "per_class"},
        "improved": {k: v for k, v in imp_m.items() if k != "per_class"},
    }
    with open(os.path.join(OUTPUT_DIR, "full_comparison.json"), "w") as f:
        json.dump(full, f, indent=2)

    print(f"\nAll comparison outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
