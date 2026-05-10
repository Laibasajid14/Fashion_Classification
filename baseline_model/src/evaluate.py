"""
baseline_model/src/evaluate.py
Full evaluation of the trained baseline model on the test split.
Produces all metrics, confusion matrix, qualitative examples, and Grad-CAM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import torch
import numpy as np
import pandas as pd

from configs.dataset_config import CLASS_NAMES, NUM_CLASSES
import configs.baseline_config as cfg
from utils.metrics import (
    compute_all_metrics, get_confusion_matrix, top_confused_pairs,
    plot_confusion_matrix, plot_per_class_f1, save_results, collect_predictions,
    save_qualitative_examples,
)
from utils.gradcam import generate_gradcam_grid
from utils.dataset import get_transforms


def evaluate_baseline(model, test_loader, test_df, device, logger):
    """
    Run full evaluation on the test set.

    Args:
        model:       loaded ResNet-18 model (weights from best checkpoint)
        test_loader: DataLoader for test split
        test_df:     DataFrame for test split
        device:      torch.device
        logger:      logging.Logger

    Saves all outputs to cfg.RESULTS_DIR, cfg.PLOTS_DIR, cfg.QUALITATIVE_DIR,
    cfg.GRADCAM_DIR.
    """
    for d in [cfg.RESULTS_DIR, cfg.PLOTS_DIR, cfg.QUALITATIVE_DIR, cfg.GRADCAM_DIR]:
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running evaluation on test set — Baseline (ResNet-18)")
    logger.info("=" * 60)

    # ── Collect predictions ────────────────────────────────────────────────────
    preds = collect_predictions(model, test_loader, device)
    y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]

    logger.info(f"Test samples: {len(y_true)}")

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    logger.info(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['f1_macro']:.4f} | "
        f"Weighted F1: {metrics['f1_weighted']:.4f}"
    )
    if metrics["top5_accuracy"]:
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    if metrics["roc_auc_macro"]:
        logger.info(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")

    # Per-class log
    for cls, vals in metrics["per_class"].items():
        logger.debug(f"  {cls:30s} P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = get_confusion_matrix(y_true, y_pred)
    confused_pairs = top_confused_pairs(cm, top_k=3)
    logger.info("Top confused pairs:")
    for pair in confused_pairs:
        logger.info(f"  {pair['true_class']} → {pair['pred_class']}: {pair['count']} times")

    plot_confusion_matrix(
        cm,
        save_path=os.path.join(cfg.PLOTS_DIR, "confusion_matrix_test.png"),
        title="Confusion Matrix — Baseline ResNet-18 (Test Set)",
    )

    # ── Per-class F1 plot ──────────────────────────────────────────────────────
    plot_per_class_f1(
        metrics,
        save_path=os.path.join(cfg.PLOTS_DIR, "per_class_f1_test.png"),
        title="Per-class F1 Score — Baseline ResNet-18 (Test Set)",
    )

    # ── Save all results ───────────────────────────────────────────────────────
    # Load training history if available
    hist_path = os.path.join(cfg.RESULTS_DIR, "training_history.csv")
    history = {}
    if os.path.exists(hist_path):
        hdf = pd.read_csv(hist_path)
        history = hdf.to_dict(orient="list")

    save_results(metrics, confused_pairs, history, cfg.RESULTS_DIR, prefix="baseline_")

    # Save raw predictions
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": y_prob[np.arange(len(y_prob)), y_pred],
        "true_class": [CLASS_NAMES[i] for i in y_true],
        "pred_class":  [CLASS_NAMES[i] for i in y_pred],
        "correct":     (y_true == y_pred).astype(int),
    })
    pred_df.to_csv(os.path.join(cfg.RESULTS_DIR, "baseline_predictions.csv"), index=False)
    logger.info("Raw predictions saved.")

    # ── Qualitative examples ───────────────────────────────────────────────────
    logger.info("Generating qualitative examples...")
    save_qualitative_examples(
        test_df, preds, cfg.QUALITATIVE_DIR,
        n_tp=5, n_fp=5, n_fn=5, n_hard=3,
    )

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    logger.info("Generating Grad-CAM visualisations...")
    eval_transform = get_transforms(cfg.IMAGE_SIZE, split="val")
    generate_gradcam_grid(
        model, test_df, device, eval_transform,
        save_dir=cfg.GRADCAM_DIR,
        class_names=CLASS_NAMES,
        n_examples=3,
        image_size=cfg.IMAGE_SIZE,
    )

    logger.info("Evaluation complete.")
    return metrics
