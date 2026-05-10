"""
improved_model/src/evaluate.py
Full evaluation of the trained improved model on the test split.
Mirrors baseline evaluate.py for direct comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import numpy as np
import pandas as pd
import torch

from configs.dataset_config import CLASS_NAMES
import configs.improved_config as cfg
from utils.metrics import (
    compute_all_metrics, get_confusion_matrix, top_confused_pairs,
    plot_confusion_matrix, plot_per_class_f1, save_results,
    collect_predictions, save_qualitative_examples,
)
from utils.gradcam import generate_gradcam_grid
from utils.dataset import get_transforms


def evaluate_improved(model, test_loader, test_df, device, logger):
    for d in [cfg.RESULTS_DIR, cfg.PLOTS_DIR, cfg.QUALITATIVE_DIR, cfg.GRADCAM_DIR]:
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running evaluation on test set — Improved (ResNet-50)")
    logger.info("=" * 60)

    preds = collect_predictions(model, test_loader, device)
    y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]

    logger.info(f"Test samples: {len(y_true)}")

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

    for cls, vals in metrics["per_class"].items():
        logger.debug(f"  {cls:30s} P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}")

    cm = get_confusion_matrix(y_true, y_pred)
    confused_pairs = top_confused_pairs(cm, top_k=3)
    logger.info("Top confused pairs:")
    for pair in confused_pairs:
        logger.info(f"  {pair['true_class']} → {pair['pred_class']}: {pair['count']} times")

    plot_confusion_matrix(
        cm,
        save_path=os.path.join(cfg.PLOTS_DIR, "confusion_matrix_test.png"),
        title="Confusion Matrix — Improved ResNet-50 (Test Set)",
    )

    plot_per_class_f1(
        metrics,
        save_path=os.path.join(cfg.PLOTS_DIR, "per_class_f1_test.png"),
        title="Per-class F1 Score — Improved ResNet-50 (Test Set)",
    )

    hist_path = os.path.join(cfg.RESULTS_DIR, "training_history.csv")
    history = {}
    if os.path.exists(hist_path):
        hdf = pd.read_csv(hist_path)
        history = hdf.to_dict(orient="list")

    save_results(metrics, confused_pairs, history, cfg.RESULTS_DIR, prefix="improved_")

    pred_df = pd.DataFrame({
        "y_true":     y_true,
        "y_pred":     y_pred,
        "confidence": y_prob[np.arange(len(y_prob)), y_pred],
        "true_class": [CLASS_NAMES[i] for i in y_true],
        "pred_class":  [CLASS_NAMES[i] for i in y_pred],
        "correct":     (y_true == y_pred).astype(int),
    })
    pred_df.to_csv(os.path.join(cfg.RESULTS_DIR, "improved_predictions.csv"), index=False)
    logger.info("Raw predictions saved.")

    logger.info("Generating qualitative examples...")
    save_qualitative_examples(
        test_df, preds, cfg.QUALITATIVE_DIR,
        n_tp=5, n_fp=5, n_fn=5, n_hard=3,
    )

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
