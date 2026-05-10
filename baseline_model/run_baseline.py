"""
baseline_model/run_baseline.py
Main entry point for baseline model training and evaluation.

Usage (from project root):
    cd baseline_model
    python run_baseline.py [--skip-training] [--eval-only]

Outputs saved to baseline_model/outputs/
"""

import sys
import os
import argparse

# Set working directory to this file's directory for relative config paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import json

import configs.baseline_config as cfg
from configs.dataset_config import TRAIN_CSV, VAL_CSV, NUM_CLASSES, CLASS_NAMES

from utils.dataset import (
    load_and_clean_df, stratified_split, compute_class_weights,
    build_dataloaders, save_splits,
)
from utils.metrics import plot_training_curves
from utils.logger import setup_logger, EpochLogger

from baseline_model.src.model import build_baseline_model, model_summary
from baseline_model.src.trainer import BaselineTrainer
from baseline_model.src.evaluate import evaluate_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline ResNet-18 training")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, load best checkpoint and run eval only")
    parser.add_argument("--eval-only", action="store_true",
                        help="Alias for --skip-training")
    return parser.parse_args()


def main():
    args = parse_args()
    skip_training = args.skip_training or args.eval_only

    # ── Setup ──────────────────────────────────────────────────────────────────
    for d in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR, cfg.PLOTS_DIR,
              cfg.RESULTS_DIR, cfg.LOGS_DIR, cfg.QUALITATIVE_DIR, cfg.GRADCAM_DIR]:
        os.makedirs(d, exist_ok=True)

    logger       = setup_logger("baseline", cfg.LOGS_DIR)
    epoch_logger = EpochLogger(cfg.LOGS_DIR, filename="epoch_log.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ───────────────────────────────────────────────────────────────────
    logger.info("Loading and cleaning dataset...")
    train_raw = load_and_clean_df(TRAIN_CSV)
    val_raw   = load_and_clean_df(VAL_CSV)

    # Combine train + val then re-split to get proper 70/15/15
    import pandas as pd
    combined = pd.concat([train_raw, val_raw], ignore_index=True)
    logger.info(f"Combined annotations: {len(combined)}")

    train_df, val_df, test_df = stratified_split(combined)
    save_splits(train_df, val_df, test_df, cfg.RESULTS_DIR)

    class_weights = compute_class_weights(train_df)
    logger.info(f"Class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f})")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        augmentations=cfg.AUGMENTATIONS,
    )
    logger.info(f"Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_baseline_model(
        dropout_rate=cfg.DROPOUT_RATE,
        pretrained=cfg.PRETRAINED,
        freeze_backbone=cfg.FREEZE_BACKBONE,
    )
    stats = model_summary(model, logger)
    with open(os.path.join(cfg.RESULTS_DIR, "model_stats.json"), "w") as f:
        json.dump({"model": "resnet18_baseline", **stats}, f, indent=2)

    # ── Training ───────────────────────────────────────────────────────────────
    if not skip_training:
        trainer = BaselineTrainer(
            model, train_loader, val_loader,
            config=cfg,
            class_weights=class_weights,
            device=device,
            logger=logger,
            epoch_logger=epoch_logger,
        )
        history = trainer.train()

        # Plot training curves
        if history:
            plot_training_curves(
                {
                    "train_loss": history.get("train_loss", []),
                    "val_loss":   history.get("val_loss",   []),
                    "train_f1":   history.get("train_f1",   []),
                    "val_f1":     history.get("val_f1",     []),
                },
                save_path=os.path.join(cfg.PLOTS_DIR, "training_curves.png"),
                title="Baseline ResNet-18 Training Curves",
            )
    else:
        logger.info("Skipping training — loading best checkpoint.")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint from {cfg.BEST_MODEL_PATH}")
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    metrics = evaluate_baseline(model, test_loader, test_df, device, logger)

    logger.info("=" * 60)
    logger.info("BASELINE FINAL RESULTS")
    logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:      {metrics['f1_macro']:.4f}")
    logger.info(f"  Weighted F1:   {metrics['f1_weighted']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
