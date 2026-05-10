"""
improved_model/run_improved.py
Main entry point for improved model training and evaluation.

Usage (from project root):
    cd improved_model
    python run_improved.py [--skip-training] [--skip-hp-search] [--skip-ablation]

Outputs saved to improved_model/outputs/
"""

import sys
import os
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import torch
import pandas as pd

import configs.improved_config as cfg
from configs.dataset_config import TRAIN_CSV, VAL_CSV, CLASS_NAMES

from utils.dataset import (
    load_and_clean_df, stratified_split, compute_class_weights,
    build_dataloaders, save_splits,
)
from utils.metrics import plot_training_curves
from utils.logger import setup_logger, EpochLogger

from improved_model.src.model import build_improved_model, model_summary
from improved_model.src.trainer import ImprovedTrainer
from improved_model.src.evaluate import evaluate_improved
from improved_model.src.hparam_search import (
    random_hp_search, lr_finder, run_ablation_study
)


def parse_args():
    parser = argparse.ArgumentParser(description="Improved ResNet-50 training")
    parser.add_argument("--skip-training",  action="store_true")
    parser.add_argument("--skip-hp-search", action="store_true")
    parser.add_argument("--skip-ablation",  action="store_true")
    parser.add_argument("--eval-only",      action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    skip_training = args.skip_training or args.eval_only

    for d in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR, cfg.PLOTS_DIR, cfg.RESULTS_DIR,
              cfg.LOGS_DIR, cfg.QUALITATIVE_DIR, cfg.GRADCAM_DIR]:
        os.makedirs(d, exist_ok=True)

    logger       = setup_logger("improved", cfg.LOGS_DIR)
    epoch_logger = EpochLogger(cfg.LOGS_DIR, filename="epoch_log.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ───────────────────────────────────────────────────────────────────
    logger.info("Loading and cleaning dataset...")
    train_raw = load_and_clean_df(TRAIN_CSV)
    val_raw   = load_and_clean_df(VAL_CSV)
    combined  = pd.concat([train_raw, val_raw], ignore_index=True)
    logger.info(f"Combined annotations: {len(combined)}")

    train_df, val_df, test_df = stratified_split(combined)
    save_splits(train_df, val_df, test_df, cfg.RESULTS_DIR)

    class_weights = compute_class_weights(train_df)
    logger.info(f"Class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f})")

    # ── Subsample training set to keep epoch time ~10 min on T4 ──────────
    SUBSET = 50000
    if len(train_df) > SUBSET:
        train_df = train_df.sample(n=SUBSET, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled train set to {SUBSET} samples for speed")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        augmentations=cfg.AUGMENTATIONS,
    )
    logger.info(f"Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Hyperparameter search (before full training) ────────────────────────────
    if not args.skip_hp_search and not skip_training:
        logger.info("Running hyperparameter search...")
        best_hp = random_hp_search(train_loader, val_loader, class_weights, device, logger)
        # Optionally override config with best found params
        logger.info(f"Using best HP: LR={best_hp['lr']}, Dropout={best_hp['dropout']}")
    else:
        logger.info("Skipping HP search.")

    # ── LR finder ──────────────────────────────────────────────────────────────
    if not args.skip_hp_search and not skip_training:
        logger.info("Running LR finder...")
        model_for_lr = build_improved_model(pretrained=True, freeze_backbone=False)
        lr_finder(model_for_lr, train_loader, class_weights, device, logger)

    # ── Ablation study ─────────────────────────────────────────────────────────
    if not args.skip_ablation and not skip_training:
        logger.info("Running ablation study...")
        ablation_results = run_ablation_study(
            train_loader, val_loader, class_weights, device, logger
        )
    else:
        logger.info("Skipping ablation study.")

    # ── Build full model ────────────────────────────────────────────────────────
    model = build_improved_model(
        dropout_rate=cfg.DROPOUT_RATE,
        pretrained=cfg.PRETRAINED,
        freeze_backbone=True,   # start frozen; unfreezing schedule handles the rest
    )
    stats = model_summary(model, logger)
    with open(os.path.join(cfg.RESULTS_DIR, "model_stats.json"), "w") as f:
        json.dump({"model": "resnet50_improved", **stats}, f, indent=2)

    # ── Training ───────────────────────────────────────────────────────────────
    if not skip_training:
        trainer = ImprovedTrainer(
            model, train_loader, val_loader,
            config=cfg,
            class_weights=class_weights,
            device=device,
            logger=logger,
            epoch_logger=epoch_logger,
        )
        history = trainer.train()

        if history:
            plot_training_curves(
                {
                    "train_loss": history.get("train_loss", []),
                    "val_loss":   history.get("val_loss",   []),
                    "train_f1":   history.get("train_f1",   []),
                    "val_f1":     history.get("val_f1",     []),
                },
                save_path=os.path.join(cfg.PLOTS_DIR, "training_curves.png"),
                title="Improved ResNet-50 Training Curves",
            )
    else:
        logger.info("Skipping training — loading best checkpoint.")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint from {cfg.BEST_MODEL_PATH}")
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    metrics = evaluate_improved(model, test_loader, test_df, device, logger)

    logger.info("=" * 60)
    logger.info("IMPROVED FINAL RESULTS")
    logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:      {metrics['f1_macro']:.4f}")
    logger.info(f"  Weighted F1:   {metrics['f1_weighted']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
