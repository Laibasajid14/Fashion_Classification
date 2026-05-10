"""
baseline_model/src/trainer.py
Training loop for the ResNet-18 baseline model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score


class BaselineTrainer:
    """
    Handles training, validation, early stopping, and checkpoint saving
    for the baseline ResNet-18 model.
    """

    def __init__(self, model, train_loader, val_loader,
                 config, class_weights, device, logger, epoch_logger):
        """
        Args:
            model:          nn.Module
            train_loader:   DataLoader
            val_loader:     DataLoader
            config:         baseline_config module
            class_weights:  tensor of shape (NUM_CLASSES,) for weighted loss
            device:         torch.device
            logger:         logging.Logger
            epoch_logger:   EpochLogger instance
        """
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = config
        self.device       = device
        self.logger       = logger
        self.epoch_logger = epoch_logger

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
        )

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.best_val_f1   = -1.0
        self.patience_ctr  = 0
        self.best_epoch    = 0

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            preds       = logits.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if (batch_idx + 1) % self.cfg.LOG_INTERVAL == 0:
                self.logger.debug(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / total
        acc      = correct / total
        f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, f1

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)

            preds       = logits.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / total
        acc      = correct / total
        f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, f1

    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting Baseline (ResNet-18) training")
        self.logger.info(f"Epochs: {self.cfg.NUM_EPOCHS} | LR: {self.cfg.LEARNING_RATE} "
                         f"| Batch: {self.cfg.BATCH_SIZE} | Device: {self.device}")
        self.logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            ep_start = time.time()

            tr_loss, tr_acc, tr_f1 = self._train_epoch(epoch)
            val_loss, val_acc, val_f1 = self._val_epoch()

            ep_time = time.time() - ep_start

            self.logger.info(
                f"Epoch {epoch:3d}/{self.cfg.NUM_EPOCHS} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"Time: {ep_time:.1f}s"
            )

            self.epoch_logger.log(
                epoch=epoch,
                train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                val_loss=val_loss,   val_acc=val_acc,   val_f1=val_f1,
                lr=self.cfg.LEARNING_RATE,
                epoch_time=ep_time,
            )

            # Checkpoint on improvement
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch  = epoch
                self.patience_ctr = 0
                torch.save({
                    "epoch":          epoch,
                    "model_state":    self.model.state_dict(),
                    "optimizer_state":self.optimizer.state_dict(),
                    "val_f1":         val_f1,
                    "val_acc":        val_acc,
                    "val_loss":       val_loss,
                }, self.cfg.BEST_MODEL_PATH)
                self.logger.info(f"  ✓ Best model saved (val_f1={val_f1:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.cfg.EARLY_STOPPING_PATIENCE:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.cfg.EARLY_STOPPING_PATIENCE} epochs)"
                    )
                    break

        # Save final model
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "val_f1":      val_f1,
        }, self.cfg.FINAL_MODEL_PATH)

        total_time = time.time() - start_time
        self.logger.info(f"Training complete | Best epoch: {self.best_epoch} "
                         f"| Best val F1: {self.best_val_f1:.4f} "
                         f"| Total time: {total_time/60:.1f} min")

        # Save training summary
        summary = {
            "best_epoch":   self.best_epoch,
            "best_val_f1":  self.best_val_f1,
            "total_epochs": epoch,
            "total_time_min": round(total_time / 60, 2),
            "model":        "resnet18_baseline",
        }
        with open(os.path.join(self.cfg.RESULTS_DIR, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return self.epoch_logger.get_history()
