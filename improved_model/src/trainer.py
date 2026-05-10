"""
improved_model/src/trainer.py
Training loop for the ResNet-50 improved model.
Supports: full fine-tuning, gradual unfreezing, cosine LR schedule, label smoothing, mixup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from improved_model.src.model import apply_unfreeze_schedule


def mixup_data(x, y, alpha=0.4, device="cpu"):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ImprovedTrainer:
    """
    Handles training for the improved ResNet-50 model with all enhancements.
    """

    def __init__(self, model, train_loader, val_loader,
                 config, class_weights, device, logger, epoch_logger):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = config
        self.device       = device
        self.logger       = logger
        self.epoch_logger = epoch_logger

        # Label smoothing CrossEntropy (Ablation C)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=config.LABEL_SMOOTHING,
        )

        # Differential learning rates: backbone gets lower LR than head
        backbone_params = [p for n, p in model.named_parameters()
                           if "fc" not in n and p.requires_grad]
        head_params     = [p for n, p in model.named_parameters()
                           if "fc" in n and p.requires_grad]

        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": config.LEARNING_RATE * config.BACKBONE_LR_MULTIPLIER},
            {"params": head_params,     "lr": config.LEARNING_RATE},
        ], weight_decay=config.WEIGHT_DECAY)

        # Cosine annealing LR scheduler (Ablation B)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        ) if config.LR_SCHEDULER == "cosine" else None

        self.best_val_f1  = -1.0
        self.patience_ctr = 0
        self.best_epoch   = 0

        for d in [config.CHECKPOINT_DIR, config.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)

    def _get_current_lr(self):
        return self.optimizer.param_groups[-1]["lr"]   # head LR

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            if self.cfg.USE_MIXUP:
                imgs, labels_a, labels_b, lam = mixup_data(
                    imgs, labels, alpha=self.cfg.MIXUP_ALPHA, device=self.device
                )
                logits = self.model(imgs)
                loss   = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                preds  = logits.argmax(dim=1)
                # For mixup, count accuracy w.r.t. dominant label
                correct += (lam * (preds == labels_a).float() +
                            (1 - lam) * (preds == labels_b).float()).sum().item()
                all_labels.extend(labels_a.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            else:
                logits = self.model(imgs)
                loss   = self.criterion(logits, labels)
                preds  = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability during fine-tuning
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)

            if (batch_idx + 1) % self.cfg.LOG_INTERVAL == 0:
                self.logger.debug(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss: {loss.item():.4f} | LR: {self._get_current_lr():.2e}"
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
            preds  = logits.argmax(dim=1)
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
        self.logger.info("Starting Improved (ResNet-50) training")
        self.logger.info(
            f"Epochs: {self.cfg.NUM_EPOCHS} | LR: {self.cfg.LEARNING_RATE} "
            f"| Batch: {self.cfg.BATCH_SIZE} | LR Scheduler: {self.cfg.LR_SCHEDULER} "
            f"| Label Smoothing: {self.cfg.LABEL_SMOOTHING} | Mixup: {self.cfg.USE_MIXUP}"
        )
        self.logger.info("=" * 60)

        start_time = time.time()
        lr_history = []

        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            ep_start = time.time()

            # Gradual unfreezing
            apply_unfreeze_schedule(
                self.model, epoch, self.cfg.UNFREEZE_SCHEDULE, self.logger
            )
            # Rebuild optimizer param groups after unfreezing (new params become trainable)
            if epoch in self.cfg.UNFREEZE_SCHEDULE:
                self._rebuild_optimizer()

            tr_loss, tr_acc, tr_f1   = self._train_epoch(epoch)
            val_loss, val_acc, val_f1 = self._val_epoch()

            current_lr = self._get_current_lr()
            lr_history.append(current_lr)

            if self.scheduler:
                self.scheduler.step()

            ep_time = time.time() - ep_start

            self.logger.info(
                f"Epoch {epoch:3d}/{self.cfg.NUM_EPOCHS} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e} | Time: {ep_time:.1f}s"
            )

            self.epoch_logger.log(
                epoch=epoch,
                train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                val_loss=val_loss,   val_acc=val_acc,   val_f1=val_f1,
                lr=current_lr, epoch_time=ep_time,
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1  = val_f1
                self.best_epoch   = epoch
                self.patience_ctr = 0
                torch.save({
                    "epoch":           epoch,
                    "model_state":     self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                    "val_f1":          val_f1,
                    "val_acc":         val_acc,
                    "val_loss":        val_loss,
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

        torch.save({"epoch": epoch, "model_state": self.model.state_dict()},
                   self.cfg.FINAL_MODEL_PATH)

        total_time = time.time() - start_time
        self.logger.info(
            f"Training complete | Best epoch: {self.best_epoch} "
            f"| Best val F1: {self.best_val_f1:.4f} "
            f"| Total time: {total_time/60:.1f} min"
        )

        summary = {
            "best_epoch":     self.best_epoch,
            "best_val_f1":    self.best_val_f1,
            "total_epochs":   epoch,
            "total_time_min": round(total_time / 60, 2),
            "model":          "resnet50_improved",
            "lr_history":     lr_history,
        }
        with open(os.path.join(self.cfg.RESULTS_DIR, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return self.epoch_logger.get_history()

    def _rebuild_optimizer(self):
        """Rebuild optimizer after unfreezing to include newly trainable params."""
        backbone_params = [p for n, p in self.model.named_parameters()
                           if "fc" not in n and p.requires_grad]
        head_params     = [p for n, p in self.model.named_parameters()
                           if "fc" in n and p.requires_grad]
        current_lr = self._get_current_lr()
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": current_lr * self.cfg.BACKBONE_LR_MULTIPLIER},
            {"params": head_params,     "lr": current_lr},
        ], weight_decay=self.cfg.WEIGHT_DECAY)
        if self.scheduler:
            remaining = self.cfg.T_MAX - (self.best_epoch if self.best_epoch else 0)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(remaining, 1), eta_min=self.cfg.ETA_MIN
            )
        self.logger.info("Optimizer rebuilt with newly unfrozen parameters.")
