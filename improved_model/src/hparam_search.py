"""
improved_model/src/hparam_search.py
Random hyperparameter search and ablation study for the improved model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

import configs.improved_config as cfg
from configs.dataset_config import NUM_CLASSES
from improved_model.src.model import build_improved_model


def _quick_train(model, train_loader, val_loader, lr, dropout, class_weights,
                 device, n_epochs, label_smoothing=0.1):
    """Short training run for a hyperparameter trial."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                    label_smoothing=label_smoothing)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    best_f1 = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        best_f1 = max(best_f1, f1)

    return best_f1


def random_hp_search(train_loader, val_loader, class_weights, device, logger):
    """
    Random search over learning_rate × dropout_rate.
    Trains for ABLATION_EPOCHS epochs per trial.
    Saves results to RESULTS_DIR.
    """
    search_space = cfg.HP_SEARCH
    lr_values      = search_space["learning_rate"]
    dropout_values = search_space["dropout_rate"]
    n_trials       = search_space["num_trials"]
    trial_epochs   = search_space["epochs_per_trial"]

    all_combos = list(itertools.product(lr_values, dropout_values))
    random.shuffle(all_combos)
    selected = all_combos[:n_trials]

    results = []
    logger.info(f"HP Search: {n_trials} trials over LR × Dropout")

    for i, (lr, dr) in enumerate(selected, 1):
        logger.info(f"  Trial {i}/{n_trials}: LR={lr}, Dropout={dr}")
        model = build_improved_model(dropout_rate=dr, pretrained=True, freeze_backbone=False)
        val_f1 = _quick_train(model, train_loader, val_loader, lr, dr,
                               class_weights, device, trial_epochs)
        results.append({"lr": lr, "dropout": dr, "val_f1_macro": round(val_f1, 4)})
        logger.info(f"    → val F1 macro: {val_f1:.4f}")

    results.sort(key=lambda x: x["val_f1_macro"], reverse=True)
    best = results[0]
    logger.info(f"Best HP: LR={best['lr']}, Dropout={best['dropout']}, "
                f"val_F1={best['val_f1_macro']}")

    out_dir = os.path.join(cfg.RESULTS_DIR, "hparam_search")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hp_search_results.json"), "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "hp_search_results.csv"), index=False)
    logger.info(f"HP search results saved to {out_dir}")
    return best


def lr_finder(model, train_loader, class_weights, device, logger,
              start_lr=1e-7, end_lr=10.0, num_iter=100):
    """
    Learning rate finder: ramps LR exponentially and records loss.
    Saves LR-loss curve data.
    """
    import copy
    model_copy = copy.deepcopy(model).to(device)
    optimizer  = optim.SGD(filter(lambda p: p.requires_grad, model_copy.parameters()),
                           lr=start_lr, momentum=0.9)
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))

    mult   = (end_lr / start_lr) ** (1.0 / num_iter)
    lrs, losses = [], []
    avg_loss, best_loss = 0.0, float("inf")
    beta = 0.98   # smoothing

    data_iter = iter(train_loader)
    model_copy.train()

    for i in range(num_iter):
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, labels = next(data_iter)

        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model_copy(imgs), labels)
        loss.backward()
        optimizer.step()

        raw_loss = loss.item()
        avg_loss = beta * avg_loss + (1 - beta) * raw_loss
        smoothed = avg_loss / (1 - beta ** (i + 1))

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(smoothed)

        if smoothed > 4 * best_loss:
            logger.info(f"LR finder: loss diverged at LR={lrs[-1]:.2e}")
            break
        if smoothed < best_loss:
            best_loss = smoothed

        for g in optimizer.param_groups:
            g["lr"] *= mult

    out_dir = os.path.join(cfg.RESULTS_DIR, "lr_finder")
    os.makedirs(out_dir, exist_ok=True)
    lr_data = {"lrs": lrs, "losses": losses}
    with open(os.path.join(out_dir, "lr_finder_data.json"), "w") as f:
        json.dump(lr_data, f)

    from utils.metrics import plot_lr_curve
    plot_lr_curve(lrs, losses, os.path.join(cfg.PLOTS_DIR, "lr_finder.png"))
    logger.info("LR finder complete.")
    return lrs, losses


def run_ablation_study(train_loader, val_loader, class_weights, device, logger):
    """
    Run each ablation variant for ABLATION_EPOCHS epochs and record val F1.
    """
    results = {}
    out_dir = os.path.join(cfg.RESULTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    for name, ablation_cfg in cfg.ABLATIONS.items():
        logger.info(f"Ablation: {name} → {ablation_cfg}")
        model = build_improved_model(
            dropout_rate=ablation_cfg.get("dropout_rate", cfg.DROPOUT_RATE),
            pretrained=True,
            freeze_backbone=ablation_cfg.get("freeze_backbone", False),
        )
        val_f1 = _quick_train(
            model, train_loader, val_loader,
            lr=cfg.LEARNING_RATE,
            dropout=cfg.DROPOUT_RATE,
            class_weights=class_weights,
            device=device,
            n_epochs=cfg.ABLATION_EPOCHS,
            label_smoothing=ablation_cfg.get("label_smoothing", 0.0),
        )
        results[name] = {"config": ablation_cfg, "val_f1_macro": round(val_f1, 4)}
        logger.info(f"  {name}: val F1 = {val_f1:.4f}")

    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    import pandas as pd
    rows = [{"ablation": k, "val_f1_macro": v["val_f1_macro"]} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    logger.info(f"Ablation results saved to {out_dir}")
    return results
