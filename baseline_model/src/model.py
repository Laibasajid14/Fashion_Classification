"""
baseline_model/src/model.py
ResNet-18 baseline — frozen backbone + custom classification head.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torchvision.models as models
from configs.dataset_config import NUM_CLASSES


def build_baseline_model(dropout_rate=0.3, pretrained=True, freeze_backbone=True):
    """
    Build ResNet-18 with a custom classification head.

    Architecture:
        ResNet-18 backbone (pretrained on ImageNet)
        → AdaptiveAvgPool (built-in, 1×1 output)
        → Dropout(p=dropout_rate)
        → Linear(512, NUM_CLASSES)

    Args:
        dropout_rate:     dropout before the final FC layer
        pretrained:       load ImageNet weights
        freeze_backbone:  if True, freeze all layers except the FC head

    Returns:
        nn.Module
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC
    in_features = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # Ensure FC head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model, logger=None):
    total, trainable = count_parameters(model)
    msg = (
        f"Model: ResNet-18 Baseline\n"
        f"  Total parameters:     {total:,}\n"
        f"  Trainable parameters: {trainable:,}\n"
        f"  Frozen parameters:    {total - trainable:,}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return {"total_params": total, "trainable_params": trainable}
