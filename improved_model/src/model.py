"""
improved_model/src/model.py
ResNet-50 with full fine-tuning and gradual unfreezing support.

Architecture justification (see improved_config.py for full rationale):
  ResNet-50 chosen over ViT and EfficientNet because:
  - Same family as baseline (ResNet-18) → clean ablation of depth effect
  - ViT requires domain-specific pretraining (DeiT/CLIP) to outperform at this scale
  - ResNet-50 has straightforward Grad-CAM support; ViT attention rollout is experimental
  - Validated on DeepFashion in the M1 literature review (Zheng et al., Ge et al.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torchvision.models as models
from configs.dataset_config import NUM_CLASSES


def build_improved_model(dropout_rate=0.5, pretrained=True, freeze_backbone=False):
    """
    Build ResNet-50 with a custom classification head.

    Architecture:
        ResNet-50 backbone (pretrained on ImageNet)
        → AdaptiveAvgPool (built-in, 1×1 output)
        → Dropout(p=dropout_rate)
        → Linear(2048, 512)
        → ReLU
        → Dropout(p=dropout_rate/2)
        → Linear(512, NUM_CLASSES)

    The two-layer head (vs single linear in baseline) gives the improved model
    a stronger non-linear classifier, contributing to measurable gains.

    Args:
        dropout_rate:     main dropout rate (halved for second dropout layer)
        pretrained:       load ImageNet weights
        freeze_backbone:  if True, freeze backbone at init (used for gradual unfreezing)

    Returns:
        nn.Module
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC with a two-layer head
    in_features = model.fc.in_features   # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(512, NUM_CLASSES),
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def apply_unfreeze_schedule(model, epoch, schedule, logger=None):
    """
    Gradually unfreeze ResNet-50 layer groups per the schedule in config.

    Args:
        model:    ResNet-50 nn.Module
        epoch:    current epoch (1-indexed)
        schedule: dict {epoch_threshold: list_of_layer_names or None}
                  None means unfreeze everything
        logger:   optional Logger

    The schedule is applied at the start of the relevant epoch.
    """
    # Find the highest threshold <= current epoch
    applicable = {k: v for k, v in schedule.items() if k <= epoch}
    if not applicable:
        return

    target_epoch = max(applicable.keys())
    target_layers = applicable[target_epoch]

    if epoch != target_epoch:
        return   # only act at the exact epoch threshold

    if target_layers is None:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        msg = f"Epoch {epoch}: All layers unfrozen"
    else:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze specified layers
        for layer_name in target_layers:
            layer = getattr(model, layer_name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True
        msg = f"Epoch {epoch}: Unfrozen layers: {target_layers}"

    if logger:
        logger.info(msg)
    else:
        print(msg)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model, logger=None):
    total, trainable = count_parameters(model)
    msg = (
        f"Model: ResNet-50 Improved\n"
        f"  Total parameters:     {total:,}\n"
        f"  Trainable parameters: {trainable:,}\n"
        f"  Frozen parameters:    {total - trainable:,}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return {"total_params": total, "trainable_params": trainable}
