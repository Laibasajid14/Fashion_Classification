"""
Configuration for the Improved Model — ResNet-50 with full fine-tuning.

Architecture choice justification:
  ResNet-50 is chosen over EfficientNet/ViT for the following reasons:
  1. Same family as the baseline (ResNet-18), making the contribution of depth
     and full fine-tuning directly measurable in the ablation study.
  2. Well-validated on fashion benchmarks in the M1 literature review.
  3. Better compute/performance trade-off than ViT on datasets of this size
     without dedicated ViT pre-training (e.g. DeiT or CLIP).
  4. Simpler to apply Grad-CAM than ViT attention maps.

  ViT would require either a much larger dataset or a domain-specific pretrained
  checkpoint (e.g. CLIP-ViT) to outperform ResNet-50 at this scale — not
  justified given the training budget. ResNet-50 is the defensible choice.

Enhancements over baseline (at least 2 required, ablated individually):
  1. Full fine-tuning with gradual unfreezing  [ABLATION A]
  2. Cosine annealing LR schedule              [ABLATION B]
  3. Label smoothing (ε=0.1)                  [ABLATION C]
  4. Mixup augmentation (α=0.4)               [ABLATION D — optional, if time allows]
"""

MODEL_NAME      = "resnet50"
PRETRAINED      = True
FREEZE_BACKBONE = False      # Full fine-tuning
DROPOUT_RATE    = 0.5

# ── Gradual unfreezing schedule ────────────────────────────────────────────────
# Epoch at which each layer group is unfrozen (0 = all frozen at start)
# ResNet-50 layers: layer4 → layer3 → layer2 → layer1 → conv1/bn1
UNFREEZE_SCHEDULE = {
    0:  ["fc"],                          # epoch 0: only head
    2:  ["layer4", "fc"],               # epoch 2: unfreeze layer4
    5:  ["layer3", "layer4", "fc"],     # epoch 5: unfreeze layer3
    8:  None,                           # epoch 8: unfreeze all
}

# ── Training ───────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
BATCH_SIZE    = 32           # Smaller batch for larger model
NUM_WORKERS   = 4
NUM_EPOCHS    = 40
LEARNING_RATE = 3e-4         # Lower initial LR for fine-tuning
BACKBONE_LR_MULTIPLIER = 0.1 # Backbone LR = LEARNING_RATE * multiplier
WEIGHT_DECAY  = 1e-4
OPTIMIZER     = "adamw"      # AdamW for better fine-tuning regularisation
LR_SCHEDULER  = "cosine"     # CosineAnnealingLR

# Cosine annealing params
T_MAX    = 40                # Full cycle = NUM_EPOCHS
ETA_MIN  = 1e-6              # Minimum LR

EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_METRIC   = "val_f1_macro"

# ── Loss function ──────────────────────────────────────────────────────────────
LABEL_SMOOTHING = 0.1        # CrossEntropyLoss smoothing parameter

# ── Mixup augmentation ─────────────────────────────────────────────────────────
USE_MIXUP    = True
MIXUP_ALPHA  = 0.4

# ── Augmentation (same 4 + stronger crop) ─────────────────────────────────────
AUGMENTATIONS = [
    "random_horizontal_flip",
    "random_resized_crop",      # scale=(0.7, 1.0) — slightly more aggressive
    "color_jitter",             # brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
    "random_rotation",          # degrees=20
    "random_grayscale",         # p=0.1 — additional for improved model
]

# ── Hyperparameter search space (random search) ────────────────────────────────
# Search over LR × dropout (3 values each = 9 combinations)
HP_SEARCH = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "dropout_rate":  [0.3, 0.5, 0.6],
    "num_trials":    9,
    "epochs_per_trial": 5,      # Short trials for search
}

# ── Ablation study config ──────────────────────────────────────────────────────
# Each ablation trains for fewer epochs to save compute
ABLATION_EPOCHS = 10
ABLATIONS = {
    "A_full_finetune":      {"freeze_backbone": False, "lr_scheduler": None,     "label_smoothing": 0.0, "use_mixup": False},
    "B_cosine_schedule":    {"freeze_backbone": False, "lr_scheduler": "cosine", "label_smoothing": 0.0, "use_mixup": False},
    "C_label_smoothing":    {"freeze_backbone": False, "lr_scheduler": "cosine", "label_smoothing": 0.1, "use_mixup": False},
    "D_mixup":              {"freeze_backbone": False, "lr_scheduler": "cosine", "label_smoothing": 0.1, "use_mixup": True},
}

# ── Output paths (relative to improved_model/) ────────────────────────────────
OUTPUT_DIR        = "outputs"
CHECKPOINT_DIR    = "outputs/checkpoints"
PLOTS_DIR         = "outputs/plots"
RESULTS_DIR       = "outputs/results"
LOGS_DIR          = "outputs/logs"
QUALITATIVE_DIR   = "outputs/qualitative"
GRADCAM_DIR       = "outputs/gradcam"
ABLATION_DIR      = "outputs/ablation"

BEST_MODEL_PATH   = "outputs/checkpoints/best_model.pth"
FINAL_MODEL_PATH  = "outputs/checkpoints/final_model.pth"

LOG_INTERVAL = 50
