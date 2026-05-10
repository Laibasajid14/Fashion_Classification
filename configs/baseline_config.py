"""
Configuration for the Baseline Model — ResNet-18 with frozen backbone.

Architecture choice justification:
  ResNet-18 is a well-understood, lightweight residual network that serves as
  a strong, interpretable baseline. Frozen backbone + custom head gives a
  performance floor that the improved model must beat convincingly. Preferred
  over MobileNetV2 because ResNet-18 is in the same family as the improved
  ResNet-50, making the architecture comparison clean and apples-to-apples.
"""

MODEL_NAME      = "resnet18"
PRETRAINED      = True
FREEZE_BACKBONE = True       # Only train the classification head
DROPOUT_RATE    = 0.3

# ── Training ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224           # Compatible with all standard ImageNet backbones
BATCH_SIZE   = 64
NUM_WORKERS  = 4
NUM_EPOCHS   = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
OPTIMIZER    = "adam"        # Adam with fixed LR — simplest baseline setup
LR_SCHEDULER = None          # No scheduler for baseline

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC   = "val_f1_macro"

# ── Augmentation (training only) ───────────────────────────────────────────────
# ≥ 3 techniques required; using 4 for robustness
AUGMENTATIONS = [
    "random_horizontal_flip",   # p=0.5
    "random_resized_crop",      # scale=(0.8, 1.0)
    "color_jitter",             # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    "random_rotation",          # degrees=15
]

# ── Output paths (relative to baseline_model/) ────────────────────────────────
OUTPUT_DIR        = "outputs"
CHECKPOINT_DIR    = "outputs/checkpoints"
PLOTS_DIR         = "outputs/plots"
RESULTS_DIR       = "outputs/results"
LOGS_DIR          = "outputs/logs"
QUALITATIVE_DIR   = "outputs/qualitative"
GRADCAM_DIR       = "outputs/gradcam"

BEST_MODEL_PATH   = "outputs/checkpoints/best_model.pth"
FINAL_MODEL_PATH  = "outputs/checkpoints/final_model.pth"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_INTERVAL = 50    # log every N batches
