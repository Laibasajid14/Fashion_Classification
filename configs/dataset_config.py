"""
Dataset configuration for DeepFashion2 clothing classification.
Adjust DATA_ROOT to point to your local or Kaggle dataset path.
"""

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = "/kaggle/input/deepfashion2-original-with-dataframes/DeepFashion2"
IMAGE_ROOT = f"{DATA_ROOT}/deepfashion2_original_images"
DATAFRAME_ROOT = f"{DATA_ROOT}/img_info_dataframes"

TRAIN_CSV   = f"{DATAFRAME_ROOT}/train.csv"
VAL_CSV     = f"{DATAFRAME_ROOT}/validation.csv"
# Test CSV has no annotations — used only for inference demo
TEST_CSV    = f"{DATAFRAME_ROOT}/test.csv"

# ── Classes ────────────────────────────────────────────────────────────────────
# DeepFashion2 category IDs are 1-indexed
CATEGORY_MAP = {
    1:  "short sleeve top",
    2:  "long sleeve top",
    3:  "short sleeve outwear",
    4:  "long sleeve outwear",
    5:  "vest",
    6:  "sling",
    7:  "shorts",
    8:  "trousers",
    9:  "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}

NUM_CLASSES  = len(CATEGORY_MAP)
CLASS_NAMES  = [CATEGORY_MAP[i] for i in range(1, NUM_CLASSES + 1)]

# ── Split settings ─────────────────────────────────────────────────────────────
# We re-split from the labelled train+val annotations for proper 70/15/15
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── EDA findings → preprocessing ──────────────────────────────────────────────
# From M1: images highly variable; warm RGB bias; need channel-wise normalisation.
# ImageNet stats used because both backbones are pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# From M1: 93 extreme aspect-ratio bboxes (<0.2 or >5.0) → filter before training
BBOX_ASPECT_MIN = 0.2
BBOX_ASPECT_MAX = 5.0

# From M1: class imbalance ratio 122.9×  → weighted loss mandatory
# Class counts from M1 (train+val combined, approximate):
CLASS_COUNTS = {
    1:  84201,   # short sleeve top
    2:  42030,   # long sleeve top
    3:  685,     # short sleeve outwear
    4:  15468,   # long sleeve outwear
    5:  18208,   # vest
    6:  2307,    # sling
    7:  40783,   # shorts
    8:  64973,   # trousers
    9:  37357,   # skirt
    10: 20338,   # short sleeve dress
    11: 9384,    # long sleeve dress
    12: 21301,   # vest dress
    13: 7641,    # sling dress
}
