# config.py

from __future__ import annotations  # list[X] syntax on Python 3.9


import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────────────
# Root is the directory that contains this file.
ROOT_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = ROOT_DIR / "data"            # Raw & cached dataset files
RESULTS_DIR: Path = ROOT_DIR / "results"      # Saved figures & CSVs
CHECKPOINTS_DIR: Path = ROOT_DIR / "checkpoints"  # Model weight files
LOGS_DIR: Path = ROOT_DIR / "logs"            # Training / eval logs

# Create directories on import so every phase can write freely.
for _d in (DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
DATASET: str = "FashionMNIST"
IMG_SIZE: int = 28          # Height & width of each input image (pixels)
IMG_CHANNELS: int = 1       # Greyscale
NUM_CLASSES: int = 10

# Class labels (used for plotting)
FASHION_MNIST_CLASSES: list[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Baseline ANN Training
# ─────────────────────────────────────────────────────────────────────────────
ANN_BATCH_SIZE: int = 128
ANN_EPOCHS: int = 20
ANN_LR: float = 1e-3              # Adam initial learning rate
ANN_LR_STEP_SIZE: int = 7        # StepLR: decay every N epochs
ANN_LR_GAMMA: float = 0.5        # StepLR: multiplicative decay factor
ANN_WEIGHT_DECAY: float = 1e-4   # L2 regularisation

# Checkpoint filename for the best ANN weights (Phase 2 loads this).
ANN_CHECKPOINT: Path = CHECKPOINTS_DIR / "baseline_ann_best.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — ANN-to-SNN Conversion
# ─────────────────────────────────────────────────────────────────────────────
# Time-steps evaluated during SNN profiling.
SNN_TIME_STEPS: list[int] = [16, 32, 64, 128]
SNN_BATCH_SIZE: int = 64

# Weight-normalisation percentile used during threshold balancing.
# Values in [99, 100]; 99.9 is a common robust choice.
SNN_NORM_PERCENTILE: float = 99.9

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Adversarial Attacks
# ─────────────────────────────────────────────────────────────────────────────
# Spatial attack (FGSM)
FGSM_EPSILONS: list[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Temporal attack (Gaussian jitter on spike trains, in ms)
TEMPORAL_JITTER_SIGMAS: list[float] = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# Time-step to use for attack evaluation (balance speed vs. accuracy).
ATTACK_EVAL_TIMESTEPS: int = 64

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — ATTF Defense
# ─────────────────────────────────────────────────────────────────────────────
# Burst detection window (in time-steps).
ATTF_WINDOW: int = 5
# Spike count threshold inside the window that triggers adaptation.
ATTF_BURST_THRESHOLD: int = 4
# Maximum multiplicative increase of V_thresh during adaptation.
ATTF_MAX_THRESH_SCALE: float = 3.0
# Decay constant — how fast the elevated threshold returns to baseline.
ATTF_THRESH_DECAY: float = 0.9
