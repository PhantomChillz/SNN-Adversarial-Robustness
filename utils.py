# utils.py

from __future__ import annotations  # Enables X | Y union syntax on Python 3.9

import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Create and return a logger that writes to stdout and, optionally, a file.

    Args:
        name:     Logger name (typically the calling module's ``__name__``).
        log_file: If provided, a ``FileHandler`` is added pointing to this path
                  inside ``config.LOGS_DIR``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers on re-import.
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler — DEBUG and above
    if log_file is not None:
        log_path = config.LOGS_DIR / log_file
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED) -> None:
    """
    Fix all random seeds for full experiment reproducibility.

    Sets seeds for Python's ``random`` module, NumPy, and PyTorch (both CPU
    and CUDA).  Also enables deterministic cuDNN algorithms, which may reduce
    GPU performance slightly but is essential for research comparisons.

    Args:
        seed: Integer seed value. Defaults to ``config.SEED``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available compute device (CUDA > MPS > CPU).

    Returns:
        :class:`torch.device` pointing to CUDA GPU, Apple MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_fashion_mnist_loaders(
    batch_size_train: int = config.ANN_BATCH_SIZE,
    batch_size_test: int = config.ANN_BATCH_SIZE,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download (if necessary) and return DataLoaders for Fashion-MNIST.

    Normalisation statistics (mean=0.2860, std=0.3530) are the true per-pixel
    statistics computed over the Fashion-MNIST training split.  Using the
    correct statistics is important because ANN-to-SNN conversion relies on
    weight normalisation that assumes zero-centred, unit-variance inputs.

    Args:
        batch_size_train: Mini-batch size for the training loader.
        batch_size_test:  Mini-batch size for the test loader.
        num_workers:      DataLoader worker processes.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Training pipeline: random horizontal flip for light augmentation.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # Fashion-MNIST channel statistics
        transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
    ])

    # Test pipeline: deterministic — no augmentation.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
    ])

    train_set = datasets.FashionMNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.FashionMNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    filepath: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Serialise a training state dictionary to disk.

    Args:
        state:    Dictionary containing at minimum ``model_state_dict``,
                  ``epoch``, and ``best_accuracy``.
        filepath: Destination ``.pt`` file path.
        logger:   Optional logger for confirmation messages.
    """
    torch.save(state, filepath)
    msg = f"Checkpoint saved → {filepath}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Load a checkpoint and restore model weights in-place.

    Args:
        filepath: Path to the ``.pt`` checkpoint file.
        model:    Model instance whose parameters will be updated.
        device:   Target device for the loaded tensors.
        logger:   Optional logger for confirmation messages.

    Returns:
        The raw checkpoint dictionary (includes metadata like epoch/accuracy).
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    msg = (
        f"Checkpoint loaded ← {filepath}  "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"acc={checkpoint.get('best_accuracy', '?'):.4f})"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# ANN FLOPs Estimation
# ─────────────────────────────────────────────────────────────────────────────

def count_ann_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Estimate the theoretical Multiply-Accumulate (MAC) operations for a
    single forward pass through an ANN.

    The FLOPs count follows the standard academic convention:
      • Conv2d:   2 × Cin × Kh × Kw × Hout × Wout × Cout
      • Linear:   2 × in_features × out_features
      (factor of 2 converts MACs → FLOPs; bias additions are ignored)

    This estimate is used as the baseline efficiency reference against which
    SNN Synaptic Operations (SynOps) are compared in Phase 2.

    Args:
        model:      The ANN module to profile.
        input_size: Input tensor shape as ``(C, H, W)``  (batch dim excluded).

    Returns:
        Total estimated integer FLOPs for a single sample.
    """
    total_flops = 0
    hooks = []

    def _conv_hook(module: nn.Conv2d, inp, out):
        nonlocal total_flops
        # out shape: (B, Cout, Hout, Wout)
        _, cout, hout, wout = out.shape
        cin = module.in_channels
        kh, kw = module.kernel_size  # type: ignore[misc]
        # MACs × 2 = FLOPs
        total_flops += 2 * cin * kh * kw * hout * wout * cout

    def _linear_hook(module: nn.Linear, inp, out):
        nonlocal total_flops
        total_flops += 2 * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    device = next(model.parameters()).device
    dummy = torch.zeros(1, *input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return total_flops


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 70) -> None:
    """Print a visually distinct section header to stdout."""
    border = "─" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(border)


class Timer:
    """Simple context-manager wall-clock timer."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.2f}s"
