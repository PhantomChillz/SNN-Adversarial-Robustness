# phase1_baseline_ann.py

from __future__ import annotations  # tuple/dict/list generic hints on Python 3.9


import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend for server/HPC compat.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import utils
from models.ann_model import BaselineANN


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
) -> tuple[float, float]:
    """
    Run a single training epoch.

    Args:
        model:     The network being trained.
        loader:    Training DataLoader.
        criterion: Loss function.
        optimiser: Parameter update rule.
        device:    Compute device.
        epoch:     Current epoch index (0-based), used for logging.
        logger:    Logger instance.

    Returns:
        Tuple ``(avg_loss, accuracy)`` over the training set.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1:>3} [train]",
                unit="batch", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping prevents exploding gradients during early epochs.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    logger.debug(
        f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | "
        f"Train Acc: {accuracy * 100:.2f}%"
    )
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "test",
) -> tuple[float, float]:
    """
    Evaluate the model on a data split without gradient computation.

    Args:
        model:      The network to evaluate (set to eval mode internally).
        loader:     DataLoader for the evaluation split.
        criterion:  Loss function.
        device:     Compute device.
        split_name: Label string used for the progress bar.

    Returns:
        Tuple ``(avg_loss, accuracy)``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"           [{split_name}]",
                               unit="batch", leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: Path) -> None:
    """
    Generate and save training/validation loss & accuracy curves.

    Args:
        history:  Dict with keys ``train_loss``, ``test_loss``,
                  ``train_acc``, ``test_acc`` — each a list of epoch values.
        save_dir: Directory where the figure is saved.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Phase 1 — Baseline ANN Training Curves\n"
        "Fashion-MNIST  |  BaselineANN (ReLU6)",
        fontsize=13, fontweight="bold",
    )
    fig.patch.set_facecolor("#1a1a2e")

    # Shared styling
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ── Loss panel ────────────────────────────────────────────────────────
    ax_loss = axes[0]
    ax_loss.plot(epochs, history["train_loss"], color="#e94560",
                 linewidth=2, marker="o", markersize=3, label="Train Loss")
    ax_loss.plot(epochs, history["test_loss"],  color="#0f3460",
                 linewidth=2, marker="s", markersize=3, linestyle="--",
                 label="Test Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(facecolor="#0f3460", labelcolor="white",
                   edgecolor="none")
    ax_loss.grid(True, alpha=0.2, color="white")
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy panel ────────────────────────────────────────────────────
    ax_acc = axes[1]
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]],
                color="#53d8fb", linewidth=2, marker="o", markersize=3,
                label="Train Acc")
    ax_acc.plot(epochs, [a * 100 for a in history["test_acc"]],
                color="#f5a623", linewidth=2, marker="s", markersize=3,
                linestyle="--", label="Test Acc")
    # 90% target line
    ax_acc.axhline(y=90, color="#7bed9f", linewidth=1.5, linestyle=":",
                   alpha=0.8, label="90% Target")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(50, 101)
    ax_acc.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none")
    ax_acc.grid(True, alpha=0.2, color="white")
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out_path = save_dir / "phase1_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Training curves saved → {out_path}")


def save_history_csv(history: dict, save_dir: Path) -> None:
    """
    Persist per-epoch metrics to a CSV file for reproducible analysis.

    Args:
        history:  Dict with keys ``train_loss``, ``test_loss``,
                  ``train_acc``, ``test_acc``.
        save_dir: Directory where the CSV is written.
    """
    out_path = save_dir / "phase1_training_history.csv"
    fieldnames = ["epoch", "train_loss", "test_loss", "train_acc", "test_acc"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            writer.writerow({
                "epoch":      i + 1,
                "train_loss": f"{history['train_loss'][i]:.6f}",
                "test_loss":  f"{history['test_loss'][i]:.6f}",
                "train_acc":  f"{history['train_acc'][i]:.6f}",
                "test_acc":   f"{history['test_acc'][i]:.6f}",
            })
    print(f"  Training history saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """End-to-end Phase 1 training pipeline."""

    logger = utils.get_logger(__name__, log_file="phase1_ann.log")
    utils.set_seed(config.SEED)
    device = utils.get_device()

    utils.print_section("PHASE 1 — Baseline ANN Training")
    logger.info(f"Device: {device}")
    logger.info(f"Seed:   {config.SEED}")

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Loading Fashion-MNIST …")
    train_loader, test_loader = utils.get_fashion_mnist_loaders(
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
    )
    logger.info(
        f"  Train samples: {len(train_loader.dataset):,}  "
        f"Test samples: {len(test_loader.dataset):,}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = BaselineANN(num_classes=config.NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: BaselineANN | Trainable params: {total_params:,}")

    # ── FLOPs (before training — architecture is fixed) ───────────────────
    flops = utils.count_ann_flops(
        model,
        input_size=(config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE),
    )
    logger.info(
        f"Theoretical ANN FLOPs per inference: {flops:,}  "
        f"({flops / 1e6:.3f} MFLOPs)"
    )

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    # label_smoothing=0.1 reduces over-confidence and helps generalisation
    # without requiring any architectural changes.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimiser = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimiser,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    # ── Training loop with early stopping ─────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [],
    }
    best_accuracy: float = 0.0
    patience_counter: int = 0

    logger.info(
        f"Starting training — {args.epochs} epochs, "
        f"patience={args.patience}, lr={args.lr}"
    )

    for epoch in range(args.epochs):
        with utils.Timer() as t:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimiser, device, epoch, logger
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device, "test"
            )
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1:>3}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc:  {test_acc * 100:.2f}%  ★ | "
            f"LR: {lr_now:.2e} | Elapsed: {t}"
        )

        # Best model tracking
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            utils.save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "best_accuracy": best_accuracy,
                    "flops": flops,
                    "total_params": total_params,
                    "config": {
                        "num_classes": config.NUM_CLASSES,
                        "dropout_p": 0.4,
                    },
                },
                filepath=config.ANN_CHECKPOINT,
                logger=logger,
            )
        else:
            patience_counter += 1
            logger.debug(
                f"No improvement for {patience_counter}/{args.patience} epochs."
            )

        if patience_counter >= args.patience:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(patience={args.patience})."
            )
            break

    # ── Final report ──────────────────────────────────────────────────────
    utils.print_section("PHASE 1 — Results Summary")
    logger.info(f"Best Test Accuracy : {best_accuracy * 100:.2f}%")
    logger.info(f"ANN FLOPs / sample : {flops:,}  ({flops / 1e6:.3f} MFLOPs)")

    target_met = "✅  TARGET MET" if best_accuracy >= 0.90 else "❌  TARGET NOT MET"
    logger.info(f"90%  Accuracy Goal : {target_met}")

    # ── Save artefacts ────────────────────────────────────────────────────
    plot_training_curves(history, config.RESULTS_DIR)
    save_history_csv(history, config.RESULTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1: Train Baseline ANN on Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--epochs", type=int, default=config.ANN_EPOCHS,
        help="Maximum number of training epochs."
    )
    p.add_argument(
        "--lr", type=float, default=config.ANN_LR,
        help="Adam initial learning rate."
    )
    p.add_argument(
        "--batch_size", type=int, default=config.ANN_BATCH_SIZE,
        help="Mini-batch size for both train and test loaders."
    )
    p.add_argument(
        "--weight_decay", type=float, default=config.ANN_WEIGHT_DECAY,
        help="L2 weight decay (Adam)."
    )
    p.add_argument(
        "--lr_step_size", type=int, default=config.ANN_LR_STEP_SIZE,
        help="StepLR: decay learning rate every N epochs."
    )
    p.add_argument(
        "--lr_gamma", type=float, default=config.ANN_LR_GAMMA,
        help="StepLR: multiplicative decay factor."
    )
    p.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience (epochs without improvement)."
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    main(args)
