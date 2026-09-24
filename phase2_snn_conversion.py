# phase2_snn_conversion.py

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import utils
from models.ann_model import BaselineANN
from models.snn_model import ConvertedSNN, SynOpsEstimator, ThresholdNormaliser


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_snn(
    snn_model: ConvertedSNN,
    loader,
    num_steps: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate SNN accuracy and accumulate SynOps on a test split.

    The SNN is run for ``num_steps`` timesteps per sample.  Class prediction
    is the argmax of the final output membrane potential over all classes.

    Args:
        snn_model:   Converted SNN in eval mode.
        loader:      Test DataLoader.
        num_steps:   Simulation duration T in timesteps.
        device:      Compute device.
        max_batches: If set, only evaluate this many batches (for quick runs).

    Returns:
        accuracy:  Float in [0, 1].
        synops:    Dict of average SynOps/sample per layer + ``"total"``.
    """
    snn_model.eval()
    correct = 0
    total = 0
    # Accumulate raw spike counts summed over batches.
    cumulative_spikes: Dict[str, float] = {
        "conv1": 0.0, "conv2": 0.0, "conv3": 0.0, "fc1": 0.0,
    }
    cumulative_samples = 0

    pbar = tqdm(
        loader,
        desc=f"  SNN T={num_steps:>3}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        # Forward pass with spike counting enabled.
        output_mem, spike_info = snn_model(
            images, num_steps, return_spike_counts=True
        )

        # Class prediction: argmax of accumulated output membrane.
        preds = output_mem.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B

        # Accumulate spike counts across batches.
        for key in cumulative_spikes:
            cumulative_spikes[key] += spike_info[key]
        cumulative_samples += B

        acc_so_far = correct / total * 100
        pbar.set_postfix(acc=f"{acc_so_far:.1f}%")

    accuracy = correct / total

    # Average SynOps per sample (divide by total samples seen).
    # The SynOpsEstimator expects batch-level raw counts; we pass the
    # aggregate and use cumulative_samples as the "batch size".
    avg_synops = SynOpsEstimator.compute(
        spike_counts=cumulative_spikes,
        batch_size=cumulative_samples,
        num_steps=num_steps,
    )

    return accuracy, avg_synops


@torch.no_grad()
def evaluate_ann(
    ann_model: BaselineANN,
    loader,
    device: torch.device,
) -> float:
    """
    Re-evaluate the ANN on the test set for the comparison baseline.

    Args:
        ann_model: Trained ANN in eval mode.
        loader:    Test DataLoader.
        device:    Compute device.

    Returns:
        Accuracy in [0, 1].
    """
    ann_model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="  ANN (baseline)", leave=False,
                                unit="batch", dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)
        preds = ann_model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / total


def compute_firing_rates(
    snn_model: ConvertedSNN,
    loader,
    num_steps: int,
    device: torch.device,
    num_batches: int = 4,
) -> Dict[str, float]:
    """
    Estimate average firing rates (spikes / neuron / timestep) per layer.

    This is a key sparsity metric for the paper: low firing rates → sparse
    computation → hardware efficiency.

    Args:
        snn_model:   Converted SNN.
        loader:      Data loader (test split).
        num_steps:   Simulation timesteps T.
        device:      Compute device.
        num_batches: Number of batches to sample (small for speed).

    Returns:
        Dict mapping layer name → average firing rate (scalar in [0, 1]).
    """
    snn_model.eval()
    # Total neurons per spiking layer (spatial × channel dims, pre-pooling).
    neuron_counts = {
        "conv1": 32 * 28 * 28,   # 25,088
        "conv2": 64 * 14 * 14,   # 12,544
        "conv3": 128 * 7 * 7,    #  6,272
        "fc1":   256,
    }

    cumulative_spikes: Dict[str, float] = {k: 0.0 for k in neuron_counts}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            B = images.size(0)
            _, spike_info = snn_model(images, num_steps, return_spike_counts=True)
            for k in cumulative_spikes:
                cumulative_spikes[k] += spike_info[k]
            total_samples += B

    # Rate = total_spikes / (total_neurons × total_timesteps × total_samples)
    rates: Dict[str, float] = {}
    for layer, n_neurons in neuron_counts.items():
        total_possible = n_neurons * num_steps * total_samples
        rates[layer] = cumulative_spikes[layer] / total_possible

    return rates


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT_1  = "#e94560"   # Coral-red
ACCENT_2  = "#53d8fb"   # Cyan
ACCENT_3  = "#f5a623"   # Amber
ACCENT_4  = "#7bed9f"   # Mint-green
GRID_COL  = "white"


def _style_ax(ax: plt.Axes) -> None:
    """Apply dark-theme styling to an Axes object."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, alpha=0.18, color=GRID_COL, linestyle="--")


def plot_accuracy_vs_timesteps(
    time_steps: List[int],
    snn_accuracies: List[float],
    ann_accuracy: float,
    save_dir: Path,
) -> None:
    """
    Plot SNN accuracy vs. number of timesteps alongside ANN baseline.

    Args:
        time_steps:      List of T values evaluated.
        snn_accuracies:  Corresponding SNN test accuracies (fractions).
        ann_accuracy:    ANN test accuracy (fraction).
        save_dir:        Output directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    snn_pct = [a * 100 for a in snn_accuracies]
    ann_pct = ann_accuracy * 100

    ax.plot(
        time_steps, snn_pct,
        color=ACCENT_2, linewidth=2.5, marker="o", markersize=7,
        label="Converted SNN",
    )
    ax.axhline(
        y=ann_pct, color=ACCENT_1, linewidth=2, linestyle="--",
        label=f"ANN Baseline ({ann_pct:.2f}%)",
    )

    # Annotate each point with its accuracy value.
    for t, acc in zip(time_steps, snn_pct):
        ax.annotate(
            f"{acc:.1f}%",
            xy=(t, acc), xytext=(0, 10), textcoords="offset points",
            ha="center", color="white", fontsize=9,
        )

    ax.set_xlabel("Simulation Timesteps (T)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Phase 2 — SNN Test Accuracy vs. Timesteps\nFashion-MNIST | IF Neurons | Direct Encoding",
        fontsize=12, fontweight="bold", color="white",
    )
    ax.set_xticks(time_steps)
    ax.set_ylim(min(snn_pct) - 5, ann_pct + 3)
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none",
              fontsize=10)
    plt.tight_layout()

    out = save_dir / "phase2_accuracy_vs_timesteps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_synops_vs_flops(
    time_steps: List[int],
    synops_list: List[float],
    ann_flops: int,
    save_dir: Path,
) -> None:
    """
    Plot SNN SynOps vs. T alongside the ANN FLOPs baseline.

    A log-scale y-axis is used because SynOps and FLOPs differ by orders of
    magnitude at small T, which is the key efficiency result.

    Args:
        time_steps:  List of T values.
        synops_list: Total SynOps per sample at each T.
        ann_flops:   ANN FLOPs per sample (from Phase 1 checkpoint).
        save_dir:    Output directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    ax.plot(
        time_steps, [s / 1e6 for s in synops_list],
        color=ACCENT_3, linewidth=2.5, marker="s", markersize=7,
        label="SNN SynOps (÷ 10⁶)",
    )
    ax.axhline(
        y=ann_flops / 1e6, color=ACCENT_1, linewidth=2, linestyle="--",
        label=f"ANN FLOPs ({ann_flops / 1e6:.2f} MFLOPs)",
    )

    # Ratio annotation on each bar.
    for t, s in zip(time_steps, synops_list):
        ratio = ann_flops / s if s > 0 else float("inf")
        ax.annotate(
            f"×{ratio:.1f} fewer",
            xy=(t, s / 1e6), xytext=(0, 10), textcoords="offset points",
            ha="center", color=ACCENT_4, fontsize=8,
        )

    ax.set_xlabel("Simulation Timesteps (T)", fontsize=12)
    ax.set_ylabel("Operations per Inference (Millions)", fontsize=12)
    ax.set_title(
        "Phase 2 — SNN SynOps vs. ANN FLOPs\nEdge Efficiency Analysis (1 SynOp ≈ 0.5 FLOPs)",
        fontsize=12, fontweight="bold", color="white",
    )
    ax.set_xticks(time_steps)
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none",
              fontsize=10)
    plt.tight_layout()

    out = save_dir / "phase2_synops_vs_flops.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_accuracy_synops_tradeoff(
    time_steps: List[int],
    snn_accuracies: List[float],
    synops_list: List[float],
    ann_accuracy: float,
    ann_flops: int,
    save_dir: Path,
) -> None:
    """
    Scatter plot: accuracy (y) vs. SynOps (x) — the efficiency frontier.

    Each dot is an SNN at a different T.  The ANN is shown as a reference
    cross.  Points to the upper-left are Pareto-optimal: same accuracy, fewer
    operations.

    Args:
        time_steps:       List of T values (used for annotation labels).
        snn_accuracies:   SNN test accuracies at each T.
        synops_list:      SNN SynOps at each T.
        ann_accuracy:     ANN test accuracy.
        ann_flops:        ANN FLOPs.
        save_dir:         Output directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    # SNN points (coloured by T — darker = higher T).
    cmap = plt.cm.plasma  # type: ignore[attr-defined]
    colours = [cmap(i / max(len(time_steps) - 1, 1))
               for i in range(len(time_steps))]

    for i, (t, acc, s) in enumerate(zip(time_steps, snn_accuracies, synops_list)):
        ax.scatter(s / 1e6, acc * 100, color=colours[i], s=120,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(
            f"T={t}", xy=(s / 1e6, acc * 100),
            xytext=(5, 4), textcoords="offset points",
            color="white", fontsize=9,
        )

    # ANN reference cross.
    ax.scatter(
        ann_flops / 1e6, ann_accuracy * 100,
        marker="*", color=ACCENT_1, s=250, zorder=6,
        edgecolors="white", linewidths=0.8,
        label=f"ANN Baseline ({ann_accuracy * 100:.1f}%, {ann_flops / 1e6:.1f}M FLOPs)",
    )

    ax.set_xlabel("Operations per Sample (Millions)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Phase 2 — Accuracy–Efficiency Frontier\nSNN (dots, T label) vs. ANN (★)",
        fontsize=12, fontweight="bold", color="white",
    )
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none",
              fontsize=10)
    plt.tight_layout()

    out = save_dir / "phase2_accuracy_efficiency_frontier.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_firing_rates(
    time_steps: List[int],
    rates_per_T: List[Dict[str, float]],
    save_dir: Path,
) -> None:
    """
    Grouped bar chart: average firing rate per layer at each T.

    Lower rates = sparser computation = more energy-efficient inference.

    Args:
        time_steps:   List of T values.
        rates_per_T:  List of {layer: rate} dicts, one per T.
        save_dir:     Output directory.
    """
    layers = ["conv1", "conv2", "conv3", "fc1"]
    layer_colours = [ACCENT_2, ACCENT_3, ACCENT_4, ACCENT_1]

    fig, axes = plt.subplots(1, len(time_steps),
                             figsize=(4 * len(time_steps), 5),
                             sharey=True)
    if len(time_steps) == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Phase 2 — Average Neuron Firing Rates per Layer\n"
        "(lower = sparser SNN = more efficient)",
        fontsize=12, fontweight="bold", color="white",
    )

    for ax, t, rates in zip(axes, time_steps, rates_per_T):
        _style_ax(ax)
        vals = [rates[l] * 100 for l in layers]  # convert to %
        bars = ax.bar(layers, vals, color=layer_colours, width=0.55,
                      edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.1f}%", ha="center", va="bottom",
                color="white", fontsize=8,
            )
        ax.set_title(f"T = {t}", color="white", fontsize=11)
        ax.set_ylabel("Avg. Firing Rate (%)" if ax is axes[0] else "",
                      fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = save_dir / "phase2_firing_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(
    time_steps: List[int],
    snn_accuracies: List[float],
    synops_per_T: List[Dict[str, float]],
    ann_accuracy: float,
    ann_flops: int,
    thresholds: List[float],
    save_dir: Path,
) -> None:
    """
    Write per-T SNN metrics to CSV, with ANN baseline appended as last row.

    Output columns:
        T, snn_accuracy, total_synops, synops_conv1, synops_conv2,
        synops_conv3, synops_fc1, flops_reduction_ratio,
        thresh_1, thresh_2, thresh_3, thresh_4
    """
    out = save_dir / "phase2_snn_profiling_results.csv"
    fields = [
        "T", "snn_accuracy", "total_synops",
        "synops_conv1", "synops_conv2", "synops_conv3", "synops_fc1",
        "flops_reduction_ratio",
        "thresh_1", "thresh_2", "thresh_3", "thresh_4",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for t, acc, so in zip(time_steps, snn_accuracies, synops_per_T):
            ratio = ann_flops / so["total"] if so["total"] > 0 else float("inf")
            w.writerow({
                "T":                   t,
                "snn_accuracy":        f"{acc:.6f}",
                "total_synops":        f"{so['total']:.2f}",
                "synops_conv1":        f"{so['conv1']:.2f}",
                "synops_conv2":        f"{so['conv2']:.2f}",
                "synops_conv3":        f"{so['conv3']:.2f}",
                "synops_fc1":          f"{so['fc1']:.2f}",
                "flops_reduction_ratio": f"{ratio:.3f}",
                "thresh_1":            f"{thresholds[0]:.6f}",
                "thresh_2":            f"{thresholds[1]:.6f}",
                "thresh_3":            f"{thresholds[2]:.6f}",
                "thresh_4":            f"{thresholds[3]:.6f}",
            })

        # ANN row for easy side-by-side comparison.
        w.writerow({
            "T":                   "ANN",
            "snn_accuracy":        f"{ann_accuracy:.6f}",
            "total_synops":        str(ann_flops),
            "synops_conv1":        "-",
            "synops_conv2":        "-",
            "synops_conv3":        "-",
            "synops_fc1":          "-",
            "flops_reduction_ratio": "1.000",
            "thresh_1": "-", "thresh_2": "-",
            "thresh_3": "-", "thresh_4": "-",
        })

    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """End-to-end Phase 2 pipeline."""

    logger = utils.get_logger(__name__, log_file="phase2_snn.log")
    utils.set_seed(config.SEED)
    device = utils.get_device()

    utils.print_section("PHASE 2 — ANN-to-SNN Conversion & Profiling")
    logger.info(f"Device       : {device}")
    logger.info(f"Time-steps   : {args.time_steps}")
    logger.info(f"Norm pctile  : {args.norm_percentile}")
    logger.info(f"Calib batches: {args.calib_batches}")

    # ── 1. Load ANN checkpoint ────────────────────────────────────────────
    logger.info("Loading ANN checkpoint …")
    ann = BaselineANN(num_classes=config.NUM_CLASSES)
    ckpt = utils.load_checkpoint(config.ANN_CHECKPOINT, ann, device, logger)
    ann = ann.to(device)
    ann.eval()

    ann_accuracy_ckpt = ckpt.get("best_accuracy", None)
    ann_flops = ckpt.get("flops", None)
    if ann_flops is None:
        # Fallback: recompute FLOPs if checkpoint pre-dates Phase 1 update.
        ann_flops = utils.count_ann_flops(
            ann,
            input_size=(config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE),
        )
    logger.info(
        f"ANN checkpoint accuracy : {ann_accuracy_ckpt * 100:.2f}%  "
        f"| FLOPs/sample : {ann_flops:,}  ({ann_flops / 1e6:.3f} MFLOPs)"
    )

    # ── 2. Data loaders ───────────────────────────────────────────────────
    logger.info("Loading Fashion-MNIST …")
    train_loader, test_loader = utils.get_fashion_mnist_loaders(
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
    )

    # ── 3. Re-evaluate ANN on full test set ──────────────────────────────
    logger.info("Re-evaluating ANN on full test set …")
    ann_accuracy = evaluate_ann(ann, test_loader, device)
    logger.info(f"ANN test accuracy (fresh eval): {ann_accuracy * 100:.2f}%")

    # ── 4. Threshold normalisation ────────────────────────────────────────
    logger.info(
        f"Computing thresholds via {args.norm_percentile}th-percentile "
        f"calibration ({args.calib_batches} batches × {args.batch_size} samples) …"
    )
    normaliser = ThresholdNormaliser(
        ann_model=ann,
        percentile=args.norm_percentile,
    )
    # Use training loader for calibration (avoids test-set leakage).
    thresholds = normaliser.compute_thresholds(
        calibration_loader=train_loader,
        num_batches=args.calib_batches,
    )
    logger.info(
        f"Calibrated thresholds:\n"
        f"  θ_1 (conv1/ReLU6) = {thresholds[0]:.4f}\n"
        f"  θ_2 (conv2/ReLU6) = {thresholds[1]:.4f}\n"
        f"  θ_3 (conv3/ReLU6) = {thresholds[2]:.4f}\n"
        f"  θ_4 (fc1 /ReLU6)  = {thresholds[3]:.4f}"
    )

    # ── 5. Build converted SNN and copy weights ───────────────────────────
    logger.info("Building ConvertedSNN and copying ANN weights …")
    snn_model = ConvertedSNN(
        thresholds=thresholds,
        num_classes=config.NUM_CLASSES,
    ).to(device)
    snn_model.load_ann_weights(ann)
    snn_model.eval()

    # Quick sanity check: count shared parameter count.
    snn_params = sum(p.numel() for p in snn_model.parameters())
    logger.info(f"SNN parameters  : {snn_params:,}  (identical to ANN)")

    # ── 6. Evaluate SNN at each T ─────────────────────────────────────────
    utils.print_section("SNN Evaluation at Variable Timesteps")
    snn_accuracies: List[float] = []
    synops_per_T: List[Dict[str, float]] = []
    rates_per_T: List[Dict[str, float]] = []

    for T in args.time_steps:
        logger.info(f"\n── T = {T} timesteps ──────────────────────────────────")

        with utils.Timer() as t_eval:
            acc, synops = evaluate_snn(
                snn_model, test_loader, T, device,
                max_batches=args.max_batches,
            )

        rates = compute_firing_rates(
            snn_model, test_loader, T, device, num_batches=4
        )

        snn_accuracies.append(acc)
        synops_per_T.append(synops)
        rates_per_T.append(rates)

        reduction = ann_flops / synops["total"] if synops["total"] > 0 else float("inf")
        acc_drop = (ann_accuracy - acc) * 100

        logger.info(
            f"  Accuracy    : {acc * 100:.2f}%  "
            f"(Δ = {acc_drop:+.2f}% vs ANN)"
        )
        logger.info(
            f"  SynOps      : {synops['total']:,.0f}  "
            f"({synops['total'] / 1e6:.3f} MSynOps)"
        )
        logger.info(
            f"  FLOPs/SynOps: {reduction:.2f}×  "
            f"(ANN is {reduction:.1f}× more expensive)"
        )
        logger.info(
            f"  Firing rates: conv1={rates['conv1']*100:.1f}%  "
            f"conv2={rates['conv2']*100:.1f}%  "
            f"conv3={rates['conv3']*100:.1f}%  "
            f"fc1={rates['fc1']*100:.1f}%"
        )
        logger.info(f"  Elapsed     : {t_eval}")

    # ── 7. Summary table ──────────────────────────────────────────────────
    utils.print_section("Phase 2 — Summary Table")
    header = (
        f"{'T':>6} | {'Accuracy':>10} | {'SynOps (M)':>12} | "
        f"{'FLOPs Reduction':>16} | {'Acc Drop':>10}"
    )
    logger.info(header)
    logger.info("─" * len(header))
    for t, acc, so in zip(args.time_steps, snn_accuracies, synops_per_T):
        reduction = ann_flops / so["total"] if so["total"] > 0 else 0
        logger.info(
            f"{t:>6} | {acc * 100:>9.2f}% | "
            f"{so['total'] / 1e6:>12.3f} | "
            f"{reduction:>15.2f}× | "
            f"{(ann_accuracy - acc) * 100:>+9.2f}%"
        )
    logger.info(
        f"{'ANN':>6} | {ann_accuracy * 100:>9.2f}% | "
        f"{ann_flops / 1e6:>12.3f} | {'1.00×':>16} | {'–':>10}"
    )

    # ── 8. Save checkpoint with thresholds for Phase 3 ───────────────────
    snn_ckpt_path = config.CHECKPOINTS_DIR / "converted_snn.pt"
    torch.save({
        "snn_state_dict":  snn_model.state_dict(),
        "thresholds":      thresholds,
        "ann_accuracy":    ann_accuracy,
        "ann_flops":       ann_flops,
        "snn_accuracies":  dict(zip(args.time_steps, snn_accuracies)),
        "synops_per_T":    dict(zip(args.time_steps, synops_per_T)),
    }, snn_ckpt_path)
    logger.info(f"\nSNN checkpoint saved → {snn_ckpt_path}")

    # ── 9. Plots ──────────────────────────────────────────────────────────
    utils.print_section("Generating Phase 2 Plots")

    plot_accuracy_vs_timesteps(
        args.time_steps, snn_accuracies, ann_accuracy, config.RESULTS_DIR
    )
    plot_synops_vs_flops(
        args.time_steps,
        [so["total"] for so in synops_per_T],
        ann_flops,
        config.RESULTS_DIR,
    )
    plot_accuracy_synops_tradeoff(
        args.time_steps, snn_accuracies,
        [so["total"] for so in synops_per_T],
        ann_accuracy, ann_flops, config.RESULTS_DIR,
    )
    plot_firing_rates(args.time_steps, rates_per_T, config.RESULTS_DIR)

    # ── 10. CSV ───────────────────────────────────────────────────────────
    save_results_csv(
        args.time_steps, snn_accuracies, synops_per_T,
        ann_accuracy, ann_flops, thresholds, config.RESULTS_DIR,
    )

    logger.info("\n✅  Phase 2 complete.  All results saved to results/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2: ANN-to-SNN Conversion & Profiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--time_steps", type=int, nargs="+",
        default=config.SNN_TIME_STEPS,
        help="List of simulation timesteps T to evaluate.",
    )
    p.add_argument(
        "--batch_size", type=int, default=config.SNN_BATCH_SIZE,
        help="Batch size for SNN evaluation.",
    )
    p.add_argument(
        "--norm_percentile", type=float, default=config.SNN_NORM_PERCENTILE,
        help="Activation percentile for threshold normalisation.",
    )
    p.add_argument(
        "--calib_batches", type=int, default=8,
        help="Number of training batches used for threshold calibration.",
    )
    p.add_argument(
        "--max_batches", type=int, default=None,
        help="Max test batches to evaluate per T (None = full test set).",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    main(args)
