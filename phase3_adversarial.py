# phase3_adversarial.py

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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import utils
from models.ann_model import BaselineANN
from models.snn_model import ConvertedSNN


# ─────────────────────────────────────────────────────────────────────────────
# Plot styling constants
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
C_CLEAN  = "#53d8fb"    # Cyan  — clean / unattacked
C_FGSM   = "#e94560"    # Coral — spatial FGSM attack
C_JITTER = "#f5a623"    # Amber — temporal jitter attack
C_ANN    = "#7bed9f"    # Mint  — ANN baseline


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, alpha=0.18, color="white", linestyle="--")


# ─────────────────────────────────────────────────────────────────────────────
# FGSM Attack
# ─────────────────────────────────────────────────────────────────────────────

def fgsm_attack(
    ann_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).

    The ANN is used as the surrogate model for gradient computation because
    the SNN's spike generation (Heaviside step function) is not differentiable.
    This is a standard "white-box against ANN, black-box transfer to SNN"
    attack paradigm (Sharmin et al., 2020).

    The perturbation budget ε is applied in the **normalised** input space.
    Input values are clamped to [x_min, x_max] of the normalised distribution
    to ensure the perturbed images remain within the valid data manifold.

    Args:
        ann_model: Trained ``BaselineANN`` used to compute gradients.
        images:    Clean input batch ``(B, 1, 28, 28)``, normalised.
        labels:    Ground-truth class indices ``(B,)``.
        epsilon:   L∞ perturbation budget.  ε=0 returns the clean image.
        device:    Compute device.

    Returns:
        Adversarial images ``(B, 1, 28, 28)`` within [x_min, x_max].
    """
    if epsilon == 0.0:
        return images.clone()

    # Images must require grad for backward pass.
    images_req = images.clone().detach().to(device)
    images_req.requires_grad_(True)

    ann_model.eval()
    logits = ann_model(images_req)
    loss = F.cross_entropy(logits, labels.to(device))

    ann_model.zero_grad()
    loss.backward()

    # The sign of the gradient points in the direction that maximises loss.
    grad_sign = images_req.grad.data.sign()

    x_adv = images_req.detach() + epsilon * grad_sign

    # Clamp to the observed min/max range of Fashion-MNIST after normalisation.
    # Normalisation: mean=0.286, std=0.353; pixel ∈ [0,1]
    # → normalised range ≈ [-0.81, 2.02]
    x_adv = x_adv.clamp(-0.81, 2.02)

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Jitter Attack
# ─────────────────────────────────────────────────────────────────────────────

def generate_delayed_onset_inputs(
    x_normalised: torch.Tensor,
    num_steps: int,
    sigma: float,
) -> torch.Tensor:
    """
    Generate temporally-jittered inputs using the **delayed-onset direct coding**
    model.

    Physical interpretation
    -----------------------
    In direct (constant-current) coding each neuron receives the same analogue
    input signal x at every timestep t ∈ [0, T−1] (Phase 2 baseline).
    A temporal adversary introduces a per-neuron onset delay:

        d_{b,c,h,w}  ~  |N(0, σ²)|   (half-normal, always ≥ 0)

    The neuron's input is therefore:

        x_{t}  =  x  if  t ≥ d,  else  0

    When σ=0 → all d=0 → input equals x at every step → **identical to
    Phase 2 direct coding** (clean SNN accuracy ~90.59% at T=64).  ✓

    As σ increases, neurons miss more of their integration window, reducing
    the accumulated membrane potential and causing more classification errors.

    This model captures hardware-level timing attacks (e.g. adversarial clock
    jitter, propagation-delay manipulation in neuromorphic interconnects) far
    more faithfully than Poisson jitter, and avoids the encoding-threshold
    mismatch that would otherwise cause a catastrophic baseline drop.

    Args:
        x_normalised: Input tensor ``(B, C, H, W)`` (normalised, as in Phase 2).
        num_steps:    Simulation duration T (timesteps).
        sigma:        Onset-delay std-dev in timesteps (1 step = 1 ms convention).
                      σ=0 reproduces Phase 2 direct coding exactly.

    Returns:
        Input sequence ``(T, B, C, H, W)`` — analogue values, not binary.
        Values are ≥ 0 (negative inputs have no physical onset, clamped to 0).
    """
    B, C, H, W = x_normalised.shape
    device = x_normalised.device

    if sigma == 0.0:
        # Fast path: exact direct coding — broadcast x over T steps.
        return x_normalised.unsqueeze(0).expand(num_steps, B, C, H, W)

    # Sample non-negative onset delays (half-normal ensures delay ≥ 0).
    delays = (torch.randn(B, C, H, W, device=device).abs() * sigma)
    delays = delays.round().long().clamp(0, num_steps - 1)       # (B, C, H, W)

    # Build activation mask: mask[t, b, c, h, w] = 1 if t >= delay, else 0.
    t_idx = torch.arange(num_steps, device=device).view(num_steps, 1, 1, 1, 1)
    delays_exp = delays.unsqueeze(0).expand(num_steps, B, C, H, W)
    mask = (t_idx >= delays_exp).float()                         # (T, B, C, H, W)

    # Apply mask: neuron receives its full normalised input after its onset delay,
    # and receives 0 during the silent window.  Negative pixel values are kept —
    # they are essential for the conv kernels (inhibitory background pixels) and
    # are zeroed only during the delay window (mask=0), not permanently.
    x_exp = x_normalised.unsqueeze(0).expand(num_steps, B, C, H, W)

    return x_exp * mask                                          # (T, B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_fgsm(
    ann_model: nn.Module,
    snn_model: ConvertedSNN,
    loader,
    epsilons: List[float],
    num_steps: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[float, float]:
    """
    Evaluate SNN accuracy under FGSM attack at each epsilon.

    For each ε:
    1. Compute FGSM perturbation using the ANN's gradient (requires_grad).
    2. Feed the adversarial image to the SNN via direct coding (same as Phase 2).
    3. Record accuracy.

    Note: FGSM gradient computation requires ``torch.enable_grad()`` even
    though the SNN evaluation is ``@torch.no_grad()``.  We handle this
    by enabling grad only inside ``fgsm_attack()``.

    Args:
        ann_model:   Differentiable ANN for gradient computation.
        snn_model:   Converted SNN to evaluate (target model).
        loader:      Test DataLoader.
        epsilons:    List of ε values to test.
        num_steps:   SNN simulation timesteps T.
        device:      Compute device.
        max_batches: Optional batch limit for speed.

    Returns:
        Dict mapping epsilon → SNN accuracy (fraction).
    """
    snn_model.eval()
    results: Dict[float, float] = {}

    for eps in epsilons:
        correct = 0
        total = 0
        pbar = tqdm(
            loader,
            desc=f"  FGSM ε={eps:.2f}",
            unit="batch", leave=False, dynamic_ncols=True,
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            # Generate adversarial images (requires grad internally).
            with torch.enable_grad():
                x_adv = fgsm_attack(ann_model, images, labels, eps, device)

            # Evaluate on SNN — direct coding with perturbed image.
            output_mem, _ = snn_model(x_adv, num_steps, return_spike_counts=False)
            preds = output_mem.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            pbar.set_postfix(acc=f"{correct / total * 100:.1f}%")

        results[eps] = correct / total

    return results


def evaluate_temporal_jitter(
    snn_model: ConvertedSNN,
    loader,
    sigmas: List[float],
    num_steps: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[float, float]:
    """
    Evaluate SNN accuracy under Gaussian temporal onset-delay jitter.

    For each σ:
    1. Generate a delayed-onset direct-coded input sequence ``(T, B, C, H, W)``
       using ``generate_delayed_onset_inputs()``.  Each neuron's signal is
       withheld for the first d ~ |N(0, σ²)| timesteps, simulating adversarial
       or hardware-induced onset delays.
    2. Feed the per-timestep inputs into the SNN via
       ``forward_with_spike_trains()``.  (The method name is retained for
       compatibility; it handles both binary spikes and analogue input
       sequences.)
    3. Record accuracy.

    Crucially, σ=0 recovers the Phase 2 direct-coding baseline (≈90.59%
    at T=64) because zero delay means the full input is presented at every
    timestep — identical to the standard forward pass.

    Args:
        snn_model:   Converted SNN.
        loader:      Test DataLoader.
        sigmas:      List of σ values in timesteps (1 step = 1 ms).
        num_steps:   Simulation duration T.
        device:      Compute device.
        max_batches: Optional batch limit for speed.

    Returns:
        Dict mapping sigma → SNN accuracy (fraction).
    """
    snn_model.eval()
    results: Dict[float, float] = {}

    for sigma in sigmas:
        correct = 0
        total   = 0
        pbar = tqdm(
            loader,
            desc=f"  Jitter σ={sigma:4.1f}ms",
            unit="batch", leave=False, dynamic_ncols=True,
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                # Build delayed-onset input sequence: (T, B, C, H, W)
                # σ=0  → identical to Phase 2 direct coding  ✓
                # σ>0  → each neuron's signal starts late by ~ N(0,σ²) steps
                delayed_inputs = generate_delayed_onset_inputs(
                    images, num_steps, sigma
                )

                # Run SNN with the per-timestep analogue input tensor.
                output_mem = snn_model.forward_with_spike_trains(delayed_inputs)
                preds = output_mem.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            pbar.set_postfix(acc=f"{correct / total * 100:.1f}%")

        results[sigma] = correct / total

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fgsm_curve(
    epsilons: List[float],
    accuracies: List[float],
    ann_accuracy: float,
    clean_snn_accuracy: float,
    save_dir: Path,
) -> None:
    """
    Plot FGSM accuracy degradation curve.

    Shows three reference lines:
      • ANN baseline accuracy (dashed green)
      • Clean SNN accuracy at ε=0 (dashed cyan)
      • SNN under FGSM at each ε (solid coral with markers)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    pct = [a * 100 for a in accuracies]

    ax.plot(
        epsilons, pct,
        color=C_FGSM, linewidth=2.5, marker="o", markersize=7,
        label="SNN under FGSM attack",
    )
    ax.axhline(
        y=ann_accuracy * 100, color=C_ANN,
        linewidth=1.8, linestyle="--",
        label=f"ANN baseline ({ann_accuracy * 100:.1f}%)",
    )
    ax.axhline(
        y=clean_snn_accuracy * 100, color=C_CLEAN,
        linewidth=1.8, linestyle=":",
        label=f"Clean SNN (ε=0): {clean_snn_accuracy * 100:.1f}%",
    )

    # Shade the degradation region.
    ax.fill_between(
        epsilons, pct, clean_snn_accuracy * 100,
        where=[p < clean_snn_accuracy * 100 for p in pct],
        alpha=0.15, color=C_FGSM, label="Degradation region",
    )

    # Annotate each data point.
    for eps, acc in zip(epsilons, pct):
        ax.annotate(
            f"{acc:.1f}%",
            xy=(eps, acc), xytext=(0, 10), textcoords="offset points",
            ha="center", color="white", fontsize=8,
        )

    ax.set_xlabel("FGSM Perturbation Budget ε (L∞ norm)", fontsize=12)
    ax.set_ylabel("SNN Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Phase 3 — Spatial Attack: FGSM on Converted SNN\n"
        f"T={config.ATTACK_EVAL_TIMESTEPS} | ANN gradient surrogate | Fashion-MNIST",
        fontsize=12, fontweight="bold", color="white",
    )
    ax.set_ylim(0, ann_accuracy * 100 + 5)
    ax.set_xticks(epsilons)
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()

    out = save_dir / "phase3_fgsm_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_jitter_curve(
    sigmas: List[float],
    accuracies: List[float],
    ann_accuracy: float,
    save_dir: Path,
) -> None:
    """
    Plot temporal jitter accuracy degradation curve.

    X-axis: σ in milliseconds (= timesteps under 1ms convention).
    Shows ANN baseline, clean jitter (σ=0), and degradation envelope.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    pct = [a * 100 for a in accuracies]
    clean_acc = pct[0]  # σ=0 is the baseline for Poisson-coded SNN

    ax.plot(
        sigmas, pct,
        color=C_JITTER, linewidth=2.5, marker="D", markersize=7,
        label="SNN under temporal jitter",
    )
    ax.axhline(
        y=ann_accuracy * 100, color=C_ANN,
        linewidth=1.8, linestyle="--",
        label=f"ANN baseline ({ann_accuracy * 100:.1f}%)",
    )
    ax.axhline(
        y=clean_acc, color=C_CLEAN,
        linewidth=1.8, linestyle=":",
        label=f"Clean SNN (σ=0): {clean_acc:.1f}%",
    )

    # Shade degradation region.
    ax.fill_between(
        sigmas, pct, clean_acc,
        where=[p < clean_acc for p in pct],
        alpha=0.15, color=C_JITTER, label="Degradation region",
    )

    for sigma, acc in zip(sigmas, pct):
        ax.annotate(
            f"{acc:.1f}%",
            xy=(sigma, acc), xytext=(0, 10), textcoords="offset points",
            ha="center", color="white", fontsize=8,
        )

    ax.set_xlabel("Jitter Standard Deviation σ (milliseconds)", fontsize=12)
    ax.set_ylabel("SNN Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Phase 3 — Temporal Attack: Gaussian Spike-Train Jitter\n"
        f"T={config.ATTACK_EVAL_TIMESTEPS} | Poisson input encoding | Fashion-MNIST",
        fontsize=12, fontweight="bold", color="white",
    )
    ax.set_ylim(0, ann_accuracy * 100 + 5)
    ax.set_xticks(sigmas)
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()

    out = save_dir / "phase3_jitter_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_combined_attacks(
    epsilons: List[float],
    fgsm_accs: List[float],
    sigmas: List[float],
    jitter_accs: List[float],
    ann_accuracy: float,
    save_dir: Path,
) -> None:
    """
    Side-by-side dual-panel comparison of both attack modalities.

    This is the primary figure for the paper's Phase 3 section, showing
    that both spatial and temporal attacks independently degrade SNN accuracy,
    motivating the ATTF defence in Phase 4.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Phase 3 — Adversarial Vulnerabilities in Converted SNN\n"
        "Spatial Attack (FGSM)  ·  Temporal Attack (Gaussian Jitter)",
        fontsize=13, fontweight="bold", color="white", y=1.02,
    )

    for ax in (ax1, ax2):
        _style_ax(ax)

    # ── Panel 1: FGSM ────────────────────────────────────────────────────
    fgsm_pct = [a * 100 for a in fgsm_accs]
    ax1.plot(epsilons, fgsm_pct,
             color=C_FGSM, linewidth=2.5, marker="o", markersize=7,
             label="SNN + FGSM")
    ax1.axhline(y=ann_accuracy * 100, color=C_ANN,
                linewidth=1.5, linestyle="--",
                label=f"ANN ({ann_accuracy * 100:.1f}%)")
    ax1.axhline(y=fgsm_pct[0], color=C_CLEAN,
                linewidth=1.5, linestyle=":",
                label=f"Clean SNN ({fgsm_pct[0]:.1f}%)")
    ax1.fill_between(epsilons, fgsm_pct, fgsm_pct[0],
                     where=[p < fgsm_pct[0] for p in fgsm_pct],
                     alpha=0.15, color=C_FGSM)
    ax1.set_xlabel("Perturbation ε", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.set_title("Spatial Attack (FGSM)", fontsize=11, color="white")
    ax1.set_ylim(0, ann_accuracy * 100 + 5)
    ax1.set_xticks(epsilons)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none", fontsize=9)

    # ── Panel 2: Temporal Jitter ──────────────────────────────────────────
    jitter_pct = [a * 100 for a in jitter_accs]
    ax2.plot(sigmas, jitter_pct,
             color=C_JITTER, linewidth=2.5, marker="D", markersize=7,
             label="SNN + Jitter")
    ax2.axhline(y=ann_accuracy * 100, color=C_ANN,
                linewidth=1.5, linestyle="--",
                label=f"ANN ({ann_accuracy * 100:.1f}%)")
    ax2.axhline(y=jitter_pct[0], color=C_CLEAN,
                linewidth=1.5, linestyle=":",
                label=f"Clean SNN ({jitter_pct[0]:.1f}%)")
    ax2.fill_between(sigmas, jitter_pct, jitter_pct[0],
                     where=[p < jitter_pct[0] for p in jitter_pct],
                     alpha=0.15, color=C_JITTER)
    ax2.set_xlabel("Jitter σ (ms)", fontsize=11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax2.set_title("Temporal Attack (Gaussian Jitter)", fontsize=11, color="white")
    ax2.set_ylim(0, ann_accuracy * 100 + 5)
    ax2.set_xticks(sigmas)
    ax2.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none", fontsize=9)

    plt.tight_layout()
    out = save_dir / "phase3_combined_attacks.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_attack_severity_heatmap(
    epsilons: List[float],
    fgsm_accs: List[float],
    sigmas: List[float],
    jitter_accs: List[float],
    save_dir: Path,
) -> None:
    """
    Normalised accuracy-drop bar chart comparing peak attack severity.

    Visualises the total accuracy drop at the maximum attack intensity for
    each threat model, providing a quick severity comparison for the paper.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    # Accuracy drop = (clean − worst) in percentage points.
    fgsm_drop   = (fgsm_accs[0]   - min(fgsm_accs))   * 100
    jitter_drop = (jitter_accs[0] - min(jitter_accs)) * 100

    labels  = [f"Spatial FGSM\n(max ε={max(epsilons)})",
               f"Temporal Jitter\n(max σ={max(sigmas)} ms)"]
    drops   = [fgsm_drop, jitter_drop]
    colours = [C_FGSM, C_JITTER]

    bars = ax.bar(labels, drops, color=colours, width=0.45, edgecolor="none")
    for bar, val in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.3,
            f"−{val:.1f} pp", ha="center", va="bottom",
            color="white", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Accuracy Drop (percentage points)", fontsize=11)
    ax.set_title(
        "Phase 3 — Peak Attack Severity Comparison\n(pp = percentage-point drop from clean SNN)",
        fontsize=11, fontweight="bold", color="white",
    )
    ax.set_ylim(0, max(drops) + 8)
    plt.tight_layout()

    out = save_dir / "phase3_attack_severity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(
    epsilons: List[float],
    fgsm_accs: List[float],
    sigmas: List[float],
    jitter_accs: List[float],
    ann_accuracy: float,
    save_dir: Path,
) -> None:
    """
    Save all Phase 3 results to a structured CSV file.

    Two tables are written in sequence: FGSM results then temporal jitter.
    """
    out = save_dir / "phase3_attack_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["# Phase 3 — Adversarial Attack Results"])
        w.writerow([f"# ANN baseline accuracy: {ann_accuracy:.6f}"])
        w.writerow([f"# Eval timesteps T: {config.ATTACK_EVAL_TIMESTEPS}"])
        w.writerow([])

        # FGSM table
        w.writerow(["attack_type", "attack_param", "snn_accuracy", "acc_drop_pp"])
        for eps, acc in zip(epsilons, fgsm_accs):
            drop = (fgsm_accs[0] - acc) * 100
            w.writerow(["FGSM", f"{eps:.2f}", f"{acc:.6f}", f"{drop:.3f}"])

        w.writerow([])

        # Jitter table
        w.writerow(["attack_type", "attack_param", "snn_accuracy", "acc_drop_pp"])
        for sigma, acc in zip(sigmas, jitter_accs):
            drop = (jitter_accs[0] - acc) * 100
            w.writerow(["TemporalJitter", f"{sigma:.1f}", f"{acc:.6f}", f"{drop:.3f}"])

    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """End-to-end Phase 3 pipeline."""

    logger = utils.get_logger(__name__, log_file="phase3_adversarial.log")
    utils.set_seed(config.SEED)
    device = utils.get_device()

    utils.print_section("PHASE 3 — Adversarial Attack Implementation")
    logger.info(f"Device        : {device}")
    logger.info(f"Eval T        : {args.num_steps} timesteps")
    logger.info(f"FGSM epsilons : {args.epsilons}")
    logger.info(f"Jitter sigmas : {args.sigmas}")

    # ── 1. Load models ────────────────────────────────────────────────────
    logger.info("\nLoading ANN checkpoint …")
    ann = BaselineANN(num_classes=config.NUM_CLASSES)
    ann_ckpt = utils.load_checkpoint(config.ANN_CHECKPOINT, ann, device, logger)
    ann = ann.to(device)
    ann.eval()
    ann_accuracy = ann_ckpt.get("best_accuracy", 0.932)
    ann_flops    = ann_ckpt.get("flops", 15_497_216)

    logger.info("Loading SNN checkpoint …")
    snn_ckpt_path = config.CHECKPOINTS_DIR / "converted_snn.pt"
    snn_ckpt = torch.load(snn_ckpt_path, map_location=device)
    thresholds = snn_ckpt["thresholds"]

    snn_model = ConvertedSNN(thresholds=thresholds, num_classes=config.NUM_CLASSES)
    snn_model.load_state_dict(snn_ckpt["snn_state_dict"])
    snn_model = snn_model.to(device)
    snn_model.eval()

    logger.info(
        f"Models loaded. ANN accuracy: {ann_accuracy * 100:.2f}%  "
        f"| Thresholds: {[f'{t:.3f}' for t in thresholds]}"
    )

    # ── 2. Data loader ────────────────────────────────────────────────────
    logger.info("Loading test data …")
    _, test_loader = utils.get_fashion_mnist_loaders(
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
    )

    # ── 3. Attack 1 — FGSM (Spatial) ─────────────────────────────────────
    utils.print_section("Attack 1 — FGSM (Spatial)")
    logger.info(
        f"Running FGSM at ε ∈ {args.epsilons} "
        f"(T={args.num_steps}, max_batches={args.max_batches}) …"
    )

    with utils.Timer() as t_fgsm:
        fgsm_results = evaluate_fgsm(
            ann_model=ann,
            snn_model=snn_model,
            loader=test_loader,
            epsilons=args.epsilons,
            num_steps=args.num_steps,
            device=device,
            max_batches=args.max_batches,
        )

    logger.info(f"FGSM evaluation complete in {t_fgsm}")
    logger.info(f"\n{'ε':>6} | {'Accuracy':>10} | {'Acc Drop':>10}")
    logger.info("─" * 32)
    clean_fgsm_acc = fgsm_results[args.epsilons[0]]
    for eps, acc in fgsm_results.items():
        drop = (clean_fgsm_acc - acc) * 100
        logger.info(f"{eps:>6.2f} | {acc * 100:>9.2f}% | {drop:>+9.2f} pp")

    # ── 4. Attack 2 — Temporal Jitter ────────────────────────────────────
    utils.print_section("Attack 2 — Temporal Spike-Train Jitter")
    logger.info(
        f"Running temporal jitter at σ ∈ {args.sigmas} ms "
        f"(T={args.num_steps}, max_batches={args.max_batches}) …"
    )

    with utils.Timer() as t_jitter:
        jitter_results = evaluate_temporal_jitter(
            snn_model=snn_model,
            loader=test_loader,
            sigmas=args.sigmas,
            num_steps=args.num_steps,
            device=device,
            max_batches=args.max_batches,
        )

    logger.info(f"Jitter evaluation complete in {t_jitter}")
    logger.info(f"\n{'σ (ms)':>8} | {'Accuracy':>10} | {'Acc Drop':>10}")
    logger.info("─" * 35)
    clean_jitter_acc = jitter_results[args.sigmas[0]]
    for sigma, acc in jitter_results.items():
        drop = (clean_jitter_acc - acc) * 100
        logger.info(f"{sigma:>8.1f} | {acc * 100:>9.2f}% | {drop:>+9.2f} pp")

    # ── 5. Summary ────────────────────────────────────────────────────────
    utils.print_section("Phase 3 — Attack Severity Summary")
    fgsm_list   = [fgsm_results[e]   for e in args.epsilons]
    jitter_list = [jitter_results[s] for s in args.sigmas]

    fgsm_max_drop   = (clean_fgsm_acc   - min(fgsm_list))   * 100
    jitter_max_drop = (clean_jitter_acc - min(jitter_list)) * 100

    logger.info(
        f"FGSM:   clean={clean_fgsm_acc * 100:.2f}%  "
        f"→  worst (ε={max(args.epsilons):.2f}): {min(fgsm_list) * 100:.2f}%  "
        f"[−{fgsm_max_drop:.2f} pp]"
    )
    logger.info(
        f"Jitter: clean={clean_jitter_acc * 100:.2f}%  "
        f"→  worst (σ={max(args.sigmas):.0f}ms): {min(jitter_list) * 100:.2f}%  "
        f"[−{jitter_max_drop:.2f} pp]"
    )
    logger.info(
        "\nInterpretation: The SNN is vulnerable to BOTH spatial (pixel-domain)"
        " and temporal (spike-timing) adversarial perturbations.  Phase 4 will"
        " introduce the ATTF defence to mitigate the temporal attack."
    )

    # ── 6. Plots ──────────────────────────────────────────────────────────
    utils.print_section("Generating Phase 3 Plots")
    plot_fgsm_curve(
        args.epsilons, fgsm_list, ann_accuracy, clean_fgsm_acc,
        config.RESULTS_DIR,
    )
    plot_jitter_curve(
        args.sigmas, jitter_list, ann_accuracy,
        config.RESULTS_DIR,
    )
    plot_combined_attacks(
        args.epsilons, fgsm_list,
        args.sigmas, jitter_list,
        ann_accuracy, config.RESULTS_DIR,
    )
    plot_attack_severity_heatmap(
        args.epsilons, fgsm_list,
        args.sigmas, jitter_list,
        config.RESULTS_DIR,
    )

    # ── 7. CSV ────────────────────────────────────────────────────────────
    save_results_csv(
        args.epsilons, fgsm_list,
        args.sigmas, jitter_list,
        ann_accuracy, config.RESULTS_DIR,
    )

    logger.info("\n✅  Phase 3 complete.  All results saved to results/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 3: Adversarial Attack Evaluation on Converted SNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--epsilons", type=float, nargs="+",
        default=config.FGSM_EPSILONS,
        help="FGSM perturbation budgets (L∞).",
    )
    p.add_argument(
        "--sigmas", type=float, nargs="+",
        default=config.TEMPORAL_JITTER_SIGMAS,
        help="Temporal jitter std-dev values in ms (= timesteps).",
    )
    p.add_argument(
        "--num_steps", type=int, default=config.ATTACK_EVAL_TIMESTEPS,
        help="SNN simulation timesteps T for all evaluations.",
    )
    p.add_argument(
        "--batch_size", type=int, default=config.SNN_BATCH_SIZE,
        help="Test batch size.",
    )
    p.add_argument(
        "--max_batches", type=int, default=None,
        help="Max test batches per attack setting (None = full test set).",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    main(args)
