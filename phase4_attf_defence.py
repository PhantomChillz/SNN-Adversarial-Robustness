# phase4_attf_defence.py

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import utils
from models.attf_snn import ATTFConfig, ATTFDefendedSNN
from models.snn_model import ConvertedSNN


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
C_CLEAN  = "#53d8fb"   # cyan    — clean baseline
C_UNDEF  = "#e94560"   # coral   — undefended SNN under attack
C_DEF    = "#7bed9f"   # mint    — ATTF-defended SNN
C_ANN    = "#f5a623"   # amber   — ANN reference
C_FGSM   = "#e94560"
C_JITTER = "#a29bfe"   # lavender

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, alpha=0.18, color="white", linestyle="--")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def generate_delayed_onset_inputs(
    x: torch.Tensor, num_steps: int, sigma: float
) -> torch.Tensor:
    """
    Delayed-onset direct coding for temporal jitter evaluation.
    sigma=0 -> identical to Phase-2 direct coding (90.59% baseline).
    """
    B, C, H, W = x.shape
    device = x.device

    if sigma == 0.0:
        return x.unsqueeze(0).expand(num_steps, B, C, H, W)

    delays = (torch.randn(B, C, H, W, device=device).abs() * sigma)
    delays = delays.round().long().clamp(0, num_steps - 1)

    t_idx      = torch.arange(num_steps, device=device).view(num_steps,1,1,1,1)
    delays_exp = delays.unsqueeze(0).expand(num_steps, B, C, H, W)
    mask       = (t_idx >= delays_exp).float()

    x_exp = x.unsqueeze(0).expand(num_steps, B, C, H, W)
    return x_exp * mask


def fgsm_attack(ann_model, images, labels, epsilon, device):
    """Standard FGSM using ANN gradient (transfer attack to SNN)."""
    if epsilon == 0.0:
        return images.clone()
    x = images.clone().detach().to(device).requires_grad_(True)
    ann_model.eval()
    loss = F.cross_entropy(ann_model(x), labels.to(device))
    ann_model.zero_grad()
    loss.backward()
    x_adv = (x.detach() + epsilon * x.grad.data.sign()).clamp(-0.81, 2.02)
    return x_adv.detach()


@torch.no_grad()
def evaluate_clean(model, loader, num_steps, device, max_batches=None,
                   use_spike_trains=False):
    """Evaluate accuracy on clean inputs."""
    model.eval()
    correct = total = 0
    for i, (imgs, labels) in enumerate(tqdm(loader, desc="  Clean eval",
                                             leave=False, dynamic_ncols=True)):
        if max_batches and i >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        if use_spike_trains:
            out = model.forward_with_spike_trains(
                generate_delayed_onset_inputs(imgs, num_steps, 0.0))
        else:
            out = model(imgs, num_steps)
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate_jitter(model, loader, sigmas, num_steps, device,
                    max_batches=None):
    """Evaluate accuracy under temporal onset-delay jitter."""
    model.eval()
    results = {}
    for sigma in sigmas:
        correct = total = 0
        for i, (imgs, labels) in enumerate(
            tqdm(loader, desc=f"  Jitter σ={sigma:.0f}ms",
                 leave=False, dynamic_ncols=True)
        ):
            if max_batches and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            delayed = generate_delayed_onset_inputs(imgs, num_steps, sigma)
            out     = model.forward_with_spike_trains(delayed)
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        results[sigma] = correct / total
    return results


def evaluate_fgsm(ann_model, snn_model, loader, epsilons, num_steps, device,
                  max_batches=None):
    """Evaluate SNN accuracy under FGSM spatial attack."""
    snn_model.eval()
    results = {}
    for eps in epsilons:
        correct = total = 0
        for i, (imgs, labels) in enumerate(
            tqdm(loader, desc=f"  FGSM ε={eps:.2f}",
                 leave=False, dynamic_ncols=True)
        ):
            if max_batches and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.enable_grad():
                x_adv = fgsm_attack(ann_model, imgs, labels, eps, device)
            out = snn_model(x_adv, num_steps)
            out = out[0] if isinstance(out, tuple) else out
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        results[eps] = correct / total
    return results


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def attf_hparam_sweep(
    snn_ckpt: dict,
    test_loader,
    sigma_target: float,
    num_steps: int,
    device: torch.device,
    windows: List[int],
    burst_thresholds: List[int],
    max_scale: float,
    decay: float,
    max_batches: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep over (window W, burst_threshold K) combinations and record:
      - clean accuracy at sigma=0
      - defended accuracy at sigma=sigma_target
      - robustness gain = defended(sigma) - undefended(sigma)

    Returns three 2-D numpy arrays indexed [i_W, i_K].
    """
    thresholds = snn_ckpt["thresholds"]
    n_W = len(windows)
    n_K = len(burst_thresholds)

    clean_grid    = np.zeros((n_W, n_K))
    defended_grid = np.zeros((n_W, n_K))

    for i, W in enumerate(windows):
        for j, K in enumerate(burst_thresholds):
            cfg = ATTFConfig(W, K, max_scale, decay)
            model = ATTFDefendedSNN(
                thresholds=thresholds,
                attf_window=W,
                attf_burst_thresh=K,
                attf_max_scale=max_scale,
                attf_decay=decay,
            ).to(device)
            model.load_from_snn_checkpoint(snn_ckpt["snn_state_dict"])
            model.eval()

            # Clean
            c_acc = evaluate_clean(
                model, test_loader, num_steps, device,
                max_batches=max_batches, use_spike_trains=True)
            clean_grid[i, j] = c_acc

            # Jitter at sigma_target
            j_res = evaluate_jitter(
                model, test_loader, [sigma_target], num_steps, device,
                max_batches=max_batches)
            defended_grid[i, j] = j_res[sigma_target]

            print(
                f"    W={W}, K={K}: clean={c_acc*100:.2f}%  "
                f"jitter({sigma_target}ms)={j_res[sigma_target]*100:.2f}%"
            )

    gain_grid = defended_grid - clean_grid   # robustness vs. clean tradeoff
    return clean_grid, defended_grid, gain_grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_jitter_comparison(sigmas, undef_accs, def_accs, ann_acc, save_dir):
    """Undefended vs. ATTF-defended accuracy under temporal jitter."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    u_pct = [a * 100 for a in undef_accs]
    d_pct = [a * 100 for a in def_accs]

    ax.plot(sigmas, u_pct, color=C_UNDEF, lw=2.5, marker="o", ms=7,
            label="Undefended SNN", linestyle="--")
    ax.plot(sigmas, d_pct, color=C_DEF,  lw=2.5, marker="D", ms=7,
            label="ATTF-Defended SNN")
    ax.axhline(ann_acc * 100, color=C_ANN, lw=1.8, linestyle=":",
               label=f"ANN baseline ({ann_acc*100:.1f}%)")

    # Shade improvement region
    ax.fill_between(sigmas, u_pct, d_pct,
                    where=[d > u for d, u in zip(d_pct, u_pct)],
                    alpha=0.18, color=C_DEF, label="ATTF improvement")

    for s, u, d in zip(sigmas, u_pct, d_pct):
        gain = d - u
        if gain > 0.1:
            ax.annotate(f"+{gain:.1f}pp", xy=(s, (u+d)/2),
                        xytext=(6, 0), textcoords="offset points",
                        color=C_DEF, fontsize=8)

    ax.set_xlabel("Jitter σ (milliseconds)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Phase 4 — ATTF Defence vs. Temporal Onset-Delay Jitter\n"
        f"T={config.ATTACK_EVAL_TIMESTEPS} | W={config.ATTF_WINDOW}, "
        f"K={config.ATTF_BURST_THRESHOLD}, "
        f"scale={config.ATTF_MAX_THRESH_SCALE}, decay={config.ATTF_THRESH_DECAY}",
        fontsize=11, fontweight="bold", color="white",
    )
    ax.set_ylim(max(0, min(u_pct) - 5), ann_acc * 100 + 4)
    ax.set_xticks(sigmas)
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()

    out = save_dir / "phase4_jitter_defended.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_fgsm_comparison(epsilons, undef_accs, def_accs, ann_acc, save_dir):
    """Undefended vs. ATTF-defended accuracy under FGSM."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    u_pct = [a * 100 for a in undef_accs]
    d_pct = [a * 100 for a in def_accs]

    ax.plot(epsilons, u_pct, color=C_UNDEF, lw=2.5, marker="o", ms=7,
            linestyle="--", label="Undefended SNN")
    ax.plot(epsilons, d_pct, color=C_DEF,  lw=2.5, marker="s", ms=7,
            label="ATTF-Defended SNN")
    ax.axhline(ann_acc * 100, color=C_ANN, lw=1.8, linestyle=":",
               label=f"ANN ({ann_acc*100:.1f}%)")

    ax.set_xlabel("FGSM Perturbation ε (L∞)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Phase 4 — ATTF vs. FGSM Spatial Attack\n"
        "(ATTF targets temporal domain; FGSM impact shown for completeness)",
        fontsize=11, fontweight="bold", color="white",
    )
    ax.set_ylim(0, ann_acc * 100 + 5)
    ax.set_xticks(epsilons)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(facecolor="#0f3460", labelcolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()

    out = save_dir / "phase4_fgsm_defended.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_combined_summary(
    sigmas, undef_j, def_j,
    epsilons, undef_f, def_f,
    ann_acc, save_dir,
):
    """Four-panel figure: the complete defence story."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Phase 4 — ATTF Defence: Complete Evaluation Summary",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    (ax_jitter, ax_fgsm, ax_gain, ax_bar) = axes.flatten()
    for ax in axes.flatten():
        _style_ax(ax)

    u_j = [a * 100 for a in undef_j]
    d_j = [a * 100 for a in def_j]
    u_f = [a * 100 for a in undef_f]
    d_f = [a * 100 for a in def_f]

    # Panel 1: Jitter comparison
    ax_jitter.plot(sigmas, u_j, color=C_UNDEF, lw=2, marker="o", ms=6,
                   linestyle="--", label="Undefended")
    ax_jitter.plot(sigmas, d_j, color=C_DEF,  lw=2, marker="D", ms=6,
                   label="ATTF Defended")
    ax_jitter.axhline(ann_acc * 100, color=C_ANN, lw=1.5, linestyle=":")
    ax_jitter.fill_between(sigmas, u_j, d_j,
                           where=[d > u for d,u in zip(d_j,u_j)],
                           alpha=0.15, color=C_DEF)
    ax_jitter.set_title("Temporal Jitter Attack", color="white", fontsize=11)
    ax_jitter.set_xlabel("σ (ms)", fontsize=10)
    ax_jitter.set_ylabel("Accuracy (%)", fontsize=10)
    ax_jitter.set_xticks(sigmas)
    ax_jitter.legend(facecolor="#0f3460", labelcolor="white",
                     edgecolor="none", fontsize=8)

    # Panel 2: FGSM comparison
    ax_fgsm.plot(epsilons, u_f, color=C_UNDEF, lw=2, marker="o", ms=6,
                 linestyle="--", label="Undefended")
    ax_fgsm.plot(epsilons, d_f, color=C_DEF,   lw=2, marker="s", ms=6,
                 label="ATTF Defended")
    ax_fgsm.axhline(ann_acc * 100, color=C_ANN, lw=1.5, linestyle=":")
    ax_fgsm.set_title("FGSM Spatial Attack", color="white", fontsize=11)
    ax_fgsm.set_xlabel("ε (L∞)", fontsize=10)
    ax_fgsm.set_ylabel("Accuracy (%)", fontsize=10)
    ax_fgsm.set_xticks(epsilons)
    ax_fgsm.tick_params(axis="x", rotation=30)
    ax_fgsm.legend(facecolor="#0f3460", labelcolor="white",
                   edgecolor="none", fontsize=8)

    # Panel 3: Robustness gain (pp improvement) per sigma
    gains_j = [d - u for d, u in zip(d_j, u_j)]
    bar_cols = [C_DEF if g > 0 else C_UNDEF for g in gains_j]
    ax_gain.bar(sigmas, gains_j, color=bar_cols, width=0.8, edgecolor="none")
    ax_gain.axhline(0, color="white", lw=0.8, alpha=0.5)
    ax_gain.set_title("ATTF Robustness Gain (pp) vs. σ", color="white", fontsize=11)
    ax_gain.set_xlabel("σ (ms)", fontsize=10)
    ax_gain.set_ylabel("Gain (percentage points)", fontsize=10)
    ax_gain.set_xticks(sigmas)
    for x, g in zip(sigmas, gains_j):
        ax_gain.annotate(f"{g:+.1f}", xy=(x, g),
                         xytext=(0, 5 if g >= 0 else -12),
                         textcoords="offset points",
                         ha="center", color="white", fontsize=8)

    # Panel 4: Summary bar (clean, jitter@10ms, FGSM@0.30)
    metrics     = ["Clean\nAccuracy", "Jitter@10ms\nAccuracy",
                   "FGSM@0.30\nAccuracy"]
    undef_vals  = [u_j[0], u_j[-1], u_f[-1]]
    def_vals    = [d_j[0], d_j[-1], d_f[-1]]
    x_pos       = np.arange(len(metrics))
    width       = 0.35

    bars1 = ax_bar.bar(x_pos - width/2, undef_vals, width, color=C_UNDEF,
                       label="Undefended", edgecolor="none")
    bars2 = ax_bar.bar(x_pos + width/2, def_vals,   width, color=C_DEF,
                       label="ATTF Defended", edgecolor="none")
    ax_bar.axhline(ann_acc * 100, color=C_ANN, lw=1.5, linestyle=":",
                   label=f"ANN ({ann_acc*100:.1f}%)")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metrics, color="white", fontsize=9)
    ax_bar.set_title("Summary: Key Metric Comparison", color="white", fontsize=11)
    ax_bar.set_ylabel("Accuracy (%)", fontsize=10)
    ax_bar.legend(facecolor="#0f3460", labelcolor="white",
                  edgecolor="none", fontsize=8)

    for bar in list(bars1) + list(bars2):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    color="white", fontsize=7)

    plt.tight_layout()
    out = save_dir / "phase4_complete_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_hparam_heatmap(
    windows, burst_thresholds,
    clean_grid, defended_grid,
    sigma_target, save_dir,
):
    """2-D heatmap of defended accuracy over (W, K) grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        f"Phase 4 — ATTF Hyperparameter Sweep  (σ={sigma_target}ms)\n"
        "Left: Clean Accuracy (σ=0)   |   Right: Defended Accuracy at σ-target",
        fontsize=12, fontweight="bold", color="white",
    )

    for ax, grid, title in [
        (ax1, clean_grid * 100, "Clean Accuracy (%)"),
        (ax2, defended_grid * 100, f"Defended Accuracy at σ={sigma_target}ms (%)"),
    ]:
        ax.set_facecolor(PANEL_BG)
        im = ax.imshow(grid, cmap="plasma",
                       vmin=grid.min() - 1, vmax=grid.max() + 1,
                       aspect="auto")
        ax.set_xticks(range(len(burst_thresholds)))
        ax.set_xticklabels([str(k) for k in burst_thresholds], color="white")
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([str(w) for w in windows], color="white")
        ax.set_xlabel("Burst threshold K", fontsize=10, color="white")
        ax.set_ylabel("Window W", fontsize=10, color="white")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Annotate cells
        for i in range(len(windows)):
            for j in range(len(burst_thresholds)):
                ax.text(j, i, f"{grid[i,j]:.1f}",
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = save_dir / "phase4_hparam_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(
    sigmas, undef_j, def_j,
    epsilons, undef_f, def_f,
    ann_acc, save_dir,
):
    out = save_dir / "phase4_attf_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# Phase 4 — ATTF Defence Results"])
        w.writerow([f"# ANN accuracy: {ann_acc:.6f}"])
        w.writerow([f"# ATTF: W={config.ATTF_WINDOW}, K={config.ATTF_BURST_THRESHOLD},"
                    f" max_scale={config.ATTF_MAX_THRESH_SCALE},"
                    f" decay={config.ATTF_THRESH_DECAY}"])
        w.writerow([])

        w.writerow(["# Temporal Jitter"])
        w.writerow(["sigma_ms", "undefended_acc", "defended_acc",
                    "gain_pp", "undefended_drop_pp", "defended_drop_pp"])
        base_u = undef_j[0]
        base_d = def_j[0]
        for s, u, d in zip(sigmas, undef_j, def_j):
            w.writerow([f"{s:.1f}", f"{u:.6f}", f"{d:.6f}",
                        f"{(d-u)*100:.3f}",
                        f"{(base_u-u)*100:.3f}",
                        f"{(base_d-d)*100:.3f}"])

        w.writerow([])
        w.writerow(["# FGSM Spatial Attack"])
        w.writerow(["epsilon", "undefended_acc", "defended_acc", "gain_pp"])
        for e, u, d in zip(epsilons, undef_f, def_f):
            w.writerow([f"{e:.2f}", f"{u:.6f}", f"{d:.6f}",
                        f"{(d-u)*100:.3f}"])

    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    logger = utils.get_logger(__name__, log_file="phase4_attf.log")
    utils.set_seed(config.SEED)
    device = utils.get_device()

    utils.print_section("PHASE 4 — ATTF Defence Mechanism")
    logger.info(f"Device      : {device}")
    logger.info(f"ATTF params : W={config.ATTF_WINDOW}, "
                f"K={config.ATTF_BURST_THRESHOLD}, "
                f"scale={config.ATTF_MAX_THRESH_SCALE}, "
                f"decay={config.ATTF_THRESH_DECAY}")
    logger.info(f"Eval T      : {args.num_steps}")

    # ── 1. Load Phase-2 SNN checkpoint ──────────────────────────────────
    logger.info("\nLoading Phase-2 SNN checkpoint …")
    snn_ckpt_path = config.CHECKPOINTS_DIR / "converted_snn.pt"
    snn_ckpt      = torch.load(snn_ckpt_path, map_location=device)
    thresholds    = snn_ckpt["thresholds"]
    ann_accuracy  = snn_ckpt["ann_accuracy"]
    ann_flops     = snn_ckpt["ann_flops"]

    # Undefended SNN
    undefended = ConvertedSNN(thresholds=thresholds).to(device)
    undefended.load_state_dict(snn_ckpt["snn_state_dict"])
    undefended.eval()

    # ATTF-defended SNN
    defended = ATTFDefendedSNN(thresholds=thresholds).to(device)
    defended.load_from_snn_checkpoint(snn_ckpt["snn_state_dict"])
    defended.eval()

    # Load ANN for FGSM gradient computation
    from models.ann_model import BaselineANN
    ann = BaselineANN(num_classes=config.NUM_CLASSES)
    ann_ckpt = utils.load_checkpoint(
        config.ANN_CHECKPOINT, ann, device, logger)
    ann = ann.to(device)
    ann.eval()

    logger.info(
        f"Thresholds: {[f'{t:.4f}' for t in thresholds]}  "
        f"| ANN acc: {ann_accuracy*100:.2f}%"
    )

    # ── 2. Data loader ───────────────────────────────────────────────────
    _, test_loader = utils.get_fashion_mnist_loaders(
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
    )

    # ── 3. Clean accuracy ────────────────────────────────────────────────
    utils.print_section("Clean Accuracy (σ=0 — identical to Phase 2)")

    with utils.Timer() as t_clean:
        clean_undef = evaluate_clean(
            undefended, test_loader, args.num_steps, device,
            max_batches=args.max_batches, use_spike_trains=True)
        clean_def   = evaluate_clean(
            defended, test_loader, args.num_steps, device,
            max_batches=args.max_batches, use_spike_trains=True)

    logger.info(f"Undefended clean : {clean_undef*100:.2f}%")
    logger.info(f"ATTF defended    : {clean_def*100:.2f}%")
    logger.info(f"Clean overhead   : {(clean_undef - clean_def)*100:+.2f} pp")
    logger.info(f"Elapsed: {t_clean}")

    # ── 4. Temporal jitter ───────────────────────────────────────────────
    utils.print_section("Temporal Jitter Attack — Defended vs. Undefended")

    with utils.Timer() as t_jit:
        jitter_undef = evaluate_jitter(
            undefended, test_loader, args.sigmas,
            args.num_steps, device, max_batches=args.max_batches)
        jitter_def   = evaluate_jitter(
            defended, test_loader, args.sigmas,
            args.num_steps, device, max_batches=args.max_batches)

    logger.info(f"\n{'σ (ms)':>8} | {'Undefended':>11} | {'ATTF Def':>10} | {'Gain':>8}")
    logger.info("─" * 46)
    for s in args.sigmas:
        u, d = jitter_undef[s], jitter_def[s]
        logger.info(f"{s:>8.1f} | {u*100:>10.2f}% | {d*100:>9.2f}% | {(d-u)*100:>+7.2f} pp")

    # ── 5. FGSM ──────────────────────────────────────────────────────────
    utils.print_section("FGSM Spatial Attack — Defended vs. Undefended")

    with utils.Timer() as t_fgsm:
        fgsm_undef = evaluate_fgsm(
            ann, undefended, test_loader, args.epsilons,
            args.num_steps, device, max_batches=args.max_batches)
        fgsm_def   = evaluate_fgsm(
            ann, defended,   test_loader, args.epsilons,
            args.num_steps, device, max_batches=args.max_batches)

    logger.info(f"\n{'ε':>6} | {'Undefended':>11} | {'ATTF Def':>10} | {'Gain':>8}")
    logger.info("─" * 44)
    for e in args.epsilons:
        u, d = fgsm_undef[e], fgsm_def[e]
        logger.info(f"{e:>6.2f} | {u*100:>10.2f}% | {d*100:>9.2f}% | {(d-u)*100:>+7.2f} pp")

    # ── 6. ATTF hyperparameter sweep ─────────────────────────────────────
    if not args.no_sweep:
        utils.print_section("ATTF Hyperparameter Sweep")
        windows     = [3, 5, 8, 10]
        burst_Ks    = [2, 3, 4, 5]
        sigma_target = 10.0

        logger.info(
            f"Sweeping W={windows}, K={burst_Ks} at σ={sigma_target}ms "
            f"(max_batches={args.max_batches}) …"
        )
        clean_g, def_g, gain_g = attf_hparam_sweep(
            snn_ckpt=snn_ckpt,
            test_loader=test_loader,
            sigma_target=sigma_target,
            num_steps=args.num_steps,
            device=device,
            windows=windows,
            burst_thresholds=burst_Ks,
            max_scale=config.ATTF_MAX_THRESH_SCALE,
            decay=config.ATTF_THRESH_DECAY,
            max_batches=args.max_batches,
        )

        # Best configuration
        best_idx = np.unravel_index(def_g.argmax(), def_g.shape)
        best_W   = windows[best_idx[0]]
        best_K   = burst_Ks[best_idx[1]]
        logger.info(
            f"\nBest hyperparams: W={best_W}, K={best_K} → "
            f"clean={clean_g[best_idx]*100:.2f}%, "
            f"defended@{sigma_target}ms={def_g[best_idx]*100:.2f}%"
        )

        plot_hparam_heatmap(
            windows, burst_Ks, clean_g, def_g,
            sigma_target, config.RESULTS_DIR,
        )
        np.save(config.RESULTS_DIR / "phase4_hparam_clean.npy",   clean_g)
        np.save(config.RESULTS_DIR / "phase4_hparam_defended.npy", def_g)

    # ── 7. Summary ────────────────────────────────────────────────────────
    utils.print_section("Phase 4 — Defence Summary")
    max_jitter_gain = max((jitter_def[s] - jitter_undef[s]) * 100
                         for s in args.sigmas)
    max_fgsm_gain   = max((fgsm_def[e] - fgsm_undef[e]) * 100
                         for e in args.epsilons)
    clean_overhead  = (clean_undef - clean_def) * 100

    logger.info(f"Clean accuracy overhead  : {clean_overhead:+.2f} pp")
    logger.info(f"Max jitter robustness gain: {max_jitter_gain:+.2f} pp")
    logger.info(f"Max FGSM gain (incidental): {max_fgsm_gain:+.2f} pp")

    # ── 8. Plots ──────────────────────────────────────────────────────────
    utils.print_section("Generating Phase 4 Plots")

    undef_j_list = [jitter_undef[s] for s in args.sigmas]
    def_j_list   = [jitter_def[s]   for s in args.sigmas]
    undef_f_list = [fgsm_undef[e]   for e in args.epsilons]
    def_f_list   = [fgsm_def[e]     for e in args.epsilons]

    plot_jitter_comparison(
        args.sigmas, undef_j_list, def_j_list,
        ann_accuracy, config.RESULTS_DIR,
    )
    plot_fgsm_comparison(
        args.epsilons, undef_f_list, def_f_list,
        ann_accuracy, config.RESULTS_DIR,
    )
    plot_combined_summary(
        args.sigmas, undef_j_list, def_j_list,
        args.epsilons, undef_f_list, def_f_list,
        ann_accuracy, config.RESULTS_DIR,
    )

    # ── 9. CSV ────────────────────────────────────────────────────────────
    save_results_csv(
        args.sigmas, undef_j_list, def_j_list,
        args.epsilons, undef_f_list, def_f_list,
        ann_accuracy, config.RESULTS_DIR,
    )

    # ── 10. Save ATTF checkpoint ──────────────────────────────────────────
    attf_ckpt_path = config.CHECKPOINTS_DIR / "attf_defended_snn.pt"
    torch.save({
        "attf_state_dict": defended.state_dict(),
        "thresholds":      thresholds,
        "attf_config": {
            "window":        config.ATTF_WINDOW,
            "burst_threshold": config.ATTF_BURST_THRESHOLD,
            "max_scale":     config.ATTF_MAX_THRESH_SCALE,
            "decay":         config.ATTF_THRESH_DECAY,
        },
        "clean_accuracy_defended":   clean_def,
        "clean_accuracy_undefended": clean_undef,
        "ann_accuracy":   ann_accuracy,
        "jitter_results": {
            "undefended": {s: jitter_undef[s] for s in args.sigmas},
            "defended":   {s: jitter_def[s]   for s in args.sigmas},
        },
    }, attf_ckpt_path)
    logger.info(f"\nATTF checkpoint saved → {attf_ckpt_path}")
    logger.info("\n✅  Phase 4 complete.  All results saved to results/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Phase 4: ATTF Defence Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=config.TEMPORAL_JITTER_SIGMAS)
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=config.FGSM_EPSILONS)
    p.add_argument("--num_steps", type=int,
                   default=config.ATTACK_EVAL_TIMESTEPS)
    p.add_argument("--batch_size", type=int, default=config.SNN_BATCH_SIZE)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Limit test batches per evaluation (None=full set).")
    p.add_argument("--no_sweep", action="store_true",
                   help="Skip the ATTF hyperparameter sweep.")
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
