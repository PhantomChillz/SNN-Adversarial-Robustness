# phase3b_perclass_jitter.py

from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config, utils
from models.snn_model import ConvertedSNN

CLASSES = config.FASHION_MNIST_CLASSES
DARK_BG = "#1a1a2e"; PANEL_BG = "#16213e"

def gen_delayed(x, T, sigma):
    B,C,H,W = x.shape; dev = x.device
    if sigma == 0.0:
        return x.unsqueeze(0).expand(T,B,C,H,W)
    d = (torch.randn(B,C,H,W,device=dev).abs()*sigma).round().long().clamp(0,T-1)
    t = torch.arange(T,device=dev).view(T,1,1,1,1)
    mask = (t >= d.unsqueeze(0).expand(T,B,C,H,W)).float()
    return x.unsqueeze(0).expand(T,B,C,H,W)*mask

def perclass_acc(model, loader, sigma, T, device):
    model.eval()
    correct = [0]*10; total = [0]*10
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            inp = gen_delayed(imgs, T, sigma)
            out = model.forward_with_spike_trains(inp)
            preds = out.argmax(1)
            for c in range(10):
                mask = (labels == c)
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                total[c]   += mask.sum().item()
    return [correct[c]/total[c]*100 if total[c]>0 else 0 for c in range(10)]

def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()
    ckpt   = torch.load(config.CHECKPOINTS_DIR/"converted_snn.pt", map_location=device)
    model  = ConvertedSNN(thresholds=ckpt["thresholds"]).to(device)
    model.load_state_dict(ckpt["snn_state_dict"])
    _, loader = utils.get_fashion_mnist_loaders(64, 64)

    print("Evaluating σ=0 (clean)…")
    acc_clean = perclass_acc(model, loader, 0.0, 64, device)
    print("Evaluating σ=10ms…")
    acc_jitter= perclass_acc(model, loader, 10.0, 64, device)

    drop = [c-j for c,j in zip(acc_clean, acc_jitter)]

    # Save CSV
    import csv
    with open(config.RESULTS_DIR/"phase3b_perclass_jitter.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","clean_acc","jitter10ms_acc","drop_pp"])
        for i,cls in enumerate(CLASSES):
            w.writerow([cls, f"{acc_clean[i]:.2f}", f"{acc_jitter[i]:.2f}", f"{drop[i]:.2f}"])

    # Plot
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Per-Class Accuracy Under Temporal Jitter (σ=10ms) — Fashion-MNIST",
                 fontsize=13, fontweight="bold", color="white")

    for ax in axes: ax.set_facecolor(PANEL_BG)

    x = np.arange(10)
    w = 0.38
    ax = axes[0]
    b1 = ax.bar(x-w/2, acc_clean,  w, label="Clean (σ=0)",    color="#53d8fb", edgecolor="none")
    b2 = ax.bar(x+w/2, acc_jitter, w, label="Jitter (σ=10ms)",color="#a29bfe", edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=40, ha="right", color="white", fontsize=8)
    ax.set_ylabel("Accuracy (%)", color="white"); ax.tick_params(colors="white")
    ax.set_title("Clean vs Jitter Accuracy by Class", color="white")
    ax.legend(facecolor="#0f3460",labelcolor="white",edgecolor="none",fontsize=8)
    ax.grid(True,alpha=0.15,color="white",linestyle="--")
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_edgecolor("#444466")

    ax2 = axes[1]
    colours = ["#e94560" if d>2 else "#f5a623" if d>0.5 else "#7bed9f" for d in drop]
    bars = ax2.bar(x, drop, color=colours, edgecolor="none")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=40, ha="right", color="white", fontsize=8)
    ax2.set_ylabel("Accuracy Drop (pp)", color="white"); ax2.tick_params(colors="white")
    ax2.set_title("Accuracy Drop at σ=10ms per Class", color="white")
    ax2.axhline(0, color="white", lw=0.8, alpha=0.4)
    ax2.grid(True,alpha=0.15,color="white",linestyle="--")
    ax2.set_facecolor(PANEL_BG)
    for spine in ax2.spines.values(): spine.set_edgecolor("#444466")
    for bar, d in zip(bars, drop):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"{d:.1f}", ha="center", color="white", fontsize=7)

    plt.tight_layout()
    out = config.RESULTS_DIR/"phase3b_perclass_jitter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved → {out}")
    print("\nPer-class results:")
    for i,cls in enumerate(CLASSES):
        print(f"  {cls:15s}: clean={acc_clean[i]:.1f}%  jitter={acc_jitter[i]:.1f}%  drop={drop[i]:+.1f}pp")

if __name__=="__main__":
    main()
