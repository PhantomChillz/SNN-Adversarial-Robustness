# run_w3k4_eval.py

import sys
import time
from pathlib import Path
import torch
import numpy as np

import config
import utils
from models.attf_snn import ATTFDefendedSNN
from phase4_attf_defence import evaluate_jitter, evaluate_clean

def run_evaluation():
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    snn_ckpt_path = config.CHECKPOINTS_DIR / "converted_snn.pt"
    snn_ckpt = torch.load(snn_ckpt_path, map_location=device)
    thresholds = snn_ckpt["thresholds"]

    _, test_loader = utils.get_fashion_mnist_loaders(
        batch_size_train=128,
        batch_size_test=128,
    )

    W = 3
    K = 4
    max_scale = 3.0
    decay = 0.9
    num_steps = 64

    model = ATTFDefendedSNN(
        thresholds=thresholds,
        attf_window=W,
        attf_burst_thresh=K,
        attf_max_scale=max_scale,
        attf_decay=decay,
    ).to(device)
    model.load_from_snn_checkpoint(snn_ckpt["snn_state_dict"])
    model.eval()

    print(f"Evaluating ATTF (W={W}, K={K}) on Clean (sigma=0)...")
    clean_acc = evaluate_clean(model, test_loader, num_steps, device, use_spike_trains=True)
    print(f"Clean accuracy: {clean_acc * 100:.2f}% ({clean_acc:.6f})")

    sigmas = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    print(f"Evaluating ATTF (W={W}, K={K}) on Sigmas: {sigmas}...")
    
    t0 = time.time()
    results = evaluate_jitter(model, test_loader, sigmas, num_steps, device)
    t1 = time.time()
    
    print(f"Done in {t1 - t0:.1f}s")
    print("Results:")
    for s in sigmas:
        print(f"sigma = {s:4.1f} ms -> {results[s] * 100:.2f}% ({results[s]:.6f})")

    # Save to a dedicated CSV for provenance
    out_csv = Path("results/phase4_attf_w3_k4_measured.csv")
    with open(out_csv, "w") as f:
        f.write("sigma_ms,defended_acc,defended_acc_pct\n")
        for s in sigmas:
            f.write(f"{s:.1f},{results[s]:.6f},{results[s]*100:.2f}\n")
    print(f"Saved real measured results to {out_csv}")

if __name__ == "__main__":
    run_evaluation()
