# SNN Adversarial Robustness

Adversarial robustness analysis of ANN-to-SNN converted networks under FGSM and temporal jitter attacks with a training-free ATTF defense for neuromorphic edge devices.

---

## Overview

This repository contains the experimental code for a dual-domain robustness study on converted spiking neural networks (SNNs). The main question is whether converted SNNs, which get weights directly from a trained ANN are equally affected by pixel-space attacks and temporal spike-timing attacks. And whether a simple post-hoc defense can bring back accuracy under timing stress without retraining.

Experiments are done on Fashion-MNIST across simulation times T ∈ {16, 32, 64, 128}.

**Key results at T = 64:**

- Converted SNN: 90.59% accuracy versus ANN baseline 93.20% at 4.18× less compute (64.75 MSynOps versus 15.50 MFLOPs)
- Spatial FGSM at ε = 0.30: accuracy goes down to 22.70% (−67.89 pp)
- Temporal jitter at σ = 10 ms: accuracy goes down to 85.10% (−5.49 pp)
- 12× difference between spatial and temporal vulnerability
- ATTF defense (W=3, K=4): brings back +0.47 pp at σ = 10 ms with no retraining and no loss in clean accuracy

---

## Repository Structure

```
.
├── models/
│   ├── ann_model.py           # Baseline CNN with Bounded-ReLU (ReLU6)
│   ├── snn_model.py           # Converted IF-neuron SNN (weight-copy + threshold balancing)
│   └── attf_snn.py            # ATTF-defended SNN with threshold neurons
│
├── phase1_baseline_ann.py     # Train and evaluate the baseline ANN
├── phase2_snn_conversion.py   # Convert ANN → SNN, profile SynOps versus FLOPs
├── phase3_adversarial.py      # FGSM spatial attack + temporal onset-delay jitter
├── phase3b_perclass_jitter.py # Per-class accuracy breakdown under jitter
├── phase4_attf_defence.py     # ATTF defense evaluation + hyperparameter sweep
├── run_w3k4_eval.py           # Standalone reproducer for W=3 K=4 result
│
├── config.py                  # Central configuration (all hyperparameters)
├── utils.py                   # Shared utilities (seeding, dataloading, checkpointing)
├── figures/                   # Publication figures (PNG + PDF)
└── requirements.txt
```

---

## Setup

**Python version 3.10 or above is needed.**

```bash
git clone https://github.com/<your-username>/snn-adversarial-robustness.git
cd snn-adversarial-robustness
pip install -r requirements.txt
```

Data is downloaded automatically via `torchvision` when it is first run.

---

## Reproducing Results

Run each step in order. Each script is self-contained and reads/writes checkpoints from `checkpoints/`.

```bash
# Phase 1. Train baseline ANN (~5 minutes on CPU <1 minute on GPU)
python phase1_baseline_ann.py

# Phase 2. Convert to SNN evaluate accuracy and SynOps across T ∈ {16,32,64,128}
python phase2_snn_conversion.py

# Phase 3. Spatial FGSM attack + temporal jitter sweep
python phase3_adversarial.py

# Phase 3b. Per-class jitter breakdown
python phase3b_perclass_jitter.py

# Phase 4. ATTF defense evaluation and hyperparameter grid search
python phase4_attf_defence.py
```

To reproduce the W=3 K=4 ATTF result (requires Phase 2 checkpoint):

```bash
python run_w3k4_eval.py
```

All outputs (figures, CSVs) are saved in `results/`.

---

## Methods Summary

### ANN-to-SNN Conversion

Weights from the trained CNN are moved directly into an Integrate-and-Fire spiking topology. Layer-wise firing thresholds are adjusted using the 99.9th percentile of activations recorded over a 512-sample calibration pass (Rueckauer et al. 2017). Neurons use subtractive reset to keep residual membrane charge.

### Spatial Attack — FGSM

The source ANN is used as a surrogate. FGSM perturbations are created on the ANN and passed directly to the SNN using the same weight tensors.

### Temporal Attack — Onset-Delay Jitter

Each input neuron gets a sampled half-normal onset delay d ~ |N(0, σ²)| hiding input current until time step d. This models AER interconnect latency, clock skew, and thermal fluctuations in physical neuromorphic hardware.

### ATTF Defense

Each IF neuron keeps track of its spike history in a circular buffer of length W. When the number of spikes in the window goes over threshold K, the effective firing threshold increases by ΔS = 0.5 (capped at 3.0×) and decreases at rate γ = 0.9 per step. No weight updates are done.

---

## Results

| | Accuracy | Compute |
|---|---|---|
| ANN Baseline | 93.20% | 15.50 MFLOPs |
| SNN T=16 | 84.16% | 13.72 MSynOps |
| SNN T=32 | 88.68% | 30.66 MSynOps |
| SNN T=64 | 90.59% | 64.75 MSynOps (4.18×↓) |
| SNN T=128 | 90.88% | 133.18 MSynOps |

**Spatial FGSM (T=64):**

| ε | Accuracy | Drop |
|---|---|---|
| 0.00 | 90.59% | — |
| 0.05 | 74.26% | −16.33 pp |
| 0.15 | 49.81% | −40.78 pp |
| 0.30 | 22.70% | −67.89 pp |

**Temporal Jitter (T=64):**

| σ (ms) | Undefended | ATTF (W=3, K=4) | Gain |
|---|---|---|---|
| 0.0 | 90.59% | 90.58% | −0.01 pp |
| 2.0 | 89.79% | 89.85% | +0.06 pp |
| 10.0 | 85.10% | 85.57% | +0.47 pp |

---

## Figures

All figures are in `figures/`. Key plots:

- `figure1_accuracy_efficiency.png` — Accuracy versus T and Pareto frontier
- `figure2_fgsm_degradation.png` — FGSM accuracy collapse curve
- `figure3_temporal_jitter.png` — Jitter robustness curve
- `figure4_perclass_jitter.png` — Per-class vulnerability spectrum
- `figure5_defense_comparison.png` — ATTF defended versus undefended
- `figure6_hyperparameter_heatmap.png` — ATTF W × K grid search

---

## Configuration

All hyperparameters are in `config.py`. Notable defaults:

```python
SEED                  = 42
ANN_EPOCHS            = 20
ANN_LR                = 1e-3
SNN_NORM_PERCENTILE   = 99.9
ATTACK_EVAL_TIMESTEPS = 64
ATTF_WINDOW           = 5
ATTF_BURST_THRESHOLD  = 4
ATTF_MAX_THRESH_SCALE = 3.0
ATTF_THRESH_DECAY     = 0.9
```

---

## Dependencies

| Package | Version |
|---|---|
| torch | ≥ 2.2.0 |
| torchvision | ≥ 0.17.0 |
| snntorch | ≥ 0.9.1 |
| numpy | ≥ 1.26.0 |
| scipy | ≥ 1.12.0 |
| matplotlib | ≥ 3.8.0 |
| seaborn | ≥ 0.13.0 |
| pandas | ≥ 2.2.0 |
| tqdm | ≥ 4.66.0 |

---

## References

1. Pfeiffer, M. & Pfeil, T. (2018). Deep learning with spiking neurons. *Front. Comput. Neurosci.*
2. Davies, M. et al. (2018). Loihi: A neuromorphic manycore processor. *IEEE Micro.*
3. Rueckauer, B. et al. (2017). Conversion of valued deep networks to efficient event-driven networks. *Front. Neurosci.*
4. Sengupta, A. et al. (2019). Going deeper in spiking networks. *Front. Neurosci.*
5. Goodfellow, I. et al. (2015). Harnessing adversarial examples. *ICLR.*
6. Xiao, H. et al. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747.*
