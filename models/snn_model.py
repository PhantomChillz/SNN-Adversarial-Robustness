# snn_model.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# snntorch: Leaky neuron with β=1 behaves as a pure Integrate-and-Fire unit.
try:
    import snntorch as snn
except ImportError as e:
    raise ImportError(
        "snntorch is required for Phase 2.  "
        "Install it with:  pip3 install snntorch>=0.9.1"
    ) from e

# Allow running from project root without explicit package installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Normalisation Helper
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdNormaliser:
    """
    Compute layer-wise IF firing thresholds from calibration data.

    Algorithm
    ---------
    1. Register forward hooks on every ReLU6 in the ANN.
    2. Run a calibration batch (from the *training* split) through the ANN.
    3. At each ReLU6, record the ``percentile``-th value of all activations
       seen across the batch.
    4. Return the list of thresholds [θ_1, θ_2, θ_3, θ_4] corresponding to
       conv1→conv2→conv3→fc1 in the BaselineANN.

    Using the training split (not test) for calibration prevents any
    inadvertent data leakage from the test set into the threshold values.

    Args:
        ann_model:  Trained ``BaselineANN`` in eval mode.
        percentile: Percentile for threshold selection (default 99.9).
    """

    def __init__(
        self,
        ann_model: nn.Module,
        percentile: float = config.SNN_NORM_PERCENTILE,
    ) -> None:
        self.model = ann_model
        self.percentile = percentile
        self._activation_buffers: Dict[str, List[torch.Tensor]] = {}
        self._hooks: List = []

    # ------------------------------------------------------------------
    def _make_hook(self, name: str):
        """Return a forward hook that accumulates post-ReLU6 activations."""
        def hook(module, inp, output):
            # Detach and move to CPU immediately to avoid CUDA OOM on large
            # calibration batches.
            self._activation_buffers[name].append(
                output.detach().cpu().float()
            )
        return hook

    # ------------------------------------------------------------------
    def compute_thresholds(
        self,
        calibration_loader,
        num_batches: int = 8,
    ) -> List[float]:
        """
        Run calibration and return per-layer thresholds.

        Args:
            calibration_loader: DataLoader yielding (images, labels) tuples.
                                Typically the training loader.
            num_batches:        Number of batches to use for calibration.
                                8 × 128 = 1024 samples is sufficient for a
                                stable percentile estimate.

        Returns:
            List of float thresholds ``[θ_1, θ_2, θ_3, θ_4]``.
        """
        self.model.eval()

        # ── Register hooks on every ReLU6 ────────────────────────────────
        layer_names: List[str] = []
        hook_idx = 0

        def register_relu6_hooks(module: nn.Module, prefix: str = "") -> None:
            nonlocal hook_idx
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.ReLU6):
                    tag = f"relu6_{hook_idx}"
                    layer_names.append(tag)
                    self._activation_buffers[tag] = []
                    h = child.register_forward_hook(self._make_hook(tag))
                    self._hooks.append(h)
                    hook_idx += 1
                else:
                    register_relu6_hooks(child, full_name)

        register_relu6_hooks(self.model)

        # ── Forward calibration pass ──────────────────────────────────────
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calibration_loader):
                if batch_idx >= num_batches:
                    break
                self.model(images.to(device))

        # ── Compute percentiles ───────────────────────────────────────────
        thresholds: List[float] = []
        for tag in layer_names:
            acts = torch.cat(self._activation_buffers[tag], dim=0)
            # Flatten all spatial/channel dims and compute global percentile.
            p_val = float(torch.quantile(acts.flatten(), self.percentile / 100.0))
            # Clamp: threshold must be > 0 and ≤ 6 (the ReLU6 ceiling).
            p_val = max(p_val, 1e-3)
            thresholds.append(p_val)

        # ── Clean up hooks ────────────────────────────────────────────────
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activation_buffers.clear()

        return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Converted SNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

class ConvertedSNN(nn.Module):
    """
    Integrate-and-Fire SNN converted from ``BaselineANN``.

    The convolutional and linear weight tensors are **identical** to those
    of the trained ANN; only the activation function changes from ReLU6 to
    spiking IF neurons.

    Architecture (mirrors BaselineANN exactly):
      Input  [B, 1, 28, 28]
        → Conv1 → IF(θ_1) → MaxPool → [B, 32, 14, 14]
        → Conv2 → IF(θ_2) → MaxPool → [B, 64, 7, 7]
        → Conv3 → IF(θ_3) → MaxPool → [B, 128, 3, 3]
        → Flatten → FC1 → IF(θ_4)   → [B, 256]
        → FC2 (membrane integrate, no spike) → argmax → class

    The output layer membrane potential (not spike count) is used as the
    class score because choosing a meaningful threshold for the output layer
    is non-trivial and membrane potential readout achieves higher accuracy.

    Args:
        thresholds:  List of 4 floats [θ_1, θ_2, θ_3, θ_4] computed by
                     ``ThresholdNormaliser``.
        num_classes: Number of output classes (10 for Fashion-MNIST).
    """

    def __init__(
        self,
        thresholds: List[float],
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        super().__init__()

        if len(thresholds) != 4:
            raise ValueError(
                f"Expected 4 thresholds (one per ReLU6 layer), "
                f"got {len(thresholds)}.  "
                "Run ThresholdNormaliser.compute_thresholds() first."
            )

        self.thresholds = thresholds

        # ── Learnable weight layers (weights copied from ANN in load_ann()) ─
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(128 * 3 * 3, 256, bias=True)  # 1152 → 256
        self.fc2   = nn.Linear(256, num_classes, bias=True)

        # ── Spiking IF neurons (snntorch.Leaky with β=1 ≡ IF) ─────────────
        # beta=1.0       : no membrane leak → pure integration (IF, not LIF)
        # threshold=θ_l  : calibrated firing threshold for layer l
        # reset_mechanism: 'subtract' (soft reset — subtract θ from membrane
        #                  after each spike).  This is mathematically closer
        #                  to the ANN ReLU6 than hard ('zero') reset because
        #                  the residual above threshold is preserved, allowing
        #                  a neuron that fired to continue contributing signal
        #                  in subsequent timesteps.
        # learn_beta, learn_threshold: False — these are fixed conversion
        #                  parameters, not trainable.
        self.lif1 = snn.Leaky(
            beta=1.0, threshold=thresholds[0],
            reset_mechanism="subtract",
            learn_beta=False, learn_threshold=False,
        )
        self.lif2 = snn.Leaky(
            beta=1.0, threshold=thresholds[1],
            reset_mechanism="subtract",
            learn_beta=False, learn_threshold=False,
        )
        self.lif3 = snn.Leaky(
            beta=1.0, threshold=thresholds[2],
            reset_mechanism="subtract",
            learn_beta=False, learn_threshold=False,
        )
        self.lif4 = snn.Leaky(
            beta=1.0, threshold=thresholds[3],
            reset_mechanism="subtract",
            learn_beta=False, learn_threshold=False,
        )
        # Output layer: plain IF membrane integrator (no firing threshold).
        # We simply accumulate input current over all T steps and read out
        # the final membrane vector as class logits.
        # Implemented manually: mem5_out += fc2(spk4_output)

    # ──────────────────────────────────────────────────────────────────────────
    def _init_membranes(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        Initialise all membrane potential tensors to zero at t=0.

        Returns 5 zero tensors (one per spiking layer + output integrator).
        """
        zeros = lambda *shape: torch.zeros(*shape, device=device)
        return (
            zeros(batch_size, 32, 28, 28),  # mem1: post-conv1, pre-pool
            zeros(batch_size, 64, 14, 14),  # mem2: post-conv2, pre-pool
            zeros(batch_size, 128, 7, 7),   # mem3: post-conv3, pre-pool
            zeros(batch_size, 256),          # mem4: post-fc1
            zeros(batch_size, config.NUM_CLASSES),  # mem5: output integrator
        )

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        num_steps: int,
        return_spike_counts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Simulate the SNN for ``num_steps`` timesteps.

        Input encoding: Direct (constant-current) coding — the identical
        normalised input tensor is presented at every timestep.  This is
        deterministic and equivalent to feeding a constant analogue current
        into the first spiking layer.

        Args:
            x:                  Input tensor ``(B, 1, 28, 28)``.
            num_steps:          Number of simulation timesteps T.
            return_spike_counts: If True, also return per-layer total spike
                                 counts (used for SynOps estimation).

        Returns:
            output_mem: Final output membrane potential ``(B, num_classes)``.
            spike_info: Dict of total spike counts per layer if
                        ``return_spike_counts=True``, else ``None``.
        """
        B = x.size(0)
        device = x.device

        mem1, mem2, mem3, mem4, mem5 = self._init_membranes(B, device)

        # Accumulators for SynOps estimation (only filled if requested).
        if return_spike_counts:
            total_spikes: Dict[str, float] = {
                "conv1": 0.0, "conv2": 0.0,
                "conv3": 0.0, "fc1":   0.0,
            }
        else:
            total_spikes = {}

        # ── Temporal simulation loop ──────────────────────────────────────
        for _ in range(num_steps):
            # ── Layer 1: Conv1 → IF(θ_1) → MaxPool ────────────────────────
            # Direct coding: same input x at every timestep.
            cur1 = self.conv1(x)            # (B, 32, 28, 28)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_pooled = self.pool(spk1)   # (B, 32, 14, 14)

            # ── Layer 2: Conv2 → IF(θ_2) → MaxPool ────────────────────────
            cur2 = self.conv2(spk1_pooled)  # (B, 64, 14, 14)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_pooled = self.pool(spk2)   # (B, 64, 7, 7)

            # ── Layer 3: Conv3 → IF(θ_3) → MaxPool ────────────────────────
            cur3 = self.conv3(spk2_pooled)  # (B, 128, 7, 7)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_pooled = self.pool(spk3)   # (B, 128, 3, 3)

            # ── Layer 4: FC1 → IF(θ_4) ────────────────────────────────────
            flat = spk3_pooled.flatten(start_dim=1)  # (B, 1152)
            cur4 = self.fc1(flat)                    # (B, 256)
            spk4, mem4 = self.lif4(cur4, mem4)

            # ── Layer 5: FC2 → membrane integrator (no threshold) ──────────
            mem5 = mem5 + self.fc2(spk4)             # (B, num_classes)

            # ── Accumulate spike counts for SynOps ────────────────────────
            if return_spike_counts:
                # .sum() counts ones across (B, C, H, W) or (B, N)
                total_spikes["conv1"] += float(spk1.sum().item())
                total_spikes["conv2"] += float(spk2.sum().item())
                total_spikes["conv3"] += float(spk3.sum().item())
                total_spikes["fc1"]   += float(spk4.sum().item())

        if return_spike_counts:
            return mem5, total_spikes
        return mem5, None

    # ──────────────────────────────────────────────────────────────────────────
    def forward_with_spike_trains(
        self,
        spike_trains: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the SNN using pre-generated binary spike trains as layer-1 input.

        This method is used in Phase 3 for the **temporal jitter attack**:
        rather than applying direct (constant-current) coding internally, the
        caller provides an explicit ``(T, B, C, H, W)`` spike train that may
        have been corrupted with Gaussian timing jitter.

        Layers 2–5 still operate normally (they receive spikes from the previous
        layer's IF neurons in real time — the jitter only affects layer-1 input).

        Args:
            spike_trains: Binary tensor of shape ``(T, B, 1, 28, 28)``
                          where T is the number of timesteps and each slice
                          ``spike_trains[t]`` is the input spike map for
                          timestep t.

        Returns:
            output_mem: Final output membrane potential ``(B, num_classes)``.
        """
        T  = spike_trains.shape[0]
        B  = spike_trains.shape[1]
        device = spike_trains.device

        mem1, mem2, mem3, mem4, mem5 = self._init_membranes(B, device)

        for t in range(T):
            x_t = spike_trains[t].float()    # (B, 1, 28, 28) — binary spikes

            # Layer 1: conv on binary spike input
            cur1 = self.conv1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_pooled = self.pool(spk1)

            # Layers 2-5: identical to normal forward pass
            cur2 = self.conv2(spk1_pooled)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_pooled = self.pool(spk2)

            cur3 = self.conv3(spk2_pooled)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_pooled = self.pool(spk3)

            flat = spk3_pooled.flatten(start_dim=1)
            cur4 = self.fc1(flat)
            spk4, mem4 = self.lif4(cur4, mem4)

            mem5 = mem5 + self.fc2(spk4)

        return mem5


    # ──────────────────────────────────────────────────────────────────────────
    def load_ann_weights(self, ann_model: nn.Module) -> None:
        """
        Copy all Conv2d and Linear weights from ``ann_model`` (BaselineANN)
        into this SNN's corresponding layers.

        The ANN's ``features`` Sequential contains:
          [0] Conv2d  [1] ReLU6  [2] MaxPool
          [3] Conv2d  [4] ReLU6  [5] MaxPool
          [6] Conv2d  [7] ReLU6  [8] MaxPool
        The ANN's ``classifier`` Sequential contains:
          [0] Flatten  [1] Linear  [2] ReLU6  [3] Dropout  [4] Linear

        We copy:  ann.features[0] → snn.conv1
                  ann.features[3] → snn.conv2
                  ann.features[6] → snn.conv3
                  ann.classifier[1] → snn.fc1
                  ann.classifier[4] → snn.fc2
        """
        ann_features   = ann_model.features

        ann_classifier = ann_model.classifier

        self.conv1.weight.data.copy_(ann_features[0].weight.data)
        self.conv1.bias.data.copy_(ann_features[0].bias.data)

        self.conv2.weight.data.copy_(ann_features[3].weight.data)
        self.conv2.bias.data.copy_(ann_features[3].bias.data)

        self.conv3.weight.data.copy_(ann_features[6].weight.data)
        self.conv3.bias.data.copy_(ann_features[6].bias.data)

        self.fc1.weight.data.copy_(ann_classifier[1].weight.data)
        self.fc1.bias.data.copy_(ann_classifier[1].bias.data)

        self.fc2.weight.data.copy_(ann_classifier[4].weight.data)
        self.fc2.bias.data.copy_(ann_classifier[4].bias.data)


# ─────────────────────────────────────────────────────────────────────────────
# SynOps Estimator
# ─────────────────────────────────────────────────────────────────────────────

class SynOpsEstimator:
    """
    Convert per-layer spike counts into Synaptic Operations (SynOps).

    SynOps formula
    --------------
    For a spike at position (c, h, w) in the input to a Conv2d(K, K, C_out)
    layer, it contributes to K×K output positions and to all C_out channels.
    Thus, over the full feature map:

        SynOps_conv = total_input_spikes × K_h × K_w × C_out

    For a spike at neuron i in the input to a Linear(C_out) layer:

        SynOps_fc = total_input_spikes × C_out

    These are summed over all timesteps and averaged per sample.

    Note: SynOps count *accumulate-only* operations (no multiplications),
    so 1 SynOp ≈ 0.5 ANN FLOPs in terms of hardware cost.
    """

    # Fixed topology constants for this specific architecture.
    LAYER_CONFIGS = {
        # layer_name: (kernel_h, kernel_w, C_out)
        # For Linear layers, kernel_h = kernel_w = 1.
        "conv1": (3, 3, 32),
        "conv2": (3, 3, 64),
        "conv3": (3, 3, 128),
        "fc1":   (1, 1, 256),   # Linear: each spike → 256 operations
    }

    @classmethod
    def compute(
        cls,
        spike_counts: Dict[str, float],
        batch_size: int,
        num_steps: int,
    ) -> Dict[str, float]:
        """
        Compute per-layer and total SynOps per sample.

        Args:
            spike_counts: Raw total spike counts returned by
                          ``ConvertedSNN.forward(..., return_spike_counts=True)``.
                          Each value is the *sum* over the batch and all T steps.
            batch_size:   Number of samples in the batch.
            num_steps:    Number of simulation timesteps T.

        Returns:
            Dict with per-layer SynOps/sample and ``"total"`` key.
        """
        synops: Dict[str, float] = {}
        total = 0.0

        for layer, (kh, kw, cout) in cls.LAYER_CONFIGS.items():
            raw_spikes = spike_counts.get(layer, 0.0)
            # Normalise: divide by batch_size to get per-sample count.
            # (raw_spikes already summed over all timesteps T)
            spikes_per_sample = raw_spikes / batch_size
            ops = spikes_per_sample * kh * kw * cout
            synops[layer] = ops
            total += ops

        synops["total"] = total
        return synops
