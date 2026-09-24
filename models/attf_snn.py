# attf_snn.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ---------------------------------------------------------------------------
# ATTF Neuron — adaptive-threshold IF cell
# ---------------------------------------------------------------------------

class ATTFNeuron(nn.Module):
    """
    Integrate-and-Fire neuron with Adaptive Temporal Thresholding Filter.

    State variables (per neuron):
      - mem           : membrane potential
      - spike_history : W-step ring buffer of spike outputs
      - thresh_scale  : current threshold multiplier S(t) in [1, max_scale]

    Firing rule (soft reset):
        fire  iff  mem >= base_threshold * thresh_scale
        mem   <- mem - base_threshold * thresh_scale  (after firing)

    Args:
        base_threshold : theta_base from Phase-2 calibration.
        window         : W — burst detection window length.
        burst_threshold: K — spikes-in-window that triggers adaptation.
        max_scale      : Maximum multiplier on theta_base.
        decay          : Per-step decay factor toward scale=1.
    """

    def __init__(
        self,
        base_threshold: float,
        window: int          = config.ATTF_WINDOW,
        burst_threshold: int = config.ATTF_BURST_THRESHOLD,
        max_scale: float     = config.ATTF_MAX_THRESH_SCALE,
        decay: float         = config.ATTF_THRESH_DECAY,
    ) -> None:
        super().__init__()
        self.base_threshold  = base_threshold
        self.window          = window
        self.burst_threshold = burst_threshold
        self.max_scale       = max_scale
        self.decay           = decay

        self._mem:           Optional[torch.Tensor] = None
        self._thresh_scale:  Optional[torch.Tensor] = None
        self._spike_history: Optional[torch.Tensor] = None
        self._history_ptr:   int = 0

    def reset_state(self, shape: Tuple[int, ...], device: torch.device) -> None:
        """Reset all state tensors. Called once per batch before the T-step loop."""
        self._mem           = torch.zeros(*shape, device=device)
        self._thresh_scale  = torch.ones(*shape,  device=device)
        self._spike_history = torch.zeros(self.window, *shape, device=device)
        self._history_ptr   = 0

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """
        Process one timestep of input current; return spike output.

        Steps:
          1. Integrate:       mem <- mem + current
          2. Effective thresh: theta_eff = base_threshold * thresh_scale
          3. Fire:            spk = (mem >= theta_eff)
          4. Soft reset:      mem <- mem - theta_eff * spk
          5. Record in history ring buffer
          6. Compute burst_count = sum over W-step window
          7. Update thresh_scale
        """
        if self._mem is None:
            self.reset_state(current.shape, current.device)

        # 1. Integrate
        self._mem = self._mem + current

        # 2. Effective threshold
        theta_eff = self.base_threshold * self._thresh_scale

        # 3. Fire
        spk = (self._mem >= theta_eff).float()

        # 4. Soft reset
        self._mem = self._mem - theta_eff * spk

        # 5. History ring buffer
        self._spike_history[self._history_ptr] = spk.detach()
        self._history_ptr = (self._history_ptr + 1) % self.window

        # 6. Burst detection
        burst_count = self._spike_history.sum(dim=0)        # (*shape)
        bursting    = (burst_count >= self.burst_threshold)  # bool

        # 7. Update scale: raise on burst, decay otherwise
        scale_up   = (self._thresh_scale / self.decay).clamp(max=self.max_scale)
        scale_down = (self._thresh_scale * self.decay).clamp(min=1.0)
        self._thresh_scale = torch.where(bursting, scale_up, scale_down)

        return spk


# ---------------------------------------------------------------------------
# Defended SNN — ATTF neurons replace standard IF neurons
# ---------------------------------------------------------------------------

class ATTFDefendedSNN(nn.Module):
    """
    Converted SNN with ATTF neurons for adversarial temporal robustness.

    Architecture: identical Conv/FC topology to ConvertedSNN (Phase 2).
    The four snntorch.Leaky neurons are replaced by ATTFNeuron instances.
    Weights are loaded directly from the Phase-2 SNN checkpoint.

    Args:
        thresholds  : [theta_1, theta_2, theta_3, theta_4] from Phase 2.
        num_classes : Number of output classes (10 for Fashion-MNIST).
        attf_window, attf_burst_thresh, attf_max_scale, attf_decay:
            ATTF hyperparameters (override config defaults for sweeps).
    """

    def __init__(
        self,
        thresholds: List[float],
        num_classes: int       = config.NUM_CLASSES,
        attf_window: int       = config.ATTF_WINDOW,
        attf_burst_thresh: int = config.ATTF_BURST_THRESHOLD,
        attf_max_scale: float  = config.ATTF_MAX_THRESH_SCALE,
        attf_decay: float      = config.ATTF_THRESH_DECAY,
    ) -> None:
        super().__init__()

        if len(thresholds) != 4:
            raise ValueError(f"Expected 4 thresholds, got {len(thresholds)}.")

        self.thresholds = thresholds

        # Weight layers (identical to ConvertedSNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(128 * 3 * 3, 256, bias=True)
        self.fc2   = nn.Linear(256, num_classes, bias=True)

        # ATTF neurons
        attf_kw = dict(
            window=attf_window,
            burst_threshold=attf_burst_thresh,
            max_scale=attf_max_scale,
            decay=attf_decay,
        )
        self.attf1 = ATTFNeuron(base_threshold=thresholds[0], **attf_kw)
        self.attf2 = ATTFNeuron(base_threshold=thresholds[1], **attf_kw)
        self.attf3 = ATTFNeuron(base_threshold=thresholds[2], **attf_kw)
        self.attf4 = ATTFNeuron(base_threshold=thresholds[3], **attf_kw)

    def _reset_all_states(self, B: int, device: torch.device) -> None:
        """Reset ATTF state for all four spiking layers before each batch."""
        self.attf1.reset_state((B, 32, 28, 28), device)
        self.attf2.reset_state((B, 64, 14, 14), device)
        self.attf3.reset_state((B, 128, 7, 7),  device)
        self.attf4.reset_state((B, 256),          device)

    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Direct-coding forward pass with ATTF adaptive thresholds.

        Args:
            x:         Input image (B, 1, 28, 28), normalised.
            num_steps: Simulation duration T.

        Returns:
            Output membrane potential (B, num_classes).
        """
        B = x.size(0)
        device = x.device
        self._reset_all_states(B, device)

        mem5 = torch.zeros(B, config.NUM_CLASSES, device=device)
        for _ in range(num_steps):
            spk1 = self.attf1(self.conv1(x))
            spk2 = self.attf2(self.conv2(self.pool(spk1)))
            spk3 = self.attf3(self.conv3(self.pool(spk2)))
            spk4 = self.attf4(self.fc1(self.pool(spk3).flatten(start_dim=1)))
            mem5 = mem5 + self.fc2(spk4)

        return mem5

    def forward_with_spike_trains(
        self, spike_trains: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed per-timestep input tensors.

        Used for temporal jitter evaluation: caller supplies a
        (T, B, 1, 28, 28) delayed-onset analogue sequence.

        Args:
            spike_trains: (T, B, 1, 28, 28) per-timestep input.

        Returns:
            Output membrane potential (B, num_classes).
        """
        T      = spike_trains.shape[0]
        B      = spike_trains.shape[1]
        device = spike_trains.device
        self._reset_all_states(B, device)

        mem5 = torch.zeros(B, config.NUM_CLASSES, device=device)
        for t in range(T):
            x_t  = spike_trains[t].float()
            spk1 = self.attf1(self.conv1(x_t))
            spk2 = self.attf2(self.conv2(self.pool(spk1)))
            spk3 = self.attf3(self.conv3(self.pool(spk2)))
            spk4 = self.attf4(self.fc1(self.pool(spk3).flatten(start_dim=1)))
            mem5 = mem5 + self.fc2(spk4)

        return mem5

    def load_from_snn_checkpoint(self, snn_state_dict: dict) -> None:
        """
        Copy Conv2d and Linear weights from a Phase-2 ConvertedSNN state_dict.
        ATTF neurons have no learnable parameters — only the five weight
        layers (conv1/2/3, fc1, fc2) are copied.
        """
        own_sd = self.state_dict()
        copy_keys = [k for k in own_sd if k.startswith(
            ("conv1.", "conv2.", "conv3.", "fc1.", "fc2.")
        )]
        for key in copy_keys:
            if key not in snn_state_dict:
                raise KeyError(
                    f"Key '{key}' not in SNN checkpoint. "
                    "Check that checkpoint comes from ConvertedSNN."
                )
            own_sd[key].copy_(snn_state_dict[key])
        self.load_state_dict(own_sd)


# ---------------------------------------------------------------------------
# ATTF Config helper for hyperparameter sweeps
# ---------------------------------------------------------------------------

class ATTFConfig:
    """Lightweight container for one ATTF hyperparameter configuration."""

    def __init__(
        self,
        window: int,
        burst_thresh: int,
        max_scale: float,
        decay: float,
    ) -> None:
        self.window       = window
        self.burst_thresh = burst_thresh
        self.max_scale    = max_scale
        self.decay        = decay

    def __repr__(self) -> str:
        return (
            f"ATTFConfig(W={self.window}, K={self.burst_thresh}, "
            f"scale={self.max_scale}, decay={self.decay})"
        )

    def label(self) -> str:
        return f"W{self.window}K{self.burst_thresh}S{self.max_scale}D{self.decay}"
