# ann_model.py

from __future__ import annotations  # list[X] generic hints on Python 3.9

import torch

import torch.nn as nn
import torch.nn.functional as F


class BaselineANN(nn.Module):
    """
    Three-layer CNN with ReLU6 activations for Fashion-MNIST classification.

    This model is the starting point for ANN-to-SNN conversion in Phase 2.
    All design choices are documented in the module docstring above.

    Args:
        num_classes: Number of output classes (default: 10 for Fashion-MNIST).
        dropout_p:   Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()

        # ── Feature extractor ──────────────────────────────────────────────
        # Three convolutional blocks.  Each block uses:
        #   Conv2d (3×3, same padding) → ReLU6 → MaxPool2d (2×2)
        #
        # Same padding (padding=1 with kernel=3) preserves spatial dimensions
        # through convolution so that each MaxPool step halves them cleanly.

        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels, 28×28 → 14×14
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU6(inplace=True),           # Bounded activation [0, 6]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 → 64 channels, 14×14 → 7×7
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 → 128 channels, 7×7 → 3×3
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Classifier head ────────────────────────────────────────────────
        # Spatial size after three 2×2 MaxPool on a 28×28 input:
        #   28 → 14 → 7 → 3  (floor division)
        # Flattened: 128 × 3 × 3 = 1152 features.
        self._flatten_size: int = 128 * 3 * 3  # 1152

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flatten_size, 256, bias=True),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes, bias=True),
            # No final activation — raw logits are expected by CrossEntropyLoss
            # and required for correct FGSM gradient computation in Phase 3.
        )

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Float tensor of shape ``(B, 1, 28, 28)``, normalised to ~N(0,1).

        Returns:
            Raw logit tensor of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    # ──────────────────────────────────────────────────────────────────────────
    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Return intermediate feature maps after each convolutional ReLU6.

        Used during Phase 2 threshold normalisation: we feed a calibration
        batch through the ANN and record the 99.9th-percentile activation at
        each layer to set the corresponding IF neuron threshold.

        Args:
            x: Input tensor ``(B, 1, 28, 28)``.

        Returns:
            List of activation tensors [after_relu6_1, after_relu6_2,
            after_relu6_3, after_relu6_4 (FC1)].
        """
        activations: list[torch.Tensor] = []
        # Walk through self.features manually to capture post-ReLU6 outputs.
        out = x
        for layer in self.features:
            out = layer(out)
            if isinstance(layer, nn.ReLU6):
                activations.append(out.detach().clone())

        # Classifier portion — capture post-ReLU6 FC1 activation.
        out = out.flatten(start_dim=1)
        for layer in self.classifier:
            out = layer(out)
            if isinstance(layer, nn.ReLU6):
                activations.append(out.detach().clone())

        return activations
