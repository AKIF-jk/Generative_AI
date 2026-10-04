"""Corruption classifier for Task 2: Hard-Routed Restoration.

A lightweight convolutional classifier that predicts one of four input
classes: clean, salt-and-pepper noise, Gaussian blur, or rectangular
occlusion.  The predicted class drives hard routing to the appropriate
specialist autoencoder.

Architecture: small CNN backbone (configurable channels) → global average
pool → dropout → linear head.  GroupNorm is used throughout so batch-size
of 1 works cleanly at inference time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NUM_CLASSES = 4  # clean / salt_pepper / gaussian_blur / occlusion


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return a GroupNorm group count that exactly divides ``channels``."""
    for groups in range(min(channels, maximum_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBnSiLU(nn.Sequential):
    """Conv → GroupNorm → SiLU block, stride configurable."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )


class CorruptionClassifier(nn.Module):
    """Four-class corruption classifier.

    Parameters
    ----------
    base_channels:
        Number of channels in the first convolutional stage.  Doubles at
        each of the four downsampling stages up to 512.
    dropout:
        Dropout probability applied between the global-average-pool and
        the linear head.

    Input
    -----
    x : Tensor [B, 3, 128, 128] in [0, 1]

    Output
    ------
    logits : Tensor [B, 4]  (raw scores; apply softmax for probabilities)
    """

    def __init__(self, base_channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        c1 = base_channels
        c2 = min(c1 * 2, 256)
        c3 = min(c1 * 4, 256)
        c4 = min(c1 * 8, 256)

        # 128 → 64 → 32 → 16 → 8
        self.stem = nn.Sequential(
            _ConvBnSiLU(3, c1, stride=2),   # 64×64
            _ConvBnSiLU(c1, c1),
        )
        self.stage2 = nn.Sequential(
            _ConvBnSiLU(c1, c2, stride=2),  # 32×32
            _ConvBnSiLU(c2, c2),
        )
        self.stage3 = nn.Sequential(
            _ConvBnSiLU(c2, c3, stride=2),  # 16×16
            _ConvBnSiLU(c3, c3),
        )
        self.stage4 = nn.Sequential(
            _ConvBnSiLU(c3, c4, stride=2),  # 8×8
            _ConvBnSiLU(c4, c4),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)   # → [B, c4, 1, 1]
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(c4, NUM_CLASSES)

        # Xavier init on the head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.base_channels = base_channels
        self.dropout_p = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logits shaped [B, 4]."""
        if x.ndim != 4 or x.shape[1:] != (3, 128, 128):
            raise ValueError("x must have shape [B, 3, 128, 128]")
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)

    def predict_probs(self, x: Tensor) -> Tensor:
        """Softmax probabilities [B, 4] (no gradient)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def predict_class(self, x: Tensor) -> Tensor:
        """Argmax class index [B] (no gradient)."""
        return self.predict_probs(x).argmax(dim=1)
