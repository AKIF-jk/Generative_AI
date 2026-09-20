"""Neural-network architectures for image restoration."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return a GroupNorm group count that exactly divides ``channels``."""
    for groups in range(min(channels, maximum_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormSiLU(nn.Sequential):
    """Convolution with normalization stable for every supported batch size."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _EncoderStage(nn.Module):
    """Stride-two downsampling followed by same-resolution refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = _ConvNormSiLU(in_channels, out_channels, stride=2)
        self.refine = _ConvNormSiLU(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.refine(self.downsample(x))


class _DecoderStage(nn.Module):
    """Interpolation-plus-convolution upsampling without checkerboard artefacts."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = _ConvNormSiLU(in_channels, out_channels)
        self.refine = _ConvNormSiLU(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.refine(self.project(x))


class _GatedSkip(nn.Module):
    """A compressed, per-channel-gated encoder-to-decoder skip."""

    def __init__(self, encoder_channels: int, decoder_channels: int) -> None:
        super().__init__()
        compressed_channels = decoder_channels // 4
        if compressed_channels < 1:
            raise ValueError("decoder_channels must be at least four for a gated skip")
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.compressed_channels = compressed_channels
        self.compress = nn.Conv2d(encoder_channels, compressed_channels, kernel_size=1)
        # sigmoid(-2) is about 0.119, so skips start as weak contributions.
        self.gate = nn.Parameter(torch.full((1, compressed_channels, 1, 1), -2.0))
        self.refine = nn.Sequential(
            _ConvNormSiLU(decoder_channels + compressed_channels, decoder_channels),
            _ConvNormSiLU(decoder_channels, decoder_channels),
        )

    def forward(self, decoder_feature: Tensor, encoder_feature: Tensor) -> Tensor:
        compressed = self.compress(encoder_feature)
        gated = compressed * torch.sigmoid(self.gate)
        return self.refine(torch.cat((decoder_feature, gated), dim=1))

    def gate_values(self) -> Tensor:
        """Detached per-channel gate strengths in the range [0, 1]."""
        return torch.sigmoid(self.gate).detach().flatten()


class UniversalRestorationAutoencoder(nn.Module):
    """Five-stage restoration autoencoder with a genuine vector bottleneck.

    ``skip_count=0`` sends all reconstruction information through the vector
    bottleneck. ``skip_count=1`` additionally enables the 16x16 skip; two
    skips enable that route and a 32x32 route. No input- or 128x128-level
    skip is created, and the model has no corruption-label input.
    """

    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 256,
        dropout: float = 0.0,
        skip_count: int = 0,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if skip_count not in (0, 1, 2):
            raise ValueError("skip_count must be one of 0, 1, or 2")

        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.skip_count = skip_count
        self.channel_progression = (
            base_channels,
            min(2 * base_channels, 512),
            min(4 * base_channels, 512),
            min(8 * base_channels, 512),
            512,
        )
        c1, c2, c3, c4, c5 = self.channel_progression

        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.ModuleList((
            _EncoderStage(3, c1),
            _EncoderStage(c1, c2),
            _EncoderStage(c2, c3),
            _EncoderStage(c3, c4),
            _EncoderStage(c4, c5),
        ))
        self.pre_latent_shape = (c5, 4, 4)
        self.flatten = nn.Flatten(start_dim=1)
        self.to_latent = nn.Linear(c5 * 4 * 4, latent_dim)
        self.latent_dropout = nn.Dropout(p=dropout)
        self.from_latent = nn.Linear(latent_dim, c5 * 4 * 4)

        # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.ModuleList((
            _DecoderStage(c5, c4),
            _DecoderStage(c4, c3),
            _DecoderStage(c3, c2),
            _DecoderStage(c2, c1),
            _DecoderStage(c1, c1),
        ))
        # Disabled variants intentionally contain no dormant skip modules.
        self.skip_16: _GatedSkip | None = _GatedSkip(c3, c3) if skip_count >= 1 else None
        self.skip_32: _GatedSkip | None = _GatedSkip(c2, c2) if skip_count >= 2 else None
        self.output = nn.Sequential(nn.Conv2d(c1, 3, kernel_size=3, padding=1), nn.Sigmoid())

    def encode_to_latent(self, x: Tensor) -> Tensor:
        """Encode an RGB batch all the way to its configured vector width."""
        for stage in self.encoder:
            x = stage(x)
        return self.latent_dropout(self.to_latent(self.flatten(x)))

    def decode_from_latent(
        self,
        latent: Tensor,
        *,
        encoder_features: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Decode a latent vector and optionally apply enabled intermediate skips."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape [B, {self.latent_dim}]")
        x = self.from_latent(latent).reshape(latent.shape[0], *self.pre_latent_shape)
        x = self.decoder[0](x)  # 8x8
        x = self.decoder[1](x)  # 16x16
        if self.skip_16 is not None:
            if encoder_features is None:
                raise ValueError("encoder_features are required when skips are enabled")
            x = self.skip_16(x, encoder_features[0])
        x = self.decoder[2](x)  # 32x32
        if self.skip_32 is not None:
            if encoder_features is None:
                raise ValueError("encoder_features are required when skips are enabled")
            x = self.skip_32(x, encoder_features[1])
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        return self.output(x)

    def forward(self, x: Tensor) -> Tensor:
        """Restore RGB tensors shaped ``[B, 3, 128, 128]`` in ``[0, 1]``."""
        if x.ndim != 4 or x.shape[1:] != (3, 128, 128):
            raise ValueError("x must have shape [B, 3, 128, 128]")
        features: list[Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            features.append(x)
        latent = self.latent_dropout(self.to_latent(self.flatten(x)))
        skip_features = (features[2], features[1]) if self.skip_count else None
        return self.decode_from_latent(latent, encoder_features=skip_features)

    def skip_gate_values(self) -> Mapping[str, Tensor]:
        """Return detached sigmoid gate values for experiment logging.

        They are model parameters only; no corruption labels or label-derived
        routing information influence these values.
        """
        values: dict[str, Tensor] = {}
        if self.skip_16 is not None:
            values["16x16"] = self.skip_16.gate_values()
        if self.skip_32 is not None:
            values["32x32"] = self.skip_32.gate_values()
        return values

    get_skip_gate_values = skip_gate_values
    inspect_skip_gates = skip_gate_values
