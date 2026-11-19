"""F2: polarization-context spatial reliability gating."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from poldehaze_v2.models.common import ConvNormAct, build_activation
from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    register_fusion,
    validate_fusion_inputs,
)

CONTEXT_KEYS = ("s0_norm", "dolp", "cos2aop", "sin2aop")


def stack_polar_context(
    polar_context: Mapping[str, Tensor],
    spatial_shape: tuple[int, int],
) -> Tensor:
    """Stack the four required reliability fields after strict validation."""
    missing = [key for key in CONTEXT_KEYS if key not in polar_context]
    if missing:
        raise KeyError(f"Fusion polar_context is missing {missing}")
    values = [polar_context[key] for key in CONTEXT_KEYS]
    if any(value.ndim != 4 or value.shape[1] != 1 for value in values):
        raise ValueError("Every polar context field must have shape (B,1,H,W)")
    if any(value.shape[-2:] != spatial_shape for value in values):
        raise ValueError("Polar context fields must match the fusion spatial shape")
    return torch.cat(values, dim=1)


class SpatialGateFusion(FusionBlock):
    """Allow polar-to-raw modulation only where learned reliability is high."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.reliability = nn.Sequential(
            nn.Conv2d(len(CONTEXT_KEYS), hidden, 3, padding=1, bias=True),
            build_activation(activation_name),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.polar_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.reduce = ConvNormAct(
            channels * 2, channels, kernel_size=1,
            activation_name=activation_name, norm_groups=norm_groups,
        )
        self.refine = ConvNormAct(
            channels, channels, kernel_size=3,
            activation_name=activation_name, norm_groups=norm_groups,
        )

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        reliability = self.reliability(context)
        raw_mod = raw_feat + reliability * self.polar_projection(polar_feat)
        return self.refine(self.reduce(torch.cat((raw_mod, polar_feat), dim=1)))


register_fusion(
    "spatial_gate",
    lambda channels, activation, groups: SpatialGateFusion(
        channels, activation, groups
    ),
)

