"""F1: opposite-stream global channel-gated fusion."""

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


class ChannelGateFusion(FusionBlock):
    """Generate each stream's channel gate only from the opposite stream."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.raw_gate_from_polar = nn.Sequential(
            nn.Linear(channels, hidden),
            build_activation(activation_name),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.polar_gate_from_raw = nn.Sequential(
            nn.Linear(channels, hidden),
            build_activation(activation_name),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
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
        del polar_context
        validate_fusion_inputs(raw_feat, polar_feat)
        raw_gate = self.raw_gate_from_polar(polar_feat.mean(dim=(-2, -1)))
        polar_gate = self.polar_gate_from_raw(raw_feat.mean(dim=(-2, -1)))
        raw_mod = raw_feat + raw_gate[:, :, None, None] * raw_feat
        polar_mod = polar_feat + polar_gate[:, :, None, None] * polar_feat
        return self.refine(self.reduce(torch.cat((raw_mod, polar_mod), dim=1)))


register_fusion(
    "channel_gate",
    lambda channels, activation, groups: ChannelGateFusion(
        channels, activation, groups
    ),
)

