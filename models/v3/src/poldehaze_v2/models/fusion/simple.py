"""F0: concatenation-based SimpleFusion baseline."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from poldehaze_v2.models.common import ConvNormAct
from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    register_fusion,
    validate_fusion_inputs,
)


class SimpleFusion(FusionBlock):
    """``concat(raw, polar) -> 1x1 -> 3x3`` reference fusion."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
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
        return self.refine(self.reduce(torch.cat((raw_feat, polar_feat), dim=1)))


register_fusion(
    "simple",
    lambda channels, activation, groups: SimpleFusion(
        channels, activation, groups
    ),
)

