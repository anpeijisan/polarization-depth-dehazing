"""Common multimodal fusion controls used by the fusion study."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from poldehaze_v2.models.common import (
    ConvNormAct,
    build_activation,
    build_group_norm,
)
from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    register_fusion,
    validate_fusion_inputs,
)


class AdditiveFusion(FusionBlock):
    """Project both modalities and combine them by elementwise addition."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.raw_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        self.polar_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        self.activation = build_activation(activation_name)
        self.refine = ConvNormAct(
            channels,
            channels,
            kernel_size=3,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        del polar_context
        validate_fusion_inputs(raw_feat, polar_feat)
        combined = self.activation(
            self.raw_projection(raw_feat) + self.polar_projection(polar_feat)
        )
        return self.refine(combined)


class FiLMFusion(FusionBlock):
    """Use the polarization stream to FiLM-condition the raw stream."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.raw_norm = build_group_norm(channels, norm_groups)
        self.conditioner = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        self.refine = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                kernel_size=3,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        del polar_context
        validate_fusion_inputs(raw_feat, polar_feat)
        gamma, beta = torch.chunk(self.conditioner(polar_feat), 2, dim=1)
        conditioned = self.raw_norm(raw_feat) * (
            1.0 + 0.5 * torch.tanh(gamma)
        ) + beta
        return raw_feat + self.layer_scale * self.refine(conditioned)


class ResidualConcatFusion(FusionBlock):
    """Parameter-matched residual concatenation without polar priors."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.raw_projection = nn.Conv2d(
            channels, channels, 1, bias=False
        )
        self.polar_projection = nn.Conv2d(
            channels, channels, 1, bias=False
        )
        self.fuse = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
                kernel_size=1,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            ConvNormAct(
                channels,
                channels,
                kernel_size=3,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        del polar_context
        validate_fusion_inputs(raw_feat, polar_feat)
        residual = self.fuse(
            torch.cat(
                (
                    self.raw_projection(raw_feat),
                    self.polar_projection(polar_feat),
                ),
                dim=1,
            )
        )
        return raw_feat + self.layer_scale * residual


def _valid_groups(channels: int, preferred: int) -> int:
    groups = min(int(channels), int(preferred))
    while channels % groups != 0:
        groups -= 1
    return groups


register_fusion(
    "additive",
    lambda channels, activation, groups: AdditiveFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "residual_concat",
    lambda channels, activation, groups: ResidualConcatFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "film",
    lambda channels, activation, groups: FiLMFusion(
        channels, activation, groups
    ),
)
