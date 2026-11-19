"""Patch-routed polarimetric residual experts on a concatenation anchor."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
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
from poldehaze_v2.models.fusion.simple import SimpleFusion
from poldehaze_v2.models.fusion.spatial_gate import stack_polar_context


class PatchRoutedPolarimetricExpertFusion(FusionBlock):
    """Refine a stable Concat path with locally routed polarimetric experts.

    The anchor is the standard concatenation-convolution fusion used by the
    reference model. Two residual experts independently process low-frequency
    veil evidence and high-frequency polarization detail. Their spatial gates
    are predicted on a coarse patch grid from full-Stokes context and aligned
    cross-stream agreement, then combined with per-channel gates. This keeps
    every correction conditional while leaving the strong anchor unaltered.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
        *,
        patch_size: int = 8,
        use_patch_routing: bool = True,
        use_frequency_experts: bool = True,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        self.use_patch_routing = bool(use_patch_routing)
        self.use_frequency_experts = bool(use_frequency_experts)
        hidden = max(channels // 4, 8)

        self.anchor = SimpleFusion(channels, activation_name, norm_groups)
        self.raw_norm = build_group_norm(channels, norm_groups)
        self.polar_norm = build_group_norm(channels, norm_groups)
        self.raw_alignment = nn.Conv2d(channels, channels, 1, bias=False)
        self.polar_alignment = nn.Conv2d(channels, channels, 1, bias=False)

        # Four Stokes fields, radiometric validity, four stream statistics,
        # cosine agreement, and mean absolute disagreement.
        router_channels = 11
        self.spatial_router = nn.Sequential(
            nn.Conv2d(router_channels, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )
        nn.init.zeros_(self.spatial_router[-1].weight)
        nn.init.zeros_(self.spatial_router[-1].bias)

        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            build_activation(activation_name),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        nn.init.zeros_(self.channel_router[-1].weight)
        nn.init.zeros_(self.channel_router[-1].bias)

        self.low_expert = self._expert(
            channels, activation_name, norm_groups, kernel_size=5
        )
        self.high_expert = self._expert(
            channels, activation_name, norm_groups, kernel_size=3
        )
        self.low_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.2))
        self.high_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.2))
        self.last_spatial_weight: Tensor | None = None

    @staticmethod
    def _expert(
        channels: int,
        activation_name: str,
        norm_groups: int,
        *,
        kernel_size: int,
    ) -> nn.Module:
        return nn.Sequential(
            ConvNormAct(
                channels * 3,
                channels,
                kernel_size=1,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        anchor = self.anchor(raw_feat, polar_feat, polar_context)
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        s0_norm, dolp, cos2aop, sin2aop = torch.chunk(context, 4, dim=1)
        raw = self.raw_alignment(self.raw_norm(raw_feat))
        polar = self.polar_alignment(self.polar_norm(polar_feat))

        agreement = functional.cosine_similarity(
            raw, polar, dim=1, eps=1e-4
        ).unsqueeze(1)
        disagreement = (raw - polar).abs().mean(dim=1, keepdim=True)
        validity = 4.0 * s0_norm.clamp(0.0, 1.0) * (
            1.0 - s0_norm.clamp(0.0, 1.0)
        )
        routing_input = torch.cat(
            (
                context,
                validity,
                raw.mean(dim=1, keepdim=True),
                raw.std(dim=1, keepdim=True, unbiased=False),
                polar.mean(dim=1, keepdim=True),
                polar.std(dim=1, keepdim=True, unbiased=False),
                agreement,
                disagreement,
            ),
            dim=1,
        )
        spatial_logits = self.spatial_router(routing_input)
        if self.use_patch_routing and min(spatial_logits.shape[-2:]) > 1:
            kernel = min(self.patch_size, *spatial_logits.shape[-2:])
            spatial_logits = functional.avg_pool2d(
                spatial_logits,
                kernel_size=kernel,
                stride=kernel,
                ceil_mode=True,
            )
            spatial_logits = functional.interpolate(
                spatial_logits,
                size=raw_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        spatial_gate = torch.sigmoid(spatial_logits)
        low_spatial, high_spatial = torch.chunk(spatial_gate, 2, dim=1)

        channel_gate = torch.sigmoid(
            self.channel_router(torch.cat((raw, polar, (raw - polar).abs()), dim=1))
        )
        low_channel, high_channel = torch.chunk(channel_gate, 2, dim=1)

        if self.use_frequency_experts:
            raw_low = self._blur(raw)
            polar_low = self._blur(polar)
            raw_high = raw - raw_low
            polar_high = polar - polar_low
        else:
            raw_low = raw_high = raw
            polar_low = polar_high = polar

        low = self.low_expert(
            torch.cat((raw_low, polar_low, polar_low - raw_low), dim=1)
        )
        high = self.high_expert(
            torch.cat((raw_high, polar_high, polar_high - raw_high), dim=1)
        )
        orientation_coherence = self._orientation_coherence(cos2aop, sin2aop)
        high_reliability = 0.25 + 0.75 * orientation_coherence * dolp
        low_gate = low_spatial * low_channel
        high_gate = high_spatial * high_channel * high_reliability
        self.last_spatial_weight = torch.cat(
            (low_spatial, high_spatial), dim=1
        ).detach()
        return (
            anchor
            + self.low_scale * low_gate * low
            + self.high_scale * high_gate * high
        )

    @staticmethod
    def _blur(inputs: Tensor) -> Tensor:
        padded = functional.pad(inputs, (2, 2, 2, 2), mode="reflect")
        return functional.avg_pool2d(padded, kernel_size=5, stride=1)

    @staticmethod
    def _orientation_coherence(cosine: Tensor, sine: Tensor) -> Tensor:
        cosine_mean = functional.avg_pool2d(
            functional.pad(cosine, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        sine_mean = functional.avg_pool2d(
            functional.pad(sine, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        return torch.sqrt(
            cosine_mean.square() + sine_mean.square() + 1e-4
        ).clamp(0.0, 1.0)


def _valid_groups(channels: int, preferred: int) -> int:
    groups = min(int(channels), int(preferred))
    while channels % groups != 0:
        groups -= 1
    return groups


register_fusion(
    "pper_no_patch",
    lambda channels, activation, groups: PatchRoutedPolarimetricExpertFusion(
        channels,
        activation,
        groups,
        use_patch_routing=False,
    ),
)
register_fusion(
    "pper_no_frequency",
    lambda channels, activation, groups: PatchRoutedPolarimetricExpertFusion(
        channels,
        activation,
        groups,
        use_frequency_experts=False,
    ),
)
register_fusion(
    "pper",
    lambda channels, activation, groups: PatchRoutedPolarimetricExpertFusion(
        channels,
        activation,
        groups,
    ),
)
