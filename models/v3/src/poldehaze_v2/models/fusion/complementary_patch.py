"""Complementarity-preserving patch fusion of polarization experts."""

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
from poldehaze_v2.models.fusion.prgf import PRGF
from poldehaze_v2.models.fusion.spatial_gate import (
    SpatialGateFusion,
    stack_polar_context,
)


class ComplementarityPreservingPatchFusion(FusionBlock):
    """Keep the strongest expert intact and route its complementary residual.

    SpatialGate is the radiometrically stable anchor. PRGF supplies a second
    polarimetric interpretation, but is never averaged directly with the
    anchor. Normalized expert consensus, signed disagreement, and absolute
    disagreement are retained as separate subspaces. A coarse spatial router
    and a channel router select the useful signed disagreement before a
    lightweight residual projection returns it to the anchor representation.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
        *,
        patch_size: int = 8,
        protect_anchor: bool = False,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        self.protect_anchor = bool(protect_anchor)
        hidden = max(channels // 8, 8)

        self.anchor_expert = SpatialGateFusion(
            channels, activation_name, norm_groups
        )
        self.complementary_expert = PRGF(
            channels, activation_name, norm_groups
        )
        self.anchor_norm = build_group_norm(channels, norm_groups)
        self.complementary_norm = build_group_norm(channels, norm_groups)

        # Four observed Stokes context fields, four expert statistics,
        # cosine agreement, and mean absolute disagreement.
        self.spatial_router = nn.Sequential(
            nn.Conv2d(10, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.spatial_router[-1].weight)
        nn.init.zeros_(self.spatial_router[-1].bias)

        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            build_activation(activation_name),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.channel_router[-1].weight)
        nn.init.zeros_(self.channel_router[-1].bias)

        self.residual_projection = nn.Sequential(
            ConvNormAct(
                channels * 4,
                channels,
                kernel_size=1,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        initial_scale = 0.0 if self.protect_anchor else 0.1
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), initial_scale)
        )
        self.last_spatial_weight: Tensor | None = None

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        anchor = self.anchor_expert(raw_feat, polar_feat, polar_context)
        complementary = self.complementary_expert(
            raw_feat, polar_feat, polar_context
        )
        anchor_normalized = self.anchor_norm(anchor)
        complementary_normalized = self.complementary_norm(complementary)
        consensus = 0.5 * (anchor_normalized + complementary_normalized)
        disagreement = complementary_normalized - anchor_normalized
        absolute_disagreement = disagreement.abs()

        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        router_input = torch.cat(
            (
                context,
                anchor_normalized.mean(dim=1, keepdim=True),
                anchor_normalized.abs().mean(dim=1, keepdim=True),
                complementary_normalized.mean(dim=1, keepdim=True),
                complementary_normalized.abs().mean(
                    dim=1, keepdim=True
                ),
                functional.cosine_similarity(
                    anchor_normalized,
                    complementary_normalized,
                    dim=1,
                    eps=1e-4,
                ).unsqueeze(1),
                absolute_disagreement.mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_logits = self.spatial_router(router_input)
        if min(spatial_logits.shape[-2:]) > 1:
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
        spatial_weight = torch.sigmoid(spatial_logits)
        channel_weight = torch.sigmoid(
            self.channel_router(
                torch.cat(
                    (
                        anchor_normalized,
                        complementary_normalized,
                        absolute_disagreement,
                    ),
                    dim=1,
                )
            )
        )
        routed_disagreement = (
            spatial_weight * channel_weight * disagreement
        )
        residual = self.residual_projection(
            torch.cat(
                (
                    consensus,
                    disagreement,
                    absolute_disagreement,
                    routed_disagreement,
                ),
                dim=1,
            )
        )
        self.last_spatial_weight = spatial_weight
        scale = (
            0.05 * torch.tanh(self.layer_scale)
            if self.protect_anchor
            else self.layer_scale
        )
        return anchor + scale * residual


def _valid_groups(channels: int, preferred: int) -> int:
    groups = min(int(channels), int(preferred))
    while channels % groups != 0:
        groups -= 1
    return groups


register_fusion(
    "cppf",
    lambda channels, activation, groups: ComplementarityPreservingPatchFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "cppf_protected",
    lambda channels, activation, groups: ComplementarityPreservingPatchFusion(
        channels, activation, groups, protect_anchor=True
    ),
)
