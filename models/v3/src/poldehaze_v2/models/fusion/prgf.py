"""F4: Polarimetric Reliability-Guided Fusion (PRGF)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.common import (
    build_activation,
    build_group_norm,
)
from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    register_fusion,
    validate_fusion_inputs,
)
from poldehaze_v2.models.fusion.spatial_gate import (
    CONTEXT_KEYS,
    stack_polar_context,
)


class PRGF(FusionBlock):
    """Fuse streams while preserving an unconditional raw residual path.

    Polar-to-raw modulation is jointly limited by a global channel gate and a
    learned spatial reliability map. The final residual projection is zero
    initialized, so the block begins as an exact raw-stream identity.
    """

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
        self.reliability = nn.Sequential(
            nn.Conv2d(len(CONTEXT_KEYS), hidden, 3, padding=1, bias=True),
            build_activation(activation_name),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.polar_to_raw = nn.Conv2d(channels, channels, 1, bias=False)
        self.raw_to_polar = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse_reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.fuse_output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        nn.init.zeros_(self.fuse_output[0].weight)

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        reliability = self.reliability(context)
        raw_channel_gate = self.raw_gate_from_polar(
            polar_feat.mean(dim=(-2, -1))
        )[:, :, None, None]
        polar_channel_gate = self.polar_gate_from_raw(
            raw_feat.mean(dim=(-2, -1))
        )[:, :, None, None]

        raw_mod = raw_feat + (
            reliability * raw_channel_gate * self.polar_to_raw(polar_feat)
        )
        polar_mod = polar_feat + (
            polar_channel_gate * self.raw_to_polar(raw_feat)
        )
        residual = self.fuse_output(
            self.fuse_reduce(torch.cat((raw_mod, polar_mod), dim=1))
        )
        return raw_feat + residual


class PRGFPlus(FusionBlock):
    """Feature-conditioned PRGF with a direct fusion path.

    The original PRGF deliberately starts as an exact raw-stream identity.
    This optimized variant keeps its polarimetric reliability and reciprocal
    channel gates, but lets the main fusion path learn from the first update.
    Reliability is conditioned on both the observed Stokes context and local
    agreement between normalized raw/polar features.  A zero-initialized
    correction remains on top of the active direct path, so extra capacity
    cannot destabilize the initial fusion output.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.raw_norm = build_group_norm(channels, norm_groups)
        self.polar_norm = build_group_norm(channels, norm_groups)
        self.raw_alignment = nn.Conv2d(channels, channels, 1, bias=False)
        self.polar_alignment = nn.Conv2d(channels, channels, 1, bias=False)

        # Four observed polarimetric fields plus raw/polar mean, standard
        # deviation, and their per-pixel aligned cosine agreement.
        reliability_channels = len(CONTEXT_KEYS) + 5
        self.reliability = nn.Sequential(
            nn.Conv2d(reliability_channels, hidden, 3, padding=1, bias=True),
            build_activation(activation_name),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            build_activation(activation_name),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
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
        self.polar_to_raw = nn.Conv2d(channels, channels, 1, bias=False)
        self.raw_to_polar = nn.Conv2d(channels, channels, 1, bias=False)

        self.direct_reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.direct_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.correction_reduce = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.correction_output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        nn.init.zeros_(self.correction_output[0].weight)

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        raw_aligned = self.raw_alignment(self.raw_norm(raw_feat))
        polar_aligned = self.polar_alignment(self.polar_norm(polar_feat))
        feature_context = torch.cat(
            (
                context,
                raw_aligned.mean(dim=1, keepdim=True),
                raw_aligned.std(dim=1, keepdim=True, unbiased=False),
                polar_aligned.mean(dim=1, keepdim=True),
                polar_aligned.std(dim=1, keepdim=True, unbiased=False),
                functional.cosine_similarity(
                    raw_aligned, polar_aligned, dim=1, eps=1e-4
                ).unsqueeze(1),
            ),
            dim=1,
        )
        reliability = self.reliability(feature_context)
        raw_channel_gate = self.raw_gate_from_polar(
            polar_aligned.mean(dim=(-2, -1))
        )[:, :, None, None]
        polar_channel_gate = self.polar_gate_from_raw(
            raw_aligned.mean(dim=(-2, -1))
        )[:, :, None, None]

        polar_delta = self.polar_to_raw(polar_aligned)
        raw_delta = self.raw_to_polar(raw_aligned)
        raw_mod = raw_feat + reliability * raw_channel_gate * polar_delta
        polar_mod = polar_feat + reliability * polar_channel_gate * raw_delta
        direct = self.direct_refine(
            self.direct_reduce(torch.cat((raw_mod, polar_mod), dim=1))
        )
        correction = self.correction_output(
            self.correction_reduce(
                torch.cat((direct, raw_feat, reliability * polar_delta), dim=1)
            )
        )
        return direct + correction


register_fusion(
    "prgf",
    lambda channels, activation, groups: PRGF(channels, activation, groups),
)
register_fusion(
    "prgf_plus",
    lambda channels, activation, groups: PRGFPlus(
        channels, activation, groups
    ),
)
