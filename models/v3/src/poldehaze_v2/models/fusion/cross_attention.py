"""F3: bidirectional channel and linear spatial cross-attention."""

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


class CrossAttentionFusion(FusionBlock):
    """Use O(C²HW) ELU+1 cross-attention with zero-initialized residuals."""

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

        self.query_raw = nn.Conv2d(channels, channels, 1, bias=False)
        self.key_polar = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_polar = nn.Conv2d(channels, channels, 1, bias=False)
        self.output_raw = nn.Conv2d(channels, channels, 1, bias=False)
        self.query_polar = nn.Conv2d(channels, channels, 1, bias=False)
        self.key_raw = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_raw = nn.Conv2d(channels, channels, 1, bias=False)
        self.output_polar = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.output_raw.weight)
        nn.init.zeros_(self.output_polar.weight)

        self.raw_norm = build_group_norm(channels, norm_groups)
        self.polar_norm = build_group_norm(channels, norm_groups)
        self.reduce = ConvNormAct(
            channels * 2, channels, kernel_size=1,
            activation_name=activation_name, norm_groups=norm_groups,
        )
        self.refine = ConvNormAct(
            channels, channels, kernel_size=3,
            activation_name=activation_name, norm_groups=norm_groups,
        )

    @staticmethod
    def _linear_cross_attention(
        query_feat: Tensor,
        key_feat: Tensor,
        value_feat: Tensor,
    ) -> Tensor:
        batch, channels, height, width = query_feat.shape
        query = (functional.elu(query_feat.flatten(2)) + 1.0).transpose(1, 2)
        key = (functional.elu(key_feat.flatten(2)) + 1.0).transpose(1, 2)
        value = value_feat.flatten(2).transpose(1, 2)
        key_value = torch.bmm(key.transpose(1, 2), value)
        key_sum = key.sum(dim=1)
        denominator = (
            query * key_sum.unsqueeze(1)
        ).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attended = torch.bmm(query, key_value) / denominator
        return attended.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        del polar_context
        validate_fusion_inputs(raw_feat, polar_feat)
        output_dtype = raw_feat.dtype
        with torch.autocast(device_type=raw_feat.device.type, enabled=False):
            raw_float = raw_feat.float()
            polar_float = polar_feat.float()
            raw_gate = self.raw_gate_from_polar(polar_float.mean(dim=(-2, -1)))
            polar_gate = self.polar_gate_from_raw(raw_float.mean(dim=(-2, -1)))
            raw_mod = raw_float * raw_gate[:, :, None, None]
            polar_mod = polar_float * polar_gate[:, :, None, None]

            raw_spatial = self.output_raw(
                self._linear_cross_attention(
                    self.query_raw(raw_mod),
                    self.key_polar(polar_mod),
                    self.value_polar(polar_mod),
                )
            )
            polar_spatial = self.output_polar(
                self._linear_cross_attention(
                    self.query_polar(polar_mod),
                    self.key_raw(raw_mod),
                    self.value_raw(raw_mod),
                )
            )
            raw_out = self.raw_norm(raw_mod + raw_spatial)
            polar_out = self.polar_norm(polar_mod + polar_spatial)
            fused = self.refine(self.reduce(torch.cat((raw_out, polar_out), dim=1)))
        return fused.to(output_dtype)


register_fusion(
    "cross_attention",
    lambda channels, activation, groups: CrossAttentionFusion(
        channels, activation, groups
    ),
)

