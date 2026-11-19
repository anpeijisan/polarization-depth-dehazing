"""Polarimetric frequency-selective raw/physics feature fusion."""

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
from poldehaze_v2.models.fusion.spatial_gate import stack_polar_context


class PolarimetricFrequencySelectiveFusion(FusionBlock):
    """Inject reliable polarization evidence without an ungated polar bypass.

    The raw stream preserves radiometric detail through an identity path.
    Full-Stokes evidence and aligned cross-stream agreement predict spatial
    reliability, while channel gates select useful polar features. In the
    frequency-selective form, low-frequency polarization primarily corrects
    the scattering veil and high-frequency polarization is admitted only when
    doubled-angle orientation is locally coherent.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
        *,
        use_frequency: bool = True,
        use_orientation: bool = True,
        use_polar_context: bool = True,
    ) -> None:
        super().__init__()
        self.use_frequency = bool(use_frequency)
        self.use_orientation = bool(use_orientation)
        self.use_polar_context = bool(use_polar_context)
        if self.use_orientation and not self.use_polar_context:
            raise ValueError(
                "Orientation coherence requires explicit polar context"
            )
        hidden = max(channels // 4, 8)

        self.raw_norm = build_group_norm(channels, norm_groups)
        self.polar_norm = build_group_norm(channels, norm_groups)
        self.raw_alignment = nn.Conv2d(channels, channels, 1, bias=False)
        self.polar_alignment = nn.Conv2d(channels, channels, 1, bias=False)

        # Four canonical Stokes context fields, radiometric validity, optional
        # doubled-angle coherence, and two aligned feature agreement fields.
        context_channels = (
            4 + 1 + int(self.use_orientation)
            if self.use_polar_context
            else 0
        )
        reliability_channels = context_channels + 2
        reliability_outputs = 2 if self.use_frequency else 1
        self.reliability = nn.Sequential(
            nn.Conv2d(
                reliability_channels, hidden, 3, padding=1, bias=False
            ),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, reliability_outputs, 1, bias=True),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

        channel_outputs = channels * (2 if self.use_frequency else 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            build_activation(activation_name),
            nn.Conv2d(hidden, channel_outputs, 1, bias=True),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

        if self.use_frequency:
            self.low_fusion = self._fusion_branch(
                channels, activation_name, norm_groups
            )
            self.high_fusion = self._fusion_branch(
                channels, activation_name, norm_groups
            )
            merge_channels = channels * 2
        else:
            self.full_fusion = self._fusion_branch(
                channels, activation_name, norm_groups
            )
            merge_channels = channels
        self.merge = nn.Sequential(
            ConvNormAct(
                merge_channels,
                channels,
                kernel_size=1,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )
        self.last_spatial_weight: Tensor | None = None

    @staticmethod
    def _fusion_branch(
        channels: int,
        activation_name: str,
        norm_groups: int,
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
                3,
                padding=1,
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
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        s0_norm, dolp, cos2aop, sin2aop = torch.chunk(context, 4, dim=1)
        raw = self.raw_alignment(self.raw_norm(raw_feat))
        polar = self.polar_alignment(self.polar_norm(polar_feat))

        radiometric_validity = (
            4.0 * s0_norm.clamp(0.0, 1.0)
            * (1.0 - s0_norm.clamp(0.0, 1.0))
        )
        agreement = functional.cosine_similarity(
            raw, polar, dim=1, eps=1e-4
        ).unsqueeze(1)
        disagreement = (raw - polar).abs().mean(dim=1, keepdim=True)
        reliability_fields = [agreement, disagreement]
        orientation_coherence = self._orientation_coherence(
            cos2aop, sin2aop
        )
        if self.use_polar_context:
            reliability_fields[:0] = [context, radiometric_validity]
            if self.use_orientation:
                reliability_fields.insert(2, orientation_coherence)
        spatial_gate = torch.sigmoid(
            self.reliability(torch.cat(reliability_fields, dim=1))
        )
        channel_gate = torch.sigmoid(
            self.channel_gate(torch.cat((raw, polar, (raw - polar).abs()), dim=1))
        )

        if self.use_frequency:
            low_spatial, high_spatial = torch.chunk(spatial_gate, 2, dim=1)
            low_channel, high_channel = torch.chunk(channel_gate, 2, dim=1)
            # Polarization directions must be locally coherent before their
            # high-frequency component is allowed to alter raw texture.
            if self.use_orientation:
                high_spatial = high_spatial * (
                    0.25 + 0.75 * orientation_coherence
                )
            raw_low = self._reflect_blur(raw)
            polar_low = self._reflect_blur(polar)
            raw_high = raw - raw_low
            polar_high = polar - polar_low
            low = self.low_fusion(
                self._gated_inputs(
                    raw_low, polar_low, low_spatial * low_channel
                )
            )
            high = self.high_fusion(
                self._gated_inputs(
                    raw_high, polar_high, high_spatial * high_channel
                )
            )
            residual = self.merge(torch.cat((low, high), dim=1))
            exposed_gate = torch.cat((low_spatial, high_spatial), dim=1)
        else:
            gate = spatial_gate * channel_gate
            residual = self.merge(
                self.full_fusion(self._gated_inputs(raw, polar, gate))
            )
            exposed_gate = spatial_gate

        self.last_spatial_weight = exposed_gate.detach()
        return raw_feat + self.layer_scale * residual

    @staticmethod
    def _gated_inputs(raw: Tensor, polar: Tensor, gate: Tensor) -> Tensor:
        gated_polar = gate * polar
        gated_innovation = gate * (polar - raw)
        return torch.cat((raw, gated_polar, gated_innovation), dim=1)

    @staticmethod
    def _reflect_blur(inputs: Tensor) -> Tensor:
        padded = functional.pad(inputs, (2, 2, 2, 2), mode="reflect")
        return functional.avg_pool2d(padded, kernel_size=5, stride=1)

    @staticmethod
    def _orientation_coherence(cosine: Tensor, sine: Tensor) -> Tensor:
        average_cosine = functional.avg_pool2d(
            functional.pad(cosine, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        average_sine = functional.avg_pool2d(
            functional.pad(sine, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        return torch.sqrt(
            average_cosine.square() + average_sine.square() + 1e-4
        ).clamp(0.0, 1.0)


def _valid_groups(channels: int, preferred: int) -> int:
    groups = min(int(channels), int(preferred))
    while channels % groups != 0:
        groups -= 1
    return groups


register_fusion(
    "pfsf_no_context",
    lambda channels, activation, groups: PolarimetricFrequencySelectiveFusion(
        channels,
        activation,
        groups,
        use_frequency=True,
        use_orientation=False,
        use_polar_context=False,
    ),
)
register_fusion(
    "pfsf_reliability",
    lambda channels, activation, groups: PolarimetricFrequencySelectiveFusion(
        channels,
        activation,
        groups,
        use_frequency=False,
        use_orientation=True,
    ),
)
register_fusion(
    "pfsf_no_orientation",
    lambda channels, activation, groups: PolarimetricFrequencySelectiveFusion(
        channels,
        activation,
        groups,
        use_frequency=True,
        use_orientation=False,
    ),
)
register_fusion(
    "pfsf",
    lambda channels, activation, groups: PolarimetricFrequencySelectiveFusion(
        channels,
        activation,
        groups,
        use_frequency=True,
        use_orientation=True,
    ),
)
