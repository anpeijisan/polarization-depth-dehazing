"""Decoder attention variants for controlled full-Stokes ablations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.common import build_group_norm
from poldehaze_v2.models.fusion.spatial_gate import stack_polar_context


class _FeatureChannelAttention(nn.Module):
    """Shared feature-statistics channel attention and residual mixing."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.average_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.std_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.last_attention_map: Tensor | None = None

    def _channel_attend(self, features: Tensor, bias: Tensor | None = None) -> Tensor:
        channel_std = features.std(dim=(-2, -1), keepdim=True)
        logits = (
            self.average_mlp(self.average_pool(features))
            + self.std_mlp(channel_std)
        )
        if bias is not None:
            logits = logits + bias
        return features * torch.sigmoid(logits)

    def _residual_mix(self, features: Tensor, attended: Tensor, mask: Tensor) -> Tensor:
        self.last_attention_map = mask.detach()
        result = attended * mask
        return features + self.gate * (result - features)


class FeatureStatisticsAttention(_FeatureChannelAttention):
    """Generic channel-spatial attention without polarization context."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__(channels, reduction)
        mid = max(channels // reduction, 4)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, mid, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        """Refine ``features`` while intentionally ignoring ``polar_context``."""
        del polar_context
        attended = self._channel_attend(features)
        spatial_input = torch.cat(
            (
                attended.mean(dim=1, keepdim=True),
                attended.std(dim=1, keepdim=True),
            ),
            dim=1,
        )
        return self._residual_mix(features, attended, self.spatial(spatial_input))


class DoLPGuidedSpatialAttention(_FeatureChannelAttention):
    """V5-compatible decoder attention conditioned only on observed DoLP."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__(channels, reduction)
        mid = max(channels // reduction, 4)
        dolp_channels = max(channels // 8, 4)
        self.dolp_projection = nn.Conv2d(
            1, dolp_channels, 1, bias=False
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                2 + dolp_channels,
                mid,
                3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        """Apply the established DoLP-only channel-spatial attention."""
        dolp = _context_field(polar_context, "dolp", features)
        attended = self._channel_attend(features)
        spatial_input = torch.cat(
            (
                attended.mean(dim=1, keepdim=True),
                attended.std(dim=1, keepdim=True),
                self.dolp_projection(dolp),
            ),
            dim=1,
        )
        return self._residual_mix(features, attended, self.spatial(spatial_input))


class FullStokesGuidedAttention(_FeatureChannelAttention):
    """Channel-spatial attention conditioned on a stable full-Stokes field.

    Context order is ``(S0_norm, DoLP, cos(2AoP), sin(2AoP))``. When
    ``reliability_aware`` is enabled, doubled-angle components are weighted by
    both DoLP and local orientation coherence before entering the attention.
    This suppresses undefined directions in weakly polarized regions without
    introducing the periodic discontinuity of a scalar AoP channel.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        *,
        reliability_aware: bool = False,
    ) -> None:
        super().__init__(channels, reduction)
        mid = max(channels // reduction, 4)
        context_channels = max(channels // 8, 4)
        self.reliability_aware = bool(reliability_aware)
        self.context_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, channels, 1, bias=False),
        )
        self.context_projection = nn.Sequential(
            nn.Conv2d(4, context_channels, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                2 + context_channels,
                mid,
                3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.last_orientation_reliability: Tensor | None = None

    def forward(
        self,
        features: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        """Refine decoder features using all four stable Stokes fields."""
        context = stack_polar_context(
            polar_context, features.shape[-2:]
        )
        if context.shape[0] != features.shape[0]:
            raise ValueError(
                "Polar context batch size must match decoder features"
            )
        context = self._prepare_context(context)
        attended = self._channel_attend(
            features, self.context_channel(context)
        )
        context_features = self.context_projection(context)
        spatial_input = torch.cat(
            (
                attended.mean(dim=1, keepdim=True),
                attended.std(dim=1, keepdim=True),
                context_features,
            ),
            dim=1,
        )
        return self._residual_mix(features, attended, self.spatial(spatial_input))

    def _prepare_context(self, context: Tensor) -> Tensor:
        if not self.reliability_aware:
            self.last_orientation_reliability = None
            return context
        s0_norm, dolp, cos2aop, sin2aop = context.split(1, dim=1)
        local_cos = functional.avg_pool2d(
            cos2aop, kernel_size=3, stride=1, padding=1
        )
        local_sin = functional.avg_pool2d(
            sin2aop, kernel_size=3, stride=1, padding=1
        )
        coherence = torch.sqrt(
            local_cos.square() + local_sin.square() + 1e-4
        ).clamp(0.0, 1.0)
        reliability = (dolp.clamp(0.0, 1.0) * coherence).clamp(0.0, 1.0)
        self.last_orientation_reliability = reliability.detach()
        return torch.cat(
            (
                s0_norm,
                dolp,
                reliability * cos2aop,
                reliability * sin2aop,
            ),
            dim=1,
        )


class PolarizationContextModulation(nn.Module):
    """Inject polarization context through bounded residual modulation.

    Unlike the historical decoder attention, this block can add a learned
    context-conditioned residual instead of only attenuating existing image
    features.  Channel and spatial gates are centred at one, while a bounded
    learnable residual scale limits the total update to half of the modulated
    residual magnitude.  A single block is resolution independent and can be
    instantiated at post-fusion, bottleneck, or decoder stages.

    Args:
        channels: Feature channels at the insertion point.
        context_mode: One of ``feature_stats``, ``dolp``, ``full_stokes``, or
            ``full_stokes_reliability``.
        reduction: Hidden-channel reduction factor.
        initial_residual_scale: Initial residual magnitude in ``(0, 0.5)``.
    """

    MODES = (
        "feature_stats",
        "dolp",
        "full_stokes",
        "full_stokes_reliability",
    )

    def __init__(
        self,
        channels: int,
        context_mode: str,
        reduction: int = 8,
        initial_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        normalized_mode = str(context_mode).strip().lower()
        if normalized_mode not in self.MODES:
            raise ValueError(
                f"Unsupported context_mode={context_mode!r}; "
                f"expected one of {self.MODES}"
            )
        if not 0.0 < initial_residual_scale < 0.5:
            raise ValueError("initial_residual_scale must lie in (0, 0.5)")
        self.context_mode = normalized_mode
        self.reliability_aware = normalized_mode == "full_stokes_reliability"
        hidden = max(channels // reduction, 8)
        self.feature_norm = build_group_norm(channels, 8)
        self.feature_value = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            build_group_norm(channels, 8),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.feature_average = nn.AdaptiveAvgPool2d(1)
        self.feature_average_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.feature_std_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        context_channels = 0
        if normalized_mode == "dolp":
            context_channels = 1
        elif normalized_mode in {"full_stokes", "full_stokes_reliability"}:
            context_channels = 4
        self.context_channels = context_channels
        spatial_channels = 2
        if context_channels:
            self.context_features = nn.Sequential(
                nn.Conv2d(context_channels, hidden, 3, padding=1, bias=False),
                build_group_norm(hidden, _valid_groups(hidden, 8)),
                nn.GELU(),
            )
            self.context_value = nn.Conv2d(hidden, channels, 1, bias=False)
            self.context_channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(context_channels, channels, 1, bias=False),
            )
            spatial_channels += hidden
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(spatial_channels, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, 8)),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, 8),
        )
        initial_fraction = initial_residual_scale / 0.5
        initial_logit = torch.logit(torch.tensor(initial_fraction))
        self.residual_logit = nn.Parameter(initial_logit)
        self.last_attention_map: Tensor | None = None
        self.last_channel_gate: Tensor | None = None
        self.last_orientation_reliability: Tensor | None = None

    @property
    def residual_scale(self) -> Tensor:
        """Return the bounded scalar residual scale in ``(0, 0.5)``."""
        return 0.5 * torch.sigmoid(self.residual_logit)

    def forward(
        self,
        features: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        normalized = self.feature_norm(features)
        value = self.feature_value(normalized)
        channel_logits = self.feature_average_mlp(
            self.feature_average(normalized)
        ) + self.feature_std_mlp(
            normalized.std(dim=(-2, -1), keepdim=True, unbiased=False)
        )
        spatial_parts = [
            normalized.mean(dim=1, keepdim=True),
            normalized.std(dim=1, keepdim=True, unbiased=False),
        ]
        context = self._select_context(polar_context, features)
        if context is not None:
            context_features = self.context_features(context)
            value = value + self.context_value(context_features)
            channel_logits = channel_logits + self.context_channel(context)
            spatial_parts.append(context_features)
        channel_gate = 2.0 * torch.sigmoid(channel_logits)
        spatial_gate = 2.0 * torch.sigmoid(
            self.spatial_gate(torch.cat(spatial_parts, dim=1))
        )
        self.last_channel_gate = channel_gate.detach()
        self.last_attention_map = spatial_gate.detach()
        residual = self.output(value) * channel_gate * spatial_gate
        return features + self.residual_scale * residual

    def _select_context(
        self,
        polar_context: Mapping[str, Tensor],
        features: Tensor,
    ) -> Tensor | None:
        if self.context_mode == "feature_stats":
            self.last_orientation_reliability = None
            return None
        if self.context_mode == "dolp":
            self.last_orientation_reliability = None
            return _context_field(polar_context, "dolp", features)
        context = stack_polar_context(polar_context, features.shape[-2:])
        if context.shape[0] != features.shape[0]:
            raise ValueError("Polar context batch size must match features")
        if not self.reliability_aware:
            self.last_orientation_reliability = None
            return context
        s0_norm, dolp, cos2aop, sin2aop = context.split(1, dim=1)
        local_cos = functional.avg_pool2d(
            cos2aop, kernel_size=3, stride=1, padding=1
        )
        local_sin = functional.avg_pool2d(
            sin2aop, kernel_size=3, stride=1, padding=1
        )
        coherence = torch.sqrt(
            local_cos.square() + local_sin.square() + 1e-4
        ).clamp(0.0, 1.0)
        reliability = (dolp.clamp(0.0, 1.0) * coherence).clamp(0.0, 1.0)
        self.last_orientation_reliability = reliability.detach()
        return torch.cat(
            (
                s0_norm,
                dolp,
                reliability * cos2aop,
                reliability * sin2aop,
            ),
            dim=1,
        )


ATTENTION_TYPES = (
    "none",
    "feature_stats",
    "dolp",
    "full_stokes",
    "full_stokes_reliability",
    "feature_stats_modulation",
    "dolp_modulation",
    "full_stokes_modulation",
    "full_stokes_reliability_modulation",
)

ATTENTION_POSITIONS = (
    "post_fusion",
    "bottleneck",
    "decoder_first",
    "decoder",
)


def build_decoder_attention(
    attention_type: str,
    channels: int,
) -> nn.Module:
    """Build one decoder attention variant from its configuration name."""
    normalized = str(attention_type).strip().lower()
    if normalized == "feature_stats":
        return FeatureStatisticsAttention(channels)
    if normalized == "dolp":
        return DoLPGuidedSpatialAttention(channels)
    if normalized == "full_stokes":
        return FullStokesGuidedAttention(channels)
    if normalized == "full_stokes_reliability":
        return FullStokesGuidedAttention(
            channels, reliability_aware=True
        )
    modulation_suffix = "_modulation"
    if normalized.endswith(modulation_suffix):
        context_mode = normalized[: -len(modulation_suffix)]
        return PolarizationContextModulation(channels, context_mode)
    if normalized == "none":
        raise ValueError("'none' does not instantiate an attention module")
    raise ValueError(
        f"Unsupported global_attention_type={attention_type!r}; "
        f"expected one of {ATTENTION_TYPES}"
    )


def _context_field(
    context: Mapping[str, Tensor],
    key: str,
    features: Tensor,
) -> Tensor:
    if key not in context:
        raise KeyError(f"Polar context is missing {key!r}")
    value = context[key]
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"Polar context field {key!r} must have shape (B,1,H,W)")
    if value.shape[0] != features.shape[0]:
        raise ValueError(f"Polar context field {key!r} has the wrong batch size")
    if value.shape[-2:] != features.shape[-2:]:
        raise ValueError(
            f"Polar context field {key!r} must match feature spatial shape"
        )
    return value


__all__ = [
    "ATTENTION_POSITIONS",
    "ATTENTION_TYPES",
    "DoLPGuidedSpatialAttention",
    "FeatureStatisticsAttention",
    "FullStokesGuidedAttention",
    "PolarizationContextModulation",
    "build_decoder_attention",
]


def _valid_groups(channels: int, preferred: int) -> int:
    """Return the largest valid GroupNorm divisor no greater than preferred."""
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1
