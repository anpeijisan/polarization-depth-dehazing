"""DoLP-guided feature refinement for difficult dehazing regions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.common import ConvNormAct, build_activation


DHDE_POSITIONS = (
    "post_fusion",
    "bottleneck",
    "decoder_first",
    "decoder_final",
    "decoder",
)
DHDE_GATE_TYPES = ("legacy", "residual_supervised")


class DoLPGuidedDehazeEnhancement(nn.Module):
    """Refine restoration features with DoLP-guided difficult-region attention.

    The module consumes an existing feature tensor and observed polarization
    context.  DoLP is used as a material-boundary cue, while feature statistics
    and normalized S0 let the learned attention depart from literal edges and
    cover any locally difficult haze region.  The module returns only a refined
    feature tensor; it does not introduce an additional image or residual head.

    Args:
        channels: Number of feature channels.
        residual_scale: Maximum scale of the gated feature correction.
        activation: Activation used by convolutional projections.
        norm_groups: Maximum number of GroupNorm groups.
    """

    def __init__(
        self,
        channels: int,
        residual_scale: float = 0.25,
        activation: str = "gelu",
        norm_groups: int = 8,
        gate_type: str = "legacy",
        gate_initial_probability: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        normalized_gate_type = str(gate_type).strip().lower()
        if normalized_gate_type not in DHDE_GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {DHDE_GATE_TYPES}, got {gate_type!r}"
            )
        if not 0.0 < float(gate_initial_probability) < 1.0:
            raise ValueError("gate_initial_probability must lie in (0,1)")
        hidden = max(8, channels // 4)
        groups = min(norm_groups, hidden)
        while hidden % groups != 0:
            groups -= 1

        self.residual_scale = float(residual_scale)
        self.gate_type = normalized_gate_type
        self.register_buffer(
            "sobel_kernel_x",
            torch.tensor(
                ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.polar_projection = ConvNormAct(
            3,
            hidden,
            kernel_size=3,
            activation_name=activation,
            norm_groups=groups,
        )
        attention_channels = 5 if self.gate_type == "legacy" else 8
        self.attention = nn.Sequential(
            nn.Conv2d(attention_channels, hidden, kernel_size=3, padding=1),
            build_activation(activation, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.feature_projection = ConvNormAct(
            channels + hidden,
            channels,
            kernel_size=1,
            activation_name=activation,
            norm_groups=norm_groups,
        )
        self.contextual_refinement = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, channels),
            build_activation(activation, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.contextual_refinement[-1].weight)
        nn.init.zeros_(self.contextual_refinement[-1].bias)
        if self.gate_type == "residual_supervised":
            gate_logit = torch.logit(
                torch.tensor(float(gate_initial_probability))
            ).item()
            nn.init.zeros_(self.attention[-2].weight)
            nn.init.constant_(self.attention[-2].bias, gate_logit)
        self.last_attention: Tensor | None = None
        self.training_attention: Tensor | None = None

    def forward(
        self,
        features: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        """Return a shape-preserving residual refinement of ``features``."""
        if features.ndim != 4:
            raise ValueError("features must be a BCHW tensor")
        for key in ("s0_norm", "dolp"):
            if key not in polar_context:
                raise KeyError(f"polar_context is missing {key!r}")
        s0_norm = self._prepare_context(
            polar_context["s0_norm"], features
        ).clamp(0.0, 1.0)
        dolp = self._prepare_context(
            polar_context["dolp"], features
        ).clamp(0.0, 1.0)
        dolp_edge = _normalized_sobel_magnitude(dolp, self.sobel_kernel_x)
        polar_features = self.polar_projection(
            torch.cat((s0_norm, dolp, dolp_edge), dim=1)
        )
        attention_inputs = [
            features.mean(dim=1, keepdim=True),
            features.amax(dim=1, keepdim=True),
            s0_norm,
            dolp,
            dolp_edge,
        ]
        if self.gate_type == "residual_supervised":
            s0_edge = _normalized_sobel_magnitude(s0_norm, self.sobel_kernel_x)
            local_mean = functional.avg_pool2d(
                s0_norm, kernel_size=5, stride=1, padding=2
            )
            local_second_moment = functional.avg_pool2d(
                s0_norm.square(), kernel_size=5, stride=1, padding=2
            )
            local_std = (
                local_second_moment - local_mean.square()
            ).clamp_min(0.0).sqrt()
            attention_inputs.extend(
                (s0_edge, (s0_edge - dolp_edge).abs(), local_std)
            )
        attention = self.attention(torch.cat(attention_inputs, dim=1))
        correction = self.contextual_refinement(
            self.feature_projection(torch.cat((features, polar_features), dim=1))
        )
        self.last_attention = attention.detach()
        self.training_attention = attention if self.training else None
        return features + self.residual_scale * attention * correction

    @staticmethod
    def _prepare_context(value: Tensor, features: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError("DHDE context values must have shape Bx1xHxW")
        if value.shape[0] != features.shape[0]:
            raise ValueError("DHDE features and context must share batch size")
        if value.shape[-2:] != features.shape[-2:]:
            value = functional.interpolate(
                value,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return value.to(dtype=features.dtype)


def _normalized_sobel_magnitude(inputs: Tensor, kernel_x: Tensor) -> Tensor:
    """Return a per-image normalized, softly dilated Sobel magnitude."""
    if kernel_x.dtype != inputs.dtype:
        kernel_x = kernel_x.to(dtype=inputs.dtype)
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = functional.conv2d(inputs, kernel_x, padding=1)
    grad_y = functional.conv2d(inputs, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
    maximum = magnitude.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    normalized = magnitude / maximum.clamp_min(1e-4)
    return functional.max_pool2d(normalized, kernel_size=3, stride=1, padding=1)


__all__ = [
    "DHDE_GATE_TYPES",
    "DHDE_POSITIONS",
    "DoLPGuidedDehazeEnhancement",
]
