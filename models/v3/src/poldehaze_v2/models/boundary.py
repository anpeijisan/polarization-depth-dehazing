"""DoLP-guided boundary routing for high-resolution Stokes restoration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import functional

from poldehaze_v2.models.common import build_activation, build_group_norm

BOUNDARY_GUIDANCE_TYPES = ("none", "dolp_gradient", "joint_reliable")
BOUNDARY_GUIDANCE_POSITIONS = (
    "decoder_first",
    "decoder_final",
    "decoder",
)

PolarContext = Union[Mapping[str, Tensor], Tensor]  # noqa: UP007


class DoLPGradientBoundaryGate(nn.Module):
    """Refine features where the observed DoLP has a strong gradient.

    This deliberately simple B1 control uses a parameter-free boundary map.
    Its residual projection is zero-initialized, so enabling the module
    preserves the anchor network at initialization.
    """

    def __init__(
        self,
        channels: int,
        residual_scale: float = 0.25,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        _validate_positive(residual_scale, "residual_scale")
        self.residual_scale = float(residual_scale)
        self.refine = _boundary_residual_branch(channels, activation_name, norm_groups)
        self.last_boundary_map: Tensor | None = None
        self.last_reliability: Tensor | None = None

    def forward(self, features: Tensor, polar_context: PolarContext) -> Tensor:
        fields = _boundary_fields(polar_context, features)
        boundary = fields["dolp_edge"]
        self.last_boundary_map = boundary.detach()
        self.last_reliability = None
        return features + self.residual_scale * boundary * self.refine(features)


class ReliableJointBoundaryGate(nn.Module):
    """Route decoder refinement with joint S0/DoLP boundary confidence.

    The candidate map is the union of normalized S0 and DoLP gradients. A
    learned selector filters texture edges, while radiometric and learned
    reliability terms suppress dark, saturated, and weak-polarization noise.
    """

    def __init__(
        self,
        channels: int,
        residual_scale: float = 0.25,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        _validate_positive(residual_scale, "residual_scale")
        self.residual_scale = float(residual_scale)
        hidden = max(8, channels // 4)
        groups = _valid_groups(hidden, norm_groups)
        self.boundary_selector = _selector(hidden, groups, activation_name)
        self.reliability_selector = _selector(hidden, groups, activation_name)
        self.refine = _boundary_residual_branch(channels, activation_name, norm_groups)
        self.last_boundary_map: Tensor | None = None
        self.last_reliability: Tensor | None = None

    def forward(self, features: Tensor, polar_context: PolarContext) -> Tensor:
        fields = _boundary_fields(polar_context, features)
        evidence = _selector_evidence(fields, features)
        candidate = torch.maximum(fields["s0_edge"], fields["dolp_edge"])
        selected = candidate * torch.sigmoid(self.boundary_selector(evidence))
        learned_reliability = torch.sigmoid(self.reliability_selector(evidence))
        reliability = fields["physical_reliability"] * learned_reliability
        boundary = selected * reliability
        self.last_boundary_map = boundary.detach()
        self.last_reliability = reliability.detach()
        return features + self.residual_scale * boundary * self.refine(features)


class DoLPBoundaryDynamicSampler(nn.Module):
    """DySample-style local resampling conditioned on reliable boundaries.

    Bilinearly upsampled decoder features receive a bounded sub-pixel offset at
    the skip resolution. Offsets are applied only at reliable joint S0/DoLP
    boundaries. The zero-initialized offset head makes the initial operation an
    identity up to ``grid_sample`` numerical precision.
    """

    def __init__(
        self,
        max_offset: float = 1.0,
        activation_name: str = "gelu",
    ) -> None:
        super().__init__()
        _validate_positive(max_offset, "max_offset")
        self.max_offset = float(max_offset)
        hidden = 12
        self.evidence_body = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, 4),
            build_activation(activation_name),
        )
        self.offset_head = nn.Conv2d(hidden, 2, 3, padding=1, bias=True)
        self.reliability_head = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)
        self.last_boundary_map: Tensor | None = None
        self.last_offsets: Tensor | None = None

    def forward(
        self,
        features: Tensor,
        skip: Tensor,
        polar_context: PolarContext,
    ) -> Tensor:
        if features.ndim != 4 or skip.ndim != 4:
            raise ValueError("Dynamic sampler inputs must be BCHW tensors")
        if features.shape[0] != skip.shape[0] or features.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Dynamic sampler inputs must share batch/spatial shapes")
        fields = _boundary_fields(polar_context, features)
        evidence = torch.cat(
            (
                fields["s0_norm"],
                fields["dolp"],
                fields["s0_edge"],
                fields["dolp_edge"],
                fields["edge_disagreement"],
                features.mean(dim=1, keepdim=True),
                features.abs().mean(dim=1, keepdim=True),
                skip.mean(dim=1, keepdim=True),
                skip.abs().mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        latent = self.evidence_body(evidence)
        candidate = torch.maximum(fields["s0_edge"], fields["dolp_edge"])
        reliability = fields["physical_reliability"] * torch.sigmoid(
            self.reliability_head(latent)
        )
        boundary = candidate * reliability
        offsets = torch.tanh(self.offset_head(latent)) * self.max_offset * boundary
        self.last_boundary_map = boundary.detach()
        self.last_offsets = offsets.detach()
        return _resample_with_pixel_offsets(features, offsets)


def build_boundary_guidance(
    guidance_type: str,
    channels: int,
    residual_scale: float = 0.25,
    activation_name: str = "gelu",
    norm_groups: int = 8,
) -> nn.Module:
    """Build one boundary-routing feature module from configuration."""
    normalized = str(guidance_type).strip().lower()
    if normalized == "dolp_gradient":
        return DoLPGradientBoundaryGate(
            channels, residual_scale, activation_name, norm_groups
        )
    if normalized == "joint_reliable":
        return ReliableJointBoundaryGate(
            channels, residual_scale, activation_name, norm_groups
        )
    if normalized == "none":
        raise ValueError("'none' does not instantiate a boundary module")
    raise ValueError(
        f"Unsupported boundary_guidance_type={guidance_type!r}; "
        f"expected one of {BOUNDARY_GUIDANCE_TYPES}"
    )


def _selector(
    hidden: int,
    norm_groups: int,
    activation_name: str,
) -> nn.Sequential:
    module = nn.Sequential(
        nn.Conv2d(7, hidden, 3, padding=1, bias=False),
        build_group_norm(hidden, norm_groups),
        build_activation(activation_name),
        nn.Conv2d(hidden, 1, 1, bias=True),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


def _boundary_fields(
    polar_context: PolarContext,
    features: Tensor,
) -> dict[str, Tensor]:
    context = _stack_context(polar_context, features)
    s0_norm = context[:, 0:1].clamp(0.0, 1.0)
    dolp = context[:, 1:2].clamp(0.0, 1.0)
    s0_edge = _normalize_edge(_sobel_magnitude(s0_norm))
    dolp_edge = _normalize_edge(_sobel_magnitude(dolp))
    edge_disagreement = (s0_edge - dolp_edge).abs()
    radiometric = (4.0 * s0_norm * (1.0 - s0_norm)).clamp(0.0, 1.0)
    polar_strength = (0.1 + 0.9 * dolp).clamp(0.0, 1.0)
    physical_reliability = torch.sqrt(
        (radiometric * polar_strength).clamp_min(0.0) + 1e-6
    ).clamp(0.0, 1.0)
    return {
        "s0_norm": s0_norm,
        "dolp": dolp,
        "s0_edge": s0_edge,
        "dolp_edge": dolp_edge,
        "edge_disagreement": edge_disagreement,
        "physical_reliability": physical_reliability,
    }


def _selector_evidence(fields: Mapping[str, Tensor], features: Tensor) -> Tensor:
    return torch.cat(
        (
            fields["s0_norm"],
            fields["dolp"],
            fields["s0_edge"],
            fields["dolp_edge"],
            fields["edge_disagreement"],
            features.mean(dim=1, keepdim=True),
            features.abs().mean(dim=1, keepdim=True),
        ),
        dim=1,
    )


def _stack_context(polar_context: PolarContext, features: Tensor) -> Tensor:
    if isinstance(polar_context, Tensor):
        context = polar_context
        if context.ndim != 4 or context.shape[1] != 4:
            raise ValueError("Tensor polar_context must have shape (B,4,H,W)")
    else:
        keys = ("s0_norm", "dolp", "cos2aop", "sin2aop")
        missing = [key for key in keys if key not in polar_context]
        if missing:
            raise KeyError(f"Polar context is missing fields: {missing}")
        context = torch.cat(tuple(polar_context[key] for key in keys), dim=1)
    if context.shape[0] != features.shape[0]:
        raise ValueError("Polar context and features must share the batch size")
    if context.shape[-2:] != features.shape[-2:]:
        context = functional.interpolate(
            context,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return context.to(device=features.device, dtype=features.dtype)


def _sobel_magnitude(inputs: Tensor) -> Tensor:
    kernel_x = (
        inputs.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
            1, 1, 3, 3
        )
        / 8.0
    )
    kernel_y = kernel_x.transpose(-1, -2)
    padded = functional.pad(inputs, (1, 1, 1, 1), mode="reflect")
    grad_x = functional.conv2d(padded, kernel_x)
    grad_y = functional.conv2d(padded, kernel_y)
    return torch.sqrt((grad_x.square() + grad_y.square()).clamp_min(0.0))


def _normalize_edge(edge: Tensor) -> Tensor:
    maximum = edge.amax(dim=(2, 3), keepdim=True)
    return (edge / maximum.clamp_min(1e-4)).clamp(0.0, 1.0)


def _boundary_residual_branch(
    channels: int,
    activation_name: str,
    norm_groups: int,
) -> nn.Sequential:
    branch = nn.Sequential(
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
        nn.Conv2d(channels, channels, 1, bias=True),
    )
    nn.init.zeros_(branch[-1].weight)
    nn.init.zeros_(branch[-1].bias)
    return branch


def _resample_with_pixel_offsets(features: Tensor, offsets: Tensor) -> Tensor:
    if offsets.shape != (features.shape[0], 2, *features.shape[-2:]):
        raise ValueError("offsets must have shape (B,2,H,W)")
    batch, _, height, width = features.shape
    y = torch.arange(height, device=features.device, dtype=features.dtype) + 0.5
    x = torch.arange(width, device=features.device, dtype=features.dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base = (
        torch.stack(
            (2.0 * grid_x / width - 1.0, 2.0 * grid_y / height - 1.0),
            dim=-1,
        )
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
    )
    normalized = torch.stack(
        (2.0 * offsets[:, 0] / width, 2.0 * offsets[:, 1] / height),
        dim=-1,
    )
    return functional.grid_sample(
        features,
        base + normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _valid_groups(channels: int, preferred: int) -> int:
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_positive(value: float, name: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "BOUNDARY_GUIDANCE_POSITIONS",
    "BOUNDARY_GUIDANCE_TYPES",
    "DoLPBoundaryDynamicSampler",
    "DoLPGradientBoundaryGate",
    "ReliableJointBoundaryGate",
    "build_boundary_guidance",
]
