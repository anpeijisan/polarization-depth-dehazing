"""Differentiable conversions for linear-polarization Stokes fields."""

from __future__ import annotations

import torch
from torch import Tensor

STOKES_S0_MAX = 1.0
FP16_SAFE_EPS = 1e-4


def angles_to_stokes(
    i0: Tensor,
    i45: Tensor,
    i90: Tensor,
    i135: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert ``I0/I45/I90/I135`` into ``S0/S1/S2``.

    Inputs may have any matching leading dimensions. Angle intensities are
    expected on ``[0, 1]``; V3 requires shared scene-total normalization with ``S0`` on ``[0, 1]``.
    """
    _require_same_shape(i0, i45, i90, i135)
    return (i0 + i45 + i90 + i135) * 0.5, i0 - i90, i45 - i135


def stokes_to_angles(
    s0: Tensor,
    s1: Tensor,
    s2: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert ``S0/S1/S2`` into ``I0/I45/I90/I135`` without clipping."""
    _require_same_shape(s0, s1, s2)
    i0 = 0.5 * (s0 + s1)
    i90 = 0.5 * (s0 - s1)
    i45 = 0.5 * (s0 + s2)
    i135 = 0.5 * (s0 - s2)
    return i0, i45, i90, i135


def stokes_to_polar_features(
    s0: Tensor,
    s1: Tensor,
    s2: Tensor,
    eps: float = FP16_SAFE_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``DoLP, cos(2 AoP), sin(2 AoP)``.

    V3 derives polarization in fp32 with a denominator floor, matching
    acquisition preprocessing without inventing polarization at zero. The
    eps argument is retained for call compatibility; the V3 floor is 1e-8.
    """
    _require_same_shape(s0, s1, s2)
    s0_f, s1_f, s2_f = s0.float(), s1.float(), s2.float()
    rho = torch.linalg.vector_norm(torch.stack((s1_f, s2_f)), dim=0)
    # No additive polarization floor: zero polarization must remain zero.
    floor = 1e-8
    dolp = torch.where(s0_f > floor, rho / s0_f.clamp_min(floor), 0.0)
    return dolp.clamp(0.0, 1.0), s1_f / rho.clamp_min(floor), s2_f / rho.clamp_min(floor)


def stokes_to_dolp_aop(
    s0: Tensor,
    s1: Tensor,
    s2: Tensor,
    eps: float = FP16_SAFE_EPS,
) -> tuple[Tensor, Tensor]:
    """Derive DoLP and AoP from a full linear-Stokes field.

    DoLP is returned on ``[0, 1]``. AoP is expressed in radians on
    ``[0, pi)``. AoP is physically unreliable where DoLP approaches zero;
    evaluation code must therefore apply the configured GT-DoLP mask.
    """
    dolp, cos2aop, sin2aop = stokes_to_polar_features(
        s0, s1, s2, eps=eps
    )
    doubled_angle = torch.atan2(sin2aop, cos2aop)
    aop = torch.remainder(0.5 * doubled_angle, torch.pi)
    return dolp, aop


def normalize_s0_tensor(s0: Tensor) -> Tensor:
    """Map physical ``S0`` from ``[0, 1]`` to display range ``[0, 1]``."""
    return (s0 / STOKES_S0_MAX).clamp(0.0, 1.0)


def scale_s0_margin(value: float) -> float:
    """Convert a unit-range intensity margin to the physical S0 scale."""
    return float(value) * STOKES_S0_MAX


def safe_clip_stokes(
    s0: Tensor,
    s1: Tensor,
    s2: Tensor,
    eps: float = FP16_SAFE_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Enforce ``0≤S0≤1`` and ``sqrt(S1²+S2²)≤S0``."""
    _require_same_shape(s0, s1, s2)
    divisor = s0.float().clamp_min(STOKES_S0_MAX) / STOKES_S0_MAX
    s0_safe = (s0.float() / divisor).clamp(0.0, STOKES_S0_MAX)
    s1_safe, s2_safe = s1.float() / divisor, s2.float() / divisor
    magnitude = torch.linalg.vector_norm(torch.stack((s1_safe, s2_safe)), dim=0)
    scale = torch.minimum(torch.ones_like(s0_safe), s0_safe / magnitude.clamp_min(1e-8))
    return s0_safe, s1_safe * scale, s2_safe * scale


def _require_same_shape(*tensors: Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Stokes tensors must have identical shapes, got {sorted(shapes)}")
