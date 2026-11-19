"""Model-ready Stokes and polarization feature construction."""

from __future__ import annotations

import torch
from torch import Tensor

from poldehaze_v2.physics.stokes import (
    STOKES_S0_MAX,
    angles_to_stokes,
    stokes_to_polar_features,
)


def raw_angles_to_stokes(raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a ``(...,4,H,W)`` raw stack while preserving a channel axis."""
    if raw.ndim < 3 or raw.shape[-3] != 4:
        raise ValueError(f"Expected (...,4,H,W), got {tuple(raw.shape)}")
    i0 = raw.narrow(-3, 0, 1)
    i45 = raw.narrow(-3, 1, 1)
    i90 = raw.narrow(-3, 2, 1)
    i135 = raw.narrow(-3, 3, 1)
    return angles_to_stokes(i0, i45, i90, i135)


def compute_physics_input(
    raw_hazy: Tensor,
    phys_channels: int = 3,
    *,
    normalize_s0: bool = True,
) -> Tensor:
    """Compute the physics-stream input from a raw hazy angle stack.

    ``phys_channels=3`` returns ``[DoLP, cos2AoP, sin2AoP]``.
    ``phys_channels=6`` returns
    ``[S0, S1, S2, DoLP, cos2AoP, sin2AoP]`` when ``normalize_s0=False``
    (the V3 convention); normalization divides by the V3 maximum of one.
    """
    physics, _ = compute_physics_input_and_stokes(
        raw_hazy,
        phys_channels,
        normalize_s0=normalize_s0,
    )
    return physics


def compute_physics_input_and_stokes(
    raw_hazy: Tensor,
    phys_channels: int = 3,
    *,
    normalize_s0: bool = True,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """Build physics features and expose their already-computed Stokes field.

    This deployment-oriented form avoids repeating the deterministic
    four-angle-to-Stokes conversion at the model boundary. Device, dtype,
    batch dimensions, and numerical conventions are preserved.
    """
    if phys_channels not in {3, 6}:
        raise ValueError("phys_channels must be 3 or 6")
    s0, s1, s2 = raw_angles_to_stokes(raw_hazy)
    dolp, cos2aop, sin2aop = stokes_to_polar_features(s0, s1, s2)
    if phys_channels == 3:
        physics = torch.cat((dolp, cos2aop, sin2aop), dim=-3)
        return physics, (s0, s1, s2)
    s0_input = s0 / STOKES_S0_MAX if normalize_s0 else s0
    physics = torch.cat((s0_input, s1, s2, dolp, cos2aop, sin2aop), dim=-3)
    return physics, (s0, s1, s2)
