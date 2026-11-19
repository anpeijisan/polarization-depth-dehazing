"""Airlight parameterization and approximate observation-based estimation."""

from __future__ import annotations

import torch
from torch import Tensor

def estimate_airlight_from_stokes(
    s0_hazy: Tensor,
    s1_hazy: Tensor,
    s2_hazy: Tensor,
    depth_rel: Tensor,
    top_frac: float = 0.001,
) -> dict[str, Tensor]:
    """Estimate global airlight from the brightest distant pixels.

    Args:
        s0_hazy: Hazy S0 field shaped ``(B,1,H,W)`` on ``[0,2]``.
        s1_hazy: Hazy S1 field with the same shape.
        s2_hazy: Hazy S2 field with the same shape.
        depth_rel: Relative depth shaped ``(B,1,H,W)`` on ``[0,1]``.
        top_frac: Fraction of far/bright pixels used per image.

    Returns:
        ``A``, ``rho_a``, ``theta_a``, ``cs``, and ``sn`` tensors, each
        shaped ``(B,1,1,1)``. ``theta_a`` is expressed in radians.
    """
    if not 0.0 < float(top_frac) <= 1.0:
        raise ValueError("top_frac must lie in (0, 1]")
    shapes = {tuple(value.shape) for value in (s0_hazy, s1_hazy, s2_hazy, depth_rel)}
    if len(shapes) != 1 or s0_hazy.ndim != 4 or s0_hazy.shape[1] != 1:
        raise ValueError(f"Expected matching (B,1,H,W) fields, got {sorted(shapes)}")

    batch = s0_hazy.shape[0]
    score = s0_hazy * (0.5 + 0.5 * depth_rel)
    score_flat = score.reshape(batch, -1)
    k = max(1, int(score_flat.shape[1] * float(top_frac)))
    indices = score_flat.topk(k, dim=1).indices

    top_s0 = torch.gather(s0_hazy.reshape(batch, -1), 1, indices)
    a0 = top_s0.mean(dim=1).reshape(batch, 1, 1, 1).clamp(0.0, 2.0)

    weight = (depth_rel * (s0_hazy / (a0 + 1e-4))).reshape(batch, -1)
    weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-6)
    s1_mean = (weight * s1_hazy.reshape(batch, -1)).sum(dim=1).reshape(batch, 1, 1, 1)
    s2_mean = (weight * s2_hazy.reshape(batch, -1)).sum(dim=1).reshape(batch, 1, 1, 1)
    rho_a = (
        torch.sqrt(s1_mean.square() + s2_mean.square()) / (a0 + 1e-4)
    ).clamp(0.0, 0.5)
    theta_a = 0.5 * torch.atan2(s2_mean, s1_mean)
    cs = rho_a * torch.cos(2.0 * theta_a)
    sn = rho_a * torch.sin(2.0 * theta_a)
    return {"A": a0, "rho_a": rho_a, "theta_a": theta_a, "cs": cs, "sn": sn}


def airlight_stokes_vector(
    a0: Tensor,
    cs: Tensor,
    sn: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(A0, A0*cs, A0*sn)`` with broadcast-compatible shapes."""
    return a0, a0 * cs, a0 * sn
