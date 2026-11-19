"""Stokes-consistent preprocessing and batched layered haze propagation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def project_linear_stokes(stokes: Tensor, eps: float = 1e-8) -> Tensor:
    """Project fields by full-vector scaling to the unit linear-Stokes cone."""
    if stokes.shape[-3] != 3:
        raise ValueError(f"Expected (...,3,H,W), got {tuple(stokes.shape)}")
    bounded = stokes / stokes[..., 0:1, :, :].clamp_min(1.0)
    s0 = bounded[..., 0:1, :, :].clamp(0.0, 1.0)
    s12 = bounded[..., 1:3, :, :]
    magnitude = torch.linalg.vector_norm(s12, dim=-3, keepdim=True)
    scale = torch.minimum(torch.ones_like(magnitude), s0 / magnitude.clamp_min(eps))
    return torch.cat((s0, s12 * scale), dim=-3)


def normalize_acquired_angles(raw_j: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize acquired angles by scene total intensity and reconstruct angles.

    Args:
        raw_j: ``(B,4,H,W)`` nonnegative values in the acquired numeric scale.

    Returns:
        ``(stokes, angles_i, scale_c)``. Stokes is ``(S0,S1,S2)`` with S0 on
        ``[0,1]`` and signed S1/S2 on the same scale. ``angles_i`` is the
        Stokes-consistent four-angle reconstruction. ``scale_c`` is the
        per-scene maximum of ``sum(J)/2`` with shape ``(B,1,1,1)``.
    """
    if raw_j.ndim != 4 or raw_j.shape[1] != 4:
        raise ValueError(f"Expected (B,4,H,W), got {tuple(raw_j.shape)}")
    if not torch.isfinite(raw_j).all() or bool((raw_j < 0).any()):
        raise ValueError("Acquired angle data must be finite and nonnegative")
    total = raw_j.sum(dim=1, keepdim=True) * 0.5
    scale_c = total.amax(dim=(-2, -1), keepdim=True)
    if bool((scale_c <= 0).any()):
        raise ValueError("Cannot normalize an all-black scene")
    stokes = torch.cat(
        (
            total,
            raw_j[:, 0:1] - raw_j[:, 2:3],
            raw_j[:, 1:2] - raw_j[:, 3:4],
        ),
        dim=1,
    ) / scale_c
    stokes = project_linear_stokes(stokes)
    return stokes, stokes_to_angles(stokes), scale_c


def stokes_to_angles(stokes: Tensor) -> Tensor:
    """Invert a ``(B,3,H,W)`` normalized linear-Stokes field to four angles."""
    if stokes.ndim != 4 or stokes.shape[1] != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(stokes.shape)}")
    s0, s1, s2 = stokes.split(1, dim=1)
    return torch.cat(
        ((s0 + s1) * 0.5, (s0 + s2) * 0.5,
         (s0 - s1) * 0.5, (s0 - s2) * 0.5),
        dim=1,
    )


def angles_to_stokes(angles: Tensor) -> Tensor:
    """Estimate Stokes using the least-squares total ``sum(I)/2``."""
    if angles.ndim != 4 or angles.shape[1] != 4:
        raise ValueError(f"Expected (B,4,H,W), got {tuple(angles.shape)}")
    return torch.cat(
        (
            angles.sum(dim=1, keepdim=True) * 0.5,
            angles[:, 0:1] - angles[:, 2:3],
            angles[:, 1:2] - angles[:, 3:4],
        ),
        dim=1,
    )


def dolp_aolp(stokes: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Return exact-zero-safe DoLP and pi-periodic AoLP in radians."""
    s0, s1, s2 = stokes.split(1, dim=1)
    magnitude = torch.hypot(s1, s2)
    dolp = torch.where(s0 > eps, magnitude / s0.clamp_min(eps), torch.zeros_like(s0))
    aolp = torch.remainder(0.5 * torch.atan2(s2, s1), torch.pi)
    return dolp.clamp(0.0, 1.0), aolp


def normalized_airlight_map(
    intensity: Tensor,
    depth: Tensor,
    density: Tensor,
) -> Tensor:
    """Build the V5 spatial airlight intensity directly on ``[0,1]``.

    ``intensity`` is the normalized scene airlight scalar. The original V5
    spatial modulation is retained, but the result is bounded in the same
    unit interval as normalized ``S0`` without a legacy-scale conversion.
    """
    if intensity.ndim != 4 or intensity.shape[1] != 1:
        raise ValueError("intensity must have shape (B,1,1,1)")
    if depth.ndim != 4 or depth.shape[1] != 1 or density.shape != depth.shape:
        raise ValueError("depth and density must have shape (B,1,H,W)")
    if not torch.isfinite(intensity).all() or bool(
        ((intensity < 0) | (intensity > 1)).any()
    ):
        raise ValueError("normalized airlight intensity must lie in [0,1]")
    depth_smooth = _gaussian_blur(depth, max(depth.shape[-2:]) / 130.0)
    density_smooth = _gaussian_blur(density, max(density.shape[-2:]) / 150.0)
    spatial = (0.88 + 0.22 * depth_smooth) * (0.94 + 0.12 * density_smooth)
    return (intensity * spatial).clamp(0.0, 1.0)


def _gaussian_blur(inputs: Tensor, sigma: float) -> Tensor:
    """Apply depthwise separable Gaussian blur on the current device."""
    if sigma <= 0:
        return inputs
    radius = max(1, round(4.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=inputs.device,
                               dtype=inputs.dtype)
    kernel = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel = kernel / kernel.sum()
    channels = inputs.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    mode = "reflect" if min(inputs.shape[-2:]) > radius else "replicate"
    value = F.pad(inputs, (radius, radius, 0, 0), mode=mode)
    value = F.conv2d(value, horizontal, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode=mode)
    return F.conv2d(value, vertical, groups=channels)


def _warp_depth(depth: Tensor, mode: str, gamma: float = 1.0,
                eps: float = 0.2) -> Tensor:
    value = depth.clamp(0.0, 1.0)
    if mode == "linear":
        return value
    if mode == "power":
        return value.pow(float(gamma))
    if mode == "perspective":
        clipped = value.clamp(0.0, 1.0 - float(eps) * 0.5)
        return clipped / (1.0 - clipped + float(eps))
    raise ValueError(f"Unsupported depth_warp: {mode}")


def integrate_v5_layers(
    clear_stokes: Tensor,
    depth: Tensor,
    density: Tensor,
    airlight: Tensor,
    phase: Tensor,
    level: Mapping[str, float | int],
    *,
    return_layers: bool = False,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Run batched V5-style layered Stokes propagation on CPU or CUDA.

    All inputs use the normalized S0 unit range. ``airlight`` is a physical
    ``(B,3,H,W)`` Stokes source. Source terms and every propagated state are
    projected to the valid unit linear-Stokes cone.
    """
    if clear_stokes.ndim != 4 or clear_stokes.shape[1] != 3:
        raise ValueError("clear_stokes must have shape (B,3,H,W)")
    for name, value in (("depth", depth), ("density", density), ("phase", phase)):
        if value.shape != clear_stokes[:, :1].shape:
            raise ValueError(f"{name} must have shape {tuple(clear_stokes[:, :1].shape)}")
    if airlight.shape != clear_stokes.shape:
        raise ValueError("airlight must match clear_stokes")
    steps = int(level["steps"])
    if steps <= 0:
        raise ValueError("steps must be positive")
    extinction = float(level["extinction"])
    albedo = float(level["scattering_albedo"])
    multi = float(level["multi_scatter"])
    if extinction < 0 or not 0 <= albedo <= 1 or not 0 <= multi <= 1:
        raise ValueError("Invalid extinction/albedo/multi_scatter")

    state = project_linear_stokes(clear_stokes)
    if not torch.isfinite(airlight).all() or bool((airlight[:, :1] < 0).any()):
        raise ValueError("airlight must be finite with nonnegative S0")
    air_magnitude = torch.linalg.vector_norm(airlight[:, 1:3], dim=1, keepdim=True)
    if bool((air_magnitude > airlight[:, :1] + 1e-6).any()):
        raise ValueError("airlight must lie inside the linear-Stokes cone")
    # Airlight and every source term already share the normalized S0 scale.
    # The complete source is constrained after phase and albedo are applied.
    source_air = airlight
    total_t = torch.ones_like(depth)
    history = [state.detach().clone()] if return_layers else []
    normalized_depth = (depth - depth.amin(dim=(-2, -1), keepdim=True))
    normalized_depth /= normalized_depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    effective_depth = _warp_depth(
        depth,
        str(level.get("depth_warp", "perspective")),
        float(level.get("depth_gamma", 1.0)),
        float(level.get("depth_warp_eps", 0.2)),
    ).clamp_min(0.015)
    base_sigma = max(max(clear_stokes.shape[-2:]) / 190.0, 0.8)
    for index in range(steps):
        fraction = (index + 0.5) / steps
        layer_density = density * (0.78 + 0.35 * fraction * normalized_depth)
        step_t = torch.exp(-extinction * layer_density * effective_depth / steps)
        total_t = total_t * step_t
        scattered = _gaussian_blur(state, base_sigma * (1.0 + 2.0 * fraction))
        source = (1.0 - multi) * source_air + multi * scattered
        source[:, 0:1] = (
            (1.0 - multi) * source_air[:, 0:1] * phase
            + multi * scattered[:, 0:1]
        )
        source[:, 1:3] = (
            (1.0 - multi) * source_air[:, 1:3]
            + multi * 0.25 * scattered[:, 1:3]
        )
        source = project_linear_stokes(albedo * source)
        state = project_linear_stokes(step_t * state + (1.0 - step_t) * source)
        if return_layers:
            history.append(state.detach().clone())
    return state, total_t, history


def apply_sensor_noise(angles: Tensor, config: Mapping[str, object]) -> Tensor:
    """Reserved sensor-noise boundary; only the disabled mode is frozen now."""
    if bool(config.get("enabled", False)):
        raise NotImplementedError(
            "Sensor noise is reserved but not frozen. Set noise.enabled=false."
        )
    return angles
