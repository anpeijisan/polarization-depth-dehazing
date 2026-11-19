"""Canonical Stokes-domain atmospheric scattering forward/inverse model."""

from __future__ import annotations

from torch import Tensor


def rehaze_stokes(
    clear_s0: Tensor,
    clear_s1: Tensor,
    clear_s2: Tensor,
    transmission: Tensor,
    airlight_s0: Tensor,
    airlight_s1: Tensor,
    airlight_s2: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply ``S_h = t*S_clear + (1-t)*A`` to all three Stokes fields."""
    haze = 1.0 - transmission
    return (
        transmission * clear_s0 + haze * airlight_s0,
        transmission * clear_s1 + haze * airlight_s1,
        transmission * clear_s2 + haze * airlight_s2,
    )


def invert_asm_stokes(
    hazy_s0: Tensor,
    hazy_s1: Tensor,
    hazy_s2: Tensor,
    transmission: Tensor,
    airlight_s0: Tensor,
    airlight_s1: Tensor,
    airlight_s2: Tensor,
    t_eps: float = 0.05,
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytically invert ASM with a minimum transmission denominator."""
    transmission_safe = transmission.clamp_min(float(t_eps))
    haze = 1.0 - transmission
    return (
        (hazy_s0 - haze * airlight_s0) / transmission_safe,
        (hazy_s1 - haze * airlight_s1) / transmission_safe,
        (hazy_s2 - haze * airlight_s2) / transmission_safe,
    )

