"""Beer--Lambert transmission and relative-depth conversions."""

from __future__ import annotations

import torch
from torch import Tensor


def depth_to_transmission(
    depth_rel: Tensor,
    beta: Tensor | float,
    minimum: float = 0.01,
) -> Tensor:
    """Compute ``t = exp(-beta * depth_rel)`` and clamp it to ``[minimum, 1]``."""
    beta_tensor = _broadcast_parameter(beta, depth_rel)
    return torch.exp(-beta_tensor * depth_rel).clamp(float(minimum), 1.0)


def transmission_to_depth(
    transmission: Tensor,
    beta: Tensor | float,
    transmission_eps: float = 0.01,
    beta_eps: float = 1e-4,
) -> Tensor:
    """Invert Beer--Lambert transmission as ``d = -log(t) / beta``."""
    beta_tensor = _broadcast_parameter(beta, transmission)
    beta_safe = beta_tensor.clamp_min(float(beta_eps))
    transmission_safe = transmission.clamp(float(transmission_eps), 1.0)
    return -torch.log(transmission_safe) / beta_safe


def optical_depth_to_transmission(
    optical_depth: Tensor,
    minimum: float = 0.01,
) -> Tensor:
    """Convert a spatial optical-thickness field ``tau`` to transmission."""
    return torch.exp(-optical_depth).clamp(float(minimum), 1.0)


def _broadcast_parameter(value: Tensor | float, reference: Tensor) -> Tensor:
    parameter = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if parameter.ndim == 1 and reference.ndim >= 2 and parameter.shape[0] == reference.shape[0]:
        parameter = parameter.reshape(parameter.shape[0], *([1] * (reference.ndim - 1)))
    elif (
        parameter.ndim == 2
        and reference.ndim >= 3
        and parameter.shape == (reference.shape[0], 1)
    ):
        parameter = parameter.reshape(parameter.shape[0], *([1] * (reference.ndim - 1)))
    return parameter
