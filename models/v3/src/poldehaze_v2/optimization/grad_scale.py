"""Forward-transparent continuous gradient scaling."""

from __future__ import annotations

import torch
from torch import Tensor


class GradientScale(torch.autograd.Function):
    """Identity forward with ``gamma``-scaled backward gradient."""

    @staticmethod
    def forward(ctx: object, inputs: Tensor, gamma: float) -> Tensor:
        value = float(gamma)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Gradient gamma must lie in [0,1], got {value}")
        ctx.gamma = value
        return inputs

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output * ctx.gamma, None


def grad_scale(inputs: Tensor, gamma: float) -> Tensor:
    """Apply :class:`GradientScale` through a functional interface."""
    return GradientScale.apply(inputs, float(gamma))

