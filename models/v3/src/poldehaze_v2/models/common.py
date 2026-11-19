"""Shared neural-network building blocks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def build_activation(name: str, inplace: bool = False) -> nn.Module:
    """Build a supported activation from its configuration name."""
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    raise ValueError(f"Unsupported activation: {name!r}")


def build_group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    """Build GroupNorm and reject channel/group combinations that change semantics."""
    if channels <= 0 or groups <= 0 or channels % groups != 0:
        raise ValueError(
            f"GroupNorm requires channels divisible by groups, got {channels}/{groups}"
        )
    return nn.GroupNorm(groups, channels)


class ChannelAffine(nn.Module):
    """Learnable per-channel affine transform used by deployment variants.

    Unlike GroupNorm, this layer does not compute input-dependent spatial
    statistics.  Its parameter names and shapes intentionally match the
    affine parameters of :class:`torch.nn.GroupNorm`, allowing a deployment
    model to inherit the frozen model's scale and bias before Validation-only
    recovery fine-tuning.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply an affine transform to a BCHW tensor without normalization."""
        if inputs.ndim != 4 or inputs.shape[1] != self.weight.numel():
            raise ValueError(
                "ChannelAffine expects BCHW input with the configured channels"
            )
        weight = self.weight.to(dtype=inputs.dtype).view(1, -1, 1, 1)
        bias = self.bias.to(dtype=inputs.dtype).view(1, -1, 1, 1)
        return inputs * weight + bias


def replace_group_norm(
    module: nn.Module,
    *,
    include: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Replace selected GroupNorm layers with parameter-compatible affine layers.

    Args:
        module: Root module to transform in place.
        include: Optional predicate over qualified child names. When omitted,
            every GroupNorm below ``module`` is replaced.

    Returns:
        Qualified names of replaced layers in deterministic traversal order.
    """
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in tuple(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.GroupNorm) and (
                include is None or include(qualified)
            ):
                affine = ChannelAffine(child.num_channels)
                with torch.no_grad():
                    if child.affine:
                        affine.weight.copy_(child.weight)
                        affine.bias.copy_(child.bias)
                affine.train(child.training)
                parent._modules[name] = affine
                replaced.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    return tuple(replaced)


class ConvNormAct(nn.Module):
    """Convolution followed by GroupNorm and a configured activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            build_group_norm(out_channels, norm_groups),
            build_activation(activation_name),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class BasicResidualBlock(nn.Module):
    """Capacity-comparable two-convolution residual fallback."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = build_group_norm(channels, norm_groups)
        self.act1 = build_activation(activation_name)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = build_group_norm(channels, norm_groups)
        self.act2 = build_activation(activation_name)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.act1(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return self.act2(inputs + residual)
