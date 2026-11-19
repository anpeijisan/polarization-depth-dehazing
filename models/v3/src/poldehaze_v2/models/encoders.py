"""Independent raw/physics stream encoders and refinement registration."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
from torch import Tensor, nn

from poldehaze_v2.models.common import (
    BasicResidualBlock,
    ConvNormAct,
    build_activation,
    build_group_norm,
)

RefinementFactory = Callable[[int, str, int], nn.Module]
_RAW_REFINEMENTS: dict[str, RefinementFactory] = {
    "identity": lambda channels, activation, groups: nn.Identity(),
    "none": lambda channels, activation, groups: BasicResidualBlock(
        channels, activation, groups
    )
}


def register_raw_refinement(name: str, factory: RefinementFactory) -> None:
    """Register a Detail-family refinement under a stable config name."""
    if not name or name in _RAW_REFINEMENTS:
        raise ValueError(f"Raw refinement is already registered or invalid: {name!r}")
    _RAW_REFINEMENTS[name] = factory


def build_raw_refinement(
    name: str,
    channels: int,
    activation_name: str,
    norm_groups: int,
) -> nn.Module:
    """Build a registered refinement without substituting another algorithm."""
    if name not in _RAW_REFINEMENTS and name in {
        "legacy", "v2", "polar_guided"
    }:
        try:
            importlib.import_module("poldehaze_v2.models.detail")
        except ModuleNotFoundError as exc:
            if exc.name != "poldehaze_v2.models.detail":
                raise
    if name not in _RAW_REFINEMENTS:
        raise ValueError(
            f"Raw refinement {name!r} is not implemented/registered; "
            f"available={sorted(_RAW_REFINEMENTS)}"
        )
    return _RAW_REFINEMENTS[name](channels, activation_name, norm_groups)


def apply_raw_refinement(
    module: nn.Module,
    inputs: Tensor,
    polar_context: Tensor | None = None,
) -> Tensor:
    """Run a Detail-family module with its declared context contract.

    Context-aware refinements receive a four-channel tensor ordered as
    ``(S0_norm, DoLP, cos(2AoP), sin(2AoP))``. Legacy and generic residual
    refinements keep their original one-tensor call signature and state keys.
    """
    if bool(getattr(module, "requires_polar_context", False)):
        if polar_context is None:
            raise ValueError(
                f"{module.__class__.__name__} requires polarimetric context"
            )
        return module(inputs, polar_context)
    return module(inputs)


class Stem(nn.Module):
    """Two-layer stream-specific input stem."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels, out_channels, activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            ConvNormAct(
                out_channels, out_channels, activation_name=activation_name,
                norm_groups=norm_groups,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    """Stride-2 downsampling kept independent for each encoder stream."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class LPMBlock(nn.Module):
    """Local/low-pass/high-frequency polarization modulation block."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.local = ConvNormAct(
            channels, channels, activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.lowpass_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.lowpass_conv = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.lowpass_norm = build_group_norm(channels, norm_groups)
        self.lowpass_act = build_activation(activation_name)
        self.highfreq_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.highfreq_norm = build_group_norm(channels, norm_groups)
        self.highfreq_act = build_activation(activation_name)
        self.fuse_conv1 = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.fuse_norm1 = build_group_norm(channels, norm_groups)
        self.fuse_act = build_activation(activation_name)
        self.fuse_conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.fuse_norm2 = build_group_norm(channels, norm_groups)
        self.out_act = build_activation(activation_name)

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.local(inputs)
        smooth = self.lowpass_pool(inputs)
        lowpass = self.lowpass_act(self.lowpass_norm(self.lowpass_conv(smooth)))
        highfreq = self.highfreq_act(
            self.highfreq_norm(self.highfreq_conv(inputs - smooth))
        )
        fused = torch.cat((local, lowpass, highfreq), dim=1)
        fused = self.fuse_act(self.fuse_norm1(self.fuse_conv1(fused)))
        fused = self.fuse_norm2(self.fuse_conv2(fused))
        return inputs + self.out_act(fused)


class RawEncoderBlock(nn.Module):
    """Raw-stream block selected by the Detail ablation configuration."""

    def __init__(
        self,
        channels: int,
        detail_type: str,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = build_raw_refinement(
            detail_type, channels, activation_name, norm_groups
        )

    def forward(
        self,
        inputs: Tensor,
        polar_context: Tensor | None = None,
    ) -> Tensor:
        return apply_raw_refinement(self.block, inputs, polar_context)


class PhysEncoderBlock(nn.Module):
    """Physics-stream LPM block with a capacity-comparable fallback."""

    def __init__(
        self,
        channels: int,
        use_lpm: bool = True,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        block_type = LPMBlock if use_lpm else BasicResidualBlock
        self.block = block_type(channels, activation_name, norm_groups)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)
