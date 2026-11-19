"""Legacy and boundary-robust detail enhancement modules."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.common import (
    ConvNormAct,
    build_activation,
    build_group_norm,
)
from poldehaze_v2.models.encoders import register_raw_refinement


class DetailBlock(nn.Module):
    """Legacy directional detail block with its original zero-shift boundary."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        kwargs = {
            "activation_name": activation_name,
            "norm_groups": norm_groups,
        }
        self.base = ConvNormAct(channels, channels, **kwargs)
        self.horizontal = ConvNormAct(channels, channels, **kwargs)
        self.vertical = ConvNormAct(channels, channels, **kwargs)
        self.center = ConvNormAct(channels, channels, **kwargs)
        self.center_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.fuse_conv1 = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.fuse_norm1 = build_group_norm(channels, norm_groups)
        self.fuse_act = build_activation(activation_name)
        self.fuse_conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.fuse_norm2 = build_group_norm(channels, norm_groups)
        self.out_act = build_activation(activation_name)

    @staticmethod
    def left_shift(inputs: Tensor) -> Tensor:
        shifted = torch.zeros_like(inputs)
        shifted[..., :, :-1] = inputs[..., :, 1:]
        return shifted

    @staticmethod
    def top_shift(inputs: Tensor) -> Tensor:
        shifted = torch.zeros_like(inputs)
        shifted[..., :-1, :] = inputs[..., 1:, :]
        return shifted

    def _branch_features(
        self,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        base = self.base(inputs)
        horizontal = self.horizontal(inputs - self.left_shift(inputs))
        vertical = self.vertical(inputs - self.top_shift(inputs))
        center = self.center(inputs - self.center_pool(inputs))
        return base, horizontal, vertical, center

    def _fuse_branches(
        self,
        branches: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        fused = torch.cat(branches, dim=1)
        fused = self.fuse_act(self.fuse_norm1(self.fuse_conv1(fused)))
        return self.fuse_norm2(self.fuse_conv2(fused))

    def forward(self, inputs: Tensor) -> Tensor:
        fused = self._fuse_branches(self._branch_features(inputs))
        return inputs + self.out_act(fused)


class PolarGuidedDetailBlock(DetailBlock):
    """Polarization-guided routing over the established directional branches.

    The context tensor has shape ``(B,4,H,W)`` in the fixed order
    ``(S0_norm, DoLP, cos(2AoP), sin(2AoP))``. A local S0 gradient magnitude
    is derived internally. The router predicts bounded branch scales whose
    sum is four. Its final layer is zero-initialized, so the initial scales are
    exactly one and the module starts as :class:`DetailBlock`.
    """

    requires_polar_context = True

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__(channels, activation_name, norm_groups)
        hidden = max(channels // 8, 8)
        router_groups = min(norm_groups, hidden)
        while hidden % router_groups != 0:
            router_groups -= 1
        self.branch_router = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, router_groups),
            build_activation(activation_name),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        nn.init.zeros_(self.branch_router[-1].weight)
        nn.init.zeros_(self.branch_router[-1].bias)
        self.last_branch_weights: Tensor | None = None

    @staticmethod
    def _intensity_gradient(s0_norm: Tensor) -> Tensor:
        horizontal = functional.pad(
            s0_norm[..., 1:] - s0_norm[..., :-1],
            (0, 1, 0, 0),
            mode="replicate",
        )
        vertical = functional.pad(
            s0_norm[..., 1:, :] - s0_norm[..., :-1, :],
            (0, 0, 0, 1),
            mode="replicate",
        )
        return 0.5 * (horizontal.abs() + vertical.abs())

    @staticmethod
    def _validate_context(inputs: Tensor, polar_context: Tensor) -> None:
        if polar_context.ndim != 4 or polar_context.shape[1] != 4:
            raise ValueError(
                "polar_context must have shape (B,4,H,W) ordered as "
                "(S0_norm, DoLP, cos2AoP, sin2AoP)"
            )
        if (
            polar_context.shape[0] != inputs.shape[0]
            or polar_context.shape[-2:] != inputs.shape[-2:]
        ):
            raise ValueError(
                "polar_context must match the feature batch and spatial shape"
            )

    def forward(self, inputs: Tensor, polar_context: Tensor) -> Tensor:
        self._validate_context(inputs, polar_context)
        intensity_gradient = self._intensity_gradient(
            polar_context[:, 0:1]
        )
        routing_input = torch.cat((polar_context, intensity_gradient), dim=1)
        logits = self.branch_router(routing_input)
        logits = functional.avg_pool2d(
            logits, kernel_size=3, stride=1, padding=1
        )
        probabilities = torch.softmax(logits, dim=1)
        self.last_branch_weights = probabilities.detach()

        # Bounded scales lie in [0.5, 2.5], sum to four, and equal one for
        # zero logits. No directional branch can be fully discarded.
        scales = 0.5 + 2.0 * probabilities
        branches = self._branch_features(inputs)
        routed = tuple(
            branch * scales[:, index:index + 1]
            for index, branch in enumerate(branches)
        )
        fused = self._fuse_branches(routed)
        return inputs + self.out_act(fused)


class _ReflectDilatedBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        activation_name: str,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.padding = nn.ReflectionPad2d(dilation)
        self.conv = nn.Conv2d(
            channels,
            channels,
            3,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.norm = build_group_norm(channels, norm_groups)
        self.activation = build_activation(activation_name)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(self.padding(inputs))))


class DetailBlockV2(nn.Module):
    """Reflect-padded, multi-scale detail residual independent of image size."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.high_depthwise = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels, channels, 3, padding=0, groups=channels, bias=False
            ),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.dilation1 = _ReflectDilatedBranch(
            channels, 1, activation_name, norm_groups
        )
        self.dilation2 = _ReflectDilatedBranch(
            channels, 2, activation_name, norm_groups
        )
        self.dilation4 = _ReflectDilatedBranch(
            channels, 4, activation_name, norm_groups
        )
        self.dilation_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            build_group_norm(channels, norm_groups),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _reflect_blur(inputs: Tensor) -> Tensor:
        padded = functional.pad(inputs, (1, 1, 1, 1), mode="reflect")
        return functional.avg_pool2d(padded, kernel_size=3, stride=1)

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.local(inputs)
        high = self.high_depthwise(inputs - self._reflect_blur(inputs))
        multi = self.dilation_fuse(
            torch.cat(
                (
                    self.dilation1(inputs),
                    self.dilation2(inputs),
                    self.dilation4(inputs),
                ),
                dim=1,
            )
        )
        detail_residual = self.fuse(torch.cat((local, high, multi), dim=1))
        return inputs + self.residual_scale * detail_residual


register_raw_refinement(
    "legacy",
    lambda channels, activation, groups: DetailBlock(
        channels, activation, groups
    ),
)
register_raw_refinement(
    "v2",
    lambda channels, activation, groups: DetailBlockV2(
        channels, activation, groups
    ),
)
register_raw_refinement(
    "polar_guided",
    lambda channels, activation, groups: PolarGuidedDetailBlock(
        channels, activation, groups
    ),
)
