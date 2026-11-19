"""Learned output-domain fusion of independently trained Stokes experts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.common import BasicResidualBlock, ConvNormAct
from poldehaze_v2.physics.stokes import safe_clip_stokes, stokes_to_angles


class StokesLateFusion(nn.Module):
    """Fuse frozen F2/F4 predictions with a smooth learned output gate.

    The zero-initialized head exactly reproduces the configured fixed blend.
    A U-shaped context network then learns component-specific spatial weights
    and a small bounded Stokes residual. This module contains no teacher
    parameters; F2/F4 remain external frozen predictors during training.
    """

    def __init__(
        self,
        *,
        raw_channels: int = 4,
        physics_channels: int = 6,
        base_channels: int = 64,
        fixed_alpha_f2: float = 0.52,
        residual_scale: Sequence[float] = (0.08, 0.05, 0.05),
        gate_smoothing: int = 9,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if not 0.0 < fixed_alpha_f2 < 1.0:
            raise ValueError("fixed_alpha_f2 must lie in (0,1)")
        if gate_smoothing <= 0 or gate_smoothing % 2 == 0:
            raise ValueError("gate_smoothing must be a positive odd integer")
        scales = tuple(float(value) for value in residual_scale)
        if len(scales) != 3 or any(value <= 0.0 for value in scales):
            raise ValueError("residual_scale must contain three positive values")
        shallow = int(base_channels)
        middle = shallow + shallow // 2
        deep = shallow * 2
        if any(width % norm_groups != 0 for width in (shallow, middle, deep)):
            raise ValueError("Late-fusion widths must be divisible by norm_groups")

        # raw + physics + F2/F4 Stokes + signed/absolute Stokes differences.
        input_channels = int(raw_channels) + int(physics_channels) + 12
        self.stem = ConvNormAct(
            input_channels,
            shallow,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.shallow = nn.Sequential(
            BasicResidualBlock(shallow, activation_name, norm_groups),
            BasicResidualBlock(shallow, activation_name, norm_groups),
        )
        self.down_middle = ConvNormAct(
            shallow,
            middle,
            stride=2,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.middle = nn.Sequential(
            BasicResidualBlock(middle, activation_name, norm_groups),
            BasicResidualBlock(middle, activation_name, norm_groups),
        )
        self.down_deep = ConvNormAct(
            middle,
            deep,
            stride=2,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.deep = nn.Sequential(
            BasicResidualBlock(deep, activation_name, norm_groups),
            BasicResidualBlock(deep, activation_name, norm_groups),
        )
        self.up_middle = ConvNormAct(
            deep,
            middle,
            kernel_size=1,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.merge_middle = ConvNormAct(
            middle * 2,
            middle,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.up_shallow = ConvNormAct(
            middle,
            shallow,
            kernel_size=1,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.merge_shallow = ConvNormAct(
            shallow * 2,
            shallow,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.refine = BasicResidualBlock(
            shallow, activation_name, norm_groups
        )
        self.output = nn.Conv2d(shallow, 6, 1, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        self.fixed_alpha_f2 = float(fixed_alpha_f2)
        self.fixed_alpha_logit = math.log(
            self.fixed_alpha_f2 / (1.0 - self.fixed_alpha_f2)
        )
        self.gate_smoothing = int(gate_smoothing)
        self.register_buffer(
            "residual_scale",
            torch.tensor(scales).view(1, 3, 1, 1),
            persistent=True,
        )

    def forward(
        self,
        raw_hazy: Tensor,
        phys_hazy: Tensor,
        f2: Mapping[str, Tensor],
        f4: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        f2_stokes = _stack_stokes(f2)
        f4_stokes = _stack_stokes(f4)
        if raw_hazy.shape[0] != f2_stokes.shape[0]:
            raise ValueError("Teacher predictions must match the input batch")
        if raw_hazy.shape[-2:] != f2_stokes.shape[-2:]:
            raise ValueError("Teacher predictions must match the input resolution")

        normalized_f2 = f2_stokes / 2.0
        normalized_f4 = f4_stokes / 2.0
        signed_difference = normalized_f2 - normalized_f4
        fusion_input = torch.cat(
            (
                raw_hazy,
                phys_hazy,
                normalized_f2,
                normalized_f4,
                signed_difference,
                signed_difference.abs(),
            ),
            dim=1,
        )
        shallow = self.shallow(self.stem(fusion_input))
        middle = self.middle(self.down_middle(shallow))
        deep = self.deep(self.down_deep(middle))
        decoded = functional.interpolate(
            self.up_middle(deep),
            size=middle.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.merge_middle(torch.cat((decoded, middle), dim=1))
        decoded = functional.interpolate(
            self.up_shallow(decoded),
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.refine(
            self.merge_shallow(torch.cat((decoded, shallow), dim=1))
        )
        raw = self.output(decoded)
        padding = self.gate_smoothing // 2
        gate_delta = functional.avg_pool2d(
            raw[:, :3],
            kernel_size=self.gate_smoothing,
            stride=1,
            padding=padding,
        )
        alpha_f2 = torch.sigmoid(self.fixed_alpha_logit + gate_delta)
        residual = (
            torch.tanh(raw[:, 3:]) * self.residual_scale.to(raw.dtype)
        )
        fixed = (
            self.fixed_alpha_f2 * f2_stokes
            + (1.0 - self.fixed_alpha_f2) * f4_stokes
        )
        mixed = alpha_f2 * f2_stokes + (1.0 - alpha_f2) * f4_stokes
        raw_stokes = mixed + residual
        s0, s1, s2 = safe_clip_stokes(
            raw_stokes[:, :1], raw_stokes[:, 1:2], raw_stokes[:, 2:3]
        )
        outputs = {
            "s0": s0,
            "s12": torch.cat((s1, s2), dim=1),
            "angles": torch.cat(stokes_to_angles(s0, s1, s2), dim=1),
            "s0_fixed": fixed[:, :1],
            "s12_fixed": fixed[:, 1:],
            "fusion_alpha": alpha_f2,
            "fusion_residual": residual,
        }
        outputs.update(_blend_physical_outputs(f2, f4, self.fixed_alpha_f2))
        return outputs


def _stack_stokes(outputs: Mapping[str, Tensor]) -> Tensor:
    s0 = outputs.get("s0")
    s12 = outputs.get("s12")
    if not isinstance(s0, Tensor) or not isinstance(s12, Tensor):
        raise KeyError("Teacher outputs must contain Tensor s0 and s12")
    if s0.ndim != 4 or s0.shape[1] != 1 or s12.shape[1] != 2:
        raise ValueError("Teacher Stokes outputs must have 1+2 BCHW channels")
    return torch.cat((s0, s12), dim=1)


def _blend_physical_outputs(
    f2: Mapping[str, Tensor],
    f4: Mapping[str, Tensor],
    alpha_f2: float,
) -> dict[str, Tensor]:
    blended: dict[str, Tensor] = {}
    for key in ("depth", "t", "beta", "airlight", "airlight_map"):
        left = f2.get(key)
        right = f4.get(key)
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            if left.shape != right.shape:
                raise ValueError(f"Teacher physical output shape mismatch for {key}")
            blended[key] = alpha_f2 * left + (1.0 - alpha_f2) * right
    return blended
