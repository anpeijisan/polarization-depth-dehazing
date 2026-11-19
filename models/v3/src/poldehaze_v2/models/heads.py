"""Direct Stokes restoration and explicit physical-diagnosis heads."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from poldehaze_v2.models.common import build_activation, build_group_norm
from poldehaze_v2.physics.stokes import STOKES_S0_MAX


class PhysicsFeatureAdapter(nn.Module):
    """Adapt detached restoration features for the physical-diagnosis heads.

    The primary restoration feature is preserved by an exact residual path.
    A zero-initialized correction learns from the primary feature and the
    corresponding physics-stream feature, so enabling the adapter does not
    perturb an existing checkpoint before fine-tuning.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.primary_norm = build_group_norm(channels, norm_groups)
        self.guidance_norm = build_group_norm(channels, norm_groups)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.correction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
        )
        nn.init.zeros_(self.correction[-2].weight)

    def forward(self, primary: Tensor, guidance: Tensor) -> Tensor:
        if primary.shape != guidance.shape or primary.ndim != 4:
            raise ValueError(
                "Physics adapter inputs must be equal-shaped BCHW tensors"
            )
        fused = self.reduce(
            torch.cat(
                (self.primary_norm(primary), self.guidance_norm(guidance)),
                dim=1,
            )
        )
        return primary + self.correction(fused)


class HeadS0(nn.Module):
    """Predict clear S0 on the physical ``[0,1]`` scale."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
            nn.Conv2d(channels, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return STOKES_S0_MAX * self.block(inputs)


class HeadPolar(nn.Module):
    """Predict two bounded polarization direction/magnitude logits."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
            nn.Conv2d(channels, 2, 3, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class HeadStokesResidual(nn.Module):
    """Predict a signed clear-minus-hazy Stokes correction.

    The final convolution is zero-initialized, so a residual-output model
    starts from the observed hazy Stokes field.  Per-component physical
    correction ranges are applied by :class:`PolarDehazeNet` before the
    result is projected back to the valid linear-Stokes cone.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            build_group_norm(channels, norm_groups),
            build_activation(activation_name),
        )
        self.output = nn.Conv2d(channels, 3, 3, padding=1, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.tanh(self.output(self.features(inputs)))


class HeadDepth(nn.Module):
    """Predict per-pixel relative depth ``d∈[0,1]``."""

    def __init__(
        self,
        in_channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        mid_channels = max(in_channels // 2, norm_groups)
        if mid_channels % norm_groups != 0:
            mid_channels = ((mid_channels + norm_groups - 1) // norm_groups) * norm_groups
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            build_group_norm(mid_channels, norm_groups),
            build_activation(activation_name),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(self.block(inputs))


class HeadBeta(nn.Module):
    """Predict one global extinction scalar in a configured closed interval."""

    def __init__(
        self,
        in_channels: int,
        beta_min: float = 0.05,
        beta_max: float = 3.0,
        activation_name: str = "gelu",
    ) -> None:
        super().__init__()
        if not 0.0 <= beta_min < beta_max:
            raise ValueError("Expected 0 <= beta_min < beta_max")
        hidden = max(in_channels // 2, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.activation = build_activation(activation_name)
        self.fc2 = nn.Linear(hidden, 1)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def forward(self, inputs: Tensor) -> Tensor:
        pooled = self.pool(inputs).flatten(1)
        raw = self.fc2(self.activation(self.fc1(pooled)))
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(raw)


class HeadResidualT(nn.Module):
    """Predict regularized spatial correction ``r(x)`` for ``exp(-beta*d)``."""

    def __init__(
        self,
        in_channels: int,
        residual_min: float = 0.1,
        residual_max: float = 3.0,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if not 0.0 < residual_min < 1.0 < residual_max:
            raise ValueError("Residual transmission range must contain 1")
        hidden = max(in_channels // 2, norm_groups)
        if hidden % norm_groups != 0:
            hidden = ((hidden + norm_groups - 1) // norm_groups) * norm_groups
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, norm_groups),
            build_activation(activation_name),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)
        self.log_min = math.log(float(residual_min))
        self.log_max = math.log(float(residual_max))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.exp(self.block(inputs).clamp(self.log_min, self.log_max))


class HeadAirlightSpatial(nn.Module):
    """Predict spatial A0 and global ``rho*cos(2theta), rho*sin(2theta)``."""

    def __init__(
        self,
        spatial_in_channels: int,
        polar_in_channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        spatial_hidden = max(spatial_in_channels // 2, 16)
        if spatial_hidden % norm_groups != 0:
            spatial_hidden = (
                (spatial_hidden + norm_groups - 1) // norm_groups
            ) * norm_groups
        self.spatial = nn.Sequential(
            nn.Conv2d(spatial_in_channels, spatial_hidden, 3, padding=1, bias=False),
            build_group_norm(spatial_hidden, norm_groups),
            build_activation(activation_name),
            nn.Conv2d(spatial_hidden, 1, 3, padding=1, bias=True),
        )
        polar_hidden = max(polar_in_channels // 2, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(polar_in_channels, polar_hidden)
        self.activation = build_activation(activation_name)
        self.fc2 = nn.Linear(polar_hidden, 2)

    def forward(self, spatial_feat: Tensor, polar_feat: Tensor) -> dict[str, Tensor]:
        a_inf_map = torch.sigmoid(self.spatial(spatial_feat)) * STOKES_S0_MAX
        pooled = self.pool(polar_feat).flatten(1)
        polar = torch.tanh(self.fc2(self.activation(self.fc1(pooled))))
        return {
            "a_inf_map": a_inf_map,
            "polar_cs": polar[:, 0:1],
            "polar_sn": polar[:, 1:2],
        }
