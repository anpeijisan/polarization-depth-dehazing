"""Unified fusion interface and lazy implementation registry."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from torch import Tensor, nn

FusionFactory = Callable[[int, str, int], nn.Module]
_FUSION_FACTORIES: dict[str, FusionFactory] = {}


class FusionBlock(nn.Module, ABC):
    """Base interface shared by all raw/physics fusion algorithms."""

    @abstractmethod
    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        """Fuse matching ``(B,C,H,W)`` stream features."""


def register_fusion(name: str, factory: FusionFactory) -> None:
    """Register one fusion implementation under its configuration name."""
    if not name or name in _FUSION_FACTORIES:
        raise ValueError(f"Fusion is already registered or invalid: {name!r}")
    _FUSION_FACTORIES[name] = factory


def build_fusion(
    name: str,
    channels: int,
    activation_name: str = "gelu",
    norm_groups: int = 8,
) -> nn.Module:
    """Build a registered fusion without silently replacing missing variants."""
    if name not in _FUSION_FACTORIES:
        try:
            importlib.import_module(f"poldehaze_v2.models.fusion.{name}")
        except ModuleNotFoundError as exc:
            expected = f"poldehaze_v2.models.fusion.{name}"
            if exc.name != expected:
                raise
    if name not in _FUSION_FACTORIES:
        raise ValueError(
            f"Fusion {name!r} is not implemented/registered; "
            f"available={sorted(_FUSION_FACTORIES)}"
        )
    return _FUSION_FACTORIES[name](channels, activation_name, norm_groups)


def validate_fusion_inputs(raw_feat: Tensor, polar_feat: Tensor) -> None:
    """Reject shape mismatches before cross-stream arithmetic."""
    if raw_feat.ndim != 4 or polar_feat.ndim != 4:
        raise ValueError("Fusion inputs must be BCHW tensors")
    if raw_feat.shape != polar_feat.shape:
        raise ValueError(
            f"Fusion inputs must have identical shapes, got "
            f"{tuple(raw_feat.shape)} and {tuple(polar_feat.shape)}"
        )
