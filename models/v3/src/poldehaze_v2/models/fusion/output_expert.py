"""Anchor-preserving fusion that exposes complementary output evidence."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    register_fusion,
    validate_fusion_inputs,
)
from poldehaze_v2.models.fusion.prgf import PRGF
from poldehaze_v2.models.fusion.spatial_gate import SpatialGateFusion


class OutputExpertAnchorFusion(FusionBlock):
    """Keep SpatialGate on the main path and expose PRGF as evidence.

    The returned tensor is exactly the SpatialGate result, so the established
    strongest single-network fusion remains the radiometric anchor.  Both
    expert tensors remain attached to autograd and are consumed by the
    output-level patch router in :class:`PolarDehazeNet`.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.anchor_expert = SpatialGateFusion(
            channels, activation_name, norm_groups
        )
        self.complementary_expert = PRGF(
            channels, activation_name, norm_groups
        )
        self.last_anchor_feature: Tensor | None = None
        self.last_complementary_feature: Tensor | None = None

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        anchor = self.anchor_expert(raw_feat, polar_feat, polar_context)
        complementary = self.complementary_expert(
            raw_feat, polar_feat, polar_context
        )
        self.last_anchor_feature = anchor
        self.last_complementary_feature = complementary
        return anchor

    def forward_experts(
        self,
        anchor_raw_feat: Tensor,
        complementary_raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Evaluate each fusion expert on its own persistent raw state."""
        validate_fusion_inputs(anchor_raw_feat, polar_feat)
        validate_fusion_inputs(complementary_raw_feat, polar_feat)
        anchor = self.anchor_expert(
            anchor_raw_feat, polar_feat, polar_context
        )
        complementary = self.complementary_expert(
            complementary_raw_feat, polar_feat, polar_context
        )
        self.last_anchor_feature = anchor
        self.last_complementary_feature = complementary
        return anchor, complementary


register_fusion(
    "output_expert_anchor",
    lambda channels, activation, groups: OutputExpertAnchorFusion(
        channels, activation, groups
    ),
)
