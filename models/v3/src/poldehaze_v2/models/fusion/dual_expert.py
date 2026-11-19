"""Static and dynamic mixtures of complementary F2/F4 fusion experts."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.common import (
    BasicResidualBlock,
    ConvNormAct,
    build_activation,
    build_group_norm,
)
from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    register_fusion,
    validate_fusion_inputs,
)
from poldehaze_v2.models.fusion.prgf import PRGF
from poldehaze_v2.models.fusion.spatial_gate import (
    CONTEXT_KEYS,
    SpatialGateFusion,
    stack_polar_context,
)


class StaticDualExpertFusion(FusionBlock):
    """Average the complementary SpatialGate and PRGF feature experts."""

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.spatial_expert = SpatialGateFusion(
            channels, activation_name, norm_groups
        )
        self.prgf_expert = PRGF(channels, activation_name, norm_groups)

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        validate_fusion_inputs(raw_feat, polar_feat)
        spatial = self.spatial_expert(raw_feat, polar_feat, polar_context)
        prgf = self.prgf_expert(raw_feat, polar_feat, polar_context)
        return 0.5 * (spatial + prgf)


class DynamicDualExpertFusion(FusionBlock):
    """Blend F2/F4 experts with a bounded, smooth spatial routing map.

    The gate is conditioned on observed polarimetric context and local
    agreement between normalized raw/polar features. Its zero-initialized
    output starts as the exact static 0.5/0.5 control. The bounded gate stays
    in ``[0.25, 0.75]`` so early optimization cannot discard either expert.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.spatial_expert = SpatialGateFusion(
            channels, activation_name, norm_groups
        )
        self.prgf_expert = PRGF(channels, activation_name, norm_groups)
        self.raw_norm = build_group_norm(channels, norm_groups)
        self.polar_norm = build_group_norm(channels, norm_groups)
        hidden = max(channels // 8, 8)
        routing_channels = len(CONTEXT_KEYS) + 5
        self.router_body = nn.Sequential(
            nn.Conv2d(routing_channels, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            build_group_norm(hidden, _valid_groups(hidden, norm_groups)),
            build_activation(activation_name),
        )
        self.router_output = nn.Conv2d(hidden, 1, 1, bias=True)
        self.last_spatial_weight: Tensor | None = None
        self.trainable_spatial_weight: Tensor | None = None
        nn.init.zeros_(self.router_output.weight)
        nn.init.zeros_(self.router_output.bias)

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        spatial, prgf, raw, polar, spatial_logits = self._route_inputs(
            raw_feat, polar_feat, polar_context
        )
        spatial_weight = 0.5 + 0.25 * torch.tanh(spatial_logits)
        self.trainable_spatial_weight = spatial_weight
        self.last_spatial_weight = spatial_weight.detach()
        return spatial_weight * spatial + (1.0 - spatial_weight) * prgf

    def _route_inputs(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return expert features, normalized streams, and spatial logits."""
        validate_fusion_inputs(raw_feat, polar_feat)
        spatial = self.spatial_expert(raw_feat, polar_feat, polar_context)
        prgf = self.prgf_expert(raw_feat, polar_feat, polar_context)
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        raw = self.raw_norm(raw_feat)
        polar = self.polar_norm(polar_feat)
        routing_input = torch.cat(
            (
                context,
                raw.mean(dim=1, keepdim=True),
                raw.std(dim=1, keepdim=True, unbiased=False),
                polar.mean(dim=1, keepdim=True),
                polar.std(dim=1, keepdim=True, unbiased=False),
                functional.cosine_similarity(
                    raw, polar, dim=1, eps=1e-4
                ).unsqueeze(1),
            ),
            dim=1,
        )
        logits = self.router_output(self.router_body(routing_input))
        logits = functional.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
        return spatial, prgf, raw, polar, logits


class FixedRouterDualExpertFusion(DynamicDualExpertFusion):
    """Parameter-matched control with the routing map fixed at 0.5.

    The complete dynamic router is retained and evaluated, so this control has
    exactly the same trainable parameter inventory and routing computation as
    :class:`DynamicDualExpertFusion`.  Multiplying its logits by zero prevents
    the router from changing the equal branch allocation.  Consequently, the
    comparison with ``dual_expert_dynamic`` isolates learned spatial weight
    allocation rather than router parameter count.
    """

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        spatial, prgf, _, _, spatial_logits = self._route_inputs(
            raw_feat, polar_feat, polar_context
        )
        raw_spatial_weight = 0.5 + 0.25 * torch.tanh(spatial_logits)
        fixed_weight = 0.5 + 0.0 * spatial_logits
        self.trainable_spatial_weight = raw_spatial_weight
        self.last_spatial_weight = raw_spatial_weight.detach()
        return fixed_weight * spatial + (1.0 - fixed_weight) * prgf


class GlobalRouterDualExpertFusion(DynamicDualExpertFusion):
    """Use one image-level expert weight per fusion level.

    This parameter-matched control computes the exact spatial routing map used
    by :class:`DynamicDualExpertFusion`, then averages over ``H`` and ``W``
    before mixing the same SpatialGate and PRGF expert outputs.  Consequently,
    its only functional difference from the dynamic RPAF route is whether the
    spatial dimensions of the routing map are retained.
    """

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        spatial, prgf, _, _, spatial_logits = self._route_inputs(
            raw_feat, polar_feat, polar_context
        )
        spatial_weight = 0.5 + 0.25 * torch.tanh(spatial_logits)
        global_weight = spatial_weight.mean(
            dim=(-2, -1), keepdim=True
        )
        self.trainable_spatial_weight = global_weight
        self.last_spatial_weight = global_weight.detach()
        return global_weight * spatial + (1.0 - global_weight) * prgf


class CenteredDualExpertFusion(DynamicDualExpertFusion):
    """Route only zero-mean spatial deviations around an equal expert mixture.

    For expert features ``E_s`` and ``E_p``, the router first computes the same
    bounded map as :class:`DynamicDualExpertFusion`,
    ``G = 0.5 + 0.25*tanh(logits)``. It then removes each image's spatial mean
    and applies only the centered deviation::

        G_c = G - mean_HW(G)
        Y = 0.5 * (E_s + E_p) + G_c * (E_s - E_p)

    No gain, projection, channel gate, or other learnable parameter is added.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__(channels, activation_name, norm_groups)
        self.last_centered_weight: Tensor | None = None

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        spatial, prgf, _, _, spatial_logits = self._route_inputs(
            raw_feat, polar_feat, polar_context
        )
        raw_spatial_weight = 0.5 + 0.25 * torch.tanh(spatial_logits)
        centered_weight = raw_spatial_weight - raw_spatial_weight.mean(
            dim=(-2, -1), keepdim=True
        )
        self.trainable_spatial_weight = raw_spatial_weight
        self.last_spatial_weight = raw_spatial_weight.detach()
        self.last_centered_weight = centered_weight.detach()
        return 0.5 * (spatial + prgf) + centered_weight * (spatial - prgf)


class ChannelDynamicDualExpertFusion(DynamicDualExpertFusion):
    """Extend spatial expert routing with zero-initialized channel routing.

    The learned N1 spatial logits remain intact when its checkpoint is loaded.
    The new channel logits start at zero, making the initial output exactly N1
    while allowing different feature channels to select complementary experts.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__(channels, activation_name, norm_groups)
        hidden = max(channels // 4, 8)
        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            build_activation(activation_name),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.channel_router[-1].weight)
        nn.init.zeros_(self.channel_router[-1].bias)

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        spatial, prgf, raw, polar, spatial_logits = self._route_inputs(
            raw_feat, polar_feat, polar_context
        )
        channel_input = torch.cat((raw, polar, (raw - polar).abs()), dim=1)
        channel_logits = self.channel_router(channel_input)
        expert_weight = 0.5 + 0.25 * torch.tanh(
            spatial_logits + channel_logits
        )
        self.trainable_spatial_weight = expert_weight
        self.last_spatial_weight = expert_weight.detach()
        return expert_weight * spatial + (1.0 - expert_weight) * prgf


class ConsensusDisagreementFusion(DynamicDualExpertFusion):
    """Preserve nonlinear disagreement discarded by convex expert routing.

    The returned consensus is exactly the established N1 dynamic mixture.
    A separate feature records signed difference, absolute difference, and
    normalized expert interaction for the downstream residual decoder.
    """

    def __init__(
        self,
        channels: int,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__(channels, activation_name, norm_groups)
        self.spatial_output_norm = build_group_norm(channels, norm_groups)
        self.prgf_output_norm = build_group_norm(channels, norm_groups)
        disagreement_channels = channels * 3 + len(CONTEXT_KEYS)
        self.disagreement_encoder = nn.Sequential(
            ConvNormAct(
                disagreement_channels,
                channels,
                kernel_size=1,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            BasicResidualBlock(channels, activation_name, norm_groups),
        )
        self.last_disagreement_feature: Tensor | None = None

    def forward(
        self,
        raw_feat: Tensor,
        polar_feat: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        spatial, prgf, _, _, spatial_logits = self._route_inputs(
            raw_feat, polar_feat, polar_context
        )
        spatial_weight = 0.5 + 0.25 * torch.tanh(spatial_logits)
        self.trainable_spatial_weight = spatial_weight
        self.last_spatial_weight = spatial_weight.detach()
        consensus = spatial_weight * spatial + (1.0 - spatial_weight) * prgf

        spatial_normalized = self.spatial_output_norm(spatial)
        prgf_normalized = self.prgf_output_norm(prgf)
        context = stack_polar_context(polar_context, raw_feat.shape[-2:])
        disagreement_input = torch.cat(
            (
                spatial_normalized - prgf_normalized,
                (spatial_normalized - prgf_normalized).abs(),
                torch.tanh(spatial_normalized * prgf_normalized),
                context,
            ),
            dim=1,
        )
        self.last_disagreement_feature = self.disagreement_encoder(
            disagreement_input
        )
        return consensus


def _valid_groups(channels: int, preferred: int) -> int:
    groups = min(int(preferred), int(channels))
    while channels % groups != 0:
        groups -= 1
    return groups


register_fusion(
    "dual_expert_static",
    lambda channels, activation, groups: StaticDualExpertFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "dual_expert_dynamic",
    lambda channels, activation, groups: DynamicDualExpertFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "dual_expert_fixed_router",
    lambda channels, activation, groups: FixedRouterDualExpertFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "dual_expert_global_router",
    lambda channels, activation, groups: GlobalRouterDualExpertFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "dual_expert_centered",
    lambda channels, activation, groups: CenteredDualExpertFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "dual_expert_channel_dynamic",
    lambda channels, activation, groups: ChannelDynamicDualExpertFusion(
        channels, activation, groups
    ),
)
register_fusion(
    "dual_expert_consensus_disagreement",
    lambda channels, activation, groups: ConsensusDisagreementFusion(
        channels, activation, groups
    ),
)
