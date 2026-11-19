"""Inter-stream fusion components."""

from poldehaze_v2.models.fusion.base import (
    FusionBlock,
    build_fusion,
    register_fusion,
)
from poldehaze_v2.models.fusion.channel_gate import ChannelGateFusion
from poldehaze_v2.models.fusion.cross_attention import CrossAttentionFusion
from poldehaze_v2.models.fusion.complementary_patch import (
    ComplementarityPreservingPatchFusion,
)
from poldehaze_v2.models.fusion.dual_expert import (
    CenteredDualExpertFusion,
    ChannelDynamicDualExpertFusion,
    ConsensusDisagreementFusion,
    DynamicDualExpertFusion,
    FixedRouterDualExpertFusion,
    GlobalRouterDualExpertFusion,
    StaticDualExpertFusion,
)
from poldehaze_v2.models.fusion.prgf import PRGF, PRGFPlus
from poldehaze_v2.models.fusion.polar_frequency import (
    PolarimetricFrequencySelectiveFusion,
)
from poldehaze_v2.models.fusion.patch_expert import (
    PatchRoutedPolarimetricExpertFusion,
)
from poldehaze_v2.models.fusion.output_expert import OutputExpertAnchorFusion
from poldehaze_v2.models.fusion.simple import SimpleFusion
from poldehaze_v2.models.fusion.spatial_gate import SpatialGateFusion
from poldehaze_v2.models.fusion.standard import (
    AdditiveFusion,
    FiLMFusion,
    ResidualConcatFusion,
)

__all__ = [
    "AdditiveFusion",
    "ChannelGateFusion",
    "ChannelDynamicDualExpertFusion",
    "CenteredDualExpertFusion",
    "ComplementarityPreservingPatchFusion",
    "ConsensusDisagreementFusion",
    "CrossAttentionFusion",
    "DynamicDualExpertFusion",
    "FixedRouterDualExpertFusion",
    "GlobalRouterDualExpertFusion",
    "FiLMFusion",
    "FusionBlock",
    "PRGF",
    "PRGFPlus",
    "PatchRoutedPolarimetricExpertFusion",
    "OutputExpertAnchorFusion",
    "PolarimetricFrequencySelectiveFusion",
    "ResidualConcatFusion",
    "SimpleFusion",
    "SpatialGateFusion",
    "StaticDualExpertFusion",
    "build_fusion",
    "register_fusion",
]
