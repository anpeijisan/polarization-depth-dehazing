"""PolarDehaze V2 model components."""

from poldehaze_v2.models.attention import (
    ATTENTION_POSITIONS,
    ATTENTION_TYPES,
    DoLPGuidedSpatialAttention,
    FeatureStatisticsAttention,
    FullStokesGuidedAttention,
    PolarizationContextModulation,
    build_decoder_attention,
)
from poldehaze_v2.models.boundary import (
    BOUNDARY_GUIDANCE_POSITIONS,
    BOUNDARY_GUIDANCE_TYPES,
    DoLPBoundaryDynamicSampler,
    DoLPGradientBoundaryGate,
    ReliableJointBoundaryGate,
    build_boundary_guidance,
)
from poldehaze_v2.models.common import ChannelAffine, replace_group_norm
from poldehaze_v2.models.decoder import (
    Bottleneck,
    DecoderUpBlock,
    DisagreementResidualRefiner,
    OutputPatchExpertRouter,
    StokesOutputPatchRouter,
)
from poldehaze_v2.models.detail import (
    DetailBlock,
    DetailBlockV2,
    PolarGuidedDetailBlock,
)
from poldehaze_v2.models.encoders import (
    DownBlock,
    LPMBlock,
    PhysEncoderBlock,
    RawEncoderBlock,
    Stem,
)
from poldehaze_v2.models.heads import (
    HeadAirlightSpatial,
    HeadBeta,
    HeadDepth,
    HeadPolar,
    HeadResidualT,
    HeadS0,
    HeadStokesResidual,
    PhysicsFeatureAdapter,
)
from poldehaze_v2.models.hard_region import (
    DHDE_GATE_TYPES,
    DHDE_POSITIONS,
    DoLPGuidedDehazeEnhancement,
)
from poldehaze_v2.models.independent_stream import IndependentRestorationStream
from poldehaze_v2.models.late_fusion import StokesLateFusion
from poldehaze_v2.models.network import PolarDehazeNet
from poldehaze_v2.models.pgde import (
    PGDE_MODES,
    PGDECueBuilder,
    PolarGuidedStokesRefiner,
    apply_stokes_confidence,
    fixed_lowpass,
)

__all__ = [
    "ATTENTION_POSITIONS",
    "ATTENTION_TYPES",
    "BOUNDARY_GUIDANCE_POSITIONS",
    "BOUNDARY_GUIDANCE_TYPES",
    "Bottleneck",
    "ChannelAffine",
    "DoLPGuidedSpatialAttention",
    "DoLPBoundaryDynamicSampler",
    "DoLPGradientBoundaryGate",
    "DecoderUpBlock",
    "DisagreementResidualRefiner",
    "OutputPatchExpertRouter",
    "StokesOutputPatchRouter",
    "DetailBlock",
    "DetailBlockV2",
    "DHDE_GATE_TYPES",
    "DHDE_POSITIONS",
    "DoLPGuidedDehazeEnhancement",
    "FeatureStatisticsAttention",
    "FullStokesGuidedAttention",
    "PolarizationContextModulation",
    "PolarGuidedDetailBlock",
    "PolarGuidedStokesRefiner",
    "PGDECueBuilder",
    "PGDE_MODES",
    "ReliableJointBoundaryGate",
    "DownBlock",
    "HeadAirlightSpatial",
    "HeadBeta",
    "HeadDepth",
    "HeadPolar",
    "HeadResidualT",
    "HeadS0",
    "HeadStokesResidual",
    "IndependentRestorationStream",
    "PhysicsFeatureAdapter",
    "LPMBlock",
    "PhysEncoderBlock",
    "PolarDehazeNet",
    "RawEncoderBlock",
    "Stem",
    "StokesLateFusion",
    "build_decoder_attention",
    "build_boundary_guidance",
    "apply_stokes_confidence",
    "fixed_lowpass",
    "replace_group_norm",
]
