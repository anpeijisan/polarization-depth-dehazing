"""Canonical polarimetric imaging physics."""

from poldehaze_v2.physics.airlight import (
    airlight_stokes_vector,
    estimate_airlight_from_stokes,
)
from poldehaze_v2.physics.rehaze import invert_asm_stokes, rehaze_stokes
from poldehaze_v2.physics.stokes import (
    FP16_SAFE_EPS,
    STOKES_S0_MAX,
    angles_to_stokes,
    normalize_s0_tensor,
    safe_clip_stokes,
    stokes_to_angles,
    stokes_to_dolp_aop,
    stokes_to_polar_features,
)
from poldehaze_v2.physics.transmission import (
    depth_to_transmission,
    optical_depth_to_transmission,
    transmission_to_depth,
)
from poldehaze_v2.physics.video_atmosphere import (
    AdvectedVideoDensityField,
    VideoAtmosphereConfig,
    map_video_optical_path,
    stabilize_video_depths,
)

__all__ = [
    "FP16_SAFE_EPS",
    "STOKES_S0_MAX",
    "AdvectedVideoDensityField",
    "VideoAtmosphereConfig",
    "airlight_stokes_vector",
    "angles_to_stokes",
    "depth_to_transmission",
    "estimate_airlight_from_stokes",
    "invert_asm_stokes",
    "map_video_optical_path",
    "normalize_s0_tensor",
    "optical_depth_to_transmission",
    "rehaze_stokes",
    "safe_clip_stokes",
    "stokes_to_angles",
    "stokes_to_dolp_aop",
    "stokes_to_polar_features",
    "stabilize_video_depths",
    "transmission_to_depth",
]
