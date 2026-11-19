"""Runtime dataset and polarization-input components."""

from poldehaze_v2.data.acquisition_pipeline import (
    angles_to_stokes,
    apply_sensor_noise,
    dolp_aolp,
    integrate_v5_layers,
    normalize_acquired_angles,
    normalized_airlight_map,
    project_linear_stokes,
    stokes_to_angles,
)
from poldehaze_v2.data.dataset import (
    AugmentationConfig,
    PolarDehazeDataset,
    remap_angle_basis,
)
from poldehaze_v2.data.io import (
    load_depth_proxy,
    load_four_angle_stack,
    load_metadata,
    load_physics_map,
    load_polar_image,
)
from poldehaze_v2.data.manifest import Manifest
from poldehaze_v2.data.polar_features import (
    compute_physics_input,
    compute_physics_input_and_stokes,
    raw_angles_to_stokes,
)
from poldehaze_v2.data.polarlitis import (
    download_polarlitis_archive,
    extract_polarlitis_archive,
    write_polarlitis_preparation_report,
)
from poldehaze_v2.data.streaming import (
    ANGLE_ORDER,
    FolderFourAngleImageSequence,
    FourAngleSource,
    NpyFourAngleSequence,
    OpenCVFourAngleCameraSource,
    RawFourAngleAviFileSequence,
    RawFourAngleAviSequence,
    RawPal8AviReader,
    discover_angle_folder_layout,
    discover_four_angle_avi_files,
)

__all__ = [
    "ANGLE_ORDER",
    "AugmentationConfig",
    "FolderFourAngleImageSequence",
    "FourAngleSource",
    "Manifest",
    "NpyFourAngleSequence",
    "OpenCVFourAngleCameraSource",
    "PolarDehazeDataset",
    "RawFourAngleAviFileSequence",
    "RawFourAngleAviSequence",
    "RawPal8AviReader",
    "angles_to_stokes",
    "apply_sensor_noise",
    "compute_physics_input",
    "compute_physics_input_and_stokes",
    "discover_angle_folder_layout",
    "discover_four_angle_avi_files",
    "dolp_aolp",
    "download_polarlitis_archive",
    "extract_polarlitis_archive",
    "integrate_v5_layers",
    "load_depth_proxy",
    "load_four_angle_stack",
    "load_metadata",
    "load_physics_map",
    "load_polar_image",
    "normalize_acquired_angles",
    "normalized_airlight_map",
    "project_linear_stokes",
    "raw_angles_to_stokes",
    "remap_angle_basis",
    "stokes_to_angles",
    "write_polarlitis_preparation_report",
]
