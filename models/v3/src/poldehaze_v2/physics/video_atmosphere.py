"""Temporally coherent depth and density fields for video RTE synthesis.

This module does not replace the frozen still-image V5 synthesis.  It supplies
video-specific state: motion-aligned depth smoothing, an optical-path mapping,
and a slowly advected multi-scale density field.  The canonical RTE integration
and the frozen light/medium/heavy material parameters remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class VideoAtmosphereConfig:
    """Configuration for continuous-video atmosphere construction."""

    temporal_alpha: float = 0.82
    photometric_sigma: float = 0.08
    depth_smoothing_sigma: float = 1.2
    use_optical_flow: bool = True
    path_length_scale: float = 1.35
    depth_floor: float = 0.03
    depth_ceiling: float = 0.96
    density_base: float = 0.45
    density_depth_gain: float = 1.35
    density_depth_power: float = 0.80
    cloud_contrast: float = 1.65
    cloud_strength: float = 2.20
    density_min: float = 0.15
    density_max: float = 3.20
    wind_x_pixels_per_second: float = 2.20
    wind_y_pixels_per_second: float = 0.25
    low_scale_fraction: float = 0.24
    middle_scale_fraction: float = 0.075
    fine_scale_fraction: float = 0.025

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VideoAtmosphereConfig":
        """Construct and validate a configuration mapping."""
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown video-atmosphere keys: {sorted(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        """Reject configurations that violate numerical contracts."""
        if not 0.0 <= self.temporal_alpha < 1.0:
            raise ValueError("temporal_alpha must be in [0,1)")
        if self.photometric_sigma <= 0.0 or self.depth_smoothing_sigma < 0.0:
            raise ValueError(
                "photometric_sigma must be positive and smoothing non-negative"
            )
        if self.path_length_scale < 1.0:
            raise ValueError("path_length_scale must be at least 1")
        if not 0.0 <= self.depth_floor < self.depth_ceiling <= 1.0:
            raise ValueError("depth bounds must satisfy 0 <= floor < ceiling <= 1")
        if self.density_base <= 0.0 or self.density_depth_gain < 0.0:
            raise ValueError(
                "density base/gain must be non-negative with positive base"
            )
        if self.density_depth_power <= 0.0:
            raise ValueError("density_depth_power must be positive")
        if self.cloud_contrast <= 0.0 or self.cloud_strength < 0.0:
            raise ValueError(
                "cloud contrast must be positive and strength non-negative"
            )
        if not 0.0 < self.density_min < self.density_max:
            raise ValueError("density bounds must be positive and ordered")
        if (
            min(
                self.low_scale_fraction,
                self.middle_scale_fraction,
                self.fine_scale_fraction,
            )
            <= 0.0
        ):
            raise ValueError("all density scale fractions must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration snapshot."""
        return asdict(self)


def stabilize_video_depths(
    displays: Sequence[np.ndarray],
    raw_depths: Sequence[np.ndarray],
    config: VideoAtmosphereConfig,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Motion-align and temporally smooth a relative-depth video.

    Args:
        displays: S0 display frames, each ``float[H,W]`` on ``[0,1]``.
        raw_depths: Per-frame relative depth, each ``float[H,W]`` on ``[0,1]``.
        config: Frozen video atmosphere configuration.

    Returns:
        Stabilized, path-mapped depth frames and temporal diagnostics.
    """
    if len(displays) != len(raw_depths) or not displays:
        raise ValueError(
            "displays and raw_depths must be equal-length non-empty sequences"
        )
    reference_shape = np.asarray(displays[0]).shape
    if len(reference_shape) != 2:
        raise ValueError("video depth inputs must be two-dimensional")

    raw_values: list[np.ndarray] = []
    for display, depth in zip(displays, raw_depths):
        gray = np.asarray(display, dtype=np.float32)
        value = np.asarray(depth, dtype=np.float32)
        if gray.shape != reference_shape or value.shape != reference_shape:
            raise ValueError("all video frames and depth maps must share one shape")
        if not np.isfinite(gray).all() or not np.isfinite(value).all():
            raise ValueError("video depth inputs contain non-finite values")
        raw_values.append(
            gaussian_filter(
                np.clip(value, 0.0, 1.0),
                sigma=config.depth_smoothing_sigma,
                mode="reflect",
            ).astype(np.float32)
        )

    stabilized: list[np.ndarray] = [raw_values[0]]
    reliability_means: list[float] = [1.0]
    for index in range(1, len(raw_values)):
        previous_depth = stabilized[-1]
        previous_gray = np.asarray(displays[index - 1], dtype=np.float32)
        current_gray = np.asarray(displays[index], dtype=np.float32)
        if config.use_optical_flow:
            warped_depth, warped_gray = _warp_previous_to_current(
                previous_depth, previous_gray, current_gray
            )
        else:
            warped_depth, warped_gray = previous_depth, previous_gray
        photometric_error = np.abs(warped_gray - current_gray)
        reliability = np.exp(-photometric_error / config.photometric_sigma).astype(
            np.float32
        )
        alpha = config.temporal_alpha * reliability
        filtered = alpha * warped_depth + (1.0 - alpha) * raw_values[index]
        filtered = gaussian_filter(
            filtered,
            sigma=config.depth_smoothing_sigma,
            mode="reflect",
        ).astype(np.float32)
        stabilized.append(np.clip(filtered, 0.0, 1.0))
        reliability_means.append(float(reliability.mean()))

    mapped = [map_video_optical_path(value, config) for value in stabilized]
    raw_delta = _mean_adjacent_delta(raw_values)
    stable_delta = _mean_adjacent_delta(stabilized)
    report = {
        "frames": len(mapped),
        "shape": list(reference_shape),
        "raw_adjacent_depth_mae": raw_delta,
        "stabilized_adjacent_depth_mae": stable_delta,
        "temporal_reduction_fraction": (
            0.0 if raw_delta <= 1e-12 else 1.0 - stable_delta / raw_delta
        ),
        "mean_flow_photometric_reliability": float(np.mean(reliability_means)),
        "mapped_depth_min": float(min(value.min() for value in mapped)),
        "mapped_depth_mean": float(np.mean([value.mean() for value in mapped])),
        "mapped_depth_max": float(max(value.max() for value in mapped)),
    }
    return mapped, report


def map_video_optical_path(
    depth: np.ndarray, config: VideoAtmosphereConfig
) -> np.ndarray:
    """Map relative depth to a stable video optical path on ``[0,1]``."""
    value = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    mapped = 1.0 - np.power(1.0 - value, config.path_length_scale)
    return np.clip(mapped, config.depth_floor, config.depth_ceiling).astype(np.float32)


class AdvectedVideoDensityField:
    """Deterministic multi-scale density field with slow continuous advection."""

    def __init__(
        self,
        shape: tuple[int, int],
        seed: int,
        config: VideoAtmosphereConfig,
    ) -> None:
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("density field shape must be positive HxW")
        self.shape = shape
        self.config = config
        rng = np.random.default_rng(seed)
        scale = float(max(shape))
        self.low = _smooth_unit_noise(rng, shape, scale * config.low_scale_fraction)
        self.middle = _smooth_unit_noise(
            rng, shape, scale * config.middle_scale_fraction
        )
        self.fine = _smooth_unit_noise(rng, shape, scale * config.fine_scale_fraction)

    def frame(
        self,
        depth: np.ndarray,
        frame_index: int,
        fps: float,
        heterogeneity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return density and latent cloud fields for one video frame."""
        if frame_index < 0 or fps <= 0.0 or heterogeneity < 0.0:
            raise ValueError("frame index, fps, and heterogeneity are invalid")
        value = np.asarray(depth, dtype=np.float32)
        if value.shape != self.shape:
            raise ValueError("depth shape does not match the density field")
        seconds = frame_index / fps
        dx = self.config.wind_x_pixels_per_second * seconds
        dy = self.config.wind_y_pixels_per_second * seconds
        low = _bilinear_wrap_shift(self.low, 0.35 * dx, 0.35 * dy)
        middle = _bilinear_wrap_shift(self.middle, dx, dy)
        fine = _bilinear_wrap_shift(self.fine, 1.8 * dx, 1.8 * dy)
        cloud = 0.55 * low + 0.30 * middle + 0.15 * fine
        cloud = np.clip(
            0.5 + self.config.cloud_contrast * (cloud - 0.5), 0.0, 1.0
        ).astype(np.float32)

        base = self.config.density_base + self.config.density_depth_gain * np.power(
            np.clip(value, 0.0, 1.0), self.config.density_depth_power
        )
        modulation = 1.0 + heterogeneity * self.config.cloud_strength * (
            2.0 * cloud - 1.0
        )
        density = np.clip(
            base * modulation,
            self.config.density_min,
            self.config.density_max,
        )
        density = gaussian_filter(density, sigma=0.8, mode="reflect")
        return density.astype(np.float32), cloud


def _warp_previous_to_current(
    previous_depth: np.ndarray,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Optical-flow video depth stabilization requires opencv-python"
        ) from exc
    current_u8 = (np.clip(current_gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    previous_u8 = (np.clip(previous_gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    backward_flow = cv2.calcOpticalFlowFarneback(
        current_u8,
        previous_u8,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    height, width = current_u8.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + backward_flow[..., 0]
    map_y = grid_y + backward_flow[..., 1]
    warped_depth = cv2.remap(
        previous_depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    warped_gray = cv2.remap(
        previous_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return warped_depth.astype(np.float32), warped_gray.astype(np.float32)


def _smooth_unit_noise(
    rng: np.random.Generator, shape: tuple[int, int], sigma: float
) -> np.ndarray:
    values = rng.standard_normal(shape).astype(np.float32)
    return _normalize01(gaussian_filter(values, sigma=max(sigma, 0.8), mode="wrap"))


def _bilinear_wrap_shift(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    x0 = int(np.floor(dx))
    y0 = int(np.floor(dy))
    fx = float(dx - x0)
    fy = float(dy - y0)
    base = np.roll(values, shift=(y0, x0), axis=(0, 1))
    right = np.roll(values, shift=(y0, x0 + 1), axis=(0, 1))
    down = np.roll(values, shift=(y0 + 1, x0), axis=(0, 1))
    diagonal = np.roll(values, shift=(y0 + 1, x0 + 1), axis=(0, 1))
    return (
        (1.0 - fy) * ((1.0 - fx) * base + fx * right)
        + fy * ((1.0 - fx) * down + fx * diagonal)
    ).astype(np.float32)


def _normalize01(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low <= 1e-8:
        return np.zeros_like(array)
    return ((array - low) / (high - low)).astype(np.float32)


def _mean_adjacent_delta(values: Sequence[np.ndarray]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(
        np.mean(
            [
                np.mean(np.abs(current - previous))
                for previous, current in zip(values[:-1], values[1:])
            ]
        )
    )
