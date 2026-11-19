"""Unified manifest-backed dataset for PolarDehaze V2 experiments."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from poldehaze_v2.data.io import (
    load_depth_proxy,
    load_four_angle_stack,
    load_metadata,
    load_physics_map,
)
from poldehaze_v2.data.manifest import Manifest
from poldehaze_v2.data.polar_features import (
    compute_physics_input,
    raw_angles_to_stokes,
)
from poldehaze_v2.physics.stokes import STOKES_S0_MAX


@dataclass(frozen=True)
class AugmentationConfig:
    """Synchronized geometry applied to all images and physical maps."""

    enabled: bool = True
    hflip: bool = True
    vflip: bool = False
    rotation: int = 90

    @classmethod
    def from_value(cls, value: bool | Mapping[str, Any]) -> AugmentationConfig:
        if isinstance(value, bool):
            return cls(enabled=value)
        if not isinstance(value, Mapping):
            raise TypeError("augment must be a bool or mapping")
        rotation = int(value.get("rotation", 90))
        if rotation not in {0, 90}:
            raise ValueError("Only disabled or 90-degree rotation augmentation is supported")
        return cls(
            enabled=bool(value.get("enabled", True)),
            hflip=bool(value.get("hflip", True)),
            vflip=bool(value.get("vflip", False)),
            rotation=rotation,
        )


from typing import Union
SpatialSize = Union[int, tuple[int, int]]


def _normalise_spatial_size(value: int | Sequence[int]) -> SpatialSize:
    if isinstance(value, int):
        size = int(value)
        if size <= 0:
            raise ValueError("image sizes must be positive")
        return size
    shape = tuple(int(component) for component in value)
    if len(shape) != 2 or any(component <= 0 for component in shape):
        raise ValueError("rectangular image sizes must be [height, width]")
    return shape


class PolarDehazeDataset(Dataset[dict[str, Any]]):
    """Load paired four-angle images and complete synthetic physical labels.

    Args:
        manifest_path: Split-specific JSONL manifest.
        data_root: Root against which all manifest paths are resolved.
        split: ``train``, ``val``, or ``test``.
        image_sizes: Candidate square resolutions. Training samples one size per
            item; validation/test use the first size deterministically.
        phys_channels: Physics-stream input width, either 3 or 6.
        load_physics_targets: Load depth, transmission, airlight and scalar
            supervision. Disable for restoration-only training and inference.
        augment: Boolean or synchronized geometry mapping.
        input_mode: ``polar_dual_stream`` or ``photo_single_stream``.
        photo_source: In photo mode, use normalized ``s0`` or ``i0``.
        photo_in_channels: Number of repeated photo input channels.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str,
        image_sizes: Sequence[int | Sequence[int]] = (256,),
        phys_channels: int = 3,
        phys_normalize_s0: bool = True,
        augment: bool | Mapping[str, Any] = True,
        *,
        input_mode: str = "polar_dual_stream",
        photo_source: str = "s0",
        photo_in_channels: int = 1,
        load_physics_targets: bool = False,
    ) -> None:
        super().__init__()
        sizes = tuple(_normalise_spatial_size(size) for size in image_sizes)
        if not sizes:
            raise ValueError("image_sizes must not be empty")
        if phys_channels not in {3, 6}:
            raise ValueError("phys_channels must be 3 or 6")
        if input_mode not in {"polar_dual_stream", "photo_single_stream"}:
            raise ValueError(f"Unsupported input_mode: {input_mode!r}")
        if photo_source not in {"s0", "i0"}:
            raise ValueError("photo_source must be 's0' or 'i0'")
        if int(photo_in_channels) <= 0:
            raise ValueError("photo_in_channels must be positive")

        self.manifest = Manifest(manifest_path, data_root, split=split)
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.image_sizes = sizes
        self.phys_channels = int(phys_channels)
        self.phys_normalize_s0 = bool(phys_normalize_s0)
        self.augmentation = AugmentationConfig.from_value(augment)
        self.input_mode = input_mode
        self.photo_source = photo_source
        self.photo_in_channels = int(photo_in_channels)
        self.load_physics_targets = bool(load_physics_targets)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int | tuple[int, SpatialSize]) -> dict[str, Any]:
        forced_size: SpatialSize | None = None
        if isinstance(index, tuple):
            index, forced_size = index
            if forced_size not in self.image_sizes:
                raise ValueError(f"Batch-requested image size is not configured: {forced_size}")
        record = self.manifest[index]
        image_size = forced_size if forced_size is not None else self._select_image_size()
        fields = self._load_spatial_fields(record, image_size)
        fields = self._apply_synchronized_geometry(fields)

        raw_hazy = fields.pop("raw_hazy")
        raw_clear = fields.pop("raw_clear")
        for domain, angles in (("clear", raw_clear), ("hazy", raw_hazy)):
            s0 = angles.sum(dim=0) * 0.5
            residual = angles[0] + angles[2] - angles[1] - angles[3]
            magnitude = torch.linalg.vector_norm(
                torch.stack((angles[0] - angles[2], angles[1] - angles[3])), dim=0
            )
            if (s0.max() > 1.0 + 1e-6 or residual.abs().max() > 1e-6
                    or (magnitude - s0).max() > 1e-6):
                raise ValueError(f"V3 unit-Stokes contract violated: {domain}/{record['sample_id']}")
        s0_hazy, s1_hazy, s2_hazy = raw_angles_to_stokes(raw_hazy)
        s0_clear, s1_clear, s2_clear = raw_angles_to_stokes(raw_clear)
        phys_hazy = compute_physics_input(
            raw_hazy, self.phys_channels, normalize_s0=self.phys_normalize_s0
        )
        phys_clear = compute_physics_input(
            raw_clear, self.phys_channels, normalize_s0=self.phys_normalize_s0
        )

        physics_scalars: dict[str, Tensor] = {}
        if self.load_physics_targets:
            metadata = load_metadata(record, self.data_root)
            physics_scalars = self._physics_scalar_targets(metadata)

        model_raw_hazy = raw_hazy
        model_phys_hazy = phys_hazy
        if self.input_mode == "photo_single_stream":
            photo = s0_hazy / STOKES_S0_MAX if self.photo_source == "s0" else raw_hazy[0:1]
            model_raw_hazy = photo.expand(self.photo_in_channels, -1, -1).clone()
            model_phys_hazy = torch.zeros_like(phys_hazy)

        item: dict[str, Any] = {
            "raw_hazy": model_raw_hazy,
            "phys_hazy": model_phys_hazy,
            "raw_clear": raw_clear,
            "phys_clear": phys_clear,
            "s0_hazy": s0_hazy,
            "s12_hazy": torch.cat((s1_hazy, s2_hazy), dim=0),
            "s0_clear": s0_clear,
            "s12_clear": torch.cat((s1_clear, s2_clear), dim=0),
            **fields,
            **physics_scalars,
            "meta": {
                "sample_id": str(record["sample_id"]),
                "stem": str(record["stem"]),
                "split": str(record["split"]),
                "fog_level": str(record["fog_level"]),
                "input_mode": self.input_mode,
                "image_size": image_size,
                "metadata": str(record.get("metadata", "")),
            },
        }
        return item

    def _select_image_size(self) -> SpatialSize:
        if self.split == "train" and len(self.image_sizes) > 1:
            return random.choice(self.image_sizes)
        return self.image_sizes[0]

    def _load_spatial_fields(
        self,
        record: Mapping[str, Any],
        image_size: SpatialSize,
    ) -> dict[str, Tensor]:
        fields = {
            "raw_hazy": load_four_angle_stack(
                record, "hazy", self.data_root, image_size=image_size
            ),
            "raw_clear": load_four_angle_stack(
                record, "clear", self.data_root, image_size=image_size
            ),
        }
        if not self.load_physics_targets:
            return fields
        fields.update({
            "t_gt": load_physics_map(
                record, "transmission_map", self.data_root, image_size=image_size
            ),
            "d_gt": load_depth_proxy(
                record, "clear", self.data_root, image_size=image_size
            ),
            "d_hazy_gt": load_depth_proxy(
                record, "hazy", self.data_root, image_size=image_size
            ),
            "airlight_map_gt": load_physics_map(
                record,
                "airlight_map",
                self.data_root,
                image_size=image_size,
                scale=STOKES_S0_MAX,
            ),
            "density_gt": load_physics_map(
                record, "density_map", self.data_root, image_size=image_size
            ),
            "airlight_s0_gt": load_physics_map(
                record,
                "airlight_s0_map",
                self.data_root,
                image_size=image_size,
                scale=STOKES_S0_MAX,
            ),
        })
        return fields

    def _apply_synchronized_geometry(
        self,
        fields: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        config = self.augmentation
        if self.split != "train" or not config.enabled:
            return fields
        flip_h = config.hflip and random.random() < 0.5
        flip_v = config.vflip and random.random() < 0.5
        rotation_k = random.randrange(4) if config.rotation == 90 else 0
        transformed: dict[str, Tensor] = {}
        for key, tensor in fields.items():
            value = tensor
            if flip_h:
                value = torch.flip(value, dims=(-1,))
            if flip_v:
                value = torch.flip(value, dims=(-2,))
            if rotation_k:
                value = torch.rot90(value, k=rotation_k, dims=(-2, -1))
            if key in {"raw_hazy", "raw_clear"}:
                value = remap_angle_basis(
                    value,
                    horizontal_flip=flip_h,
                    vertical_flip=flip_v,
                    rotation_k=rotation_k,
                )
            transformed[key] = value
        return transformed

    @staticmethod
    def _physics_scalar_targets(metadata: Mapping[str, Any]) -> dict[str, Tensor]:
        required = (
            "extinction",
            "scattering_albedo",
            "anisotropy_g",
            "multi_scatter",
            "heterogeneity",
            "steps",
            "a_inf",
            "rho_a",
            "theta_a_deg",
            "density_mean",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise KeyError(f"Metadata is missing required physical values: {missing}")

        extinction = float(metadata["extinction"])
        density_mean = float(metadata["density_mean"])
        a_inf = float(metadata["a_inf"])
        rho_a = float(metadata["rho_a"])
        theta_a = math.radians(float(metadata["theta_a_deg"]))
        cs = rho_a * math.cos(2.0 * theta_a)
        sn = rho_a * math.sin(2.0 * theta_a)
        def scalar(value: float) -> Tensor:
            return torch.tensor([float(value)], dtype=torch.float32)

        return {
            "extinction_gt": scalar(extinction),
            "beta_effective_gt": scalar(extinction * density_mean),
            "scattering_albedo_gt": scalar(metadata["scattering_albedo"]),
            "anisotropy_g_gt": scalar(metadata["anisotropy_g"]),
            "multi_scatter_gt": scalar(metadata["multi_scatter"]),
            "heterogeneity_gt": scalar(metadata["heterogeneity"]),
            "steps_gt": scalar(metadata["steps"]),
            "a_inf_gt": scalar(a_inf),
            "rho_a_gt": scalar(rho_a),
            "theta_a_gt": scalar(theta_a),
            "airlight_params_gt": torch.tensor([a_inf, cs, sn], dtype=torch.float32),
            "airlight_stokes_gt": torch.tensor(
                [a_inf, a_inf * cs, a_inf * sn], dtype=torch.float32
            ),
        }


def remap_angle_basis(
    angle_stack: Tensor,
    *,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotation_k: int,
) -> Tensor:
    """Transform I0/I45/I90/I135 channels with the image reference frame.

    A reflection negates S2 and an odd 90-degree rotation negates both S1
    and S2. In four-angle form these are exact channel permutations.
    """
    if angle_stack.shape[-3] != 4:
        raise ValueError("Polarization basis remapping requires four angle channels")
    value = angle_stack
    if bool(horizontal_flip) ^ bool(vertical_flip):
        value = value.index_select(
            -3, torch.tensor([0, 3, 2, 1], device=value.device)
        )
    if int(rotation_k) % 2:
        value = value.index_select(
            -3, torch.tensor([2, 3, 0, 1], device=value.device)
        )
    return value

