"""Image, map, depth, and metadata loading for relative-path manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional

ANGLE_KEYS = ("0", "45", "90", "135")


def resolve_data_path(data_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a relative path without allowing traversal outside ``data_root``."""
    root = Path(data_root).expanduser().resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"Expected a data-root-relative path, got {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes data root: {relative}") from exc
    return resolved


def load_polar_image(
    path: str | Path,
    image_size: int | tuple[int, int] | None = None,
) -> Tensor:
    """Load one grayscale PNG as ``(1,H,W)`` float32 on ``[0,1]``."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    with Image.open(image_path) as image:
        if image.mode == "L":
            maximum = 255.0
        elif image.mode in {"I;16", "I;16L", "I;16B"} or (
            image.mode == "I" and image.format == "PNG"
        ):
            maximum = 65535.0
        else:
            raise ValueError(f"Expected grayscale 8/16-bit data, got {image.mode}: {path}")
        array = np.array(image, dtype=np.float32, copy=True)
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > maximum:
            raise ValueError(f"Invalid encoded intensity range: {path}")
        array /= maximum
    tensor = torch.from_numpy(array).unsqueeze(0)
    return resize_tensor(tensor, image_size) if image_size is not None else tensor


def resize_tensor(tensor: Tensor, image_size: int | tuple[int, int]) -> Tensor:
    """Resize a ``(C,H,W)`` tensor with bilinear interpolation."""
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"Invalid image size: {image_size}")
    return functional.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)


def load_four_angle_stack(
    record: Mapping[str, Any],
    domain: str,
    data_root: str | Path,
    image_size: int | tuple[int, int] | None = None,
) -> Tensor:
    """Load ``0/45/90/135`` images and stack them as ``(4,H,W)``."""
    if domain not in {"clear", "hazy"}:
        raise ValueError("domain must be 'clear' or 'hazy'")
    payload = record.get(domain)
    if not isinstance(payload, Mapping):
        raise KeyError(f"Manifest record has no mapping for {domain!r}")
    if set(map(str, payload.keys())) != set(ANGLE_KEYS):
        raise ValueError(f"{domain!r} angle mapping must contain exactly {ANGLE_KEYS}")
    images = [
        load_polar_image(
            resolve_data_path(data_root, str(payload[angle])), image_size=image_size
        )
        for angle in ANGLE_KEYS
    ]
    return torch.cat(images, dim=0)


def load_metadata(record: Mapping[str, Any], data_root: str | Path) -> dict[str, Any]:
    """Load the synthesis metadata JSON referenced by a manifest record."""
    path = resolve_data_path(data_root, str(record["metadata"]))
    if not path.is_file():
        raise FileNotFoundError(f"Metadata does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Metadata root must be a mapping: {path}")
    return payload


def load_depth_proxy(
    record: Mapping[str, Any],
    domain: str,
    data_root: str | Path,
    image_size: int | tuple[int, int] | None = None,
) -> Tensor:
    """Load clear or hazy depth proxy as ``(1,H,W)`` on ``[0,1]``."""
    key = {"clear": "depth_clear", "hazy": "depth_hazy"}.get(domain)
    if key is None:
        raise ValueError("domain must be 'clear' or 'hazy'")
    return load_polar_image(
        resolve_data_path(data_root, str(record[key])), image_size=image_size
    )


def load_physics_map(
    record: Mapping[str, Any],
    key: str,
    data_root: str | Path,
    image_size: int | tuple[int, int] | None = None,
    scale: float = 1.0,
) -> Tensor:
    """Load one manifest-referenced physical map with an optional scale."""
    if key not in record:
        raise KeyError(f"Manifest record has no physics map {key!r}")
    tensor = load_polar_image(
        resolve_data_path(data_root, str(record[key])), image_size=image_size
    )
    return tensor * float(scale)
