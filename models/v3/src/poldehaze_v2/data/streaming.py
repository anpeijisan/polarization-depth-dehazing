"""Synchronized four-angle sources for real-time deployment.

Every source returns one ``float32[4,H,W]`` frame ordered as
``(I0, I45, I90, I135)`` on the numeric range ``[0,1]``.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np

ANGLE_ORDER = (0, 45, 90, 135)


class FourAngleSource(Protocol):
    """Minimal synchronized polarization-source interface."""

    @property
    def frame_count(self) -> int | None:
        """Return the finite frame count, or ``None`` for a live source."""

    @property
    def fps(self) -> float | None:
        """Return the nominal capture FPS when known."""

    @property
    def label(self) -> str:
        """Return a short human-readable source label."""

    def read(self) -> np.ndarray:
        """Read the next synchronized four-angle frame."""

    def close(self) -> None:
        """Release source resources."""


class NpyFourAngleSequence:
    """Memory-map a finite ``[T,4,H,W]`` four-angle sequence."""

    def __init__(
        self, path: str | Path, *, fps: float = 22.21, loop: bool = True
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._array = np.load(self.path, mmap_mode="r")
        if self._array.ndim != 4 or self._array.shape[1] != len(ANGLE_ORDER):
            raise ValueError(
                f"Expected [T,4,H,W] sequence, received {self._array.shape}"
            )
        if self._array.shape[0] <= 0 or min(self._array.shape[-2:]) <= 0:
            raise ValueError("The four-angle sequence must contain non-empty frames")
        if self._array.dtype not in (np.uint8, np.float16, np.float32, np.float64):
            raise TypeError(f"Unsupported sequence dtype: {self._array.dtype}")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self._fps = float(fps)
        self.loop = bool(loop)
        self.index = 0

    @property
    def frame_count(self) -> int:
        return int(self._array.shape[0])

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def label(self) -> str:
        return self.path.name

    def read(self) -> np.ndarray:
        if self.index >= self.frame_count:
            if not self.loop:
                raise EOFError(f"Sequence is exhausted: {self.path}")
            self.index = 0
        frame = np.asarray(self._array[self.index])
        self.index += 1
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0
        result = frame.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError(f"Sequence contains non-finite values: {self.path}")
        minimum, maximum = float(result.min()), float(result.max())
        if minimum < -1e-4 or maximum > 1.0001:
            raise ValueError(
                f"Floating sequence range must be [0,1], received [{minimum},{maximum}]"
            )
        return np.clip(result, 0.0, 1.0)

    def close(self) -> None:
        mmap = getattr(self._array, "_mmap", None)
        if mmap is not None:
            mmap.close()


class RawPal8AviReader:
    """Read grayscale indices from a fixed-size uncompressed pal8 AVI.

    The acquisition files contain valid 8-bit indices but an invalid palette,
    so direct ``00db``/``00dc`` payload access is required to preserve
    intensities. Some cameras mark uncompressed Y800 frames as ``00dc`` even
    though ``00db`` is the more usual uncompressed AVI chunk identifier.
    """

    def __init__(self, path: str | Path, width: int, height: int) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if min(width, height) <= 0:
            raise ValueError("AVI dimensions must be positive")
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height
        self._handle = self.path.open("rb")
        header = self._handle.read(1024 * 1024)
        marker, chunk_id = self._find_first_video_chunk(header)
        if marker < 0:
            self.close()
            raise ValueError(
                f"No fixed-size 00db/00dc video chunk found in {self.path}"
            )
        payload_size = struct.unpack_from("<I", header, marker + 4)[0]
        if payload_size != self.frame_bytes:
            self.close()
            raise ValueError(
                f"Unexpected frame size in {self.path}: "
                f"{payload_size} != {self.frame_bytes}"
            )
        self.chunk_id = chunk_id
        self.first_chunk = marker
        self.stride = 8 + self.frame_bytes + (self.frame_bytes & 1)
        self._frame_offsets = self._scan_frame_offsets()
        self._frame_count = len(self._frame_offsets)

    def _find_first_video_chunk(self, header: bytes) -> tuple[int, bytes]:
        """Return the earliest fixed-size ``00db`` or ``00dc`` frame chunk."""
        candidates: list[tuple[int, bytes]] = []
        for chunk_id in (b"00db", b"00dc"):
            start = 0
            while True:
                marker = header.find(chunk_id, start)
                if marker < 0:
                    break
                if marker + 8 <= len(header):
                    payload_size = struct.unpack_from("<I", header, marker + 4)[0]
                    if payload_size == self.frame_bytes:
                        candidates.append((marker, chunk_id))
                        break
                start = marker + 1
        return min(candidates, default=(-1, b""), key=lambda item: item[0])

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _scan_frame_offsets(self) -> list[int]:
        """Index frames, including OpenDML AVI files split into RIFF/AVIX lists."""
        offsets: list[int] = []
        offset = self.first_chunk
        while True:
            self._handle.seek(offset)
            chunk = self._handle.read(8)
            if len(chunk) != 8 or chunk[:4] != self.chunk_id:
                break
            if struct.unpack("<I", chunk[4:])[0] != self.frame_bytes:
                break
            offsets.append(offset)
            expected = offset + self.stride
            self._handle.seek(expected)
            next_header = self._handle.read(8)
            if (
                len(next_header) == 8
                and next_header[:4] == self.chunk_id
                and struct.unpack("<I", next_header[4:])[0] == self.frame_bytes
            ):
                offset = expected
                continue
            offset = self._find_next_video_chunk(expected)
            if offset < 0:
                break
        if not offsets:
            raise ValueError(f"AVI contains no readable frames: {self.path}")
        return offsets

    def _find_next_video_chunk(self, start_offset: int) -> int:
        """Find the next frame after an OpenDML list boundary or metadata block."""
        block_size = 1024 * 1024
        overlap = len(self.chunk_id) + 4 - 1
        self._handle.seek(start_offset)
        file_offset = start_offset
        tail = b""
        while True:
            block = self._handle.read(block_size)
            if not block:
                return -1
            data = tail + block
            data_offset = file_offset - len(tail)
            search_from = 0
            while True:
                marker = data.find(self.chunk_id, search_from)
                if marker < 0:
                    break
                if marker + 8 <= len(data):
                    payload_size = struct.unpack_from("<I", data, marker + 4)[0]
                    if payload_size == self.frame_bytes:
                        return data_offset + marker
                search_from = marker + 1
            tail = data[-overlap:]
            file_offset += len(block)

    def read(self, frame_index: int) -> np.ndarray:
        """Read one frame as ``uint8[H,W]`` without palette conversion."""
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if frame_index >= self.frame_count:
            raise EOFError(f"Frame {frame_index} is unavailable in {self.path}")
        chunk_offset = self._frame_offsets[frame_index]
        self._handle.seek(chunk_offset)
        header = self._handle.read(8)
        if len(header) != 8 or header[:4] != self.chunk_id:
            raise EOFError(f"Frame {frame_index} is unavailable in {self.path}")
        payload_size = struct.unpack("<I", header[4:])[0]
        if payload_size != self.frame_bytes:
            raise ValueError(
                f"Frame {frame_index} has unexpected size {payload_size} in {self.path}"
            )
        payload = self._handle.read(self.frame_bytes)
        if len(payload) != self.frame_bytes:
            raise EOFError(f"Frame {frame_index} is truncated in {self.path}")
        return (
            np.frombuffer(payload, dtype=np.uint8)
            .reshape(self.height, self.width)
            .copy()
        )

    def close(self) -> None:
        self._handle.close()


class RawFourAngleAviFileSequence:
    """Synchronously read four explicitly resolved raw pal8 AVI files."""

    def __init__(
        self,
        files: Mapping[int, str | Path],
        *,
        source_width: int = 1224,
        source_height: int = 1024,
        output_size: int = 256,
        rotation_degrees: int = 180,
        fps: float = 22.21,
        loop: bool = True,
        label: str = "four-angle AVI",
    ) -> None:
        if rotation_degrees not in (0, 180):
            raise ValueError("rotation_degrees must be 0 or 180")
        if output_size <= 0 or fps <= 0.0:
            raise ValueError("output_size and fps must be positive")
        if set(files) != set(ANGLE_ORDER):
            raise ValueError(f"AVI files must use angle keys {ANGLE_ORDER}")
        self.output_size = int(output_size)
        self.rotation_degrees = int(rotation_degrees)
        self._fps = float(fps)
        self.loop = bool(loop)
        self.index = 0
        self._label = str(label)
        self.readers = {
            angle: RawPal8AviReader(
                files[angle],
                source_width,
                source_height,
            )
            for angle in ANGLE_ORDER
        }
        counts = {reader.frame_count for reader in self.readers.values()}
        if len(counts) != 1:
            self.close()
            raise ValueError(
                f"Four angle AVI files have unequal frame counts: {counts}"
            )
        self._frame_count = counts.pop()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def label(self) -> str:
        return self._label

    def read(self) -> np.ndarray:
        if self.index >= self.frame_count:
            if not self.loop:
                raise EOFError(f"AVI sequence is exhausted: {self.label}")
            self.index = 0
        frames = [
            self._prepare(self.readers[angle].read(self.index)) for angle in ANGLE_ORDER
        ]
        self.index += 1
        return np.stack(frames, axis=0).astype(np.float32) / 255.0

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        height, width = frame.shape
        side = min(height, width)
        x = (width - side) // 2
        y = (height - side) // 2
        crop = frame[y : y + side, x : x + side]
        if self.rotation_degrees == 180:
            crop = np.rot90(crop, 2)
        return cv2.resize(
            np.ascontiguousarray(crop),
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def close(self) -> None:
        for reader in getattr(self, "readers", {}).values():
            reader.close()


class RawFourAngleAviSequence(RawFourAngleAviFileSequence):
    """Read the established ``angleXXX_segment_YYY.avi`` layout."""

    def __init__(
        self,
        root: str | Path,
        *,
        segment: int,
        source_width: int = 1224,
        source_height: int = 1024,
        output_size: int = 256,
        rotation_degrees: int = 180,
        fps: float = 22.21,
        loop: bool = True,
    ) -> None:
        resolved_root = Path(root).expanduser().resolve()
        segment = int(segment)
        files = {
            angle: resolved_root / f"angle{angle:03d}_segment_{segment:03d}.avi"
            for angle in ANGLE_ORDER
        }
        super().__init__(
            files,
            source_width=source_width,
            source_height=source_height,
            output_size=output_size,
            rotation_degrees=rotation_degrees,
            fps=fps,
            loop=loop,
            label=f"segment {segment:03d} · four-angle AVI",
        )
        self.root = resolved_root
        self.segment = segment


class FolderFourAngleImageSequence:
    """Load paired images from neighboring 0/45/90/135 degree folders.

    The user selects one file below the 0-degree tree. The loader locates the
    other angle roots beside that tree, pairs identical relative filenames,
    and starts playback at the selected file.
    """

    IMAGE_SUFFIXES: ClassVar[frozenset[str]] = frozenset(
        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    def __init__(
        self,
        selected_zero_file: str | Path,
        *,
        output_size: int = 256,
        rotation_degrees: int = 0,
        fps: float = 22.21,
        loop: bool = True,
    ) -> None:
        selected = Path(selected_zero_file).expanduser().resolve()
        if selected.suffix.lower() not in self.IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported angle image type: {selected.suffix}")
        if rotation_degrees not in (0, 180):
            raise ValueError("rotation_degrees must be 0 or 180")
        if output_size <= 0 or fps <= 0.0:
            raise ValueError("output_size and fps must be positive")
        roots, selected_relative = discover_angle_folder_layout(selected)
        relative_parent = selected_relative.parent
        angle_directories = {
            angle: root / relative_parent for angle, root in roots.items()
        }
        for angle, directory in angle_directories.items():
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Missing I{angle} sequence directory near I0: {directory}"
                )

        zero_names = sorted(
            (
                item.name
                for item in angle_directories[0].iterdir()
                if item.is_file() and item.suffix.lower() in self.IMAGE_SUFFIXES
            ),
            key=_natural_key,
        )
        common_names = [
            name
            for name in zero_names
            if all((angle_directories[angle] / name).is_file() for angle in ANGLE_ORDER)
        ]
        if not common_names:
            raise ValueError("No identically named four-angle image pairs were found")
        missing = {
            angle: len(zero_names)
            - sum((angle_directories[angle] / name).is_file() for name in zero_names)
            for angle in ANGLE_ORDER[1:]
        }
        if any(missing.values()):
            details = ", ".join(
                f"I{angle} missing {count}" for angle, count in missing.items()
            )
            raise ValueError(f"Four-angle folders are not fully paired: {details}")
        if selected.name not in common_names:
            raise ValueError(
                f"Selected I0 file has no complete four-angle pair: {selected.name}"
            )

        self.paths = {
            angle: [angle_directories[angle] / name for name in common_names]
            for angle in ANGLE_ORDER
        }
        self.output_size = int(output_size)
        self.rotation_degrees = int(rotation_degrees)
        self._fps = float(fps)
        self.loop = bool(loop)
        self.index = common_names.index(selected.name)
        self.start_index = self.index
        self._frame_count = len(common_names)
        self._label = (
            f"{roots[0].parent.name}/{relative_parent.as_posix()} · "
            f"four-angle folders · {self._frame_count} frames"
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def label(self) -> str:
        return self._label

    def read(self) -> np.ndarray:
        from PIL import Image

        if self.index >= self.frame_count:
            if not self.loop:
                raise EOFError(f"Image sequence is exhausted: {self.label}")
            self.index = 0
        frames = []
        original_shapes = set()
        for angle in ANGLE_ORDER:
            with Image.open(self.paths[angle][self.index]) as image:
                frame = np.asarray(image.convert("L"), dtype=np.uint8)
            original_shapes.add(frame.shape)
            frames.append(
                _prepare_grayscale_frame(
                    frame,
                    output_size=self.output_size,
                    rotation_degrees=self.rotation_degrees,
                )
            )
        if len(original_shapes) != 1:
            raise ValueError(
                f"Four-angle images have unequal dimensions at frame {self.index}: "
                f"{original_shapes}"
            )
        self.index += 1
        return np.stack(frames, axis=0).astype(np.float32) / 255.0

    def close(self) -> None:
        """Image files are opened per frame and need no persistent handles."""


def discover_angle_folder_layout(
    selected_zero_file: str | Path,
) -> tuple[dict[int, Path], Path]:
    """Find sibling angle roots and the selected file's relative path."""
    selected = Path(selected_zero_file).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(selected)
    nearby_parents = tuple(selected.parents)[1:4]
    for ancestor in (selected.parent, *nearby_parents):
        if _folder_angle(ancestor.name) != 0:
            continue
        roots: dict[int, Path] = {}
        for sibling in ancestor.parent.iterdir():
            if not sibling.is_dir():
                continue
            angle = _folder_angle(sibling.name)
            if angle is not None and angle not in roots:
                roots[angle] = sibling
        if set(roots) == set(ANGLE_ORDER):
            return roots, selected.relative_to(ancestor)
    raise ValueError(
        "Could not find neighboring angle folders. Expected sibling directories "
        "named 0, 45, 90, and 135 (or angle000/angle045/angle090/angle135)."
    )


def discover_four_angle_avi_files(selected_zero_file: str | Path) -> dict[int, Path]:
    """Resolve four AVI files from angle folders or a shared legacy directory."""
    selected = Path(selected_zero_file).expanduser().resolve()
    if selected.suffix.lower() != ".avi":
        raise ValueError("The selected 0-degree file is not an AVI")
    try:
        roots, relative = discover_angle_folder_layout(selected)
        files = {angle: root / relative for angle, root in roots.items()}
        if all(path.is_file() for path in files.values()):
            return files
    except ValueError:
        pass

    selected_key = _angle_filename_key(selected.name)
    if selected_key is None or selected_key[0] != 0:
        raise ValueError(
            "The selected AVI must be under the 0-degree folder or contain an "
            "explicit angle000/I0 token in its filename"
        )
    canonical = selected_key[1]
    matches: dict[int, Path] = {}
    for candidate in selected.parent.glob("*.avi"):
        key = _angle_filename_key(candidate.name)
        if key is not None and key[1] == canonical:
            matches[key[0]] = candidate
    if set(matches) != set(ANGLE_ORDER):
        raise ValueError(
            f"Could not resolve four matching AVI files beside {selected.name}"
        )
    return matches


def _folder_angle(name: str) -> int | None:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    for angle in ANGLE_ORDER:
        values = {str(angle), f"{angle:03d}"}
        aliases = {
            *values,
            *(
                f"{prefix}{value}"
                for prefix in ("angle", "i", "pol", "deg", "degree")
                for value in values
            ),
            *(f"{value}{suffix}" for suffix in ("deg", "degree") for value in values),
        }
        if normalized in aliases:
            return angle
    return None


def _angle_filename_key(name: str) -> tuple[int, str] | None:
    patterns = (
        re.compile(r"(?i)(?<![a-z0-9])angle[_-]?(000|045|090|135)(?![0-9])"),
        re.compile(r"(?i)(?<![a-z0-9])i[_-]?(0|45|90|135)(?![0-9])"),
        re.compile(r"(?i)^(0|45|90|135)(?=[_-])"),
    )
    for pattern in patterns:
        match = pattern.search(name)
        if match is None:
            continue
        angle = int(match.group(1))
        canonical = name[: match.start()] + "<angle>" + name[match.end() :]
        return angle, canonical.lower()
    return None


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", value)
    )


def _prepare_grayscale_frame(
    frame: np.ndarray,
    *,
    output_size: int,
    rotation_degrees: int,
) -> np.ndarray:
    import cv2

    height, width = frame.shape
    side = min(height, width)
    x = (width - side) // 2
    y = (height - side) // 2
    crop = frame[y : y + side, x : x + side]
    if rotation_degrees == 180:
        crop = np.rot90(crop, 2)
    return cv2.resize(
        np.ascontiguousarray(crop),
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )


class OpenCVFourAngleCameraSource:
    """Acquire synchronized analyzer views from four OpenCV camera devices."""

    def __init__(
        self,
        indices: tuple[int, int, int, int],
        *,
        output_size: int = 256,
        rotation_degrees: int = 180,
        requested_width: int | None = None,
        requested_height: int | None = None,
        requested_fps: float | None = None,
    ) -> None:
        import cv2

        if len(indices) != len(ANGLE_ORDER) or len(set(indices)) != len(indices):
            raise ValueError("Camera indices must contain four distinct device IDs")
        if rotation_degrees not in (0, 180) or output_size <= 0:
            raise ValueError("Invalid camera orientation or output size")
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.captures = [cv2.VideoCapture(index, backend) for index in indices]
        self.indices = tuple(int(index) for index in indices)
        self.output_size = int(output_size)
        self.rotation_degrees = int(rotation_degrees)
        for capture in self.captures:
            if requested_width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
            if requested_height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
            if requested_fps is not None:
                capture.set(cv2.CAP_PROP_FPS, requested_fps)
        unavailable = [
            index
            for index, capture in zip(self.indices, self.captures)
            if not capture.isOpened()
        ]
        if unavailable:
            self.close()
            raise RuntimeError(f"Unable to open camera devices: {unavailable}")

    @property
    def frame_count(self) -> None:
        return None

    @property
    def fps(self) -> float | None:
        import cv2

        values = [capture.get(cv2.CAP_PROP_FPS) for capture in self.captures]
        valid = [value for value in values if value > 0.0]
        return float(min(valid)) if valid else None

    @property
    def label(self) -> str:
        return "camera " + "/".join(str(index) for index in self.indices)

    def read(self) -> np.ndarray:
        import cv2

        frames: list[np.ndarray] = []
        for angle, capture in zip(ANGLE_ORDER, self.captures):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera I{angle} failed to return a frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            height, width = gray.shape
            side = min(height, width)
            x = (width - side) // 2
            y = (height - side) // 2
            crop = gray[y : y + side, x : x + side]
            if self.rotation_degrees == 180:
                crop = np.rot90(crop, 2)
            frames.append(
                cv2.resize(
                    np.ascontiguousarray(crop),
                    (self.output_size, self.output_size),
                    interpolation=cv2.INTER_AREA,
                )
            )
        return np.stack(frames, axis=0).astype(np.float32) / 255.0

    def close(self) -> None:
        for capture in getattr(self, "captures", []):
            capture.release()


__all__ = [
    "ANGLE_ORDER",
    "FolderFourAngleImageSequence",
    "FourAngleSource",
    "NpyFourAngleSequence",
    "OpenCVFourAngleCameraSource",
    "RawFourAngleAviFileSequence",
    "RawFourAngleAviSequence",
    "RawPal8AviReader",
    "discover_angle_folder_layout",
    "discover_four_angle_avi_files",
]
