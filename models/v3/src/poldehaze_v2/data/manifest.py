"""Strict JSONL manifest loading for scene-disjoint PolarDehaze data."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

REQUIRED_RECORD_KEYS = {
    "sample_id",
    "stem",
    "split",
    "fog_level",
    "clear",
    "hazy",
}
OPTIONAL_PHYSICS_KEYS = {
    "metadata",
    "depth_clear",
    "depth_hazy",
    "transmission_map",
    "airlight_map",
    "density_map",
    "airlight_s0_map",
}
ANGLE_KEYS = ("0", "45", "90", "135")
FOG_LEVELS = ("light", "medium", "heavy")
SPLITS = ("train", "val", "test")


class Manifest:
    """Load, validate, and query one relative-path JSONL manifest."""

    def __init__(
        self,
        path: str | Path,
        data_root: str | Path,
        split: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        if not self.path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.path}")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")
        self._records = self._load_records()
        if not self._records:
            raise ValueError(f"Manifest contains no records: {self.path}")

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
                if not isinstance(payload, Mapping):
                    raise TypeError(f"Manifest record must be a mapping at line {line_number}")
                record = dict(payload)
                self._validate_record(record, line_number)
                sample_id = str(record["sample_id"])
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id {sample_id!r} at line {line_number}")
                sample_ids.add(sample_id)
                records.append(record)
        return records

    def _validate_record(self, record: dict[str, Any], line_number: int) -> None:
        missing = sorted(REQUIRED_RECORD_KEYS - record.keys())
        if missing:
            raise KeyError(f"Missing keys {missing} at {self.path}:{line_number}")
        record_split = str(record["split"])
        if record_split not in SPLITS:
            raise ValueError(f"Unknown record split {record_split!r} at line {line_number}")
        if self.split is not None and record_split != self.split:
            raise ValueError(
                f"Split mismatch at {self.path}:{line_number}: "
                f"record={record_split!r}, requested={self.split!r}"
            )
        fog_level = str(record["fog_level"])
        if fog_level not in FOG_LEVELS:
            raise ValueError(f"Unknown fog level {fog_level!r} at line {line_number}")
        for domain in ("clear", "hazy"):
            angles = record[domain]
            if not isinstance(angles, Mapping) or set(map(str, angles.keys())) != set(ANGLE_KEYS):
                raise ValueError(
                    f"{domain!r} must contain exactly {ANGLE_KEYS} at line {line_number}"
                )
            for value in angles.values():
                self.resolve_path(str(value))
        for key in OPTIONAL_PHYSICS_KEYS & record.keys():
            self.resolve_path(str(record[key]))

    def resolve_path(self, relative_path: str | Path) -> Path:
        """Resolve a manifest path while rejecting absolute/path-traversal entries."""
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(f"Manifest paths must be relative, got {relative}")
        resolved = (self.data_root / relative).resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes data root: {relative}") from exc
        return resolved

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self._records[index])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for record in self._records:
            yield copy.deepcopy(record)

    def get_split_stats(self) -> dict[str, Any]:
        """Return sample, scene, and fog-level counts for the loaded records."""
        split_samples = Counter(str(record["split"]) for record in self._records)
        split_stems: dict[str, set[str]] = defaultdict(set)
        fog_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in self._records:
            split = str(record["split"])
            split_stems[split].add(str(record["stem"]))
            fog_counts[split][str(record["fog_level"])] += 1
        return {
            split: {
                "samples": split_samples.get(split, 0),
                "scenes": len(split_stems.get(split, set())),
                "fog_levels": {
                    level: fog_counts[split].get(level, 0) for level in FOG_LEVELS
                },
            }
            for split in SPLITS
        }

