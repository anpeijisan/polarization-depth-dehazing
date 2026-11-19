"""Acquisition and integrity checks for the frozen PolarLITIS v1 archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


POLARLITIS_URL = (
    "https://zenodo.org/api/records/5547760/files/PolarLITIS.zip/content"
)
POLARLITIS_SIZE = 8_405_168_797
POLARLITIS_MD5 = "265ca71f68aa6982a6e48e5191076c23"


def download_polarlitis_archive(
    data_root: str | Path,
    *,
    chunk_size: int = 256 * 1024,
    workers: int = 8,
) -> Path:
    """Download or resume the immutable archive with verified byte ranges."""
    root = Path(data_root).expanduser().resolve()
    raw_dir = root / "external" / "polarlitis_v1" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "PolarLITIS.zip"
    partial = archive.with_suffix(".zip.part")

    if archive.is_file():
        _verify_file(archive)
        return archive

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > POLARLITIS_SIZE:
        raise ValueError(
            f"Partial archive is larger than the published file: {partial}"
        )
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if offset < POLARLITIS_SIZE:
        ranges = _partition_ranges(offset, POLARLITIS_SIZE - 1, workers)
        segment_paths = [
            partial.with_name(f"{partial.name}.segment{index:02d}")
            for index in range(len(ranges))
        ]
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(
                    _download_range,
                    start,
                    end,
                    segment,
                    chunk_size,
                )
                for (start, end), segment in zip(ranges, segment_paths)
            ]
            for future in futures:
                future.result()
        assembled = partial.with_name(f"{partial.name}.assembled")
        if assembled.exists():
            raise FileExistsError(
                f"Refusing to overwrite interrupted assembly: {assembled}"
            )
        with assembled.open("xb") as destination:
            if partial.exists():
                with partial.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=chunk_size)
            for segment in segment_paths:
                with segment.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=chunk_size)
        if assembled.stat().st_size != POLARLITIS_SIZE:
            raise ValueError(
                "Assembled PolarLITIS size mismatch: "
                f"{assembled.stat().st_size} != {POLARLITIS_SIZE}"
            )
        os.replace(assembled, partial)
        for segment in segment_paths:
            segment.unlink()

    _verify_file(partial)
    os.replace(partial, archive)
    return archive


def extract_polarlitis_archive(
    archive: str | Path,
    data_root: str | Path,
) -> Path:
    """Extract the verified archive into a non-overwriting versioned directory."""
    archive_path = Path(archive).expanduser().resolve()
    _verify_file(archive_path)
    root = Path(data_root).expanduser().resolve()
    target = root / "external" / "polarlitis_v1" / "source"
    staging = target.with_name("source.extracting")
    if target.exists() or staging.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing PolarLITIS extraction: "
            f"target={target}, staging={staging}"
        )
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as bundle:
        _validate_zip_members(bundle, staging)
        bundle.extractall(staging)
    os.replace(staging, target)
    return target


def write_polarlitis_preparation_report(
    report_path: str | Path,
    archive: str | Path,
    extracted: str | Path | None,
) -> None:
    """Atomically write archive provenance and extraction state."""
    destination = Path(report_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset": "PolarLITIS",
        "version": "v1",
        "doi": "10.5281/zenodo.5547760",
        "source_url": POLARLITIS_URL,
        "archive": str(Path(archive).resolve()),
        "archive_size_bytes": POLARLITIS_SIZE,
        "archive_md5": POLARLITIS_MD5,
        "extracted": str(Path(extracted).resolve()) if extracted else None,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _verify_file(path: Path) -> None:
    size = path.stat().st_size
    if size != POLARLITIS_SIZE:
        raise ValueError(
            f"PolarLITIS size mismatch for {path}: {size} != {POLARLITIS_SIZE}"
        )
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != POLARLITIS_MD5:
        raise ValueError(
            f"PolarLITIS MD5 mismatch for {path}: {actual} != {POLARLITIS_MD5}"
        )


def _partition_ranges(
    start: int,
    end: int,
    workers: int,
) -> list[tuple[int, int]]:
    remaining = end - start + 1
    count = min(workers, remaining)
    width = (remaining + count - 1) // count
    return [
        (range_start, min(range_start + width - 1, end))
        for range_start in range(start, end + 1, width)
    ]


def _download_range(
    start: int,
    end: int,
    destination: Path,
    chunk_size: int,
) -> None:
    expected_size = end - start + 1
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        raise ValueError(
            f"Segment exceeds its assigned range: {destination}"
        )
    request_start = start + existing_size
    mode = "ab" if destination.exists() else "xb"
    with destination.open(mode) as handle:
        while request_start <= end:
            request_end = min(
                request_start + 64 * 1024 * 1024 - 1,
                end,
            )
            request = urllib.request.Request(
                POLARLITIS_URL,
                headers={"Range": f"bytes={request_start}-{request_end}"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or not content_range.startswith(
                    f"bytes {request_start}-{request_end}/"
                ):
                    raise RuntimeError(
                        "The server returned an unexpected range response: "
                        f"status={status}, Content-Range={content_range!r}"
                    )
                shutil.copyfileobj(response, handle, length=chunk_size)
            request_start = request_end + 1
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Segment size mismatch for {destination}: "
            f"{actual_size} != {expected_size}"
        )


def _validate_zip_members(bundle: zipfile.ZipFile, target: Path) -> None:
    resolved_target = target.resolve()
    for member in bundle.infolist():
        destination = (target / member.filename).resolve()
        try:
            destination.relative_to(resolved_target)
        except ValueError as error:
            raise ValueError(
                f"Archive member escapes extraction root: {member.filename}"
            ) from error
