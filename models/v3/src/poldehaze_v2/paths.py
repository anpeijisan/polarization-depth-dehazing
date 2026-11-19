"""Project path resolution shared by command-line entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SERVER_DATA_ROOT = Path("/root/rivermind-data/poldehazeV2/data")


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest ancestor containing ``pyproject.toml``."""
    current = Path(start).expanduser().resolve() if start else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate project root from {current}")


def default_data_root(project_root: Path) -> Path:
    """Resolve the configured, server, or repository-local data root."""
    configured = os.environ.get("POLDEHAZE_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if SERVER_DATA_ROOT.parent.exists():
        return SERVER_DATA_ROOT
    return (project_root / "data").resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical filesystem locations for a single run."""

    root: Path
    data: Path
    outputs: Path
    runs: Path
    checkpoints: Path
    predictions: Path
    reports: Path

    @classmethod
    def create(
        cls,
        project_root: str | Path | None = None,
        data_root: str | Path | None = None,
        outputs_root: str | Path | None = None,
    ) -> "ProjectPaths":
        root = find_project_root(project_root)
        data = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else default_data_root(root)
        )
        outputs = (
            Path(outputs_root).expanduser().resolve()
            if outputs_root is not None
            else (root / "outputs").resolve()
        )
        return cls(
            root=root,
            data=data,
            outputs=outputs,
            runs=outputs / "runs",
            checkpoints=outputs / "checkpoints",
            predictions=outputs / "predictions",
            reports=outputs / "reports",
        )

    def ensure_output_dirs(self) -> None:
        """Create writable output directories without touching dataset files."""
        for path in (
            self.outputs,
            self.runs,
            self.checkpoints,
            self.predictions,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)

