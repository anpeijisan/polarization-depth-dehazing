"""YAML configuration loading, inheritance, and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


ConfigDict = dict[str, Any]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge mappings without mutating either input.

    Mapping values are merged recursively. All other values, including lists,
    replace the corresponding base value in full.
    """
    merged: ConfigDict = copy.deepcopy(dict(base))
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(previous, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> ConfigDict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration: {path}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return dict(payload)


def _resolve_parent(reference: str, current_file: Path, project_root: Path) -> Path:
    candidate = Path(reference).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    local_candidate = (current_file.parent / candidate).resolve()
    root_candidate = (project_root / candidate).resolve()
    if local_candidate.is_file():
        return local_candidate
    if root_candidate.is_file():
        return root_candidate
    raise ConfigError(
        f"Cannot resolve parent configuration {reference!r} from {current_file}"
    )


def _normalise_extends(value: Any, source: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(isinstance(item, str) for item in value):
            return list(value)
    raise ConfigError(f"'extends' must be a path or list of paths in {source}")


def _load_recursive(
    path: Path,
    project_root: Path,
    active_stack: tuple[Path, ...],
) -> ConfigDict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigError(f"Configuration file does not exist: {resolved}")
    if resolved in active_stack:
        chain = " -> ".join(str(item) for item in (*active_stack, resolved))
        raise ConfigError(f"Circular configuration inheritance detected: {chain}")

    payload = _read_yaml(resolved)
    parents = _normalise_extends(payload.pop("extends", None), resolved)
    inherited: ConfigDict = {}
    next_stack = (*active_stack, resolved)
    for reference in parents:
        parent_path = _resolve_parent(reference, resolved, project_root)
        inherited = deep_merge(
            inherited,
            _load_recursive(parent_path, project_root, next_stack),
        )
    return deep_merge(inherited, payload)


def parse_override(expression: str) -> tuple[list[str], Any]:
    """Parse ``section.key=value`` using YAML scalar/list syntax for values."""
    if "=" not in expression:
        raise ConfigError(f"Override must use key.path=value syntax: {expression!r}")
    raw_path, raw_value = expression.split("=", 1)
    keys = [part.strip() for part in raw_path.split(".")]
    if not keys or any(not key for key in keys):
        raise ConfigError(f"Override contains an empty key: {expression!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid override value: {expression!r}") from exc
    return keys, value


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> ConfigDict:
    """Apply dotted-path overrides to a deep copy of ``config``.

    Intermediate mappings may be created, but an existing non-mapping value
    cannot be traversed because that usually indicates a misspelled key path.
    """
    updated: ConfigDict = copy.deepcopy(dict(config))
    for expression in overrides:
        keys, value = parse_override(expression)
        cursor: MutableMapping[str, Any] = updated
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                cursor[key] = {}
                existing = cursor[key]
            if not isinstance(existing, MutableMapping):
                raise ConfigError(
                    f"Cannot traverse non-mapping key {key!r} in override {expression!r}"
                )
            cursor = existing
        cursor[keys[-1]] = value
    return updated


def load_config(
    config_path: str | Path,
    overrides: Iterable[str] = (),
    project_root: str | Path | None = None,
) -> ConfigDict:
    """Load a YAML configuration with recursive inheritance and overrides."""
    path = Path(config_path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve() if project_root else _find_project_root(path)
    config = _load_recursive(path, root, ())
    return apply_overrides(config, overrides)


def _find_project_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(f"Cannot locate pyproject.toml above {start}")

