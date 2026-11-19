"""Task-gradient geometry monitoring without modifying optimizer gradients."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Any, Optional

import torch
from torch import Tensor, nn

DEFAULT_SHARED_PREFIXES = (
    "raw_stem",
    "phys_stem",
    "raw_encoders",
    "phys_encoders",
    "fusions",
    "post_fusion",
    "raw_down",
    "phys_down",
    "bottleneck",
    "decoder",
)

NamedParameter = tuple[str, nn.Parameter]
GradientTuple = tuple[Optional[Tensor], ...]  # noqa: UP045


def select_shared_parameters(
    model: nn.Module,
    shared_param_names: Sequence[str] | None = None,
) -> list[NamedParameter]:
    """Select trainable shared parameters by exact name or module prefix."""
    selectors = tuple(shared_param_names or DEFAULT_SHARED_PREFIXES)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(
            name == selector
            or name.startswith(f"{selector}.")
            or _strip_compile_prefix(name) == selector
            or _strip_compile_prefix(name).startswith(f"{selector}.")
            for selector in selectors
        )
    ]
    if not selected:
        raise ValueError(f"No shared parameters matched selectors: {selectors}")
    return selected


def _strip_compile_prefix(name: str) -> str:
    """Remove the torch.compile wrapper prefix from a parameter name."""
    for prefix in ("_orig_mod.", "_wrapped."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def task_gradients(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool = True,
) -> GradientTuple:
    """Return task gradients with ``None`` for graph-unused parameters."""
    if loss.ndim != 0 or not loss.requires_grad:
        raise ValueError("Task loss must be a differentiable scalar tensor")
    return tuple(
        torch.autograd.grad(
            loss,
            tuple(parameters),
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
    )


def gradient_geometry(
    gradients_a: Sequence[Tensor | None],
    gradients_b: Sequence[Tensor | None],
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Compute global dot product, norms, and cosine without flatten concatenation."""
    if len(gradients_a) != len(gradients_b):
        raise ValueError("Gradient sequences must have equal lengths")
    reference = next(
        (gradient for gradient in (*gradients_a, *gradients_b) if gradient is not None),
        None,
    )
    if reference is None:
        raise ValueError("Both gradient sequences are entirely unused")
    dot = reference.new_zeros((), dtype=torch.float32)
    norm_a_sq = reference.new_zeros((), dtype=torch.float32)
    norm_b_sq = reference.new_zeros((), dtype=torch.float32)
    for grad_a, grad_b in zip(gradients_a, gradients_b):
        if grad_a is not None:
            norm_a_sq = norm_a_sq + grad_a.detach().float().square().sum()
        if grad_b is not None:
            norm_b_sq = norm_b_sq + grad_b.detach().float().square().sum()
        if grad_a is not None and grad_b is not None:
            dot = dot + (grad_a.detach().float() * grad_b.detach().float()).sum()
    norm_a = torch.sqrt(norm_a_sq)
    norm_b = torch.sqrt(norm_b_sq)
    cosine = dot / (norm_a * norm_b).clamp_min(float(eps))
    return {
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine": cosine.clamp(-1.0, 1.0),
    }


class GradientConflictMonitor:
    """Record global, per-stage, and per-fog task-gradient conflicts."""

    def __init__(
        self,
        model: nn.Module,
        shared_param_names: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.named_parameters = select_shared_parameters(model, shared_param_names)
        self._global: list[float] = []
        self._per_stage: dict[str, list[float]] = defaultdict(list)
        self._per_fog: dict[str, list[float]] = defaultdict(list)

    def capture_gradients(
        self,
        loss_restoration: Tensor,
        loss_physics: Tensor,
        *,
        fog_levels: str | Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Capture geometry while retaining the graph for the actual backward pass."""
        parameters = [parameter for _, parameter in self.named_parameters]
        gradients_r = task_gradients(loss_restoration, parameters, retain_graph=True)
        gradients_p = task_gradients(loss_physics, parameters, retain_graph=True)
        geometry = gradient_geometry(gradients_r, gradients_p)
        cosine = float(geometry["cosine"].detach().cpu())
        self._global.append(cosine)

        stage_indices: dict[str, list[int]] = defaultdict(list)
        for index, (name, _) in enumerate(self.named_parameters):
            stage_indices[_parameter_stage(name)].append(index)
        for stage, indices in stage_indices.items():
            stage_r = tuple(gradients_r[index] for index in indices)
            stage_p = tuple(gradients_p[index] for index in indices)
            if not any(value is not None for value in (*stage_r, *stage_p)):
                continue
            stage_geometry = gradient_geometry(stage_r, stage_p)
            self._per_stage[stage].append(
                float(stage_geometry["cosine"].detach().cpu())
            )

        if fog_levels is not None:
            labels = [fog_levels] if isinstance(fog_levels, str) else list(fog_levels)
            for label in sorted(set(map(str, labels))):
                self._per_fog[label].append(cosine)
        return {
            "cosine": cosine,
            "restoration_grad_norm": float(geometry["norm_a"].detach().cpu()),
            "physics_grad_norm": float(geometry["norm_b"].detach().cpu()),
            "conflict": float(cosine < 0.0),
        }

    def compute_stats(self) -> dict[str, Any]:
        """Aggregate all captured batches without clearing state."""
        return {
            **_summarize(self._global),
            "per_stage": {
                key: _summarize(values)
                for key, values in sorted(self._per_stage.items())
            },
            "per_fog_level": {
                key: _summarize(values) for key, values in sorted(self._per_fog.items())
            },
        }

    def reset(self) -> None:
        self._global.clear()
        self._per_stage.clear()
        self._per_fog.clear()


def _parameter_stage(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] in {
        "raw_encoders",
        "phys_encoders",
        "fusions",
        "post_fusion",
        "raw_down",
        "phys_down",
        "decoder",
    }:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "cosine_mean": 0.0, "conflict_ratio": 0.0}
    return {
        "count": len(values),
        "cosine_mean": fmean(values),
        "conflict_ratio": sum(value < 0.0 for value in values) / len(values),
    }
