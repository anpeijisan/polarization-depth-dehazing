"""Coupled, partial, cosine-gated, and projected-balanced task gradients."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from poldehaze_v2.optimization.conflict_monitor import (
    gradient_geometry,
    select_shared_parameters,
    task_gradients,
)

MEDIATION_MODES = ("coupled", "static", "cosine_gate", "pcgrad")


class GradientMediator:
    """Compute and accumulate mediated gradients for restoration/physics tasks.

    The method performs gradient accumulation but never calls an optimizer or
    GradScaler step. Existing ``param.grad`` values are preserved and added to,
    so the class remains compatible with gradient accumulation.
    """

    def __init__(
        self,
        mode: str = "coupled",
        gamma_max: float = 0.5,
        gamma_static: float = 0.1,
        ema_momentum: float = 0.9,
        *,
        shared_param_names: Sequence[str] | None = None,
        pcgrad_rho: float = 1.0,
        lambda_max: float = 1.0,
    ) -> None:
        if mode not in MEDIATION_MODES:
            raise ValueError(f"Unknown mediation mode {mode!r}; expected {MEDIATION_MODES}")
        if not 0.0 <= gamma_static <= 1.0:
            raise ValueError("gamma_static must lie in [0,1]")
        if not 0.0 <= gamma_max <= 1.0:
            raise ValueError("gamma_max must lie in [0,1]")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError("ema_momentum must lie in [0,1)")
        if pcgrad_rho < 0.0 or lambda_max < 0.0:
            raise ValueError("pcgrad_rho and lambda_max must be non-negative")
        self.mode = mode
        self.gamma_max = float(gamma_max)
        self.gamma_static = float(gamma_static)
        self.ema_momentum = float(ema_momentum)
        self.shared_param_names = tuple(shared_param_names) if shared_param_names else None
        self.pcgrad_rho = float(pcgrad_rho)
        self.lambda_max = float(lambda_max)
        self.gamma_ema = self.gamma_max

    def mediate(
        self,
        model: nn.Module,
        loss_restoration: Tensor,
        loss_physics: Tensor,
    ) -> dict[str, Any]:
        """Accumulate one mediated backward pass and return gradient statistics."""
        all_named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not all_named:
            raise ValueError("Model has no trainable parameters")
        all_parameters = [parameter for _, parameter in all_named]
        shared_named = select_shared_parameters(model, self.shared_param_names)
        shared_ids = {id(parameter) for _, parameter in shared_named}
        shared_indices = [
            index
            for index, (_, parameter) in enumerate(all_named)
            if id(parameter) in shared_ids
        ]

        gradients_r = task_gradients(
            loss_restoration, all_parameters, retain_graph=True
        )
        gradients_p = task_gradients(
            loss_physics, all_parameters, retain_graph=False
        )
        shared_r = tuple(gradients_r[index] for index in shared_indices)
        shared_p = tuple(gradients_p[index] for index in shared_indices)
        geometry = gradient_geometry(shared_r, shared_p)
        cosine = float(geometry["cosine"].detach().cpu())

        gamma = 1.0
        balance_lambda = 1.0
        projected_shared_p = shared_p
        if self.mode == "static":
            gamma = self.gamma_static
        elif self.mode == "cosine_gate":
            instantaneous = self.gamma_max * max(
                0.0, min(1.0, 0.5 * (1.0 + cosine))
            )
            self.gamma_ema = (
                self.ema_momentum * self.gamma_ema
                + (1.0 - self.ema_momentum) * instantaneous
            )
            gamma = self.gamma_ema
        elif self.mode == "pcgrad":
            projected_shared_p = self._project_physics(shared_r, shared_p, geometry)
            projected_geometry = gradient_geometry(shared_r, projected_shared_p)
            norm_restoration = projected_geometry["norm_a"]
            norm_physics = projected_geometry["norm_b"]
            if float(norm_physics.detach().cpu()) <= 1e-12:
                balance_lambda = 0.0
            else:
                raw_lambda = self.pcgrad_rho * norm_restoration / norm_physics.clamp_min(1e-12)
                balance_lambda = float(
                    raw_lambda.clamp(0.0, self.lambda_max).detach().cpu()
                )

        shared_projected_by_index = {
            index: gradient
            for index, gradient in zip(shared_indices, projected_shared_p)
        }
        combined: list[Tensor | None] = []
        for index, (gradient_r, gradient_p) in enumerate(zip(gradients_r, gradients_p)):
            if index in shared_projected_by_index:
                mediated_p = shared_projected_by_index[index]
                physics_scale = balance_lambda if self.mode == "pcgrad" else gamma
                combined.append(_add_gradients(gradient_r, mediated_p, physics_scale))
            else:
                combined.append(_add_gradients(gradient_r, gradient_p, 1.0))

        backward_parameters: list[nn.Parameter] = []
        backward_gradients: list[Tensor] = []
        for parameter, gradient in zip(all_parameters, combined):
            if gradient is not None:
                backward_parameters.append(parameter)
                backward_gradients.append(gradient)
        if not backward_parameters:
            raise RuntimeError("Both task losses produced no trainable gradients")
        torch.autograd.backward(
            tuple(backward_parameters),
            grad_tensors=tuple(backward_gradients),
        )
        return {
            "mode": self.mode,
            "cosine": cosine,
            "conflict": cosine < 0.0,
            "gamma": gamma,
            "lambda": balance_lambda,
            "restoration_grad_norm": float(geometry["norm_a"].detach().cpu()),
            "physics_grad_norm": float(geometry["norm_b"].detach().cpu()),
            "shared_parameter_count": len(shared_indices),
        }

    @staticmethod
    def _project_physics(
        gradients_r: Sequence[Tensor | None],
        gradients_p: Sequence[Tensor | None],
        geometry: MappingGeometry,
    ) -> tuple[Tensor | None, ...]:
        if float(geometry["dot"].detach().cpu()) >= 0.0:
            return tuple(gradients_p)
        coefficient = geometry["dot"] / geometry["norm_a"].square().clamp_min(1e-12)
        projected: list[Tensor | None] = []
        for gradient_r, gradient_p in zip(gradients_r, gradients_p):
            if gradient_p is None:
                projected.append(None)
            elif gradient_r is None:
                projected.append(gradient_p)
            else:
                projected.append(gradient_p - coefficient.to(gradient_p.dtype) * gradient_r)
        return tuple(projected)


MappingGeometry = dict[str, Tensor]


def _add_gradients(
    gradient_a: Tensor | None,
    gradient_b: Tensor | None,
    scale_b: float,
) -> Tensor | None:
    if gradient_a is None:
        return None if gradient_b is None else gradient_b * float(scale_b)
    if gradient_b is None:
        return gradient_a
    return gradient_a + float(scale_b) * gradient_b
