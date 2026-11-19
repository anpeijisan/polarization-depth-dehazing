"""Output-domain polarization-guided Stokes refinement variants."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from poldehaze_v2.models.common import BasicResidualBlock, ConvNormAct
from poldehaze_v2.physics.stokes import (
    STOKES_S0_MAX,
    safe_clip_stokes,
    stokes_to_polar_features,
)

PGDE_MODES = (
    "none",
    "single_residual",
    "dual_frequency",
    "iterative_shared",
)

_CUE_CHANNELS = 16


def fixed_lowpass(value: Tensor, kernel: int) -> Tensor:
    """Apply the protocol's fixed, non-normalized spatial low-pass filter."""
    if value.ndim != 4:
        raise ValueError("fixed_lowpass expects a BCHW tensor")
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("low-pass kernel must be a positive odd integer")
    return functional.avg_pool2d(
        value,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )


class PGDECueBuilder(nn.Module):
    """Build fixed-scale inference cues for every PGDE-V2 mode.

    Decoder features determine the target spatial size but are projected by the
    mode-specific shared stem. Stokes inputs use physical ranges: every Stokes
    component uses the V3 unit-total scale, while DoLP remains on ``[0, 1]``. No cue uses
    ground truth or per-image maximum normalization.
    """

    cue_channels = _CUE_CHANNELS

    def __init__(self, local_kernel: int = 15) -> None:
        super().__init__()
        if local_kernel <= 0 or local_kernel % 2 == 0:
            raise ValueError("local_kernel must be a positive odd integer")
        self.local_kernel = int(local_kernel)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 4.0
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False
        )

    def forward(
        self,
        decoder_feature: Tensor,
        coarse: Mapping[str, Tensor],
        observed_stokes: tuple[Tensor, Tensor, Tensor],
        *,
        step_index: int = 0,
    ) -> Tensor:
        """Return a ``B x 16 x H x W`` tensor of fixed-semantics cues."""
        if decoder_feature.ndim != 4:
            raise ValueError("decoder_feature must be BCHW")
        coarse_stokes = _mapping_stokes(coarse)
        observed = _tuple_stokes(observed_stokes)
        size = decoder_feature.shape[-2:]
        if coarse_stokes.shape[-2:] != size:
            coarse_stokes = functional.interpolate(
                coarse_stokes, size=size, mode="bilinear", align_corners=False
            )
        if observed.shape[-2:] != size:
            observed = functional.interpolate(
                observed, size=size, mode="bilinear", align_corners=False
            )
        if coarse_stokes.shape[0] != decoder_feature.shape[0] or (
            observed.shape[0] != decoder_feature.shape[0]
        ):
            raise ValueError("PGDE cues and decoder features must share batch size")

        coarse_normalized = coarse_stokes / STOKES_S0_MAX
        observed_normalized = observed / STOKES_S0_MAX
        difference = coarse_normalized - observed_normalized
        coarse_dolp, _, _ = stokes_to_polar_features(
            coarse_stokes[:, 0:1], coarse_stokes[:, 1:2], coarse_stokes[:, 2:3]
        )
        observed_dolp, _, _ = stokes_to_polar_features(
            observed[:, 0:1], observed[:, 1:2], observed[:, 2:3]
        )
        coarse_s0_normalized = coarse_normalized[:, 0:1]
        s0_gradient = self._sobel_magnitude(coarse_s0_normalized)
        dolp_gradient = self._sobel_magnitude(coarse_dolp)
        local_mean = fixed_lowpass(coarse_s0_normalized, self.local_kernel)
        local_second_moment = fixed_lowpass(
            coarse_s0_normalized.square(), self.local_kernel
        )
        local_std = (
            local_second_moment - local_mean.square()
        ).clamp_min(0.0).add(1e-6).sqrt()
        step = coarse_s0_normalized.new_full(
            coarse_s0_normalized.shape, float(step_index)
        )
        cues = torch.cat(
            (
                coarse_normalized,
                observed_normalized,
                difference,
                coarse_dolp,
                observed_dolp,
                s0_gradient,
                dolp_gradient,
                local_mean,
                local_std,
                step,
            ),
            dim=1,
        )
        if cues.shape[1] != self.cue_channels:
            raise RuntimeError("PGDE cue channel contract changed unexpectedly")
        return cues

    def _sobel_magnitude(self, value: Tensor) -> Tensor:
        kernel_x = self.sobel_x.to(device=value.device, dtype=value.dtype)
        kernel_y = self.sobel_y.to(device=value.device, dtype=value.dtype)
        grad_x = functional.conv2d(value, kernel_x, padding=1)
        grad_y = functional.conv2d(value, kernel_y, padding=1)
        return (grad_x.square() + grad_y.square() + 1e-6).sqrt()


class _SharedCueStem(nn.Module):
    """Project decoder features and common cues into one hidden field."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        lowpass_kernel: int,
        activation: str,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.cue_builder = PGDECueBuilder(lowpass_kernel)
        self.feature_projection = ConvNormAct(
            feature_channels,
            hidden_channels,
            kernel_size=1,
            activation_name=activation,
            norm_groups=norm_groups,
        )
        self.cue_projection = ConvNormAct(
            self.cue_builder.cue_channels,
            hidden_channels,
            kernel_size=3,
            activation_name=activation,
            norm_groups=norm_groups,
        )
        self.body = nn.Sequential(
            ConvNormAct(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=3,
                activation_name=activation,
                norm_groups=norm_groups,
            ),
            BasicResidualBlock(hidden_channels, activation, norm_groups),
            BasicResidualBlock(hidden_channels, activation, norm_groups),
        )

    def forward(
        self,
        decoder_feature: Tensor,
        current: Mapping[str, Tensor],
        observed_stokes: tuple[Tensor, Tensor, Tensor],
        *,
        step_index: int = 0,
    ) -> Tensor:
        feature = self.feature_projection(decoder_feature)
        cues = self.cue_projection(
            self.cue_builder(
                decoder_feature,
                current,
                observed_stokes,
                step_index=step_index,
            )
        )
        return self.body(torch.cat((feature, cues), dim=1))


class _ResidualStep(nn.Module):
    """One explicit signed-Stokes correction step."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        lowpass_kernel: int,
        activation: str,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.stem = _SharedCueStem(
            feature_channels,
            hidden_channels,
            lowpass_kernel,
            activation,
            norm_groups,
        )
        self.delta_head = nn.Conv2d(hidden_channels, 3, 3, padding=1)
        self.confidence_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        _initialize_delta_head(self.delta_head)
        _initialize_confidence_head(self.confidence_head, probability=0.1)

    def forward(
        self,
        decoder_feature: Tensor,
        current: Mapping[str, Tensor],
        observed_stokes: tuple[Tensor, Tensor, Tensor],
        *,
        step_index: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.stem(
            decoder_feature,
            current,
            observed_stokes,
            step_index=step_index,
        )
        delta = torch.tanh(self.delta_head(hidden))
        confidence = torch.sigmoid(self.confidence_head(hidden))
        correction = apply_stokes_confidence(
            delta, confidence[:, 0:1], confidence[:, 1:2]
        )
        return correction, confidence[:, 0:1], confidence[:, 1:2]


class PolarGuidedStokesRefiner(nn.Module):
    """Polarization-guided refinement of a coarse Stokes field."""

    def __init__(
        self,
        feature_channels: int,
        mode: str,
        hidden_channels: int = 32,
        residual_scales: Sequence[float] = (0.25, 0.25, 0.25),
        lowpass_kernel: int = 15,
        iterations: int = 1,
        detach_conditioning: bool = False,
        activation: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in PGDE_MODES or normalized_mode == "none":
            raise ValueError(
                "PolarGuidedStokesRefiner mode must be one of "
                f"{PGDE_MODES[1:]}"
            )
        if hidden_channels <= 0 or hidden_channels % norm_groups != 0:
            raise ValueError("hidden_channels must be positive and divisible by norm_groups")
        scales = tuple(float(value) for value in residual_scales)
        if len(scales) != 3 or any(value <= 0.0 for value in scales):
            raise ValueError("residual_scales must contain three positive values")
        if lowpass_kernel <= 0 or lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be a positive odd integer")
        if normalized_mode == "iterative_shared" and int(iterations) != 2:
            raise ValueError("iterative_shared requires exactly two iterations")
        if normalized_mode != "iterative_shared" and int(iterations) != 1:
            raise ValueError("non-iterative PGDE modes require iterations=1")

        self.mode = normalized_mode
        self.lowpass_kernel = int(lowpass_kernel)
        self.iterations = int(iterations)
        self.detach_conditioning = bool(detach_conditioning)
        self.register_buffer(
            "residual_scales",
            torch.tensor(scales, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.mode in {"single_residual", "iterative_shared"}:
            self.shared_step = _ResidualStep(
                feature_channels,
                hidden_channels,
                self.lowpass_kernel,
                activation,
                norm_groups,
            )
        else:
            self.stem = _SharedCueStem(
                feature_channels,
                hidden_channels,
                self.lowpass_kernel,
                activation,
                norm_groups,
            )
            self.low_frequency_expert = nn.Sequential(
                ConvNormAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=5,
                    activation_name=activation,
                    norm_groups=norm_groups,
                ),
                BasicResidualBlock(hidden_channels, activation, norm_groups),
                nn.Conv2d(hidden_channels, 3, 3, padding=1),
            )
            self.high_frequency_expert = nn.Sequential(
                ConvNormAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    activation_name=activation,
                    norm_groups=norm_groups,
                ),
                BasicResidualBlock(hidden_channels, activation, norm_groups),
                nn.Conv2d(hidden_channels, 3, 3, padding=1),
            )
            self.confidence_head = nn.Conv2d(hidden_channels, 4, 3, padding=1)
            _initialize_delta_head(self.low_frequency_expert[-1])
            _initialize_delta_head(self.high_frequency_expert[-1])
            _initialize_confidence_head(self.confidence_head, probability=0.1)

    def forward(
        self,
        decoder_feature: Tensor,
        coarse: Mapping[str, Tensor],
        observed_stokes: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, Tensor | str | dict[str, Tensor | str]]:
        """Refine coarse Stokes while preserving the canonical output contract."""
        coarse_stokes = _mapping_stokes(coarse)
        conditioning_feature = decoder_feature
        conditioning_coarse = coarse
        conditioning_observed = observed_stokes
        if self.training and self.detach_conditioning:
            conditioning_feature = decoder_feature.detach()
            conditioning_coarse = _pack_minimal(coarse_stokes.detach())
            conditioning_observed = tuple(value.detach() for value in observed_stokes)
        if self.mode == "single_residual":
            return self._single(
                conditioning_feature,
                conditioning_coarse,
                coarse_stokes,
                conditioning_observed,
            )
        if self.mode == "dual_frequency":
            return self._dual_frequency(
                conditioning_feature,
                conditioning_coarse,
                coarse_stokes,
                conditioning_observed,
            )
        return self._iterative(
            conditioning_feature,
            coarse_stokes,
            conditioning_observed,
            detach_conditioning=(self.training and self.detach_conditioning),
        )

    def _single(
        self,
        decoder_feature: Tensor,
        coarse: Mapping[str, Tensor],
        coarse_stokes: Tensor,
        observed_stokes: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, Tensor | str | dict[str, Tensor | str]]:
        correction, c0, c12 = self.shared_step(
            decoder_feature, coarse, observed_stokes, step_index=0
        )
        applied = self._scales(correction) * correction
        final_raw = coarse_stokes + applied
        final = _project_stokes(final_raw)
        diagnostics: dict[str, Tensor | str] = {
            "pgde_mode": self.mode,
            "pgde_coarse_s0": coarse_stokes[:, 0:1],
            "pgde_coarse_s12": coarse_stokes[:, 1:3],
            "pgde_correction": correction,
            "pgde_applied_correction": applied,
            "pgde_confidence_s0": c0,
            "pgde_confidence_s12": c12,
        }
        return self._result(final_raw, final, coarse_stokes, c0, c12, diagnostics)

    def _dual_frequency(
        self,
        decoder_feature: Tensor,
        coarse: Mapping[str, Tensor],
        coarse_stokes: Tensor,
        observed_stokes: tuple[Tensor, Tensor, Tensor],
    ) -> dict[str, Tensor | str | dict[str, Tensor | str]]:
        hidden = self.stem(decoder_feature, coarse, observed_stokes, step_index=0)
        low_raw = torch.tanh(self.low_frequency_expert(hidden))
        high_raw = torch.tanh(self.high_frequency_expert(hidden))
        low_delta = fixed_lowpass(low_raw, self.lowpass_kernel)
        high_delta = high_raw - fixed_lowpass(high_raw, self.lowpass_kernel)
        confidence = torch.sigmoid(self.confidence_head(hidden))
        c_lf_s0, c_lf_s12, c_hf_s0, c_hf_s12 = torch.chunk(
            confidence, 4, dim=1
        )
        low_correction = apply_stokes_confidence(
            low_delta, c_lf_s0, c_lf_s12
        )
        high_correction = apply_stokes_confidence(
            high_delta, c_hf_s0, c_hf_s12
        )
        scales = self._scales(low_correction)
        low_residual = scales * low_correction
        high_residual = scales * high_correction
        final_raw = coarse_stokes + low_residual + high_residual
        final = _project_stokes(final_raw)
        c0 = 0.5 * (c_lf_s0 + c_hf_s0)
        c12 = 0.5 * (c_lf_s12 + c_hf_s12)
        diagnostics: dict[str, Tensor | str] = {
            "pgde_mode": self.mode,
            "pgde_coarse_s0": coarse_stokes[:, 0:1],
            "pgde_coarse_s12": coarse_stokes[:, 1:3],
            "pgde_low_correction": low_correction,
            "pgde_high_correction": high_correction,
            "pgde_low_delta": low_delta,
            "pgde_high_delta": high_delta,
            "pgde_low_residual": low_residual,
            "pgde_high_residual": high_residual,
            "pgde_low_confidence_s0": c_lf_s0,
            "pgde_low_confidence_s12": c_lf_s12,
            "pgde_high_confidence_s0": c_hf_s0,
            "pgde_high_confidence_s12": c_hf_s12,
            "pgde_confidence_s0": c0,
            "pgde_confidence_s12": c12,
        }
        return self._result(final_raw, final, coarse_stokes, c0, c12, diagnostics)

    def _iterative(
        self,
        decoder_feature: Tensor,
        coarse_stokes: Tensor,
        observed_stokes: tuple[Tensor, Tensor, Tensor],
        *,
        detach_conditioning: bool,
    ) -> dict[str, Tensor | str | dict[str, Tensor | str]]:
        coarse_mapping = _pack_minimal(
            coarse_stokes.detach() if detach_conditioning else coarse_stokes
        )
        correction_1, c0_1, c12_1 = self.shared_step(
            decoder_feature, coarse_mapping, observed_stokes, step_index=0
        )
        raw_1 = coarse_stokes + self._scales(correction_1) * correction_1
        stokes_1 = _project_stokes(raw_1)
        step1_mapping = _pack_minimal(
            stokes_1.detach() if detach_conditioning else stokes_1
        )
        correction_2, c0_2, c12_2 = self.shared_step(
            decoder_feature, step1_mapping, observed_stokes, step_index=1
        )
        final_raw = stokes_1 + self._scales(correction_2) * correction_2
        final = _project_stokes(final_raw)
        diagnostics: dict[str, Tensor | str] = {
            "pgde_mode": self.mode,
            "pgde_coarse_s0": coarse_stokes[:, 0:1],
            "pgde_coarse_s12": coarse_stokes[:, 1:3],
            "pgde_step1_s0": stokes_1[:, 0:1],
            "pgde_step1_s12": stokes_1[:, 1:3],
            "pgde_step1_correction": correction_1,
            "pgde_step2_correction": correction_2,
            "pgde_confidence_step1": torch.cat((c0_1, c12_1), dim=1),
            "pgde_confidence_step2": torch.cat((c0_2, c12_2), dim=1),
            "pgde_confidence_s0": c0_2,
            "pgde_confidence_s12": c12_2,
        }
        return self._result(
            final_raw, final, coarse_stokes, c0_2, c12_2, diagnostics
        )

    def _scales(self, reference: Tensor) -> Tensor:
        return self.residual_scales.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _result(
        final_raw: Tensor,
        final: Tensor,
        coarse: Tensor,
        confidence_s0: Tensor,
        confidence_s12: Tensor,
        diagnostics: dict[str, Tensor | str],
    ) -> dict[str, Tensor | str | dict[str, Tensor | str]]:
        return {
            "final_stokes_raw": final_raw,
            "final_stokes": final,
            "correction": final - coarse,
            "confidence_s0": confidence_s0,
            "confidence_s12": confidence_s12,
            "diagnostics": diagnostics,
        }


def apply_stokes_confidence(
    delta: Tensor,
    confidence_s0: Tensor,
    confidence_s12: Tensor,
) -> Tensor:
    """Apply separate scalar gates to S0 and the joint S1/S2 correction."""
    if delta.ndim != 4 or delta.shape[1] != 3:
        raise ValueError("delta must have shape Bx3xHxW")
    expected = (delta.shape[0], 1, delta.shape[2], delta.shape[3])
    if confidence_s0.shape != expected or confidence_s12.shape != expected:
        raise ValueError("PGDE confidence tensors must have shape Bx1xHxW")
    return torch.cat(
        (
            confidence_s0 * delta[:, 0:1],
            confidence_s12 * delta[:, 1:2],
            confidence_s12 * delta[:, 2:3],
        ),
        dim=1,
    )


def _initialize_delta_head(head: nn.Module) -> None:
    if not isinstance(head, nn.Conv2d):
        raise TypeError("PGDE delta head must be a Conv2d")
    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def _initialize_confidence_head(head: nn.Conv2d, probability: float) -> None:
    nn.init.zeros_(head.weight)
    bias = math.log(float(probability) / (1.0 - float(probability)))
    if head.bias is None:
        raise ValueError("PGDE confidence head requires a bias")
    nn.init.constant_(head.bias, bias)


def _mapping_stokes(values: Mapping[str, Tensor]) -> Tensor:
    tensors = tuple(values.get(key) for key in ("s0", "s1", "s2"))
    if not all(isinstance(value, Tensor) for value in tensors):
        raise KeyError("PGDE coarse mapping requires tensor s0, s1, and s2")
    return _tuple_stokes(tensors)  # type: ignore[arg-type]


def _tuple_stokes(values: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    if len(values) != 3 or any(value.ndim != 4 for value in values):
        raise ValueError("Stokes input must contain three BCHW tensors")
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1 or values[0].shape[1] != 1:
        raise ValueError("Stokes components must be matching single-channel tensors")
    return torch.cat(values, dim=1)


def _project_stokes(stokes: Tensor) -> Tensor:
    if stokes.ndim != 4 or stokes.shape[1] != 3:
        raise ValueError("Stokes field must have shape Bx3xHxW")
    s0, s1, s2 = safe_clip_stokes(
        stokes[:, 0:1], stokes[:, 1:2], stokes[:, 2:3]
    )
    return torch.cat((s0, s1, s2), dim=1)


def _pack_minimal(stokes: Tensor) -> dict[str, Tensor]:
    return {
        "s0": stokes[:, 0:1],
        "s1": stokes[:, 1:2],
        "s2": stokes[:, 2:3],
    }


__all__ = [
    "PGDE_MODES",
    "PGDECueBuilder",
    "PolarGuidedStokesRefiner",
    "apply_stokes_confidence",
    "fixed_lowpass",
]
