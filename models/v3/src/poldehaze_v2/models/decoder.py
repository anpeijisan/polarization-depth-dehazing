"""Bottleneck and skip-connected decoder components."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from poldehaze_v2.models.boundary import DoLPBoundaryDynamicSampler
from poldehaze_v2.models.common import BasicResidualBlock, ConvNormAct
from poldehaze_v2.models.encoders import (
    PhysEncoderBlock,
    apply_raw_refinement,
    build_raw_refinement,
)


class Bottleneck(nn.Module):
    """Alternating LPM and configured Detail-family refinements."""

    def __init__(
        self,
        channels: int,
        detail_type: str,
        use_lpm: bool = True,
        num_pairs: int = 1,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        blocks: list[nn.Module] = []
        for _ in range(num_pairs):
            blocks.append(
                PhysEncoderBlock(
                    channels, use_lpm, activation_name, norm_groups
                )
            )
            blocks.append(
                build_raw_refinement(
                    detail_type, channels, activation_name, norm_groups
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(
        self,
        inputs: Tensor,
        polar_context: Tensor | None = None,
    ) -> Tensor:
        features = inputs
        for block in self.blocks:
            if bool(getattr(block, "requires_polar_context", False)):
                features = apply_raw_refinement(block, features, polar_context)
            else:
                features = block(features)
        return features


class DecoderUpBlock(nn.Module):
    """Bilinear upsample, skip merge, LPM, and Detail refinement."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        detail_type: str,
        use_lpm: bool = True,
        use_boundary_dynamic_upsample: bool = False,
        boundary_max_offset: float = 1.0,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.merge = ConvNormAct(
            out_channels + skip_channels,
            out_channels,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.lpm = PhysEncoderBlock(
            out_channels, use_lpm, activation_name, norm_groups
        )
        self.detail = build_raw_refinement(
            detail_type, out_channels, activation_name, norm_groups
        )
        self.dynamic_upsampler = (
            DoLPBoundaryDynamicSampler(
                max_offset=boundary_max_offset,
                activation_name=activation_name,
            )
            if use_boundary_dynamic_upsample
            else None
        )

    def forward(
        self,
        inputs: Tensor,
        skip: Tensor,
        polar_context: Tensor | None = None,
    ) -> Tensor:
        upsampled = functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        projected = self.projection(upsampled)
        if self.dynamic_upsampler is not None:
            if polar_context is None:
                raise ValueError(
                    "Boundary dynamic upsampling requires polar_context"
                )
            projected = self.dynamic_upsampler(projected, skip, polar_context)
        merged = self.merge(torch.cat((projected, skip), dim=1))
        return apply_raw_refinement(
            self.detail,
            self.lpm(merged),
            polar_context,
        )


class DisagreementResidualRefiner(nn.Module):
    """Decode a three-level expert-disagreement pyramid into Stokes residuals.

    Args:
        channels: Shallow-to-deep feature widths for exactly three scales.
        context_channels: Number of full-resolution polarization context maps.
        residual_scale: Maximum absolute correction for ``S0/S1/S2``.

    Inputs and outputs use BCHW tensors. The final convolution is initialized
    to zero, so enabling this module leaves an existing restoration checkpoint
    exactly unchanged before optimization.
    """

    def __init__(
        self,
        channels: Sequence[int],
        context_channels: int,
        residual_scale: Sequence[float],
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        scales = tuple(float(value) for value in residual_scale)
        if len(widths) != 3:
            raise ValueError("Disagreement refiner requires exactly three scales")
        if len(scales) != 3 or any(value <= 0.0 for value in scales):
            raise ValueError("residual_scale must contain three positive values")
        shallow, middle, deep = widths
        self.deep_projection = ConvNormAct(
            deep,
            middle,
            kernel_size=1,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.middle_merge = ConvNormAct(
            middle * 2,
            middle,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.middle_refine = BasicResidualBlock(
            middle, activation_name, norm_groups
        )
        self.shallow_projection = ConvNormAct(
            middle,
            shallow,
            kernel_size=1,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.shallow_merge = ConvNormAct(
            shallow * 3 + int(context_channels),
            shallow,
            activation_name=activation_name,
            norm_groups=norm_groups,
        )
        self.shallow_refine = nn.Sequential(
            BasicResidualBlock(shallow, activation_name, norm_groups),
            BasicResidualBlock(shallow, activation_name, norm_groups),
        )
        self.output = nn.Sequential(
            ConvNormAct(
                shallow,
                shallow,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(shallow, 6, 1, bias=True),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer(
            "residual_scale",
            torch.tensor(scales).view(1, 3, 1, 1),
            persistent=True,
        )

    def forward(
        self,
        base_feature: Tensor,
        disagreements: Sequence[Tensor],
        polar_context: Tensor,
    ) -> dict[str, Tensor]:
        if len(disagreements) != 3:
            raise ValueError("Expected three disagreement feature tensors")
        shallow, middle, deep = disagreements
        if base_feature.shape != shallow.shape:
            raise ValueError("Base decoder and shallow disagreement shapes must match")
        if polar_context.ndim != 4 or polar_context.shape[0] != shallow.shape[0]:
            raise ValueError("polar_context must be a batch-aligned BCHW tensor")
        if polar_context.shape[-2:] != shallow.shape[-2:]:
            raise ValueError("polar_context must match the shallow spatial shape")

        decoded = self.deep_projection(deep)
        decoded = functional.interpolate(
            decoded,
            size=middle.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.middle_refine(
            self.middle_merge(torch.cat((decoded, middle), dim=1))
        )
        decoded = functional.interpolate(
            decoded,
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.shallow_projection(decoded)
        decoded = self.shallow_refine(
            self.shallow_merge(
                torch.cat(
                    (decoded, shallow, base_feature, polar_context), dim=1
                )
            )
        )
        raw = self.output(decoded)
        delta = torch.tanh(raw[:, :3]) * self.residual_scale.to(raw.dtype)
        confidence = torch.sigmoid(raw[:, 3:])
        return {
            "residual": delta * confidence,
            "confidence": confidence,
            "feature": decoded,
        }


class OutputPatchExpertRouter(nn.Module):
    """Decode expert evidence into two Stokes candidates and a patch route.

    SpatialGate remains the anchor candidate while PRGF supplies a separately
    supervised complementary candidate.  The route is predicted at the same
    patch granularity used by the F2/F4 oracle analysis and is then expanded
    without smoothing, so one route value corresponds to one output patch.

    Args:
        channels: Shallow-to-deep feature widths for exactly three scales.
        context_channels: Number of observed polarization context fields.
        residual_scale: Maximum absolute corrections for ``S0/S1/S2``.
        patch_size: Full-resolution routing patch side length.
    """

    def __init__(
        self,
        channels: Sequence[int],
        context_channels: int,
        residual_scale: Sequence[float],
        patch_size: int = 8,
        activation_name: str = "gelu",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        scales = tuple(float(value) for value in residual_scale)
        if len(widths) != 3:
            raise ValueError("Output expert router requires exactly three scales")
        if len(scales) != 3 or any(value <= 0.0 for value in scales):
            raise ValueError("residual_scale must contain three positive values")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        latent_channels = 8
        shallow = widths[0]
        self.patch_size = int(patch_size)
        self.disagreement_projections = nn.ModuleList(
            ConvNormAct(
                width,
                latent_channels,
                kernel_size=1,
                activation_name=activation_name,
                norm_groups=latent_channels,
            )
            for width in widths
        )
        candidate_channels = (
            shallow + latent_channels * 2 + int(context_channels)
        )
        self.candidate_body = nn.Sequential(
            ConvNormAct(
                candidate_channels,
                shallow,
                activation_name=activation_name,
                norm_groups=norm_groups,
            ),
            BasicResidualBlock(shallow, activation_name, norm_groups),
        )
        self.candidate_output = nn.Conv2d(shallow, 6, 1, bias=True)
        nn.init.zeros_(self.candidate_output.weight)
        nn.init.zeros_(self.candidate_output.bias)

        # Context (4), candidate difference (3), decoded evidence (2), and
        # decoder activity (2) form the inference-time routing evidence.
        route_channels = int(context_channels) + 7
        route_hidden = 8
        self.route_body = nn.Sequential(
            nn.Conv2d(route_channels, route_hidden, 3, padding=1, bias=False),
            nn.GroupNorm(route_hidden, route_hidden),
            nn.GELU(),
            nn.Conv2d(route_hidden, route_hidden, 3, padding=1, bias=False),
            nn.GroupNorm(route_hidden, route_hidden),
            nn.GELU(),
        )
        self.route_output = nn.Conv2d(route_hidden, 1, 1, bias=True)
        nn.init.zeros_(self.route_output.weight)
        # The established fixed mixture starts with 0.48 assigned to PRGF.
        nn.init.constant_(self.route_output.bias, -0.0800427)
        self.register_buffer(
            "residual_scale",
            torch.tensor(scales).view(1, 3, 1, 1),
            persistent=True,
        )

    def forward(
        self,
        decoder_feature: Tensor,
        anchor_features: Sequence[Tensor],
        complementary_features: Sequence[Tensor],
        polar_context: Tensor,
    ) -> dict[str, Tensor]:
        if len(anchor_features) != 3 or len(complementary_features) != 3:
            raise ValueError("Expected three feature levels from both experts")
        if polar_context.ndim != 4 or polar_context.shape[0] != decoder_feature.shape[0]:
            raise ValueError("polar_context must be a batch-aligned BCHW tensor")
        if polar_context.shape[-2:] != decoder_feature.shape[-2:]:
            raise ValueError("polar_context must match decoder spatial shape")

        projected: list[Tensor] = []
        for projection, anchor, complementary in zip(
            self.disagreement_projections,
            anchor_features,
            complementary_features,
        ):
            if anchor.shape != complementary.shape:
                raise ValueError("Paired expert features must have identical shapes")
            projected.append(projection(complementary - anchor))
        shallow, middle, deep = projected
        decoded = deep
        decoded = functional.interpolate(
            decoded, size=middle.shape[-2:], mode="bilinear", align_corners=False
        ) + middle
        decoded = functional.interpolate(
            decoded, size=shallow.shape[-2:], mode="bilinear", align_corners=False
        ) + shallow
        if decoded.shape[-2:] != decoder_feature.shape[-2:]:
            decoded = functional.interpolate(
                decoded,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        candidate_feature = self.candidate_body(
            torch.cat(
                (decoder_feature, decoded, decoded.abs(), polar_context), dim=1
            )
        )
        raw_residuals = self.candidate_output(candidate_feature)
        anchor_residual, complementary_residual = torch.chunk(
            torch.tanh(raw_residuals), 2, dim=1
        )
        scale = self.residual_scale.to(raw_residuals.dtype)
        anchor_residual = anchor_residual * scale
        complementary_residual = complementary_residual * scale

        residual_difference = (complementary_residual - anchor_residual).abs()
        route_input = torch.cat(
            (
                polar_context,
                residual_difference,
                decoded.mean(dim=1, keepdim=True),
                decoded.abs().mean(dim=1, keepdim=True),
                decoder_feature.mean(dim=1, keepdim=True),
                decoder_feature.abs().mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        logits = self.route_output(self.route_body(route_input))
        if min(logits.shape[-2:]) > 1:
            kernel = min(self.patch_size, *logits.shape[-2:])
            logits = functional.avg_pool2d(
                logits, kernel_size=kernel, stride=kernel, ceil_mode=True
            )
            logits = functional.interpolate(
                logits, size=decoder_feature.shape[-2:], mode="nearest"
            )
        return {
            "anchor_residual": anchor_residual,
            "complementary_residual": complementary_residual,
            "complementary_weight": torch.sigmoid(logits),
            "feature": candidate_feature,
        }


class StokesOutputPatchRouter(nn.Module):
    """Route two genuine shared-weight decoder predictions patch by patch."""

    def __init__(
        self,
        decoder_channels: int,
        context_channels: int,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        # Context (4), absolute Stokes disagreement (3), candidate S0 (2),
        # and decoder disagreement summaries (2).
        route_channels = int(context_channels) + 7
        hidden = 12
        self.decoder_difference = nn.Conv2d(
            int(decoder_channels), hidden, 1, bias=False
        )
        self.route_body = nn.Sequential(
            nn.Conv2d(route_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
        )
        self.route_output = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.route_output.weight)
        nn.init.constant_(self.route_output.bias, -0.0800427)

    def forward(
        self,
        anchor_stokes: Tensor,
        complementary_stokes: Tensor,
        anchor_decoder: Tensor,
        complementary_decoder: Tensor,
        polar_context: Tensor,
    ) -> Tensor:
        if anchor_stokes.shape != complementary_stokes.shape:
            raise ValueError("Output expert Stokes candidates must match")
        if anchor_stokes.ndim != 4 or anchor_stokes.shape[1] != 3:
            raise ValueError("Output expert Stokes candidates must be Bx3xHxW")
        if anchor_decoder.shape != complementary_decoder.shape:
            raise ValueError("Output expert decoder features must match")
        if polar_context.shape[-2:] != anchor_stokes.shape[-2:]:
            raise ValueError("polar_context must match output spatial shape")
        decoded_difference = self.decoder_difference(
            complementary_decoder - anchor_decoder
        )
        route_input = torch.cat(
            (
                polar_context,
                (complementary_stokes - anchor_stokes).abs(),
                anchor_stokes[:, :1],
                complementary_stokes[:, :1],
                decoded_difference.mean(dim=1, keepdim=True),
                decoded_difference.abs().mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        logits = self.route_output(self.route_body(route_input))
        if min(logits.shape[-2:]) > 1:
            kernel = min(self.patch_size, *logits.shape[-2:])
            logits = functional.avg_pool2d(
                logits, kernel_size=kernel, stride=kernel, ceil_mode=True
            )
            logits = functional.interpolate(
                logits, size=anchor_stokes.shape[-2:], mode="nearest"
            )
        return torch.sigmoid(logits)
