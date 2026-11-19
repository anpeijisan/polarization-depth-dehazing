"""The single configurable PolarDehazeNet architecture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from poldehaze_v2.models.attention import (
    ATTENTION_POSITIONS,
    ATTENTION_TYPES,
    build_decoder_attention,
)
from poldehaze_v2.models.common import replace_group_norm
from poldehaze_v2.models.boundary import (
    BOUNDARY_GUIDANCE_POSITIONS,
    BOUNDARY_GUIDANCE_TYPES,
    build_boundary_guidance,
)
from poldehaze_v2.models.decoder import (
    Bottleneck,
    DecoderUpBlock,
    DisagreementResidualRefiner,
    OutputPatchExpertRouter,
    StokesOutputPatchRouter,
)
from poldehaze_v2.models.encoders import (
    DownBlock,
    PhysEncoderBlock,
    RawEncoderBlock,
    Stem,
)
from poldehaze_v2.models.fusion import build_fusion
from poldehaze_v2.models.fusion.spatial_gate import (
    CONTEXT_KEYS,
    stack_polar_context,
)
from poldehaze_v2.models.heads import (
    HeadAirlightSpatial,
    HeadBeta,
    HeadDepth,
    HeadPolar,
    HeadResidualT,
    HeadS0,
    HeadStokesResidual,
    PhysicsFeatureAdapter,
)
from poldehaze_v2.models.hard_region import (
    DHDE_GATE_TYPES,
    DHDE_POSITIONS,
    DoLPGuidedDehazeEnhancement,
)
from poldehaze_v2.models.independent_stream import IndependentRestorationStream
from poldehaze_v2.models.pgde import PGDE_MODES, PolarGuidedStokesRefiner
from poldehaze_v2.optimization.grad_scale import grad_scale
from poldehaze_v2.physics.stokes import (
    STOKES_S0_MAX,
    angles_to_stokes,
    safe_clip_stokes,
    stokes_to_angles,
    stokes_to_dolp_aop,
    stokes_to_polar_features,
)


class PolarDehazeNet(nn.Module):
    """Configurable network for direct clear full-Stokes restoration.

    ``stream_mode="dual"`` uses raw-angle and derived-physics
    encoders with learned inter-stream fusion. ``stream_mode="raw_only"`` is
    the controlled single-encoder comparison: it never constructs or executes
    the physics encoder or fusion modules. Polarization context used by the
    shared decoder is still derived deterministically from the four raw angle
    images, so both modes solve the same full-Stokes restoration task.

    The physical diagnosis branch is retained only for historical-checkpoint
    compatibility and is disabled in the paper restoration task.
    """

    def __init__(
        self,
        *,
        raw_in_channels: int = 4,
        phys_in_channels: int = 3,
        num_encoder_levels: int = 3,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        stream_mode: str = "dual",
        fusion_type: str = "simple",
        fusion_types: Sequence[str] | None = None,
        fusion_shared_weights: bool = False,
        fusion_feedback_to_raw: bool = False,
        detail_type: str = "legacy",
        raw_detail_type: str | None = None,
        post_fusion_detail_type: str | None = None,
        bottleneck_detail_type: str | None = None,
        decoder_detail_type: str | None = None,
        raw_detail_types: Sequence[str] | None = None,
        post_fusion_detail_types: Sequence[str] | None = None,
        decoder_detail_types: Sequence[str] | None = None,
        use_lpm: bool = True,
        dhde_positions: Sequence[str] | None = None,
        dhde_residual_scale: float = 0.25,
        dhde_gate_type: str = "legacy",
        dhde_gate_supervision: bool = False,
        dhde_gate_initial_probability: float = 0.1,
        pgde: Mapping[str, Any] | None = None,
        use_global_attn: bool = False,
        global_attention_type: str | None = None,
        global_attention_positions: Sequence[str] | None = None,
        boundary_guidance_type: str = "none",
        boundary_guidance_positions: Sequence[str] | None = None,
        boundary_residual_scale: float = 0.25,
        boundary_dynamic_upsample: bool = False,
        boundary_max_offset: float = 1.0,
        output_mode: str = "absolute",
        stokes_residual_scales: Sequence[float] = (2.0, 2.0, 2.0),
        deep_supervision: bool = False,
        use_disagreement_refiner: bool = False,
        disagreement_residual_scale: Sequence[float] = (0.25, 0.15, 0.15),
        use_output_expert_router: bool = False,
        output_expert_mode: str = "residual",
        output_expert_residual_scale: Sequence[float] = (0.25, 0.15, 0.15),
        output_expert_patch_size: int = 8,
        bottleneck_pairs: int = 1,
        use_physics_head: bool = False,
        use_physics_adapter: bool = False,
        decouple_phys_grad: bool = False,
        physics_gradient_scale: float | None = None,
        beta_min: float = 0.05,
        beta_max: float = 3.0,
        use_residual_t: bool = False,
        residual_t_min: float = 0.1,
        residual_t_max: float = 3.0,
        transmission_min: float = 0.05,
        return_diagnostics: bool = False,
        activation: str = "gelu",
        norm_groups: int = 8,
        deployment_norm_policy: str = "group",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if num_encoder_levels != 3:
            raise ValueError("The specified architecture currently requires 3 encoder levels")
        if len(channel_multipliers) != num_encoder_levels:
            raise ValueError("channel_multipliers must match num_encoder_levels")
        fusion_schedule = (
            (str(fusion_type),) * num_encoder_levels
            if fusion_types is None
            else tuple(str(value).strip().lower() for value in fusion_types)
        )
        if len(fusion_schedule) != num_encoder_levels:
            raise ValueError(
                "fusion_types must contain exactly one entry per encoder level"
            )
        self.fusion_types = fusion_schedule
        if fusion_shared_weights:
            raise NotImplementedError(
                "Cross-scale fusion weight sharing is undefined for unequal channel widths"
            )
        if not 0.0 < transmission_min < 1.0:
            raise ValueError("transmission_min must lie in (0,1)")
        self.stream_mode = str(stream_mode).strip().lower()
        supported_stream_modes = {
            "dual", "raw_only", "polar_only", "dual_independent"
        }
        if self.stream_mode not in supported_stream_modes:
            raise ValueError(
                f"stream_mode must be one of {sorted(supported_stream_modes)}"
            )
        if self.stream_mode != "dual":
            if any(name != "none" for name in fusion_schedule):
                raise ValueError(
                    "Non-fused stream modes require every fusion type to be 'none'"
                )
            if fusion_feedback_to_raw:
                raise ValueError(
                    "Non-fused stream modes do not support fusion feedback"
                )
            if use_disagreement_refiner or use_output_expert_router:
                raise ValueError(
                    "Non-fused stream modes do not support fusion expert refiners"
                )
            if use_physics_head:
                raise ValueError(
                    "Single/independent stream controls cannot enable the physics head"
                )

        normalized_dhde_positions = tuple(
            dict.fromkeys(
                str(position).strip().lower()
                for position in (dhde_positions or ())
            )
        )
        invalid_dhde_positions = tuple(
            position
            for position in normalized_dhde_positions
            if position not in DHDE_POSITIONS
        )
        if invalid_dhde_positions:
            raise ValueError(
                "dhde_positions must contain only "
                f"{DHDE_POSITIONS}, got {invalid_dhde_positions}"
            )
        if "decoder" in normalized_dhde_positions and any(
            position in normalized_dhde_positions
            for position in ("decoder_first", "decoder_final")
        ):
            raise ValueError(
                "Do not combine 'decoder' with decoder_first/decoder_final"
            )
        if float(dhde_residual_scale) <= 0.0:
            raise ValueError("dhde_residual_scale must be positive")
        normalized_dhde_gate_type = str(dhde_gate_type).strip().lower()
        if normalized_dhde_gate_type not in DHDE_GATE_TYPES:
            raise ValueError(
                "dhde_gate_type must be one of "
                f"{DHDE_GATE_TYPES}, got {dhde_gate_type!r}"
            )
        if not 0.0 < float(dhde_gate_initial_probability) < 1.0:
            raise ValueError("dhde_gate_initial_probability must lie in (0,1)")
        if dhde_gate_supervision and (
            normalized_dhde_gate_type != "residual_supervised"
            or normalized_dhde_positions != ("decoder_final",)
        ):
            raise ValueError(
                "dhde_gate_supervision requires dhde_gate_type="
                "'residual_supervised' and dhde_positions=[decoder_final]"
            )
        self.dhde_positions = normalized_dhde_positions
        self.dhde_gate_type = normalized_dhde_gate_type
        self.dhde_gate_supervision = bool(dhde_gate_supervision)
        self.dhde_gate_initial_probability = float(
            dhde_gate_initial_probability
        )
        pgde_config = dict(pgde or {})
        allowed_pgde_keys = {
            "mode",
            "hidden_channels",
            "residual_scales",
            "lowpass_kernel",
            "iterations",
            "detach_conditioning",
        }
        unexpected_pgde_keys = sorted(set(pgde_config) - allowed_pgde_keys)
        if unexpected_pgde_keys:
            raise ValueError(
                f"Unsupported model.pgde keys: {unexpected_pgde_keys}"
            )
        self.pgde_mode = str(pgde_config.get("mode", "none")).strip().lower()
        if self.pgde_mode not in PGDE_MODES:
            raise ValueError(
                f"model.pgde.mode must be one of {PGDE_MODES}, got {self.pgde_mode!r}"
            )
        if self.pgde_mode != "none" and self.dhde_positions:
            raise ValueError(
                "Legacy feature DHDE and output-domain PGDE cannot be enabled together"
            )
        self.pgde_hidden_channels = int(pgde_config.get("hidden_channels", 32))
        self.pgde_residual_scales = tuple(
            float(value)
            for value in pgde_config.get(
                "residual_scales", (0.25, 0.25, 0.25)
            )
        )
        self.pgde_lowpass_kernel = int(pgde_config.get("lowpass_kernel", 15))
        self.pgde_iterations = int(pgde_config.get("iterations", 1))
        self.pgde_detach_conditioning = bool(
            pgde_config.get("detach_conditioning", False)
        )
        self.pgde_output_enabled = True

        self.raw_in_channels = int(raw_in_channels)
        self.phys_in_channels = int(phys_in_channels)
        self.use_physics_head = bool(use_physics_head)
        self.use_physics_adapter = bool(use_physics_adapter)
        if not self.use_physics_head and (
            self.use_physics_adapter or use_residual_t or decouple_phys_grad
        ):
            raise ValueError(
                "Physics adapter, residual transmission, and gradient decoupling "
                "require use_physics_head=true"
            )
        self.fusion_feedback_to_raw = bool(fusion_feedback_to_raw)
        if global_attention_type is None:
            global_attention_type = "dolp" if use_global_attn else "none"
        self.global_attention_type = str(global_attention_type).strip().lower()
        if self.global_attention_type not in ATTENTION_TYPES:
            raise ValueError(
                "global_attention_type must be one of "
                f"{ATTENTION_TYPES}, got {global_attention_type!r}"
            )
        self.use_global_attn = self.global_attention_type != "none"
        if global_attention_positions is None:
            global_attention_positions = (
                ("decoder_first",) if self.use_global_attn else ()
            )
        normalized_positions = tuple(
            dict.fromkeys(
                str(position).strip().lower()
                for position in global_attention_positions
            )
        )
        invalid_positions = tuple(
            position
            for position in normalized_positions
            if position not in ATTENTION_POSITIONS
        )
        if invalid_positions:
            raise ValueError(
                "global_attention_positions must contain only "
                f"{ATTENTION_POSITIONS}, got {invalid_positions}"
            )
        if self.use_global_attn and not normalized_positions:
            raise ValueError(
                "An enabled global_attention_type requires at least one "
                "global_attention_positions entry"
            )
        if not self.use_global_attn and normalized_positions:
            raise ValueError(
                "global_attention_positions must be empty when attention is disabled"
            )
        if "decoder_first" in normalized_positions and "decoder" in normalized_positions:
            raise ValueError(
                "Use either 'decoder_first' or 'decoder', not both"
            )
        self.global_attention_positions = normalized_positions
        self.boundary_guidance_type = str(boundary_guidance_type).strip().lower()
        if self.boundary_guidance_type not in BOUNDARY_GUIDANCE_TYPES:
            raise ValueError(
                "boundary_guidance_type must be one of "
                f"{BOUNDARY_GUIDANCE_TYPES}, got {boundary_guidance_type!r}"
            )
        self.use_boundary_guidance = self.boundary_guidance_type != "none"
        if boundary_guidance_positions is None:
            boundary_guidance_positions = (
                ("decoder_final",) if self.use_boundary_guidance else ()
            )
        normalized_boundary_positions = tuple(
            dict.fromkeys(
                str(position).strip().lower()
                for position in boundary_guidance_positions
            )
        )
        invalid_boundary_positions = tuple(
            position
            for position in normalized_boundary_positions
            if position not in BOUNDARY_GUIDANCE_POSITIONS
        )
        if invalid_boundary_positions:
            raise ValueError(
                "boundary_guidance_positions must contain only "
                f"{BOUNDARY_GUIDANCE_POSITIONS}, got "
                f"{invalid_boundary_positions}"
            )
        if self.use_boundary_guidance and not normalized_boundary_positions:
            raise ValueError(
                "Enabled boundary guidance requires at least one position"
            )
        if not self.use_boundary_guidance and normalized_boundary_positions:
            raise ValueError(
                "Boundary positions must be empty when guidance is disabled"
            )
        if "decoder" in normalized_boundary_positions and len(
            normalized_boundary_positions
        ) > 1:
            raise ValueError(
                "Use 'decoder' alone instead of combining it with first/final"
            )
        if boundary_dynamic_upsample and (
            self.boundary_guidance_type != "joint_reliable"
        ):
            raise ValueError(
                "Boundary dynamic upsampling requires joint_reliable guidance"
            )
        if float(boundary_residual_scale) <= 0.0:
            raise ValueError("boundary_residual_scale must be positive")
        if float(boundary_max_offset) <= 0.0:
            raise ValueError("boundary_max_offset must be positive")
        self.boundary_guidance_positions = normalized_boundary_positions
        self.boundary_dynamic_upsample = bool(boundary_dynamic_upsample)
        self.output_mode = str(output_mode).strip().lower()
        if self.output_mode not in {"absolute", "hazy_residual"}:
            raise ValueError(
                "output_mode must be 'absolute' or 'hazy_residual'"
            )
        residual_scales = tuple(float(value) for value in stokes_residual_scales)
        if len(residual_scales) != 3 or any(value <= 0.0 for value in residual_scales):
            raise ValueError(
                "stokes_residual_scales must contain three positive values"
            )
        self.register_buffer(
            "stokes_residual_scales",
            torch.tensor(residual_scales, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.deep_supervision = bool(deep_supervision)
        self.use_disagreement_refiner = bool(use_disagreement_refiner)
        self.use_output_expert_router = bool(use_output_expert_router)
        self.return_diagnostics = bool(return_diagnostics)
        self.output_expert_mode = str(output_expert_mode)
        if self.output_expert_mode not in {"residual", "shared_dual_decode"}:
            raise ValueError(
                "output_expert_mode must be 'residual' or 'shared_dual_decode'"
            )
        if (
            self.use_disagreement_refiner
            and any(
                name != "dual_expert_consensus_disagreement"
                for name in fusion_schedule
            )
        ):
            raise ValueError(
                "use_disagreement_refiner requires "
                "fusion_type='dual_expert_consensus_disagreement'"
            )
        if (
            self.use_output_expert_router
            and any(name != "output_expert_anchor" for name in fusion_schedule)
        ):
            raise ValueError(
                "use_output_expert_router requires "
                "fusion_type='output_expert_anchor'"
            )
        if self.use_output_expert_router and self.use_disagreement_refiner:
            raise ValueError(
                "Output expert routing and disagreement refinement are "
                "mutually exclusive fusion candidates"
            )
        self.decouple_phys_grad = bool(decouple_phys_grad)
        if physics_gradient_scale is None:
            physics_gradient_scale = 0.0 if self.decouple_phys_grad else 1.0
        self.set_physics_gradient_scale(physics_gradient_scale)
        self.use_residual_t = bool(use_residual_t)
        self.transmission_min = float(transmission_min)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        raw_detail_type = raw_detail_type or detail_type
        post_fusion_detail_type = post_fusion_detail_type or detail_type
        bottleneck_detail_type = bottleneck_detail_type or detail_type
        decoder_detail_type = decoder_detail_type or detail_type
        raw_detail_schedule = self._detail_schedule(
            raw_detail_types,
            fallback=raw_detail_type,
            count=num_encoder_levels,
            name="raw_detail_types",
        )
        post_fusion_detail_schedule = self._detail_schedule(
            post_fusion_detail_types,
            fallback=post_fusion_detail_type,
            count=num_encoder_levels,
            name="post_fusion_detail_types",
        )
        decoder_detail_schedule = self._detail_schedule(
            decoder_detail_types,
            fallback=decoder_detail_type,
            count=num_encoder_levels - 1,
            name="decoder_detail_types",
        )
        self.detail_types = {
            "raw": raw_detail_type,
            "post_fusion": post_fusion_detail_type,
            "bottleneck": bottleneck_detail_type,
            "decoder": decoder_detail_type,
        }
        self.detail_type_schedules = {
            "raw": raw_detail_schedule,
            "post_fusion": post_fusion_detail_schedule,
            "decoder": decoder_detail_schedule,
        }
        channels = tuple(base_channels * int(multiplier) for multiplier in channel_multipliers)
        if any(width % norm_groups != 0 for width in channels):
            raise ValueError("Every encoder width must be divisible by norm_groups")
        self.channels = channels

        if self.stream_mode in {"polar_only", "dual_independent"}:
            unsupported = {
                "dhde_positions": bool(self.dhde_positions),
                "global_attention": self.use_global_attn,
                "boundary_guidance": self.use_boundary_guidance,
                "deep_supervision": self.deep_supervision,
            }
            enabled = sorted(name for name, value in unsupported.items() if value)
            if enabled:
                raise ValueError(
                    "Independent output streams do not support: " + ", ".join(enabled)
                )
            if self.output_mode != "absolute":
                raise ValueError(
                    "Independent output streams require output_mode='absolute'"
                )
            stream_kwargs = {
                "channels": channels,
                "encoder_detail_types": raw_detail_schedule,
                "post_detail_types": post_fusion_detail_schedule,
                "bottleneck_detail_type": bottleneck_detail_type,
                "decoder_detail_types": decoder_detail_schedule,
                "use_lpm": use_lpm,
                "bottleneck_pairs": bottleneck_pairs,
                "activation": activation,
                "norm_groups": norm_groups,
                "gradient_checkpointing": self.gradient_checkpointing,
            }
            if self.stream_mode == "dual_independent":
                self.independent_raw_stream = IndependentRestorationStream(
                    in_channels=self.raw_in_channels,
                    encoder_kind="raw",
                    **stream_kwargs,
                )
            self.independent_polar_stream = IndependentRestorationStream(
                in_channels=self.phys_in_channels,
                encoder_kind="polar",
                **stream_kwargs,
            )
            if self.pgde_mode != "none":
                self.pgde_refiner = PolarGuidedStokesRefiner(
                    channels[0],
                    self.pgde_mode,
                    hidden_channels=self.pgde_hidden_channels,
                    residual_scales=self.pgde_residual_scales,
                    lowpass_kernel=self.pgde_lowpass_kernel,
                    iterations=self.pgde_iterations,
                    detach_conditioning=self.pgde_detach_conditioning,
                    activation=activation,
                    norm_groups=norm_groups,
                )
            self.deployment_norm_policy = str(deployment_norm_policy).strip().lower()
            self.replaced_group_norms = self._apply_deployment_norm_policy(
                self.deployment_norm_policy
            )
            return

        self.raw_stem = Stem(
            self.raw_in_channels, channels[0], activation, norm_groups
        )
        self.raw_encoders = nn.ModuleList(
            RawEncoderBlock(width, level_detail, activation, norm_groups)
            for width, level_detail in zip(channels, raw_detail_schedule)
        )
        if self.stream_mode == "dual":
            self.phys_stem = Stem(
                self.phys_in_channels, channels[0], activation, norm_groups
            )
            self.phys_encoders = nn.ModuleList(
                PhysEncoderBlock(width, use_lpm, activation, norm_groups)
                for width in channels
            )
            self.fusions = nn.ModuleList(
                build_fusion(name, width, activation, norm_groups)
                for width, name in zip(channels, fusion_schedule)
            )
        else:
            self.phys_encoders = nn.ModuleList()
            self.fusions = nn.ModuleList()
        self.post_fusion = nn.ModuleList(
            RawEncoderBlock(
                width, level_detail, activation, norm_groups
            )
            for width, level_detail in zip(
                channels, post_fusion_detail_schedule
            )
        )
        self.raw_down = nn.ModuleList(
            DownBlock(channels[index], channels[index + 1], activation, norm_groups)
            for index in range(num_encoder_levels - 1)
        )
        self.phys_down = nn.ModuleList(
            DownBlock(channels[index], channels[index + 1], activation, norm_groups)
            for index in range(num_encoder_levels - 1)
        ) if self.stream_mode == "dual" else nn.ModuleList()

        self.bottleneck = Bottleneck(
            channels[-1], bottleneck_detail_type, use_lpm, bottleneck_pairs,
            activation, norm_groups,
        )
        decoder_specs = tuple(reversed(range(num_encoder_levels - 1)))
        self.decoder = nn.ModuleList(
            DecoderUpBlock(
                channels[index + 1],
                channels[index],
                channels[index],
                level_detail,
                use_lpm,
                self.boundary_dynamic_upsample,
                boundary_max_offset,
                activation,
                norm_groups,
            )
            for index, level_detail in zip(
                decoder_specs, decoder_detail_schedule
            )
        )
        decoder_widths = tuple(reversed(channels[:-1]))
        if "post_fusion" in self.dhde_positions:
            self.post_fusion_dhde = nn.ModuleList(
                DoLPGuidedDehazeEnhancement(
                    width,
                    residual_scale=dhde_residual_scale,
                    activation=activation,
                    norm_groups=norm_groups,
                    gate_type=self.dhde_gate_type,
                    gate_initial_probability=self.dhde_gate_initial_probability,
                )
                for width in channels
            )
        if "bottleneck" in self.dhde_positions:
            self.bottleneck_dhde = DoLPGuidedDehazeEnhancement(
                channels[-1],
                residual_scale=dhde_residual_scale,
                activation=activation,
                norm_groups=norm_groups,
                gate_type=self.dhde_gate_type,
                gate_initial_probability=self.dhde_gate_initial_probability,
            )
        active_dhde_levels: set[int] = set()
        if "decoder" in self.dhde_positions:
            active_dhde_levels.update(range(len(decoder_widths)))
        if "decoder_first" in self.dhde_positions:
            active_dhde_levels.add(0)
        if "decoder_final" in self.dhde_positions:
            active_dhde_levels.add(len(decoder_widths) - 1)
        if active_dhde_levels:
            self.decoder_dhde = nn.ModuleDict(
                {
                    str(level): DoLPGuidedDehazeEnhancement(
                        decoder_widths[level],
                        residual_scale=dhde_residual_scale,
                        activation=activation,
                        norm_groups=norm_groups,
                        gate_type=self.dhde_gate_type,
                        gate_initial_probability=self.dhde_gate_initial_probability,
                    )
                    for level in sorted(active_dhde_levels)
                }
            )
        active_boundary_levels: set[int] = set()
        if "decoder" in self.boundary_guidance_positions:
            active_boundary_levels.update(range(len(decoder_widths)))
        if "decoder_first" in self.boundary_guidance_positions:
            active_boundary_levels.add(0)
        if "decoder_final" in self.boundary_guidance_positions:
            active_boundary_levels.add(len(decoder_widths) - 1)
        if active_boundary_levels:
            self.decoder_boundary_guides = nn.ModuleDict(
                {
                    str(level): build_boundary_guidance(
                        self.boundary_guidance_type,
                        decoder_widths[level],
                        boundary_residual_scale,
                        activation,
                        norm_groups,
                    )
                    for level in sorted(active_boundary_levels)
                }
            )
        if "post_fusion" in self.global_attention_positions:
            self.post_fusion_attn = nn.ModuleList(
                build_decoder_attention(self.global_attention_type, width)
                for width in channels
            )
        if "bottleneck" in self.global_attention_positions:
            self.bottleneck_attn = build_decoder_attention(
                self.global_attention_type, channels[-1]
            )
        if "decoder" in self.global_attention_positions:
            self.decoder_attn = nn.ModuleList(
                build_decoder_attention(self.global_attention_type, width)
                for width in decoder_widths
            )
        if "decoder_first" in self.global_attention_positions:
            self.global_attn = build_decoder_attention(
                self.global_attention_type, channels[-2]
            )
        if self.use_disagreement_refiner:
            self.disagreement_refiner = DisagreementResidualRefiner(
                channels,
                len(CONTEXT_KEYS),
                disagreement_residual_scale,
                activation,
                norm_groups,
            )
        if self.use_output_expert_router:
            if self.output_expert_mode == "shared_dual_decode":
                self.output_expert_router = StokesOutputPatchRouter(
                    channels[0],
                    len(CONTEXT_KEYS),
                    output_expert_patch_size,
                )
            else:
                self.output_expert_router = OutputPatchExpertRouter(
                    channels,
                    len(CONTEXT_KEYS),
                    output_expert_residual_scale,
                    output_expert_patch_size,
                    activation,
                    norm_groups,
                )

        decoder_channels = channels[0]
        auxiliary_channels = (channels[-1], channels[-2])
        if self.output_mode == "absolute":
            self.head_s0 = HeadS0(decoder_channels, activation, norm_groups)
            self.head_polar = HeadPolar(decoder_channels, activation, norm_groups)
            if self.deep_supervision:
                self.auxiliary_s0_heads = nn.ModuleList(
                    HeadS0(width, activation, norm_groups)
                    for width in auxiliary_channels
                )
                self.auxiliary_polar_heads = nn.ModuleList(
                    HeadPolar(width, activation, norm_groups)
                    for width in auxiliary_channels
                )
        else:
            self.head_stokes_residual = HeadStokesResidual(
                decoder_channels, activation, norm_groups
            )
            if self.deep_supervision:
                self.auxiliary_residual_heads = nn.ModuleList(
                    HeadStokesResidual(width, activation, norm_groups)
                    for width in auxiliary_channels
                )
        if self.pgde_mode != "none":
            self.pgde_refiner = PolarGuidedStokesRefiner(
                decoder_channels,
                self.pgde_mode,
                hidden_channels=self.pgde_hidden_channels,
                residual_scales=self.pgde_residual_scales,
                lowpass_kernel=self.pgde_lowpass_kernel,
                iterations=self.pgde_iterations,
                detach_conditioning=self.pgde_detach_conditioning,
                activation=activation,
                norm_groups=norm_groups,
            )
        if self.use_physics_head:
            if self.use_physics_adapter:
                self.physics_decoder_adapter = PhysicsFeatureAdapter(
                    decoder_channels, activation, norm_groups
                )
                self.physics_bottleneck_adapter = PhysicsFeatureAdapter(
                    channels[-1], activation, norm_groups
                )
            self.head_depth = HeadDepth(decoder_channels, activation, norm_groups)
            self.head_beta = HeadBeta(
                channels[-1], beta_min, beta_max, activation
            )
            self.head_airlight = HeadAirlightSpatial(
                decoder_channels, channels[-1], activation, norm_groups
            )
            if self.use_residual_t:
                self.head_residual_t = HeadResidualT(
                    decoder_channels,
                    residual_t_min,
                    residual_t_max,
                    activation,
                    norm_groups,
                )

        self.deployment_norm_policy = str(deployment_norm_policy).strip().lower()
        self.replaced_group_norms = self._apply_deployment_norm_policy(
            self.deployment_norm_policy
        )

    @staticmethod
    def _detail_schedule(
        configured: Sequence[str] | None,
        *,
        fallback: str,
        count: int,
        name: str,
    ) -> tuple[str, ...]:
        """Resolve an explicit per-level refinement schedule."""
        if configured is None:
            return (str(fallback),) * count
        schedule = tuple(str(value).strip().lower() for value in configured)
        if len(schedule) != count:
            raise ValueError(f"{name} must contain exactly {count} entries")
        return schedule

    def _apply_deployment_norm_policy(self, policy: str) -> tuple[str, ...]:
        """Apply an explicitly requested deployment normalization policy."""
        supported = {
            "group",
            "affine_refinement",
            "affine_backbone",
            "affine_all",
        }
        if policy not in supported:
            raise ValueError(
                f"deployment_norm_policy must be one of {sorted(supported)}"
            )
        if policy == "group":
            return ()
        if policy == "affine_all":
            return replace_group_norm(self)
        if policy == "affine_refinement":
            prefixes = (
                "independent_raw_stream.",
                "independent_polar_stream.",
                "raw_encoders.",
                "post_fusion.",
                "bottleneck.",
                "decoder.",
                "pgde_refiner.",
            )
        else:
            prefixes = (
                "independent_raw_stream.",
                "independent_polar_stream.",
                "raw_stem.",
                "raw_encoders.",
                "phys_stem.",
                "phys_encoders.",
                "post_fusion.",
                "raw_down.",
                "phys_down.",
                "bottleneck.",
                "decoder.",
                "head_s0.",
                "head_polar.",
                "pgde_refiner.",
            )
        return replace_group_norm(
            self, include=lambda name: name.startswith(prefixes)
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PolarDehazeNet":
        """Construct from the ``model`` mapping without silently dropping keys."""
        model_config = config.get("model", config)
        if not isinstance(model_config, Mapping):
            raise TypeError("model configuration must be a mapping")
        payload = dict(model_config)
        name = payload.pop("name", "PolarDehazeNet")
        if name != "PolarDehazeNet":
            raise ValueError(f"Unsupported model name: {name!r}")
        # These documented keys are consumed by the Trainer/GradientMediator,
        # not by the network constructor itself.
        for policy_key in (
            "gradient_mediation",
            "gamma_static",
            "gamma_max",
            "gradient_ema_momentum",
            "pcgrad_rho",
            "pcgrad_lambda_max",
            "rehaze_type",
        ):
            payload.pop(policy_key, None)
        return cls(**payload)

    def forward(
        self,
        raw_hazy: Tensor,
        phys_hazy: Tensor,
        *,
        observed_stokes: tuple[Tensor, Tensor, Tensor] | None = None,
        return_derived_outputs: bool = True,
        s0_only: bool = False,
    ) -> dict[str, Any]:
        """Restore a clear full-Stokes field from four analyzer angles.

        ``observed_stokes`` may be supplied when it was already constructed
        together with ``phys_hazy``. ``return_derived_outputs=False`` keeps the
        learned ``S0/S1/S2`` result while deferring deterministic DoLP, AoP and
        angle-view generation until a consumer actually requests those views.
        ``s0_only=True`` additionally skips the polarization output head and is
        the common single-intensity dehazing deployment path.
        """
        self._validate_inputs(raw_hazy, phys_hazy)
        observed = (
            self._validate_observed_stokes(observed_stokes, raw_hazy)
            if observed_stokes is not None
            else self._observed_stokes(raw_hazy, phys_hazy)
        )
        base_context = (
            self._precomputed_polar_context(observed, phys_hazy)
            if observed_stokes is not None
            else self._polar_context(*observed)
        )

        if self.stream_mode in {"polar_only", "dual_independent"}:
            return self._forward_independent_streams(
                raw_hazy,
                phys_hazy,
                observed,
                base_context,
                return_derived_outputs=return_derived_outputs,
                s0_only=s0_only,
            )

        raw_feat = self._run(self.raw_stem, raw_hazy)
        phys_feat = (
            self._run(self.phys_stem, phys_hazy)
            if self.stream_mode == "dual"
            else None
        )
        dual_decode = (
            self.use_output_expert_router
            and self.output_expert_mode == "shared_dual_decode"
        )
        complementary_raw_feat = raw_feat
        skip_features: list[Tensor] = []
        complementary_skip_features: list[Tensor] = []
        raw_features: list[Tensor] = []
        complementary_raw_features: list[Tensor] = []
        phys_features: list[Tensor] = []
        disagreement_features: list[Tensor] = []
        anchor_expert_features: list[Tensor] = []
        complementary_expert_features: list[Tensor] = []

        for level, (raw_encoder, refinement) in enumerate(
            zip(self.raw_encoders, self.post_fusion)
        ):
            context = self._resize_context(base_context, raw_feat.shape[-2:])
            detail_context = stack_polar_context(
                context, raw_feat.shape[-2:]
            )
            raw_feat = self._run(raw_encoder, raw_feat, detail_context)
            if dual_decode:
                complementary_raw_feat = self._run(
                    raw_encoder, complementary_raw_feat, detail_context
                )
            complementary_refined: Tensor | None = None
            if self.stream_mode == "raw_only":
                fused = raw_feat
            elif dual_decode:
                if phys_feat is None:
                    raise RuntimeError("Dual-stream mode lost its physics feature")
                phys_encoder = self.phys_encoders[level]
                fusion = self.fusions[level]
                phys_feat = self._run(phys_encoder, phys_feat)
                forward_experts = getattr(fusion, "forward_experts", None)
                if not callable(forward_experts):
                    raise RuntimeError(
                        "Configured output fusion has no dual-state interface"
                    )
                anchor_feature, complementary_feature = forward_experts(
                    raw_feat,
                    complementary_raw_feat,
                    phys_feat,
                    context,
                )
                fused = anchor_feature
                complementary_refined = self._run(
                    refinement, complementary_feature, detail_context
                )
            else:
                if phys_feat is None:
                    raise RuntimeError("Dual-stream mode lost its physics feature")
                phys_encoder = self.phys_encoders[level]
                fusion = self.fusions[level]
                phys_feat = self._run(phys_encoder, phys_feat)
                fused = fusion(raw_feat, phys_feat, context)
            if self.use_output_expert_router and not dual_decode:
                anchor_feature = getattr(fusion, "last_anchor_feature", None)
                complementary_feature = getattr(
                    fusion, "last_complementary_feature", None
                )
                if not isinstance(anchor_feature, Tensor) or not isinstance(
                    complementary_feature, Tensor
                ):
                    raise RuntimeError(
                        "Configured fusion did not expose both output experts"
                    )
                anchor_expert_features.append(anchor_feature)
                complementary_expert_features.append(complementary_feature)
            elif dual_decode:
                anchor_expert_features.append(anchor_feature)
                complementary_expert_features.append(complementary_feature)
            if self.use_disagreement_refiner:
                disagreement = getattr(
                    fusion, "last_disagreement_feature", None
                )
                if not isinstance(disagreement, Tensor):
                    raise RuntimeError(
                        "Configured fusion did not expose a disagreement feature"
                    )
                disagreement_features.append(disagreement)
            refined = self._run(refinement, fused, detail_context)
            if hasattr(self, "post_fusion_attn"):
                refined = self._run(
                    self.post_fusion_attn[level], refined, context
                )
            if hasattr(self, "post_fusion_dhde"):
                refined = self._run(
                    self.post_fusion_dhde[level], refined, context
                )
            raw_features.append(raw_feat)
            if dual_decode:
                complementary_raw_features.append(complementary_raw_feat)
            if phys_feat is not None:
                phys_features.append(phys_feat)
            skip_features.append(refined)
            if dual_decode:
                if complementary_refined is None:
                    raise RuntimeError("Missing complementary refined feature")
                complementary_skip_features.append(complementary_refined)
            if level < len(self.raw_down):
                raw_transition = refined if self.fusion_feedback_to_raw else raw_feat
                raw_feat = self._run(self.raw_down[level], raw_transition)
                if dual_decode:
                    complementary_transition = (
                        complementary_refined
                        if self.fusion_feedback_to_raw
                        else complementary_raw_feat
                    )
                    complementary_raw_feat = self._run(
                        self.raw_down[level], complementary_transition
                    )
                if self.stream_mode == "dual":
                    if phys_feat is None:
                        raise RuntimeError(
                            "Dual-stream mode lost its physics feature"
                        )
                    phys_feat = self._run(self.phys_down[level], phys_feat)

        bottleneck_context_mapping = self._resize_context(
            base_context, skip_features[-1].shape[-2:]
        )
        bottleneck_context = stack_polar_context(
            bottleneck_context_mapping, skip_features[-1].shape[-2:]
        )
        bottleneck = self._run(
            self.bottleneck, skip_features[-1], bottleneck_context
        )
        if hasattr(self, "bottleneck_attn"):
            bottleneck = self._run(
                self.bottleneck_attn,
                bottleneck,
                bottleneck_context_mapping,
            )
        if hasattr(self, "bottleneck_dhde"):
            bottleneck = self._run(
                self.bottleneck_dhde,
                bottleneck,
                bottleneck_context_mapping,
            )
        collect_dhde_supervision = self.dhde_gate_supervision and self.training
        decoder_feat, decoder_stages, dhde_supervision = self._decode_features(
            bottleneck,
            skip_features,
            base_context,
            collect_dhde_supervision=collect_dhde_supervision,
        )
        complementary_bottleneck: Tensor | None = None
        complementary_decoder_feat: Tensor | None = None
        if dual_decode:
            complementary_bottleneck = self._run(
                self.bottleneck,
                complementary_skip_features[-1],
                bottleneck_context,
            )
            complementary_decoder_feat, _, _ = self._decode_features(
                complementary_bottleneck,
                complementary_skip_features,
                base_context,
            )

        if s0_only:
            if self.output_mode != "absolute":
                raise ValueError("s0_only currently requires output_mode='absolute'")
            if (
                self.use_output_expert_router
                or self.use_disagreement_refiner
                or self.use_physics_head
                or self.pgde_mode != "none"
            ):
                raise ValueError(
                    "s0_only is incompatible with enabled output/physics refiners"
                )
            return {"s0": self.head_s0(decoder_feat)}

        needs_internal_derivatives = (
            self.use_output_expert_router
            or self.use_disagreement_refiner
            or self.use_physics_head
            or self.pgde_mode != "none"
        )
        outputs = self._direct_outputs(
            decoder_feat,
            observed,
            include_derived=(return_derived_outputs or needs_internal_derivatives),
        )
        if self.pgde_mode != "none":
            pgde_result = self.pgde_refiner(
                decoder_feature=decoder_feat,
                coarse=outputs,
                observed_stokes=observed,
            )
            final_stokes = (
                pgde_result["final_stokes"]
                if self.pgde_output_enabled
                else torch.cat((outputs["s0"], outputs["s12"]), dim=1)
            )
            if not isinstance(final_stokes, Tensor):
                raise TypeError("PGDE final_stokes must be a Tensor")
            outputs = self._pack_stokes_outputs(
                final_stokes[:, 0:1],
                final_stokes[:, 1:2],
                final_stokes[:, 2:3],
                include_derived=(
                    return_derived_outputs or needs_internal_derivatives
                ),
            )
            diagnostics = pgde_result.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise TypeError("PGDE diagnostics must be a mapping")
            outputs.update(diagnostics)
            outputs["pgde_output_enabled"] = self.pgde_output_enabled
        if collect_dhde_supervision:
            if dhde_supervision is None:
                raise RuntimeError("DHDE gate supervision state was not collected")
            with torch.no_grad():
                pre_outputs = self._direct_outputs(
                    dhde_supervision["pre_feature"], observed
                )
            outputs.update(
                {
                    "dhde_attention": dhde_supervision["attention"],
                    "dhde_pre_s0": pre_outputs["s0"],
                    "dhde_pre_s12": pre_outputs["s12"],
                    "dhde_pre_dolp": pre_outputs["dolp"],
                }
            )
        if self.deep_supervision and self.training:
            outputs["aux_stokes"] = self._auxiliary_outputs(
                (bottleneck, decoder_stages[0]), observed
            )
        if self.use_output_expert_router:
            context_tensor = stack_polar_context(
                base_context, decoder_feat.shape[-2:]
            )
            if dual_decode:
                if complementary_decoder_feat is None:
                    raise RuntimeError("Missing complementary decoder feature")
                complementary_outputs = self._direct_outputs(
                    complementary_decoder_feat, observed
                )
                anchor_stokes = torch.cat(
                    (outputs["s0"], outputs["s12"]), dim=1
                )
                complementary_stokes = torch.cat(
                    (
                        complementary_outputs["s0"],
                        complementary_outputs["s12"],
                    ),
                    dim=1,
                )
                weight = self.output_expert_router(
                    anchor_stokes,
                    complementary_stokes,
                    decoder_feat,
                    complementary_decoder_feat,
                    context_tensor,
                )
                outputs = self._mix_output_expert_predictions(
                    outputs, complementary_outputs, weight
                )
            else:
                routed = self.output_expert_router(
                    decoder_feat,
                    anchor_expert_features,
                    complementary_expert_features,
                    context_tensor,
                )
                outputs = self._apply_output_expert_routing(outputs, routed)
        if self.use_disagreement_refiner:
            context_tensor = stack_polar_context(
                base_context, decoder_feat.shape[-2:]
            )
            refinement = self.disagreement_refiner(
                decoder_feat,
                disagreement_features,
                context_tensor,
            )
            outputs = self._apply_disagreement_refinement(outputs, refinement)
        fusion_weights = (
            [outputs["output_route_weight"]]
            if isinstance(outputs.get("output_route_weight"), Tensor)
            else self._fusion_weights()
        )
        if self.return_diagnostics:
            outputs.update(
                {
                    "decoder_feat": decoder_feat,
                    "ref_feats": skip_features,
                    "raw_feats": raw_features,
                    "complementary_raw_feats": complementary_raw_features,
                    "phys_feats": phys_features,
                    "bottleneck_feat": bottleneck,
                    "complementary_bottleneck_feat": complementary_bottleneck,
                    "complementary_decoder_feat": complementary_decoder_feat,
                    "complementary_ref_feats": complementary_skip_features,
                    "fusion_weights": fusion_weights,
                    "detail_branch_weights": self._detail_branch_weights(),
                    **self._global_attention_diagnostics(),
                    **self._dhde_diagnostics(),
                    **self._boundary_guidance_diagnostics(),
                    "disagreement_features": disagreement_features,
                    "anchor_expert_features": anchor_expert_features,
                    "complementary_expert_features": complementary_expert_features,
                }
            )
        if not self.use_physics_head:
            return (
                outputs
                if return_derived_outputs
                else self._stokes_only_outputs(outputs)
            )
        outputs.update(
            self._physics_outputs(
                decoder_feat,
                bottleneck,
                observed,
                phys_features[0],
                phys_features[-1],
            )
        )
        return (
            outputs
            if return_derived_outputs
            else self._stokes_only_outputs(outputs)
        )

    def _forward_independent_streams(
        self,
        raw_hazy: Tensor,
        phys_hazy: Tensor,
        observed: tuple[Tensor, Tensor, Tensor],
        base_context: Mapping[str, Tensor],
        *,
        return_derived_outputs: bool,
        s0_only: bool,
    ) -> dict[str, Any]:
        """Run polar-only or fixed-average dual-independent restoration."""
        if s0_only:
            raise ValueError(
                "s0_only is not implemented for independent output streams"
            )
        include_internal = return_derived_outputs or self.pgde_mode != "none"
        polar_feature = self.independent_polar_stream(phys_hazy, base_context)
        polar_outputs = self._absolute_outputs(
            polar_feature,
            self.independent_polar_stream.head_s0,
            self.independent_polar_stream.head_polar,
            include_derived=include_internal,
        )
        decoder_feature = polar_feature
        stream_diagnostics: dict[str, Tensor] = {}
        if self.stream_mode == "dual_independent":
            raw_feature = self.independent_raw_stream(raw_hazy, base_context)
            raw_outputs = self._absolute_outputs(
                raw_feature,
                self.independent_raw_stream.head_s0,
                self.independent_raw_stream.head_polar,
                include_derived=include_internal,
            )
            coarse_s0, coarse_s1, coarse_s2 = safe_clip_stokes(
                0.5 * (raw_outputs["s0"] + polar_outputs["s0"]),
                0.5 * (raw_outputs["s1"] + polar_outputs["s1"]),
                0.5 * (raw_outputs["s2"] + polar_outputs["s2"]),
            )
            outputs = self._pack_stokes_outputs(
                coarse_s0,
                coarse_s1,
                coarse_s2,
                include_derived=include_internal,
            )
            # Only G4 consumes this fixed-average guidance feature. The two
            # restoration predictions above have no feature interaction.
            decoder_feature = 0.5 * (raw_feature + polar_feature)
            if self.return_diagnostics:
                stream_diagnostics = {
                    "independent_raw_s0": raw_outputs["s0"],
                    "independent_raw_s12": torch.cat(
                        (raw_outputs["s1"], raw_outputs["s2"]), dim=1
                    ),
                    "independent_polar_s0": polar_outputs["s0"],
                    "independent_polar_s12": torch.cat(
                        (polar_outputs["s1"], polar_outputs["s2"]), dim=1
                    ),
                }
        else:
            outputs = polar_outputs

        if self.pgde_mode != "none":
            pgde_result = self.pgde_refiner(
                decoder_feature=decoder_feature,
                coarse=outputs,
                observed_stokes=observed,
            )
            final_stokes = (
                pgde_result["final_stokes"]
                if self.pgde_output_enabled
                else torch.cat(
                    (outputs["s0"], outputs["s1"], outputs["s2"]), dim=1
                )
            )
            if not isinstance(final_stokes, Tensor):
                raise TypeError("PGDE final_stokes must be a Tensor")
            outputs = self._pack_stokes_outputs(
                final_stokes[:, 0:1],
                final_stokes[:, 1:2],
                final_stokes[:, 2:3],
                include_derived=True,
            )
            diagnostics = pgde_result.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise TypeError("PGDE diagnostics must be a mapping")
            outputs.update(diagnostics)
            outputs["pgde_output_enabled"] = self.pgde_output_enabled
        if stream_diagnostics:
            outputs.update(stream_diagnostics)
        return (
            outputs
            if return_derived_outputs
            else self._stokes_only_outputs(outputs)
        )

    def _decode_features(
        self,
        bottleneck: Tensor,
        skip_features: Sequence[Tensor],
        base_context: Mapping[str, Tensor],
        *,
        collect_dhde_supervision: bool = False,
    ) -> tuple[Tensor, list[Tensor], dict[str, Tensor] | None]:
        """Run the shared decoder for one persistent fusion-expert state."""
        decoder_feat = bottleneck
        decoder_stages: list[Tensor] = []
        dhde_supervision: dict[str, Tensor] | None = None
        decoder_skips = list(reversed(skip_features[:-1]))
        for decoder_level, (block, skip) in enumerate(
            zip(self.decoder, decoder_skips)
        ):
            context_mapping = self._resize_context(
                base_context, skip.shape[-2:]
            )
            detail_context = stack_polar_context(
                context_mapping, skip.shape[-2:]
            )
            decoder_feat = self._run(
                block, decoder_feat, skip, detail_context
            )
            if hasattr(self, "decoder_attn"):
                decoder_feat = self._run(
                    self.decoder_attn[decoder_level],
                    decoder_feat,
                    context_mapping,
                )
            elif decoder_level == 0 and hasattr(self, "global_attn"):
                decoder_feat = self.global_attn(
                    decoder_feat, context_mapping
                )
            if hasattr(self, "decoder_dhde"):
                level_key = str(decoder_level)
                if level_key in self.decoder_dhde:
                    dhde_module = self.decoder_dhde[level_key]
                    is_supervised_level = (
                        collect_dhde_supervision
                        and decoder_level == len(self.decoder) - 1
                    )
                    if is_supervised_level:
                        pre_feature = decoder_feat
                        decoder_feat = dhde_module(
                            decoder_feat, context_mapping
                        )
                        attention = getattr(
                            dhde_module, "training_attention", None
                        )
                        if not isinstance(attention, Tensor):
                            raise RuntimeError(
                                "Supervised DHDE did not expose its attention map"
                            )
                        dhde_supervision = {
                            "pre_feature": pre_feature,
                            "attention": attention,
                        }
                    else:
                        decoder_feat = self._run(
                            dhde_module,
                            decoder_feat,
                            context_mapping,
                        )
            if hasattr(self, "decoder_boundary_guides"):
                level_key = str(decoder_level)
                if level_key in self.decoder_boundary_guides:
                    decoder_feat = self._run(
                        self.decoder_boundary_guides[level_key],
                        decoder_feat,
                        context_mapping,
                    )
            decoder_stages.append(decoder_feat)
        return decoder_feat, decoder_stages, dhde_supervision

    @staticmethod
    def _pack_stokes_outputs(
        s0: Tensor,
        s1: Tensor,
        s2: Tensor,
        *,
        unbounded_dolp: Tensor | None = None,
        include_derived: bool = True,
    ) -> dict[str, Tensor]:
        """Expose the predicted Stokes field and parameter-free derivatives.

        ``S0/S1/S2`` are the only learned output quantities. DoLP, AoP and
        analyzer-angle intensities are deterministic post-processing views.
        The legacy ``s12`` and ``dolp_mag`` keys remain available so existing
        checkpoints, losses and historical experiment tools stay loadable.
        """
        if not include_derived:
            return {"s0": s0, "s1": s1, "s2": s2}
        s12 = torch.cat((s1, s2), dim=1)
        dolp, aop = stokes_to_dolp_aop(s0, s1, s2)
        _, cos2aop, sin2aop = stokes_to_polar_features(s0, s1, s2)
        return {
            "s0": s0,
            "s1": s1,
            "s2": s2,
            "s12": s12,
            "stokes": torch.cat((s0, s12), dim=1),
            "dolp": dolp,
            "aop": aop,
            "cos2aop": cos2aop,
            "sin2aop": sin2aop,
            "angles": torch.cat(stokes_to_angles(s0, s1, s2), dim=1),
            "dolp_mag": dolp if unbounded_dolp is None else unbounded_dolp,
        }

    def _direct_outputs(
        self,
        decoder_feat: Tensor,
        observed: tuple[Tensor, Tensor, Tensor],
        *,
        include_derived: bool = True,
    ) -> dict[str, Tensor]:
        if self.output_mode == "hazy_residual":
            return self._residual_outputs(
                decoder_feat,
                observed,
                self.head_stokes_residual,
                include_derived=include_derived,
            )
        return self._absolute_outputs(
            decoder_feat,
            self.head_s0,
            self.head_polar,
            include_derived=include_derived,
        )

    def _absolute_outputs(
        self,
        features: Tensor,
        head_s0: nn.Module,
        head_polar: nn.Module,
        *,
        include_derived: bool = True,
    ) -> dict[str, Tensor]:
        s0_raw = head_s0(features)
        v1, v2 = torch.chunk(head_polar(features), 2, dim=1)
        magnitude = torch.sqrt(v1.square() + v2.square() + 1e-4)
        denominator = torch.maximum(magnitude, torch.ones_like(magnitude))
        s1 = (v1 / denominator) * s0_raw
        s2 = (v2 / denominator) * s0_raw
        s0, s1, s2 = safe_clip_stokes(s0_raw, s1, s2)
        return self._pack_stokes_outputs(
            s0,
            s1,
            s2,
            unbounded_dolp=magnitude,
            include_derived=include_derived,
        )

    def _residual_outputs(
        self,
        features: Tensor,
        observed: tuple[Tensor, Tensor, Tensor],
        residual_head: nn.Module,
        *,
        include_derived: bool = True,
    ) -> dict[str, Tensor]:
        residual = residual_head(features)
        scales = self.stokes_residual_scales.to(
            device=residual.device, dtype=residual.dtype
        )
        residual = residual * scales
        observed_at_scale = tuple(
            functional.interpolate(
                component,
                size=features.shape[-2:],
                mode="area",
            )
            if component.shape[-2:] != features.shape[-2:]
            else component
            for component in observed
        )
        s0, s1, s2 = safe_clip_stokes(
            observed_at_scale[0] + residual[:, 0:1],
            observed_at_scale[1] + residual[:, 1:2],
            observed_at_scale[2] + residual[:, 2:3],
        )
        outputs = self._pack_stokes_outputs(
            s0, s1, s2, include_derived=include_derived
        )
        outputs["output_residual"] = residual
        return outputs

    def set_pgde_output_enabled(self, enabled: bool) -> None:
        """Enable or bypass the PGDE correction without changing its parameters.

        The bypass is intended for a configured training warm-up. The refiner
        still executes and exposes its auxiliary tensors, while canonical
        outputs remain equal to the independently supervised coarse Stokes
        prediction.
        """
        if self.pgde_mode == "none" and not enabled:
            raise ValueError("Cannot schedule a PGDE output bypass when PGDE is disabled")
        self.pgde_output_enabled = bool(enabled)

    def _auxiliary_outputs(
        self,
        features: Sequence[Tensor],
        observed: tuple[Tensor, Tensor, Tensor],
    ) -> list[dict[str, Tensor]]:
        if len(features) != 2:
            raise ValueError("Deep supervision requires bottleneck and decoder features")
        if self.output_mode == "hazy_residual":
            return [
                self._residual_outputs(feature, observed, head)
                for feature, head in zip(
                    features, self.auxiliary_residual_heads
                )
            ]
        return [
            self._absolute_outputs(feature, head_s0, head_polar)
            for feature, head_s0, head_polar in zip(
                features,
                self.auxiliary_s0_heads,
                self.auxiliary_polar_heads,
            )
        ]

    @staticmethod
    def _mix_output_expert_predictions(
        anchor: Mapping[str, Tensor],
        complementary: Mapping[str, Tensor],
        complementary_weight: Tensor,
    ) -> dict[str, Tensor]:
        """Mix two genuine Stokes predictions with one output patch route."""
        if complementary_weight.ndim != 4 or complementary_weight.shape[1] != 1:
            raise ValueError("Output expert route must have shape Bx1xHxW")
        anchor_s0 = anchor["s0"]
        anchor_s1, anchor_s2 = torch.chunk(anchor["s12"], 2, dim=1)
        complementary_s0 = complementary["s0"]
        complementary_s1, complementary_s2 = torch.chunk(
            complementary["s12"], 2, dim=1
        )
        mixed_s0 = (
            (1.0 - complementary_weight) * anchor_s0
            + complementary_weight * complementary_s0
        )
        mixed_s1 = (
            (1.0 - complementary_weight) * anchor_s1
            + complementary_weight * complementary_s1
        )
        mixed_s2 = (
            (1.0 - complementary_weight) * anchor_s2
            + complementary_weight * complementary_s2
        )
        s0, s1, s2 = safe_clip_stokes(mixed_s0, mixed_s1, mixed_s2)
        outputs = PolarDehazeNet._pack_stokes_outputs(s0, s1, s2)
        outputs.update({
            "s0_base": anchor_s0,
            "s12_base": anchor["s12"],
            "s0_anchor": anchor_s0,
            "s12_anchor": anchor["s12"],
            "s0_complementary": complementary_s0,
            "s12_complementary": complementary["s12"],
            "output_route_weight": complementary_weight,
        })
        return outputs

    @staticmethod
    def _apply_output_expert_routing(
        base: Mapping[str, Tensor],
        routed: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Form two valid Stokes candidates and route the final prediction."""
        anchor_residual = routed.get("anchor_residual")
        complementary_residual = routed.get("complementary_residual")
        complementary_weight = routed.get("complementary_weight")
        feature = routed.get("feature")
        if not all(
            isinstance(value, Tensor)
            for value in (
                anchor_residual,
                complementary_residual,
                complementary_weight,
                feature,
            )
        ):
            raise TypeError("Output expert routing values must be tensors")
        if anchor_residual.shape[1] != 3 or complementary_residual.shape != anchor_residual.shape:
            raise ValueError("Output expert residuals must have shape Bx3xHxW")
        if complementary_weight.shape[1] != 1:
            raise ValueError("Output expert route must have shape Bx1xHxW")

        base_stokes = torch.cat((base["s0"], base["s12"]), dim=1)
        anchor_raw = base_stokes + anchor_residual
        complementary_raw = base_stokes + complementary_residual
        anchor_s0, anchor_s1, anchor_s2 = safe_clip_stokes(
            anchor_raw[:, :1], anchor_raw[:, 1:2], anchor_raw[:, 2:3]
        )
        complementary_s0, complementary_s1, complementary_s2 = safe_clip_stokes(
            complementary_raw[:, :1],
            complementary_raw[:, 1:2],
            complementary_raw[:, 2:3],
        )
        mixed_s0 = (
            (1.0 - complementary_weight) * anchor_s0
            + complementary_weight * complementary_s0
        )
        mixed_s1 = (
            (1.0 - complementary_weight) * anchor_s1
            + complementary_weight * complementary_s1
        )
        mixed_s2 = (
            (1.0 - complementary_weight) * anchor_s2
            + complementary_weight * complementary_s2
        )
        s0, s1, s2 = safe_clip_stokes(mixed_s0, mixed_s1, mixed_s2)
        outputs = PolarDehazeNet._pack_stokes_outputs(s0, s1, s2)
        outputs.update({
            "s0_base": base["s0"],
            "s12_base": base["s12"],
            "s0_anchor": anchor_s0,
            "s12_anchor": torch.cat((anchor_s1, anchor_s2), dim=1),
            "s0_complementary": complementary_s0,
            "s12_complementary": torch.cat(
                (complementary_s1, complementary_s2), dim=1
            ),
            "output_anchor_residual": anchor_residual,
            "output_complementary_residual": complementary_residual,
            "output_route_weight": complementary_weight,
            "output_expert_feature": feature,
        })
        return outputs

    @staticmethod
    def _apply_disagreement_refinement(
        base: Mapping[str, Tensor],
        refinement: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        residual = refinement.get("residual")
        confidence = refinement.get("confidence")
        feature = refinement.get("feature")
        if not all(
            isinstance(value, Tensor)
            for value in (residual, confidence, feature)
        ):
            raise TypeError("Disagreement refinement outputs must be tensors")
        if residual.shape[1] != 3 or confidence.shape != residual.shape:
            raise ValueError("Stokes residual and confidence must have shape Bx3xHxW")
        base_s0 = base["s0"]
        base_s1, base_s2 = torch.chunk(base["s12"], 2, dim=1)
        delta_s0, delta_s1, delta_s2 = torch.chunk(residual, 3, dim=1)
        raw_s0 = base_s0 + delta_s0
        raw_s1 = base_s1 + delta_s1
        raw_s2 = base_s2 + delta_s2
        raw_magnitude = torch.sqrt(raw_s1.square() + raw_s2.square() + 1e-4)
        dolp_magnitude = raw_magnitude / raw_s0.clamp_min(1e-4)
        s0, s1, s2 = safe_clip_stokes(raw_s0, raw_s1, raw_s2)
        outputs = PolarDehazeNet._pack_stokes_outputs(
            s0, s1, s2, unbounded_dolp=dolp_magnitude
        )
        outputs.update({
            "s0_base": base_s0,
            "s12_base": base["s12"],
            "stokes_residual": residual,
            "refinement_confidence": confidence,
            "disagreement_decoder_feat": feature,
        })
        return outputs

    def _physics_outputs(
        self,
        decoder_feat: Tensor,
        bottleneck: Tensor,
        observed: tuple[Tensor, Tensor, Tensor],
        decoder_guidance: Tensor,
        bottleneck_guidance: Tensor,
    ) -> dict[str, Tensor | None]:
        physics_decoder = grad_scale(decoder_feat, self.physics_gradient_scale)
        physics_bottleneck = grad_scale(bottleneck, self.physics_gradient_scale)
        if self.use_physics_adapter:
            physics_decoder = self.physics_decoder_adapter(
                physics_decoder,
                grad_scale(decoder_guidance, self.physics_gradient_scale),
            )
            physics_bottleneck = self.physics_bottleneck_adapter(
                physics_bottleneck,
                grad_scale(bottleneck_guidance, self.physics_gradient_scale),
            )
        depth = self.head_depth(physics_decoder)
        beta = self.head_beta(physics_bottleneck)
        transmission = torch.exp(-beta.view(-1, 1, 1, 1) * depth)
        residual = None
        if self.use_residual_t:
            residual = self.head_residual_t(physics_decoder)
            transmission = transmission * residual
        transmission = transmission.clamp(self.transmission_min, 1.0)

        airlight = self.head_airlight(physics_decoder, physics_bottleneck)
        a0 = airlight["a_inf_map"]
        cs = airlight["polar_cs"]
        sn = airlight["polar_sn"]
        cs_map = cs.view(-1, 1, 1, 1)
        sn_map = sn.view(-1, 1, 1, 1)
        s0_hazy, s1_hazy, s2_hazy = observed
        haze = 1.0 - transmission

        s0_phys = (s0_hazy - a0 * haze) / transmission
        s1_phys = (s1_hazy - a0 * cs_map.detach() * haze) / transmission
        s2_phys = (s2_hazy - a0 * sn_map.detach() * haze) / transmission
        s0_phys, s1_phys, s2_phys = safe_clip_stokes(s0_phys, s1_phys, s2_phys)

        a0_detached = a0.detach()
        haze_detached = haze.detach()
        transmission_detached = transmission.detach()
        s1_phys_raw = (
            s1_hazy - a0_detached * cs_map * haze_detached
        ) / transmission_detached
        s2_phys_raw = (
            s2_hazy - a0_detached * sn_map * haze_detached
        ) / transmission_detached
        a0_global = a0.mean(dim=(2, 3))
        return {
            "s0_phys": s0_phys,
            "s12_phys": torch.cat((s1_phys, s2_phys), dim=1),
            "s12_phys_raw": torch.cat((s1_phys_raw, s2_phys_raw), dim=1),
            "t": transmission,
            "t_residual": residual,
            "depth": depth,
            "beta": beta,
            "airlight_map": a0,
            "airlight_polar": torch.cat((cs, sn), dim=1),
            "airlight": torch.cat((a0_global, cs, sn), dim=1),
        }

    def _fusion_weights(self) -> list[Tensor]:
        """Return detached routing maps exposed by dynamic fusion levels."""
        weights: list[Tensor] = []
        for fusion in self.fusions:
            value = getattr(fusion, "last_spatial_weight", None)
            if isinstance(value, Tensor):
                weights.append(value)
        return weights

    def _detail_branch_weights(self) -> dict[str, list[Tensor]]:
        """Return detached branch probabilities from contextual Detail blocks."""

        def collect(modules: Sequence[nn.Module]) -> list[Tensor]:
            values: list[Tensor] = []
            for module in modules:
                block = getattr(module, "block", module)
                value = getattr(block, "last_branch_weights", None)
                if isinstance(value, Tensor):
                    values.append(value)
            return values

        return {
            "raw": collect(tuple(self.raw_encoders)),
            "post_fusion": collect(tuple(self.post_fusion)),
            "bottleneck": collect(tuple(self.bottleneck.blocks)),
            "decoder": collect(tuple(block.detail for block in self.decoder)),
        }

    def _attention_modules(self) -> list[nn.Module]:
        """Return attention blocks in encoder-to-decoder execution order."""
        modules: list[nn.Module] = []
        if hasattr(self, "post_fusion_attn"):
            modules.extend(self.post_fusion_attn)
        if hasattr(self, "bottleneck_attn"):
            modules.append(self.bottleneck_attn)
        if hasattr(self, "decoder_attn"):
            modules.extend(self.decoder_attn)
        elif hasattr(self, "global_attn"):
            modules.append(self.global_attn)
        return modules

    def _dhde_diagnostics(self) -> dict[str, Any]:
        """Expose detached DHDE attention maps when diagnostics are enabled."""
        modules: list[nn.Module] = []
        if hasattr(self, "post_fusion_dhde"):
            modules.extend(self.post_fusion_dhde)
        if hasattr(self, "bottleneck_dhde"):
            modules.append(self.bottleneck_dhde)
        if hasattr(self, "decoder_dhde"):
            modules.extend(self.decoder_dhde.values())
        maps = [
            value
            for module in modules
            if isinstance(
                value := getattr(module, "last_attention", None), Tensor
            )
        ]
        return {
            "dhde_attention_map": maps[-1] if maps else None,
            "dhde_attention_maps": maps,
        }

    def _global_attention_diagnostics(self) -> dict[str, Any]:
        """Expose the final map and all multi-scale attention diagnostics."""
        if not self.use_global_attn:
            return {
                "global_attention_map": None,
                "global_attention_reliability": None,
                "global_attention_gate": None,
                "global_attention_maps": [],
                "global_attention_reliabilities": [],
                "global_attention_gates": [],
            }
        modules = self._attention_modules()
        maps = [
            value
            for module in modules
            if isinstance(
                value := getattr(module, "last_attention_map", None), Tensor
            )
        ]
        reliabilities = [
            value
            for module in modules
            if isinstance(
                value := getattr(
                    module, "last_orientation_reliability", None
                ),
                Tensor,
            )
        ]
        gates: list[Tensor] = []
        for module in modules:
            value = getattr(module, "gate", None)
            if not isinstance(value, Tensor):
                value = getattr(module, "residual_scale", None)
            if isinstance(value, Tensor):
                gates.append(value.detach())
        return {
            "global_attention_map": maps[-1] if maps else None,
            "global_attention_reliability": (
                reliabilities[-1] if reliabilities else None
            ),
            "global_attention_gate": gates[-1] if gates else None,
            "global_attention_maps": maps,
            "global_attention_reliabilities": reliabilities,
            "global_attention_gates": gates,
        }

    def _boundary_guidance_diagnostics(self) -> dict[str, Any]:
        """Expose detached boundary confidence and dynamic sample offsets."""
        guides = (
            list(self.decoder_boundary_guides.values())
            if hasattr(self, "decoder_boundary_guides")
            else []
        )
        boundary_maps = [
            value
            for module in guides
            if isinstance(
                value := getattr(module, "last_boundary_map", None), Tensor
            )
        ]
        reliabilities = [
            value
            for module in guides
            if isinstance(
                value := getattr(module, "last_reliability", None), Tensor
            )
        ]
        dynamic_modules = [
            block.dynamic_upsampler
            for block in self.decoder
            if block.dynamic_upsampler is not None
        ]
        dynamic_maps = [
            value
            for module in dynamic_modules
            if isinstance(value := module.last_boundary_map, Tensor)
        ]
        offsets = [
            value
            for module in dynamic_modules
            if isinstance(value := module.last_offsets, Tensor)
        ]
        return {
            "boundary_map": boundary_maps[-1] if boundary_maps else None,
            "boundary_reliability": (
                reliabilities[-1] if reliabilities else None
            ),
            "boundary_maps": boundary_maps,
            "boundary_reliabilities": reliabilities,
            "boundary_dynamic_maps": dynamic_maps,
            "boundary_offsets": offsets,
        }

    def set_physics_gradient_scale(self, gamma: float) -> None:
        """Set physics-to-backbone gradient transmission in ``[0,1]``.

        Dynamic loss-gradient mediation should keep this value at ``1`` and
        let :class:`GradientMediator` perform the single shared-gradient
        adjustment. The method primarily supports hard detach compatibility
        and the static partial-gradient ablation.
        """
        value = float(gamma)
        if not 0.0 <= value <= 1.0:
            raise ValueError("physics_gradient_scale must lie in [0,1]")
        self.physics_gradient_scale = value
        self.decouple_phys_grad = value == 0.0

    def _run(self, module: nn.Module, *inputs: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def _observed_stokes(
        self,
        raw_hazy: Tensor,
        phys_hazy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if raw_hazy.shape[1] == 4:
            return angles_to_stokes(*raw_hazy.split(1, dim=1))
        if phys_hazy.shape[1] >= 6:
            return (
                phys_hazy[:, 0:1] * STOKES_S0_MAX,
                phys_hazy[:, 1:2],
                phys_hazy[:, 2:3],
            )
        if raw_hazy.shape[1] >= 1 and phys_hazy.shape[1] >= 3:
            s0 = raw_hazy[:, 0:1] * STOKES_S0_MAX
            dolp = phys_hazy[:, 0:1]
            return (
                s0,
                dolp * phys_hazy[:, 1:2] * s0,
                dolp * phys_hazy[:, 2:3] * s0,
            )
        raise ValueError("Cannot derive observed Stokes from the configured inputs")

    @staticmethod
    def _validate_observed_stokes(
        observed: tuple[Tensor, Tensor, Tensor],
        raw_hazy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Validate a preprocessing-produced Stokes tuple without copying it."""
        if len(observed) != 3:
            raise ValueError("observed_stokes must contain exactly S0, S1, and S2")
        expected_shape = (raw_hazy.shape[0], 1, *raw_hazy.shape[-2:])
        for name, value in zip(("S0", "S1", "S2"), observed):
            if not isinstance(value, Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"observed_stokes {name} must have shape {expected_shape}"
                )
            if value.device != raw_hazy.device or value.dtype != raw_hazy.dtype:
                raise ValueError(
                    "observed_stokes must share raw_hazy device and dtype"
                )
        return observed

    @staticmethod
    def _stokes_only_outputs(outputs: Mapping[str, Any]) -> dict[str, Tensor]:
        """Return only the complete learned Stokes field for deployment."""
        result: dict[str, Tensor] = {}
        for key in ("s0", "s1", "s2"):
            value = outputs.get(key)
            if not isinstance(value, Tensor):
                raise KeyError(f"Model output is missing required tensor {key!r}")
            result[key] = value
        return result

    @staticmethod
    def _polar_context(
        s0: Tensor,
        s1: Tensor,
        s2: Tensor,
    ) -> dict[str, Tensor]:
        dolp, cos2aop, sin2aop = stokes_to_polar_features(s0, s1, s2)
        return {
            "s0": s0,
            "s0_norm": s0 / STOKES_S0_MAX,
            "dolp": dolp,
            "cos2aop": cos2aop,
            "sin2aop": sin2aop,
        }

    @staticmethod
    def _precomputed_polar_context(
        observed: tuple[Tensor, Tensor, Tensor],
        phys_hazy: Tensor,
    ) -> dict[str, Tensor]:
        """Reuse polarization features already built for the physics stream."""
        if phys_hazy.shape[1] == 6:
            offset = 3
        elif phys_hazy.shape[1] == 3:
            offset = 0
        else:
            raise ValueError(
                "Precomputed polarization context requires 3 or 6 physics channels"
            )
        s0, _, _ = observed
        return {
            "s0": s0,
            "s0_norm": s0 / STOKES_S0_MAX,
            "dolp": phys_hazy[:, offset:offset + 1],
            "cos2aop": phys_hazy[:, offset + 1:offset + 2],
            "sin2aop": phys_hazy[:, offset + 2:offset + 3],
        }

    @staticmethod
    def _resize_context(
        context: Mapping[str, Tensor],
        size: tuple[int, int],
    ) -> dict[str, Tensor]:
        keys = tuple(context)
        values = tuple(context[key] for key in keys)
        if not values:
            return {}
        if any(value.shape[-2:] != values[0].shape[-2:] for value in values):
            raise ValueError("Polar context tensors must share one spatial shape")
        if values[0].shape[-2:] == size:
            return dict(zip(keys, values))
        stacked = torch.cat(values, dim=1)
        resized = functional.interpolate(
            stacked, size=size, mode="bilinear", align_corners=False
        )
        return {
            key: value
            for key, value in zip(keys, resized.split(1, dim=1))
        }

    def _validate_inputs(self, raw_hazy: Tensor, phys_hazy: Tensor) -> None:
        if raw_hazy.ndim != 4 or phys_hazy.ndim != 4:
            raise ValueError("raw_hazy and phys_hazy must be BCHW tensors")
        if raw_hazy.shape[0] != phys_hazy.shape[0] or raw_hazy.shape[-2:] != phys_hazy.shape[-2:]:
            raise ValueError("raw_hazy and phys_hazy must share batch/spatial shapes")
        if raw_hazy.shape[1] != self.raw_in_channels:
            raise ValueError(
                f"Expected {self.raw_in_channels} raw channels, got {raw_hazy.shape[1]}"
            )
        if (
            self.stream_mode in {"dual", "polar_only", "dual_independent"}
            and phys_hazy.shape[1] != self.phys_in_channels
        ):
            raise ValueError(
                f"Expected {self.phys_in_channels} physics channels, got {phys_hazy.shape[1]}"
            )
