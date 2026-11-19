"""Independent single-input restoration stream used by output-level controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from poldehaze_v2.models.decoder import Bottleneck, DecoderUpBlock
from poldehaze_v2.models.encoders import (
    DownBlock,
    PhysEncoderBlock,
    RawEncoderBlock,
    Stem,
)
from poldehaze_v2.models.fusion.spatial_gate import stack_polar_context
from poldehaze_v2.models.heads import HeadPolar, HeadS0


class IndependentRestorationStream(nn.Module):
    """Restore Stokes features without interacting with another stream.

    Args:
        in_channels: Number of BCHW input channels.
        channels: Shallow-to-deep encoder widths.
        encoder_kind: ``"raw"`` uses Detail blocks; ``"polar"`` uses the
            physics encoder block selected by ``use_lpm``.

    The returned decoder feature has shape ``B x channels[0] x H x W``. The
    stream owns its output heads so two instances are fully parameter-
    independent through their final Stokes predictions.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: Sequence[int],
        encoder_kind: str,
        encoder_detail_types: Sequence[str],
        post_detail_types: Sequence[str],
        bottleneck_detail_type: str,
        decoder_detail_types: Sequence[str],
        use_lpm: bool,
        bottleneck_pairs: int,
        activation: str,
        norm_groups: int,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        kind = str(encoder_kind).strip().lower()
        if kind not in {"raw", "polar"}:
            raise ValueError("encoder_kind must be 'raw' or 'polar'")
        if len(widths) != 3:
            raise ValueError("Independent restoration streams require three levels")
        if len(encoder_detail_types) != len(widths):
            raise ValueError("encoder_detail_types must match channels")
        if len(post_detail_types) != len(widths):
            raise ValueError("post_detail_types must match channels")
        if len(decoder_detail_types) != len(widths) - 1:
            raise ValueError("decoder_detail_types must contain two entries")

        self.encoder_kind = kind
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.stem = Stem(in_channels, widths[0], activation, norm_groups)
        if kind == "raw":
            self.encoders = nn.ModuleList(
                RawEncoderBlock(width, detail, activation, norm_groups)
                for width, detail in zip(widths, encoder_detail_types)
            )
        else:
            self.encoders = nn.ModuleList(
                PhysEncoderBlock(width, use_lpm, activation, norm_groups)
                for width in widths
            )
        self.post_encoder = nn.ModuleList(
            RawEncoderBlock(width, detail, activation, norm_groups)
            for width, detail in zip(widths, post_detail_types)
        )
        self.down = nn.ModuleList(
            DownBlock(widths[index], widths[index + 1], activation, norm_groups)
            for index in range(len(widths) - 1)
        )
        self.bottleneck = Bottleneck(
            widths[-1],
            bottleneck_detail_type,
            use_lpm,
            bottleneck_pairs,
            activation,
            norm_groups,
        )
        decoder_indices = tuple(reversed(range(len(widths) - 1)))
        self.decoder = nn.ModuleList(
            DecoderUpBlock(
                widths[index + 1],
                widths[index],
                widths[index],
                detail,
                use_lpm,
                False,
                1.0,
                activation,
                norm_groups,
            )
            for index, detail in zip(decoder_indices, decoder_detail_types)
        )
        self.head_s0 = HeadS0(widths[0], activation, norm_groups)
        self.head_polar = HeadPolar(widths[0], activation, norm_groups)

    def forward(
        self,
        inputs: Tensor,
        polar_context: Mapping[str, Tensor],
    ) -> Tensor:
        """Return the full-resolution decoder feature for one input stream."""
        feature = self._run(self.stem, inputs)
        skips: list[Tensor] = []
        for level, (encoder, refinement) in enumerate(
            zip(self.encoders, self.post_encoder)
        ):
            context = self._resize_context(polar_context, feature.shape[-2:])
            detail_context = stack_polar_context(context, feature.shape[-2:])
            if self.encoder_kind == "raw":
                encoded = self._run(encoder, feature, detail_context)
            else:
                encoded = self._run(encoder, feature)
            refined = self._run(refinement, encoded, detail_context)
            skips.append(refined)
            if level < len(self.down):
                # No post-encoder feature is fed into the next level. This is
                # the no-fusion-feedback path used by the paper controls.
                feature = self._run(self.down[level], encoded)

        bottleneck_context = self._resize_context(
            polar_context, skips[-1].shape[-2:]
        )
        decoder_feature = self._run(
            self.bottleneck,
            skips[-1],
            stack_polar_context(bottleneck_context, skips[-1].shape[-2:]),
        )
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            context = self._resize_context(polar_context, skip.shape[-2:])
            decoder_feature = self._run(
                block,
                decoder_feature,
                skip,
                stack_polar_context(context, skip.shape[-2:]),
            )
        return decoder_feature

    def _run(self, module: nn.Module, *inputs: Tensor) -> Tensor:
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and any(value.requires_grad for value in inputs)
        ):
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    @staticmethod
    def _resize_context(
        context: Mapping[str, Tensor],
        size: tuple[int, int],
    ) -> dict[str, Tensor]:
        keys = tuple(context)
        values = tuple(context[key] for key in keys)
        if not values:
            return {}
        if values[0].shape[-2:] == size:
            return dict(zip(keys, values))
        stacked = torch.cat(values, dim=1)
        resized = torch.nn.functional.interpolate(
            stacked, size=size, mode="bilinear", align_corners=False
        )
        return {
            key: value
            for key, value in zip(keys, resized.split(1, dim=1))
        }


__all__ = ["IndependentRestorationStream"]
