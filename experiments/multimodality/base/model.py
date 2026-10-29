# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SigLIP image prefixes conditioned into a Delphi (Qwen3) language model."""

from dataclasses import dataclass
from typing import cast

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
from haliax import Axis, NamedArray
from haliax.partitioning import ResourceMapping
from jaxtyping import PRNGKeyArray
from levanter.layers.attention import AttentionMask
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.models.siglip import SiglipVisionConfig, SiglipVisionModel


@dataclass(frozen=True)
class VlmConfig:
    vision: SiglipVisionConfig
    text: Qwen3Config
    vocab_size: int

    @property
    def VisionEmbed(self) -> Axis:
        return Axis("vision_embed", self.vision.hidden_size)


class Vlm(eqx.Module):
    config: VlmConfig = eqx.field(static=True)
    vision: SiglipVisionModel
    projector: hnn.Linear
    language: Qwen3LMHeadModel

    @classmethod
    def init(cls, config: VlmConfig, *, key: PRNGKeyArray) -> "Vlm":
        vision_key, projector_key, text_key = jax.random.split(key, 3)
        return cls(
            config,
            SiglipVisionModel.init(Axis("vocab", 0), config.vision, key=vision_key),
            hnn.Linear.init(config.VisionEmbed, config.text.Embed, key=projector_key, use_bias=True),
            Qwen3LMHeadModel.init(Axis("vocab", config.vocab_size), config.text, key=text_key),
        )

    def activations(self, input_ids: NamedArray, pixel_values: NamedArray, *, key=None) -> NamedArray:
        """Return text-position states after attending to all projected image patches."""
        text_position = input_ids.resolve_axis("position")
        prefix_length = self.config.vision.NumPatches.size
        if prefix_length + text_position.size > self.config.text.max_seq_len:
            raise ValueError("Image patches plus text exceed the language model context length")
        vision_key, text_key = (None, None) if key is None else jax.random.split(key)
        pixel_values = pixel_values.astype(self.vision.vision_model.embeddings.patch_embedding.weight.dtype)
        features = self.vision(pixel_values, key=vision_key).last_hidden_state
        features = features.rename({"embed": "vision_embed", "num_patches": "position"})
        prefix = self.projector(features)
        tokens = self.language.embeddings.embed(input_ids)
        hidden = hax.concatenate("position", [prefix, tokens])
        hidden = self.language.transformer(hidden, AttentionMask.causal(), key=text_key)
        return hax.slice(hidden, "position", start=prefix_length, length=text_position.size)

    def __call__(self, input_ids: NamedArray, pixel_values: NamedArray, *, key=None) -> NamedArray:
        hidden = self.activations(input_ids, pixel_values, key=key)
        return hax.dot(hidden, self.language.get_lm_head(), axis=self.config.text.Embed)


def load_pretrained(
    config: VlmConfig,
    vision_checkpoint: str,
    text_checkpoint: str,
    *,
    key: PRNGKeyArray,
    axis_mapping: ResourceMapping | None = None,
) -> Vlm:
    """Load both backbones and initialize only the image projection from scratch."""
    vision = config.vision.hf_checkpoint_converter(vision_checkpoint).load_pretrained(
        SiglipVisionModel, config=config.vision, axis_mapping=axis_mapping
    )
    language = config.text.hf_checkpoint_converter(text_checkpoint).load_pretrained(
        Qwen3LMHeadModel, config=config.text, axis_mapping=axis_mapping
    )
    projector = hnn.Linear.init(config.VisionEmbed, config.text.Embed, key=key, use_bias=True)
    return Vlm(config, cast(SiglipVisionModel, vision), projector, cast(Qwen3LMHeadModel, language))
