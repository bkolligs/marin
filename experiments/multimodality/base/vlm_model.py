# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts implemented by model configurations used in multimodal experiments."""

from typing import Any, Protocol

from haliax import Axis, NamedArray
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HfSerializableConfig


class VlmConfig(HfSerializableConfig, Protocol):
    vision_config: Any
    text_config: Any
    image_grid_pinpoints: list[list[int]] | None

    @property
    def TextEmbed(self) -> Axis: ...

    def hf_checkpoint_converter(self, ref_checkpoint: str | None = None) -> HFCheckpointConverter: ...


class VlmModel(Protocol):
    config: VlmConfig
    Vocab: Axis

    def forward_with_activations(
        self, input_ids: NamedArray, *, pixel_values=None, grid_mask=None, unpad_indices=None, key=None
    ) -> tuple[NamedArray, NamedArray]: ...
