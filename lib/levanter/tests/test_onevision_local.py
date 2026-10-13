# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
from levanter.models.qwen import QwenConfig
from levanter.models.siglip import SiglipVisionConfig


def test_jitted_multimodal_predictions_depend_on_image_pixels():
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
    )
    text = QwenConfig(
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
    )
    config = LlavaOnevisionConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=30,
        pad_token_id=0,
        image_grid_pinpoints=[[32, 32]],
    )
    model = LlavaOnevisionModel.init(hax.Axis("vocab", 32), config, key=jax.random.PRNGKey(0))
    batch, position, tiles = hax.Axis("batch", 1), hax.Axis("position", 6), hax.Axis("num_patches", 1)
    tokens = hax.named(jnp.array([[1, 30, 30, 30, 30, 2]]), (batch, position))
    axes = (batch, tiles, hax.Axis("channels", 3), hax.Axis("height", 32), hax.Axis("width", 32))
    pixels = hax.random.normal(jax.random.PRNGKey(1), axes)
    mask = hax.ones((batch, tiles), dtype=jnp.bool_)
    forward = eqx.filter_jit(lambda image: model(tokens, pixel_values=image, grid_mask=mask).array)
    original, changed = forward(pixels), forward(-pixels)
    assert np.isfinite(original).all() and np.isfinite(changed).all()
    # Compare the final text position, which can attend to preceding visual embeddings.
    assert not np.allclose(original[:, -1], changed[:, -1])
