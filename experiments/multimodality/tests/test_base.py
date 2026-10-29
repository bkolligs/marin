# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.layers.attention import AttentionBackend
from levanter.models.qwen import Qwen3Config
from levanter.models.siglip import SiglipVisionConfig
from levanter.testing.helpers import use_test_mesh
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, SiglipImageProcessor

from experiments.multimodality.base.model import Vlm, VlmConfig, load_pretrained
from experiments.multimodality.base.train import CaptionBatch, caption_batch, compute_loss


@pytest.fixture
def config():
    return VlmConfig(
        SiglipVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=4,
            patch_size=2,
            attn_backend=AttentionBackend.VANILLA,
            use_flash_attention=False,
        ),
        Qwen3Config(
            hidden_dim=8,
            intermediate_dim=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=16,
            head_dim=4,
            use_sliding_window=False,
            attn_backend=AttentionBackend.VANILLA,
        ),
        vocab_size=8,
    )


@pytest.fixture
def batch():
    batch_axis, position = hax.Axis("batch", 1), hax.Axis("position", 4)
    return CaptionBatch(
        hax.named(jnp.array([[1, 3, 4, 2]]), (batch_axis, position)),
        hax.named(jnp.array([[3, 4, 2, 2]]), (batch_axis, position)),
        hax.named(jnp.array([[1.0, 1.0, 1.0, 0.0]]), (batch_axis, position)),
        hax.random.normal(
            jax.random.PRNGKey(1), (batch_axis, hax.Axis("channels", 3), hax.Axis("height", 4), hax.Axis("width", 4))
        ),
    )


def test_loss_and_gradient_match_masked_caption_reference(config, batch):
    model = Vlm.init(config, key=jax.random.PRNGKey(0))

    def reference(network):
        logits = network(batch.input_ids, batch.pixel_values).rearrange(("batch", "position", "vocab")).array
        log_probs = jax.nn.log_softmax(logits)
        losses = -jnp.take_along_axis(log_probs, batch.targets.array[..., None], axis=-1)[..., 0]
        return jnp.sum(losses * batch.loss_mask.array) / jnp.sum(batch.loss_mask.array)

    actual, gradient = eqx.filter_jit(eqx.filter_value_and_grad(compute_loss))(model, batch)
    expected, expected_gradient = eqx.filter_jit(eqx.filter_value_and_grad(reference))(model)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected_gradient), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-4, atol=1e-5)
    for component in (gradient.vision, gradient.projector, gradient.language):
        assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(component))
    updated = eqx.apply_updates(model, jax.tree.map(lambda leaf: -1e-3 * leaf, gradient))
    assert compute_loss(updated, batch) < actual


def test_image_conditioning_and_causal_text_prefix(config, batch):
    model = Vlm.init(config, key=jax.random.PRNGKey(0))
    predict = eqx.filter_jit(lambda ids, pixels: model(ids, pixels).rearrange(("batch", "position", "vocab")).array)
    original = predict(batch.input_ids, batch.pixel_values)
    other_image = predict(batch.input_ids, batch.pixel_values * -1)
    assert not np.allclose(original[:, :1], other_image[:, :1])
    later_tokens = dataclasses.replace(batch.input_ids, array=batch.input_ids.array.at[:, 2:].set(6))
    changed = predict(later_tokens, batch.pixel_values)
    np.testing.assert_allclose(original[:, :2], changed[:, :2], rtol=1e-5, atol=1e-5)
    changed_padding = dataclasses.replace(
        batch, input_ids=later_tokens, loss_mask=hax.named(jnp.array([[1.0, 1.0, 0.0, 0.0]]), batch.loss_mask.axes)
    )
    original_prefix = dataclasses.replace(batch, loss_mask=changed_padding.loss_mask)
    np.testing.assert_allclose(
        compute_loss(model, original_prefix), compute_loss(model, changed_padding), rtol=1e-5, atol=1e-5
    )


def test_bfloat16_model_accepts_float32_images(config, batch):
    model = Vlm.init(config, key=jax.random.PRNGKey(0))
    model = jmp.get_policy("p=f32,c=bf16,o=f32").cast_to_compute(model)
    forward = eqx.filter_jit(lambda pixels: model(batch.input_ids, pixels).array)
    actual = forward(batch.pixel_values)
    expected = forward(batch.pixel_values.astype(jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32))


def test_caption_processing_aligns_bos_eos_truncation_and_padding():
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "red": 3, "blue": 4, "green": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="[BOS]", eos_token="[EOS]")
    processor = SiglipImageProcessor(size={"height": 4, "width": 4})
    records = [
        {"image": Image.new("RGB", (8, 6), "red"), "text": "red"},
        {"image": Image.new("RGB", (3, 7), "blue"), "text": "red blue green red blue"},
    ]
    processed = caption_batch(records, tokenizer, processor, text_length=4)
    np.testing.assert_array_equal(processed.input_ids.array, [[1, 3, 2, 2], [1, 3, 4, 5]])
    np.testing.assert_array_equal(processed.targets.array, [[3, 2, 2, 2], [3, 4, 5, 2]])
    np.testing.assert_array_equal(processed.loss_mask.array, [[1, 1, 0, 0], [1, 1, 1, 1]])
    assert processed.pixel_values.array.shape == (2, 3, 4, 4)
    np.testing.assert_allclose(processed.pixel_values.array[0, :, 0, 0], [1.0, -1.0, -1.0])


def test_local_pretrained_backbones_and_model_roundtrip(config, batch, tmp_path):
    model = Vlm.init(config, key=jax.random.PRNGKey(0))
    vision_path, text_path = tmp_path / "vision", tmp_path / "text"
    with use_test_mesh():
        model.vision.config.hf_checkpoint_converter().save_pretrained(
            model.vision, str(vision_path), save_tokenizer=False
        )
        converter = dataclasses.replace(model.language.config.hf_checkpoint_converter(), reference_checkpoint=None)
        converter.save_pretrained(model.language, str(text_path), save_tokenizer=False)
    # The LM converter reads tokenizer metadata to establish its vocabulary axis.
    backend = Tokenizer(WordLevel({str(i): i for i in range(config.vocab_size)}, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(text_path)
    with use_test_mesh():
        loaded = load_pretrained(config, str(vision_path), str(text_path), key=jax.random.PRNGKey(3))
    for original, restored in ((model.vision, loaded.vision), (model.language, loaded.language)):
        for expected, actual in zip(jax.tree.leaves(original), jax.tree.leaves(restored), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    expected = loaded(batch.input_ids, batch.pixel_values)
    path = tmp_path / "vlm.eqx"
    eqx.tree_serialise_leaves(path, loaded)
    restored = eqx.tree_deserialise_leaves(path, Vlm.init(config, key=jax.random.PRNGKey(4)))
    np.testing.assert_allclose(restored(batch.input_ids, batch.pixel_values).array, expected.array, rtol=1e-5, atol=1e-5)
