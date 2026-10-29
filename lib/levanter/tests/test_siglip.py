# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


import tempfile

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import PretrainedConfig
from haliax.partitioning import ResourceAxis
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict
from jax.sharding import Mesh

from levanter.models.siglip import SiglipAttention, SiglipVisionConfig, SiglipVisionModel
from levanter.layers.attention import AttentionMask
from levanter.utils.activation import ActivationFunctionEnum
from PIL import Image
from levanter.testing.helpers import use_test_mesh

from levanter.testing.helpers import skip_if_no_torch


def _hf_siglip_vision_config() -> PretrainedConfig:
    """Return a tiny SiglipVisionConfig for testing."""
    from transformers import (  # noqa: PLC0415  # optional reference
        SiglipVisionConfig as HfSiglipVisionConfig,
    )  # noqa: PLC0415  # optional reference

    return HfSiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )


@skip_if_no_torch
def test_siglip_vision_from_hf_config() -> None:
    """Test conversion from HuggingFace config to Levanter config."""
    hf_config = _hf_siglip_vision_config()
    config = SiglipVisionConfig.from_hf_config(hf_config)

    assert config.hidden_size == hf_config.hidden_size
    assert config.intermediate_size == hf_config.intermediate_size
    assert config.num_hidden_layers == hf_config.num_hidden_layers
    assert config.num_attention_heads == hf_config.num_attention_heads
    assert config.hidden_act == ActivationFunctionEnum.gelu_new


@skip_if_no_torch
def test_siglip_vision_to_hf_config() -> None:
    """Test conversion from Levanter config to HuggingFace config."""
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_act=ActivationFunctionEnum.gelu_new,
    )

    hf_config = config.to_hf_config()

    assert hf_config.hidden_size == config.hidden_size
    assert hf_config.intermediate_size == config.intermediate_size
    assert hf_config.num_hidden_layers == config.num_hidden_layers
    assert hf_config.hidden_act == "gelu_pytorch_tanh"


@skip_if_no_torch
def test_siglip_vision_config_roundtrip() -> None:
    """Test that converting HF -> Levanter -> HF preserves the config."""
    hf_config_1 = _hf_siglip_vision_config()
    levanter_config = SiglipVisionConfig.from_hf_config(hf_config_1)
    hf_config_2 = levanter_config.to_hf_config()

    assert hf_config_2 == hf_config_1


@skip_if_no_torch
def test_siglip_vision_activation_function_conversion() -> None:
    """Test various activation function conversions between HF and Levanter."""
    from transformers import (  # noqa: PLC0415  # optional reference
        SiglipVisionConfig as HfSiglipVisionConfig,
    )  # noqa: PLC0415  # optional reference

    test_cases = [
        ("gelu_pytorch_tanh", ActivationFunctionEnum.gelu_new),
        ("gelu", ActivationFunctionEnum.gelu),
        ("quick_gelu", ActivationFunctionEnum.quick_gelu),
    ]

    for hf_act, expected_lev_act in test_cases:
        hf_config = HfSiglipVisionConfig(hidden_act=hf_act)
        levanter_config = SiglipVisionConfig.from_hf_config(hf_config)
        assert levanter_config.hidden_act == expected_lev_act


@skip_if_no_torch
def test_siglip_vision_model_vs_hf() -> None:
    """Compare SiglipVisionModel output with HuggingFace using a small model."""
    import torch  # noqa: PLC0415  # optional reference
    from transformers import (  # noqa: PLC0415  # optional reference
        SiglipVisionConfig as HfSiglipVisionConfig,
    )  # noqa: PLC0415  # optional reference
    from transformers import (  # noqa: PLC0415  # optional reference
        SiglipVisionModel as HfSiglipVisionModel,
    )  # noqa: PLC0415  # optional reference

    jax.config.update("jax_default_matmul_precision", "float32")

    hf_config = HfSiglipVisionConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )

    torch.manual_seed(42)
    hf_model = HfSiglipVisionModel(hf_config)
    hf_model.eval()

    batch_size = 2
    pixel_values_torch = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        hf_output = hf_model(pixel_values_torch, output_hidden_states=True)
        hf_last_hidden_np = hf_output.last_hidden_state.detach().cpu().numpy()
        hf_pooler_np = hf_output.pooler_output.detach().cpu().numpy()

    lev_config = SiglipVisionConfig.from_hf_config(hf_config)
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        hf_model.save_pretrained(f"{tmpdir}/hf_model")

        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, lev_config, key=jax.random.PRNGKey(0))

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", 224)
        Width = hax.Axis("width", 224)

        pixel_values_jax = hax.named(
            jnp.array(pixel_values_torch.numpy(), dtype=jnp.float32), (Batch, Channels, Height, Width)
        )

        lev_output = lev_model(pixel_values_jax, key=jax.random.PRNGKey(1))

    lev_last_hidden_np = np.array(lev_output.last_hidden_state.array)
    assert lev_output.pooler_output is not None
    lev_pooler_np = np.array(lev_output.pooler_output.array)

    # 4-layer model: use 1e-3 tolerance
    np.testing.assert_allclose(hf_last_hidden_np, lev_last_hidden_np, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(hf_pooler_np, lev_pooler_np, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
@pytest.mark.timeout(600)
@skip_if_no_torch
def test_siglip_vision_pretrained_random_pixels() -> None:
    """Test SigLIP vision model with generated pixels using the HF processor."""
    import torch  # noqa: PLC0415  # optional reference
    from transformers import (  # noqa: PLC0415  # optional reference
        SiglipImageProcessor,
    )  # noqa: PLC0415  # optional reference
    from transformers import (  # noqa: PLC0415  # optional reference
        SiglipVisionModel as HfSiglipVisionModel,
    )  # noqa: PLC0415  # optional reference

    jax.config.update("jax_default_matmul_precision", "float32")

    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    model_name = "google/siglip-base-patch16-224"

    processor = SiglipImageProcessor.from_pretrained(model_name)
    torch_model = HfSiglipVisionModel.from_pretrained(model_name).eval().float()

    inputs = processor(images=image, return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"].float()

    with torch.no_grad():
        vision_outputs = torch_model(pixel_values_torch, output_hidden_states=True)
        torch_output = vision_outputs.last_hidden_state.detach().cpu().numpy()
        torch_pooler = vision_outputs.pooler_output.detach().cpu().numpy()

    lev_config = SiglipVisionConfig.from_hf_config(torch_model.config)
    single_device_mesh = Mesh(np.array([[jax.devices()[0]]]), (ResourceAxis.DATA, ResourceAxis.MODEL))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh(mesh=single_device_mesh):
        torch_model.save_pretrained(f"{tmpdir}/hf_model")

        model_template = eqx.filter_eval_shape(SiglipVisionModel.init, lev_config, key=jax.random.PRNGKey(0))

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=f"{tmpdir}/hf_model")
        state_dict = converter.load_state_dict(f"{tmpdir}/hf_model")
        lev_model = from_torch_compatible_state_dict(model_template, state_dict)

        pixel_values_np = pixel_values_torch.cpu().numpy()
        batch_size, num_channels, height, width = pixel_values_np.shape

        Batch = hax.Axis("batch", batch_size)
        Channels = hax.Axis("channels", num_channels)
        Height = hax.Axis("height", height)
        Width = hax.Axis("width", width)

        pixel_values_jax = hax.named(jnp.array(pixel_values_np, dtype=jnp.float32), (Batch, Channels, Height, Width))

        lev_output = lev_model(pixel_values_jax, key=jax.random.PRNGKey(1))

    lev_output_np = np.array(lev_output.last_hidden_state.array)
    assert lev_output.pooler_output is not None
    lev_pooler_np = np.array(lev_output.pooler_output.array)

    assert torch_output.shape == lev_output_np.shape
    assert not np.any(np.isnan(lev_output_np))
    assert not np.any(np.isinf(lev_output_np))

    # 12-layer full model: numerical differences compound, use looser tolerance
    np.testing.assert_allclose(torch_output, lev_output_np, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(torch_pooler, lev_pooler_np, rtol=1e-3, atol=1e-2)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/converted_model"

        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        converter.save_pretrained(lev_model, save_path)

        converted_hf_model = HfSiglipVisionModel.from_pretrained(save_path)
        converted_hf_model.eval()
        converted_hf_model = converted_hf_model.float()

        with torch.no_grad():
            converted_outputs = converted_hf_model(pixel_values_torch)
            converted_output_np = converted_outputs.last_hidden_state.detach().cpu().numpy()
            converted_pooler_np = converted_outputs.pooler_output.detach().cpu().numpy()

        assert not np.any(np.isnan(converted_output_np))
        np.testing.assert_allclose(lev_output_np, converted_output_np, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(lev_pooler_np, converted_pooler_np, rtol=1e-3, atol=1e-2)


def test_vision_checkpoint_roundtrip_preserves_jitted_predictions() -> None:
    parameters = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=16)
    batch = hax.Axis("batch", 1)
    config = SiglipVisionConfig(**parameters, image_size=32)
    axes = (batch, hax.Axis("channels", 3), hax.Axis("height", 32), hax.Axis("width", 32))
    model = SiglipVisionModel.init(config, key=jax.random.PRNGKey(0))
    other = SiglipVisionModel.init(config, key=jax.random.PRNGKey(1))
    pixels = hax.random.normal(jax.random.PRNGKey(2), axes)

    def forward(network: SiglipVisionModel) -> jax.Array:
        output = network(pixels)
        assert output.pooler_output is not None
        return jnp.concatenate([output.last_hidden_state.array.reshape(-1), output.pooler_output.array.reshape(-1)])

    forward = eqx.filter_jit(forward)
    expected = forward(model)
    assert not np.allclose(expected, forward(other))
    restored = from_torch_compatible_state_dict(other, to_torch_compatible_state_dict(model))
    np.testing.assert_allclose(forward(restored), expected, rtol=1e-05, atol=1e-05)

    def objective(network: SiglipVisionModel) -> jax.Array:
        features = network(pixels).last_hidden_state.array
        return jnp.sum(features * jnp.arange(features.shape[-1]))

    gradients = eqx.filter_grad(objective)(restored)
    leaves = jax.tree.leaves(eqx.filter(gradients, eqx.is_array))
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert any(np.any(leaf != 0) for leaf in leaves)


@skip_if_no_torch
def test_public_vision_checkpoint_loader(tmp_path):
    from transformers import SiglipVisionModel as HfSiglipVisionModel  # noqa: PLC0415 - optional Torch reference
    from transformers import SiglipConfig as HfSiglipConfig  # noqa: PLC0415 - optional Torch reference

    parameters = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=2)
    config = SiglipVisionConfig(**parameters, image_size=4)
    hf_model = HfSiglipVisionModel(config.to_hf_config())
    axes = (hax.Axis("batch", 1), hax.Axis("channels", 3), hax.Axis("height", 4), hax.Axis("width", 4))
    hf_model.save_pretrained(tmp_path)
    HfSiglipConfig(vision_config=hf_model.config.to_dict()).save_pretrained(tmp_path)
    converter = config.hf_checkpoint_converter(str(tmp_path))
    with use_test_mesh():
        loaded = converter.load_pretrained(config.model_type)
        template = config.model_type.init(config, key=jax.random.PRNGKey(1))
        expected = from_torch_compatible_state_dict(template, converter.load_state_dict(str(tmp_path)))
    pixels = hax.random.normal(jax.random.PRNGKey(2), axes)
    forward = eqx.filter_jit(lambda model: model(pixels).last_hidden_state.array)
    np.testing.assert_allclose(forward(loaded), forward(expected), rtol=1e-05, atol=1e-05)
    exported = tmp_path / "exported"
    with use_test_mesh():
        converter.save_pretrained(loaded, str(exported), save_reference_code=False)
        restored = config.hf_checkpoint_converter(str(exported)).load_pretrained(config.model_type)
    np.testing.assert_allclose(forward(restored), forward(loaded), rtol=1e-05, atol=1e-05)


def test_siglip_attention_respects_explicit_mask():
    config = SiglipVisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, patch_size=2
    )
    attention = SiglipAttention.init(config, key=jax.random.PRNGKey(0))
    sequence = hax.Axis("num_patches", 5)
    pixels = hax.random.normal(jax.random.PRNGKey(1), (hax.Axis("batch", 1), sequence, config.Embed))
    mask = AttentionMask.explicit(hax.named(jnp.eye(5, dtype=bool), (sequence, hax.Axis("key_position", 5))))
    actual = eqx.filter_jit(lambda x: attention(x, mask=mask))(pixels)
    # Diagonal attention gives each query exactly its own projected value.
    position = hax.Axis("position", sequence.size)
    values = attention.v_proj(pixels.rename({sequence: position}))
    values = values.broadcast_axis(config.attention_config().QHeadsPerGroup)
    values = values.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
    expected = attention.o_proj(values).rename({position: sequence})
    np.testing.assert_allclose(actual.array, expected.array, rtol=1e-05, atol=1e-05)
