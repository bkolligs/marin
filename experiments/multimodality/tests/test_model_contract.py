# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the experiment loss with a model that does not use OneVision."""

from dataclasses import dataclass
from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from experiments.multimodality.base.train import compute_vlm_loss


@dataclass(frozen=True)
class TinyConfig:
    TextEmbed: hax.Axis


class TinyModel(eqx.Module):
    config: TinyConfig = eqx.field(static=True)
    Vocab: hax.Axis = eqx.field(static=True)
    features: hax.NamedArray
    head: hax.NamedArray

    def forward_with_activations(self, input_ids, *, pixel_values=None, grid_mask=None, unpad_indices=None, key=None):
        axes = (*input_ids.axes, self.config.TextEmbed)
        features = hax.named(jnp.broadcast_to(self.features.array, tuple(axis.size for axis in axes)), axes)
        return features, self.head


def test_alternative_model_masked_loss_and_gradients_match_reference():
    embed, vocab = hax.Axis("embed", 2), hax.Axis("vocab", 4)
    model = TinyModel(
        TinyConfig(embed),
        vocab,
        hax.named(jnp.array([0.2, -0.3]), embed),
        hax.named(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]), (vocab, embed)),
    )
    position = hax.Axis("position", 4)
    example = SimpleNamespace(
        input_ids=hax.named(jnp.array([0, 1, 2, 3]), position),
        pixel_values=None,
        grid_mask=None,
        unpad_indices=None,
        loss_mask=hax.named(jnp.array([0.0, 0.0, 1.0, 1.0]), position),
    )

    def reference(features):
        log_probs = jax.nn.log_softmax(model.head.array @ features)
        return -(log_probs[2] + log_probs[3]) / 2

    actual = compute_vlm_loss(model, example, block_size=None)
    np.testing.assert_allclose(actual.scalar(), reference(model.features.array), rtol=1e-5, atol=1e-5)
    gradients = eqx.filter_grad(lambda network: compute_vlm_loss(network, example, block_size=None).scalar())(model)
    np.testing.assert_allclose(gradients.features.array, jax.grad(reference)(model.features.array), rtol=1e-5, atol=1e-5)
