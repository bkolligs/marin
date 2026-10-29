# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-process image-caption training for the copy-and-edit VLM template."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis, NamedArray
from levanter.data.image import ImageDatasetSourceConfig, load_image
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.optim.config import AdamConfig
from levanter.trainer import Trainer, TrainerConfig
from transformers import PreTrainedTokenizerBase, SiglipImageProcessor

from experiments.multimodality.base.model import Vlm, VlmConfig, load_pretrained


class CaptionBatch(eqx.Module):
    input_ids: NamedArray
    targets: NamedArray
    loss_mask: NamedArray
    pixel_values: NamedArray


@dataclass(frozen=True)
class TrainConfig:
    model: VlmConfig
    vision_checkpoint: str
    text_checkpoint: str
    data: ImageDatasetSourceConfig
    text_length: int
    trainer: TrainerConfig
    optimizer: AdamConfig


def caption_batch(
    records: Sequence[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    image_processor: SiglipImageProcessor,
    *,
    text_length: int,
    batch_axis_name: str = "batch",
) -> CaptionBatch:
    """Encode BOS+caption inputs and caption+EOS targets, masking right padding."""
    if text_length < 1:
        raise ValueError("text_length must be positive")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Caption training requires tokenizer BOS and EOS tokens")
    batch, position = Axis(batch_axis_name, len(records)), Axis("position", text_length)
    input_ids = np.full((batch.size, text_length), tokenizer.eos_token_id, dtype=np.int32)
    targets = input_ids.copy()
    mask = np.zeros(input_ids.shape, dtype=np.float32)
    for index, record in enumerate(records):
        caption = tokenizer.encode(record["text"], add_special_tokens=False, truncation=True, max_length=text_length - 1)
        tokens = [tokenizer.bos_token_id, *caption, tokenizer.eos_token_id]
        length = len(tokens) - 1
        input_ids[index, :length] = tokens[:-1]
        targets[index, :length] = tokens[1:]
        mask[index, :length] = 1
    pixels = np.asarray(
        image_processor(images=[load_image(record["image"]) for record in records], return_tensors="np")["pixel_values"]
    )
    _, channels, height, width = pixels.shape
    return CaptionBatch(
        hax.named(jnp.asarray(input_ids), (batch, position)),
        hax.named(jnp.asarray(targets), (batch, position)),
        hax.named(jnp.asarray(mask), (batch, position)),
        hax.named(
            jnp.asarray(pixels), (batch, Axis("channels", channels), Axis("height", height), Axis("width", width))
        ),
    )


def compute_loss(model: Vlm, batch: CaptionBatch, *, key=None) -> jax.Array:
    """Mean caption-token cross entropy, excluding image positions and padding."""
    hidden = model.activations(batch.input_ids, batch.pixel_values, key=key)
    losses = fused_cross_entropy_loss_and_logsumexp_penalty(
        hidden,
        model.language.get_lm_head(),
        model.config.text.Embed,
        model.language.Vocab,
        batch.targets,
        reduction=None,
        block_size=4096,
    )
    return (hax.sum(losses * batch.loss_mask) / hax.sum(batch.loss_mask)).scalar()


def training_batches(
    config: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    image_processor: SiglipImageProcessor,
) -> Iterator[CaptionBatch]:
    """Repeat source-order epochs, carrying partial batches across epoch boundaries."""
    pending = []
    batch_size = config.trainer.TrainBatch.size
    while True:
        saw_record = False
        for record in config.data.doc_iterator(config.data.train_split):
            saw_record = True
            pending.append(record)
            if len(pending) == batch_size:
                yield caption_batch(
                    pending,
                    tokenizer,
                    image_processor,
                    text_length=config.text_length,
                    batch_axis_name=config.trainer.TrainBatch.name,
                )
                pending = []
        if not saw_record:
            raise ValueError("Training data contains no image-caption records")


def train(config: TrainConfig, tokenizer: PreTrainedTokenizerBase, image_processor: SiglipImageProcessor) -> None:
    """Train all model parameters with Levanter optimizer state and checkpoints."""
    config.trainer.initialize()
    if jax.process_count() != 1:
        raise ValueError("This template's streaming caption loader supports one JAX process")
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    with Trainer(config.trainer, optimizer, compute_loss) as trainer:
        model_key, training_key = jax.random.split(jax.random.PRNGKey(config.trainer.seed))
        _, restore = trainer.checkpoint_load_plan()
        if restore:
            state = trainer.initial_state(training_key, model_init=lambda: Vlm.init(config.model, key=model_key))
        else:
            model = load_pretrained(
                config.model,
                config.vision_checkpoint,
                config.text_checkpoint,
                key=model_key,
                axis_mapping=trainer.parameter_axis_mapping,
            )
            model = hax.shard(trainer.mp.cast_to_param(model), trainer.parameter_axis_mapping)
            state = trainer.initial_state(training_key, model=model)
        batches = islice(training_batches(config, tokenizer, image_processor), int(state.step), None)
        trainer.train(state, (hax.shard(batch, trainer.compute_axis_mapping) for batch in batches))

    trainer.tracker.finish()
