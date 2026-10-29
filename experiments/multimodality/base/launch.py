# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the SigLIP + Delphi 998M image-caption base experiment."""

import argparse
import dataclasses
import json

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.data.image import ImageDatasetSourceConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from transformers import AutoTokenizer, SiglipImageProcessor

from experiments.multimodality.base.model import VlmConfig
from experiments.multimodality.base.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, nargs="+", help="Image-caption JSONL, JSON, or Parquet paths")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--vision-checkpoint")
    parser.add_argument("--text-checkpoint")
    parser.add_argument("--text-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Resolve model configs without loading weights or data")
    args = parser.parse_args()
    model = VlmConfig.from_checkpoints(args.vision_checkpoint, args.text_checkpoint)
    if args.text_length < 1 or args.text_length + model.vision.NumPatches.size > model.text.max_seq_len:
        raise ValueError("text_length must fit alongside the image prefix within the language model context")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    config = TrainConfig(
        model=model,
        data=ImageDatasetSourceConfig(train_urls=args.train_data),
        text_length=args.text_length,
        trainer=TrainerConfig(
            id="siglip-delphi",
            num_train_steps=args.steps,
            train_batch_size=args.batch_size,
            seed=args.seed,
            mp=jmp.get_policy("p=f32,c=bf16,o=f32"),
            tracker=NoopConfig(),
            checkpointer=CheckpointerConfig(base_path=args.output_path, append_run_id_to_base_path=False),
        ),
        optimizer=AdamConfig(learning_rate=args.learning_rate),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "vision_checkpoint": config.model.vision_checkpoint,
                    "text_checkpoint": config.model.text_checkpoint,
                    "vision": dataclasses.asdict(model.vision),
                    "text": dataclasses.asdict(model.text),
                    "vocab_size": model.vocab_size,
                    "image_tokens": model.vision.NumPatches.size,
                    "text_length": config.text_length,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "train_data": args.train_data,
                    "output_path": args.output_path,
                },
                indent=2,
                default=str,
            )
        )
        return
    vision_ref, text_ref = RepoRef.from_string(config.model.vision_checkpoint), RepoRef.from_string(
        config.model.text_checkpoint
    )
    tokenizer = AutoTokenizer.from_pretrained(text_ref.model_name_or_path, revision=text_ref.revision)
    processor = SiglipImageProcessor.from_pretrained(vision_ref.model_name_or_path, revision=vision_ref.revision)
    train(config, tokenizer, processor)


if __name__ == "__main__":
    main()
