#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Launch supervised VLM training from the multimodality template.

Inspect options without starting training:
    uv run python -m experiments.multimodality.base.launch --help

Train with explicit image/conversation data:
    Supply --model_factory MODULE:CALLABLE, --model_name PROCESSOR, and --train_data DATA.

Architecture factories live in recipe modules; the shared training mechanism is in train.py.
"""

import argparse
import asyncio
import dataclasses
import importlib
import logging
from collections.abc import Callable
from typing import cast

import jmp  # For mixed precision policy
from levanter.checkpoint import CheckpointerConfig
from levanter.data.image import ConversationDatasetSourceConfig, ImageMixtureDatasetConfig
from levanter.distributed import DistributedConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem.storage_path import prefix_join

import experiments.multimodality.base.train as train_vlm
from experiments.multimodality.base.vlm_model import VlmConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a configured multimodal experiment")

    parser.add_argument(
        "--model_factory",
        required=True,
        help="Python module:callable accepting the parsed arguments and returning a model configuration",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Build the configuration without loading datasets or starting training"
    )
    parser.add_argument("--vision_checkpoint", default=None, help="Optional separate vision checkpoint")
    parser.add_argument("--llm_checkpoint", default=None, help="Optional separate language-model checkpoint")

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default="./output",
        help="Path to training data. Can be: a single parquet file, a directory containing parquet files, "
        "or a glob pattern (e.g., '/path/to/*.parquet')",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data. Same format as --train_data (defaults to train_data)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/vlm_cache",
        help="Directory for data caching",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching and use streaming mode (processes images on-the-fly, saves disk space)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name for processor and optional weight initialization",
    )
    parser.add_argument(
        "--initialize_from_hf",
        action="store_true",  # Default is False; we use custom weight loading for SigLIP + Qwen3
        help="Initialize model weights from HuggingFace checkpoint (for unified llava-onevision models)",
    )
    parser.add_argument(
        "--use_hf_model_config",
        action="store_true",  # Default is False; use custom SigLIP + Qwen3 config
        help="Use model config from HuggingFace checkpoint (set to True to load full llava-onevision model)",
    )
    parser.add_argument(
        "--use_small_model",
        action="store_true",
        help="Use small model config for testing (overrides --use_hf_model_config)",
    )

    # Training arguments
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=20000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of epochs to train (default: 1). If 0, train indefinitely until num_train_steps is reached. "
        "If > 0, dataset will cycle through the data for the specified number of epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )

    # === Performance Optimization Arguments ===
    parser.add_argument(
        "--mp",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", None],
        help="Mixed precision mode: bfloat16 (recommended for TPU), float16 (GPU), or float32 (full precision)",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention (enabled by default for memory-efficient attention computation)",
    )
    parser.add_argument(
        "--flash_attention_block_size",
        type=int,
        default=1024,
        help="Block size for flash attention (default: 512, use smaller values if OOM)",
    )
    parser.add_argument(
        "--per_device_parallelism",
        type=int,
        default=-1,
        help="Number of examples to process per device. -1 means train_batch_size/num_devices. "
        "Set lower for gradient accumulation to save memory.",
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        action="store_true",
        help="Freeze vision encoder weights (only train projector and LLM). "
        "Reduces compute by ~30% and often improves fine-tuning results.",
    )
    parser.add_argument(
        "--freeze_llm",
        action="store_true",
        help="Freeze LLM weights (only train projector and vision encoder). "
        "Useful for vision encoder fine-tuning or projector-only training.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default to reduce memory usage)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/vlm_output",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--hf_save_path",
        type=str,
        default=None,
        help="Path to save HuggingFace format checkpoints",
    )
    parser.add_argument(
        "--hf_save_steps",
        type=int,
        default=1000,
        help="Save HF checkpoint every N steps",
    )
    parser.add_argument(
        "--checkpointer_path",
        type=str,
        default=None,
        help="Path for Levanter checkpoints (defaults to output_dir/checkpoints)",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="marin-vlm",
        help="Weights & Biases project name (None to disable)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    # Distributed arguments
    parser.add_argument(
        "--no_distributed",
        action="store_true",
        help="Disable JAX distributed initialization",
    )

    # Evaluation arguments
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=10,
        help="Maximum number of evaluation batches",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=500,  # Default to less frequent eval to reduce memory pressure from dual JIT
        help="How often to run evaluation (in steps). Higher values reduce JIT compilation memory overhead.",
    )
    parser.add_argument(
        "--per_device_eval_parallelism",
        type=int,
        default=-1,  # Same as training to potentially reuse XLA compilation cache
        help="Number of examples to process per device during evaluation. " "Default: -1 (same as training batch size).",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Disable evaluation completely to save memory",
    )

    return parser.parse_args()


def get_model_config(args: argparse.Namespace) -> VlmConfig:
    """Build a model configuration through the experiment's explicit factory."""
    module_name, separator, function_name = args.model_factory.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("model_factory must have the form module:callable")
    module = importlib.import_module(module_name)
    factory = cast(Callable[[argparse.Namespace], VlmConfig], getattr(module, function_name))
    return factory(args)


def main():
    args = parse_args()

    # Set validation data to train data if not specified
    if args.val_data is None:
        args.val_data = args.train_data

    logger.info("=" * 60)
    logger.info("VLM Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Initialize from HF: {args.initialize_from_hf}")
    logger.info(f"Num train steps: {args.num_train_steps}")
    logger.info(f"Batch size: {args.train_batch_size}")

    # Log performance optimization settings
    logger.info("-" * 60)
    logger.info("Performance Optimizations:")
    logger.info(f"  Mixed precision: {args.mp or 'disabled (float32)'}")
    logger.info(f"  Flash attention: {not args.no_flash_attention}")
    logger.info(f"  Freeze vision encoder: {args.freeze_vision_encoder}")
    logger.info(f"  Per-device parallelism: {args.per_device_parallelism}")
    logger.info(f"  Gradient checkpointing: {not args.no_gradient_checkpointing}")
    logger.info("-" * 60)

    data_config = ImageMixtureDatasetConfig(
        cache_dir=args.cache_dir,
        configs={
            "train": ConversationDatasetSourceConfig(
                train_urls=[f"file://{args.train_data}"],
                validation_urls=[f"file://{args.val_data}"],
                cache_dir=prefix_join(args.cache_dir, "train"),
            ),
        },
        train_weights={"train": 1.0},
        processor=args.model_name,
        max_length=args.max_length,
        use_cache=not args.no_cache,  # Use streaming mode if --no_cache is set
    )

    if args.no_cache:
        logger.info("Using streaming mode (no caching) - images will be processed on-the-fly")

    if args.dry_run:
        model = get_model_config(args)
        logger.info(
            "Configured model: %s; processor: %s; training steps: %s",
            model.model_type.__name__,
            args.model_name,
            args.num_train_steps,
        )
        return

    # Log dataset file count
    logger.info("-" * 60)
    logger.info("Dataset Files:")
    for name, source_config in data_config.configs.items():
        train_urls = source_config.urls_for_split("train")
        val_urls = source_config.urls_for_split("validation")
        logger.info(f"  {name}: {len(train_urls)} train file(s), {len(val_urls)} validation file(s)")
    logger.info("-" * 60)

    # Calculate num_train_steps based on epoch if specified
    num_train_steps = args.num_train_steps
    if args.epoch > 0:
        # Build training datasets to get the actual dataset size
        logger.info("Building training datasets to calculate epoch-based steps...")
        train_datasets = data_config.training_sets()

        # Calculate total dataset size from all training datasets
        total_dataset_size = 0
        for name, ds in train_datasets.items():
            ds_len = asyncio.run(ds.async_len())
            total_dataset_size += ds_len
            logger.info(f"  Dataset '{name}': {ds_len:,} samples")

        if total_dataset_size > 0:
            # Calculate steps needed for the specified number of epochs
            steps_per_epoch = total_dataset_size // args.train_batch_size
            epoch_based_steps = steps_per_epoch * args.epoch
            num_train_steps = epoch_based_steps
            logger.info(
                f"Epoch-based training: {args.epoch} epoch(s) = {num_train_steps:,} steps "
                f"({total_dataset_size:,} samples / {args.train_batch_size} batch_size * {args.epoch} epochs)"
            )
        else:
            logger.warning("Could not determine dataset size, using --num_train_steps instead")

    # Create model config with optimizations
    model_config = get_model_config(args)

    # Create optimizer config
    warmup_steps = int(num_train_steps * args.warmup_ratio)
    optimizer_config = AdamConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup=warmup_steps,
    )

    # Create tracker config
    if args.wandb_project:
        tracker_config = WandbConfig(
            project=args.wandb_project,
            name=args.wandb_run_name,
        )
    else:
        tracker_config = NoopConfig()

    # Create distributed config
    distributed_config = DistributedConfig(initialize_jax_distributed=not args.no_distributed)

    # Set checkpoint path
    checkpointer_path = args.checkpointer_path or prefix_join(args.output_dir, "checkpoints")
    checkpointer_config = CheckpointerConfig(base_path=checkpointer_path)

    # Convert mixed precision string to jmp.Policy
    # jmp.get_policy accepts strings like "f32", "bf16", "bfloat16", or
    # "compute=bfloat16,params=float32,output=float32"
    if args.mp:
        mp_policy = jmp.get_policy(args.mp)
    else:
        mp_policy = jmp.get_policy("f32")  # Default to full precision

    # Create trainer config with performance optimizations
    trainer_config = train_vlm.TrainerConfig(
        num_train_steps=num_train_steps,
        train_batch_size=args.train_batch_size,
        per_device_parallelism=args.per_device_parallelism,
        per_device_eval_parallelism=args.per_device_eval_parallelism,  # Smaller eval batch to save memory
        max_eval_batches=args.max_eval_batches,
        steps_per_eval=args.steps_per_eval,
        tracker=tracker_config,
        checkpointer=checkpointer_config,
        distributed=distributed_config,
        # Mixed precision configuration
        mp=mp_policy,
    )

    # Create main training config
    config = train_vlm.TrainVLMConfig(
        data=data_config,
        model=model_config,
        trainer=trainer_config,
        optimizer=optimizer_config,
        # Disable HF loading when using custom config - we'll load weights separately
        initialize_from_hf=args.model_name if args.initialize_from_hf else None,
        use_hf_model_config=args.use_hf_model_config and not args.use_small_model,
        hf_save_path=args.hf_save_path,
        hf_save_steps=args.hf_save_steps,
        # Custom weight loading paths for hybrid model
        vision_checkpoint=args.vision_checkpoint,
        llm_checkpoint=args.llm_checkpoint,
        # Evaluation control
        no_eval=args.no_eval,
        # Epoch control
        epoch=args.epoch,
    )

    # Handle freezing if requested
    if args.freeze_vision_encoder:
        config = dataclasses.replace(config, freeze_vision_encoder=True)
    if args.freeze_llm:
        config = dataclasses.replace(config, freeze_llm=True)

    logger.info("=" * 60)
    logger.info("Starting VLM training...")
    logger.info(f"Checkpoints will be saved to: {checkpointer_path}")
    if args.hf_save_path:
        logger.info(f"HF checkpoints will be saved to: {args.hf_save_path}")
    if args.epoch > 0:
        logger.info(f"Training for {args.epoch} epoch(s) ({num_train_steps:,} steps)")
    else:
        logger.info(f"Training for {num_train_steps:,} steps (no epoch limit)")

    # Run training
    train_vlm.main(config)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
