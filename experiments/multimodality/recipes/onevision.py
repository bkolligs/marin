# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the OneVision baseline to the generic experiment launcher."""

import argparse
import dataclasses

from levanter.layers.attention import AttentionBackend
from levanter.models.llava_onevision import LlavaOnevisionConfig
from levanter.models.qwen import QwenConfig
from levanter.models.siglip import SiglipVisionConfig
from transformers import AutoConfig


def model_config(args: argparse.Namespace) -> LlavaOnevisionConfig:
    if args.use_hf_model_config:
        config = LlavaOnevisionConfig.from_hf_config(AutoConfig.from_pretrained(args.model_name))
    elif args.use_small_model:
        config = LlavaOnevisionConfig(
            vision_config=SiglipVisionConfig(
                hidden_size=64,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=384,
            ),
            text_config=QwenConfig(
                hidden_dim=128,
                intermediate_dim=512,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                max_seq_len=args.max_length,
            ),
        )
    else:
        raise ValueError("Select --use_small_model or --use_hf_model_config for the OneVision recipe.")
    checkpointing = not args.no_gradient_checkpointing
    attention = AttentionBackend.VANILLA if args.no_flash_attention else AttentionBackend.DEFAULT
    return dataclasses.replace(
        config,
        gradient_checkpointing=checkpointing,
        vision_config=dataclasses.replace(
            config.vision_config,
            gradient_checkpointing=checkpointing,
            use_flash_attention=not args.no_flash_attention,
            attn_backend=attention,
            flash_attention_block_size=args.flash_attention_block_size,
        ),
        text_config=dataclasses.replace(
            config.text_config,
            gradient_checkpointing=checkpointing,
            attn_backend=attention,
            flash_attention_block_size=args.flash_attention_block_size,
        ),
    )
