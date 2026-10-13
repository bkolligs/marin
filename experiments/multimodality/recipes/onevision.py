# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the OneVision baseline to the generic experiment launcher."""

import argparse

from levanter.models.llava_onevision import LlavaOnevisionConfig
from levanter.models.qwen import QwenConfig
from levanter.models.siglip import SiglipVisionConfig
from transformers import AutoConfig


def model_config(args: argparse.Namespace) -> LlavaOnevisionConfig:
    if args.use_hf_model_config:
        return LlavaOnevisionConfig.from_hf_config(AutoConfig.from_pretrained(args.model_name))
    if not args.use_small_model:
        raise ValueError("Select --use_small_model or --use_hf_model_config for the OneVision recipe.")
    return LlavaOnevisionConfig(
        vision_config=SiglipVisionConfig(
            hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, image_size=384
        ),
        text_config=QwenConfig(
            hidden_dim=128, intermediate_dim=512, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=args.max_length
        ),
    )
