# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Image processing and fixed-shape feature ordering for the OneVision baseline."""

# =============================================================================
# LLaVA-OneVision Processor Classes
# =============================================================================
# Adapted from HuggingFace Transformers library:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/processing_llava_onevision.py
#
# Original code copyright 2024 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
#
# We acknowledge the LLaVA-OneVision team for their excellent work:
# https://github.com/LLaVA-VL/LLaVA-NeXT
# Paper: https://arxiv.org/abs/2408.03326
#
# These classes provide custom processor implementation for LLaVA-OneVision models
# with additional support for padding mode and fixed-shape processing.


import math
from collections.abc import Iterable
from typing import Any, Optional, Union

import numpy as np
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.image_processing_utils import select_best_resolution
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import MultiModalData, ProcessingKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput
from transformers.utils import logging as transformers_logging

_processor_logger = transformers_logging.get_logger(__name__)


class LlavaOnevisionProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    pass


LlavaOnevisionProcessorKwargs._defaults = {
    "text_kwargs": {
        "padding": False,
        "return_mm_token_type_ids": False,
    },
    "image_kwargs": {},
}


class LlavaOnevisionProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-Onevision processor which wraps a LLaVa-Onevision video processor, LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaOnevisionVideoProcessor`], [`LlavaOnevisionImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaOnevisionVideoProcessor.__call__`], [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaOnevisionImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`LlavaOnevisionVideoProcessor`], *optional*):
            The video processor is a required input.
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LlavaOnevisionImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    video_processor_class = "LlavaOnevisionVideoProcessor"
    optional_attributes = ["video_processor", "chat_template"]

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        video_processor: Any = None,
        num_image_tokens: int | None = None,
        vision_feature_select_strategy: str | None = None,
        chat_template: str | None = None,
        image_token: str = "<image>",
        video_token: str = "<video>",
        vision_aspect_ratio: str = "anyres_max_9",
        max_image_tiles: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if num_image_tokens is None:
            raise ValueError("OneVision requires the vision encoder token count")
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_aspect_ratio = vision_aspect_ratio

        # For padding mode: max_image_tiles is the total number of tiles (including base)
        # e.g., for anyres_max_9, max_image_tiles = 9 + 1 = 10
        self.max_image_tokens: int | None
        self.max_image_tiles = max_image_tiles
        if max_image_tiles is not None and num_image_tokens is not None:
            self.max_image_tokens = max_image_tiles * num_image_tokens
        else:
            self.max_image_tokens = None

        super().__init__(image_processor, tokenizer, video_processor=video_processor, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        videos: Optional[VideoInput] = None,
        padding_mode: bool = False,
        **kwargs: Any,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is None:
            raise ValueError("Text is required for multimodal sequence processing.")
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or not all(isinstance(item, str) for item in text):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = video_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            batch_num_images = iter(image_inputs["batch_num_images"])
            image_sizes = iter(image_inputs["image_sizes"])
            height, width = get_image_size(
                to_numpy_array(image_inputs["pixel_values"][0][0]),
                channel_dim=output_kwargs["images_kwargs"].get("data_format"),
            )
            text, num_image_tokens = self._expand_image_tokens(
                text,
                image_sizes,
                height,
                width,
                self.image_token,
                batch_num_images,
                padding_mode=padding_mode,
            )

        if videos is not None:
            video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])

            one_video = video_inputs.get("pixel_values_videos")[0]
            if isinstance(video_inputs.get("pixel_values_videos")[0], (list, tuple)):
                one_video = np.array(one_video)
            else:
                one_video = to_numpy_array(one_video)
            height, width = get_image_size(one_video[0], channel_dim=output_kwargs["images_kwargs"].get("data_format"))
            num_frames = one_video.shape[0]  # frame dim is always after batch dim
            patches_height_width = int(math.sqrt(self.num_image_tokens))
            pooled_height_width = math.ceil(patches_height_width / 2)
            num_video_tokens = num_frames * pooled_height_width * pooled_height_width
            text = [sample.replace(self.video_token, self.video_token * num_video_tokens) for sample in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=return_tensors)

    def _expand_image_tokens(
        self,
        text: list[TextInput],
        image_sizes: Iterable[Union[list[int], int]],
        height: int,
        width: int,
        special_token: str,
        batch_num_images: Iterable[int],
        padding_mode: bool = False,
    ) -> tuple[list[str], int]:
        prompt_strings = []
        max_num_vision_tokens = 0
        for sample in text:
            if special_token in sample:
                # Count actual number of image tokens in the sample
                # batch_num_images may not be reliable for multi-image
                num_images = sample.count(special_token)
                _ = next(batch_num_images)  # consume iterator to stay in sync
                is_multi_image = num_images != 1
            else:
                is_multi_image = False
                num_images = 0
            while special_token in sample:
                original_size = next(image_sizes)  # should consume iterable

                # In padding mode:
                # - Multi-image: use base tokens only (729) - no anyres for multi-image
                # - Single image: use max tokens (7290) for JIT compatibility
                if padding_mode and self.max_image_tokens is not None:
                    if is_multi_image:
                        num_image_tokens = self.num_image_tokens  # Base patch only
                    else:
                        num_image_tokens = self.max_image_tokens  # Full anyres
                elif is_multi_image:
                    num_image_tokens = self.num_image_tokens
                else:
                    if not isinstance(original_size, (list, tuple)):
                        # cast to list to avoid numerical precision errors when calculating unpadding
                        original_size = original_size.tolist()
                    orig_height, orig_width = original_size
                    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)

                assert num_image_tokens is not None  # Always assigned in branches above
                max_num_vision_tokens = max(max_num_vision_tokens, num_image_tokens)
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1

                sample = sample.replace(special_token, "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        text = [sample.replace("<placeholder>", special_token) for sample in prompt_strings]
        return text, max_num_vision_tokens

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens
        num_image_tokens = unpadded_features + base_features
        return num_image_tokens

    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(
        self, height: int, width: int, patches_height: int, patches_width: int, scale_height: int, scale_width: int
    ) -> tuple[int, int]:
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        max_num_patches = int(self.vision_aspect_ratio.strip("anyres_max_"))
        ratio = math.sqrt(current_height * current_width / (max_num_patches * patches_height**2))
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(current_width // ratio)
            newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    def _compute_unpad_indices(
        self,
        orig_height: int,
        orig_width: int,
        patches_height: int,
        patches_width: int,
        scale_height: int,
        scale_width: int,
        features_per_patch: int,
    ) -> np.ndarray:
        """
        Compute indices to reorder Levanter's padded features to HF's unpadded order.

        HF's pack_image_features applies spatial unpadding based on original image aspect ratio.
        This function computes the mapping from HF's feature positions to Levanter's sequential
        feature layout.

        Args:
            orig_height: Original image height
            orig_width: Original image width
            patches_height: Number of patches per tile in height (e.g., 27)
            patches_width: Number of patches per tile in width (e.g., 27)
            scale_height: Number of tiles in height (e.g., 3 for 3x3 grid)
            scale_width: Number of tiles in width (e.g., 3 for 3x3 grid)
            features_per_patch: Features per patch/tile (e.g., 729)

        Returns:
            unpad_indices: Array of shape (num_unpadded_features,) where
                          unpad_indices[i] = Levanter index for HF position i
        """
        # Base features are identity mapping (base patch is always first)
        base_indices = np.arange(features_per_patch)

        # Grid spatial dimensions after combining all tiles
        curr_height = patches_height * scale_height  # e.g., 81 for 3x3 grid of 27x27 patches
        curr_width = patches_width * scale_width

        # Compute unpadding bounds based on original aspect ratio
        # This matches HF's unpad_image logic
        original_aspect_ratio = orig_width / orig_height
        current_aspect_ratio = curr_width / curr_height

        if original_aspect_ratio > current_aspect_ratio:
            # Wider image - remove top/bottom padding
            scale_factor = curr_width / orig_width
            new_height = int(round(orig_height * scale_factor, 7))
            padding = (curr_height - new_height) // 2
            row_start = padding
            row_end = curr_height - padding  # Symmetric padding like HF
            col_start = 0
            col_end = curr_width
        else:
            # Taller image - remove left/right padding
            scale_factor = curr_height / orig_height
            new_width = int(round(orig_width * scale_factor, 7))
            padding = (curr_width - new_width) // 2
            row_start = 0
            row_end = curr_height
            col_start = padding
            col_end = curr_width - padding  # Symmetric padding like HF

        # Build mapping from HF grid position to Levanter grid index
        # HF order: row-major through unpadded region
        # Levanter order: patch-by-patch (tile-by-tile), then row-major within each patch
        grid_indices = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                # Convert global (row, col) to Levanter's patch-based index
                # Which tile (patch) does this position belong to?
                tile_row = row // patches_height
                tile_col = col // patches_width
                # Local position within the tile
                local_row = row % patches_height
                local_col = col % patches_width

                # Tile index in row-major order (0-indexed grid patch, excluding base)
                tile_idx = tile_row * scale_width + tile_col
                # Local feature index within the tile
                local_idx = local_row * patches_width + local_col

                # Levanter index: base_features + tile_idx * features_per_patch + local_idx
                # +1 because tile_idx=0 is the first grid tile, but Levanter's patch 0 is the base
                lev_idx = features_per_patch + tile_idx * features_per_patch + local_idx
                grid_indices.append(lev_idx)

        return np.concatenate([base_indices, np.array(grid_indices, dtype=np.int32)])

    def compute_unpad_indices(
        self,
        image_sizes: list,
        height: int,
        width: int,
        max_num_features: int,
    ) -> np.ndarray:
        """
        Compute unpad indices for a batch of images.

        Args:
            image_sizes: List of (orig_height, orig_width) tuples for each image
            height: Processed tile height (e.g., 384)
            width: Processed tile width (e.g., 384)
            max_num_features: Maximum number of features to pad to

        Returns:
            unpad_indices: Array of shape (batch, max_num_features) padded with zeros
        """
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints
        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))

        batch_indices = []
        for orig_height, orig_width in image_sizes:
            # Find best resolution for this image
            height_best_resolution, width_best_resolution = select_best_resolution(
                [orig_height, orig_width], image_grid_pinpoints
            )
            scale_height = height_best_resolution // height
            scale_width = width_best_resolution // width

            # Compute unpad indices for this image
            indices = self._compute_unpad_indices(
                orig_height,
                orig_width,
                patches_height,
                patches_width,
                scale_height,
                scale_width,
                self.num_image_tokens,
            )
            batch_indices.append(indices)

        # Pad all indices to max_num_features
        padded_indices = np.zeros((len(batch_indices), max_num_features), dtype=np.int32)
        for i, indices in enumerate(batch_indices):
            padded_indices[i, : len(indices)] = indices

        return padded_indices

    def _get_num_multimodal_tokens(
        self, image_sizes: list[list[int]] | None = None, video_sizes: list[list[int]] | None = None, **kwargs: Any
    ) -> MultiModalData:
        """
        Computes the number of placeholder tokens needed for image inputs with the given sizes.
        Args:
            image_sizes (list[list[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
        Returns:
            MultiModalData with num_image_tokens and num_image_patches.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LlavaOnevisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            assert isinstance(size, dict)  # size should be a dict with height/width or shortest_edge
            size = (
                (size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"]))
            )
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)  # llava-ov doesn't batch pixels as Idefics, thus `1` patch`
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, processed_height, processed_width
                )
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                batch_num_image_tokens.append(num_image_tokens)
            vision_data.update({"num_image_tokens": batch_num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


DEFAULT_IMAGE_GRID_PINPOINTS = [
    [384, 384],
    [384, 768],
    [384, 1152],
    [768, 384],
    [768, 768],
    [768, 1152],
    [1152, 384],
    [1152, 768],
    [1152, 1152],
]


def create_custom_processor(
    model_name: str,
    do_pad: bool = True,
    image_grid_pinpoints: list[list[int]] | None = None,
    max_image_tiles: int | None = None,
) -> LlavaOnevisionProcessor:
    """
    Create a LlavaOnevisionProcessor with custom do_pad setting.

    Args:
        model_name: HuggingFace model name
        do_pad: Whether to pad image patches (True for Levanter, False for HF reference)
        image_grid_pinpoints: Optional custom grid pinpoints. If None, uses DEFAULT_IMAGE_GRID_PINPOINTS.
        max_image_tiles: Maximum number of image tiles (including base) for padding mode.
                         For anyres_max_9, this would be 10 (9 + 1 base).
                         Required when using padding_mode=True when calling the processor.
    """

    if image_grid_pinpoints is None:
        image_grid_pinpoints = DEFAULT_IMAGE_GRID_PINPOINTS

    # Load config
    config = AutoConfig.from_pretrained(model_name)

    # Load the HF processor to get the chat template
    hf_processor = AutoProcessor.from_pretrained(model_name)
    chat_template = hf_processor.chat_template

    # Load tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load image processor from HF and configure do_pad
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image_processor.do_pad = do_pad
    image_processor.image_grid_pinpoints = image_grid_pinpoints

    # Calculate num_image_tokens (patches per image = (image_size / patch_size)^2)
    image_size = config.vision_config.image_size  # e.g., 384
    patch_size = config.vision_config.patch_size  # e.g., 14
    num_image_tokens = (image_size // patch_size) ** 2  # e.g., 729

    # Create the custom processor with required parameters
    processor = LlavaOnevisionProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=hf_processor.video_processor,
        num_image_tokens=num_image_tokens,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        vision_aspect_ratio=config.vision_aspect_ratio,
        chat_template=chat_template,
        max_image_tiles=max_image_tiles,
    )
    return processor
