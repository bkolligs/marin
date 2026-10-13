# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


"""
Test utilities for VLM (Vision-Language Model) testing.

Provides unified data preparation for HF vs Levanter comparison testing.
Uses Levanter's BatchImageProcessor for consistent processing with grid_mask support.
"""

from levanter.data.onevision import create_custom_processor
from transformers import ProcessorMixin
from levanter.data.image import BatchImageProcessor
from levanter.data.image import load_image
import jax.numpy as jnp
import haliax as hax
from haliax import Axis

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Any
import numpy as np
from PIL import Image
from PIL import Image as PILImage

from datasets import Dataset, load_dataset


HF_DATASET = "ruili0/demo-vlm-test-dataset"


@lru_cache(maxsize=1)
def _load_single_image_split() -> Dataset:
    return load_dataset(HF_DATASET, split="single_image")


@lru_cache(maxsize=1)
def _load_multi_image_split() -> Dataset:
    return load_dataset(HF_DATASET, split="multi_image")


@lru_cache(maxsize=1)
def _load_real_data_split() -> Dataset:
    return load_dataset(HF_DATASET, split="real_data")


def get_single_image() -> PILImage.Image:
    ds = _load_single_image_split()
    return ds[0]["images"][0]


def get_multi_images() -> list[PILImage.Image]:
    ds = _load_multi_image_split()
    return ds[0]["images"]


def get_real_data(num_samples: int = 20) -> Dataset:
    ds = _load_real_data_split()
    return ds.select(range(min(num_samples, len(ds))))


@dataclass
class HFProcessedData:

    input_ids: np.ndarray  # (seq_len,)
    pixel_values: np.ndarray  # (num_patches, C, H, W) - variable num_patches
    attention_mask: np.ndarray  # (seq_len,)
    image_sizes: np.ndarray  # (num_images, 2) - (height, width) per image


@dataclass
class LevProcessedData:

    input_ids: np.ndarray  # (seq_len,)
    pixel_values: np.ndarray  # (TOTAL_PATCHES, C, H, W) - fixed size, padded
    attention_mask: np.ndarray  # (seq_len,)
    grid_mask: np.ndarray  # (TOTAL_PATCHES,) - True for valid patches
    unpad_indices: Optional[np.ndarray]  # (num_image_tokens,) - for HF compatibility
    loss_mask: np.ndarray  # (seq_len,) float32 - 1.0 for compute loss, 0.0 for ignore


@dataclass
class TestDataPair:

    hf: HFProcessedData
    lev: LevProcessedData
    raw_images: list[Image.Image]  # Original PIL images for reference
    messages: list[dict[str, Any]]  # Original messages from dataset


# Default grid pinpoints for anyres_max_9 configuration
DEFAULT_GRID_PINPOINTS = [
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


def _create_processors(
    model_name: str,
    grid_pinpoints: list[list[int]],
    max_length: int,
    max_num_patches: int,
    patch_size: int,
    vision_feature_height: int,
    add_generation_prompt: bool = False,
) -> tuple[ProcessorMixin, BatchImageProcessor]:
    # HF processor with do_pad=True and max_image_tiles for padding_mode support
    hf_processor = create_custom_processor(
        model_name,
        do_pad=True,
        image_grid_pinpoints=grid_pinpoints,
        max_image_tiles=max_num_patches + 1,  # e.g., 10 for anyres_max_9
    )

    # Levanter processor with do_pad=True (padding, fixed shape)
    lev_processor = create_custom_processor(
        model_name,
        do_pad=True,
        image_grid_pinpoints=grid_pinpoints,
        max_image_tiles=max_num_patches + 1,
    )

    # Wrap Levanter processor in BatchImageProcessor for consistent grid_mask handling

    lev_batch_processor = BatchImageProcessor(
        processor=lev_processor,
        max_length=max_length,
        padding=True,
        max_num_patches=max_num_patches,
        grid_pinpoints=grid_pinpoints,
        patch_size=patch_size,
        vision_feature_height=vision_feature_height,
        add_generation_prompt=add_generation_prompt,
    )

    return hf_processor, lev_batch_processor


def prepare_test_data(
    parquet_path: str,
    sample_indices: list[int],
    model_name: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
    max_length: int = 8192,
    max_num_patches: int = 9,
    grid_pinpoints: Optional[list[list[int]]] = None,
    patch_size: int = 384,
    vision_feature_height: int = 27,
    add_generation_prompt: bool = False,
) -> list[TestDataPair]:

    # Use default grid_pinpoints if not provided
    if grid_pinpoints is None:
        grid_pinpoints = DEFAULT_GRID_PINPOINTS

    # Load dataset
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")

    # Create processors
    hf_processor, lev_batch_processor = _create_processors(
        model_name=model_name,
        grid_pinpoints=grid_pinpoints,
        max_length=max_length,
        max_num_patches=max_num_patches,
        patch_size=patch_size,
        vision_feature_height=vision_feature_height,
        add_generation_prompt=add_generation_prompt,
    )

    results = []
    for idx in sample_indices:
        example = dataset[idx]
        messages = example["messages"]
        images_data = example.get("images", [])

        # Load raw images
        raw_images = [load_image(img) for img in images_data]

        # --- HF Processing (no padding, variable shape) ---
        hf_text = hf_processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        hf_processed = hf_processor(
            images=raw_images,
            text=hf_text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=max_length,
        )

        hf_data = HFProcessedData(
            input_ids=hf_processed["input_ids"][0],
            pixel_values=hf_processed["pixel_values"][0],
            attention_mask=hf_processed["attention_mask"][0],
            image_sizes=hf_processed["image_sizes"][0],
        )

        # --- Levanter Processing (with padding + grid_mask) ---
        # Use BatchImageProcessor for consistent processing
        lev_results = lev_batch_processor([example])
        lev_result = lev_results[0]  # ImageTextDict

        lev_data = LevProcessedData(
            input_ids=lev_result["input_ids"],
            pixel_values=lev_result["pixel_values"],
            attention_mask=lev_result["attention_mask"],
            grid_mask=lev_result["grid_mask"],
            unpad_indices=lev_result.get("unpad_indices"),
            loss_mask=lev_result["loss_mask"],
        )

        results.append(
            TestDataPair(
                hf=hf_data,
                lev=lev_data,
                raw_images=raw_images,
                messages=messages,
            )
        )

    return results


def prepare_test_data_single(
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    model_name: str = "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
    max_length: int = 8192,
    max_num_patches: int = 9,
    grid_pinpoints: Optional[list[list[int]]] = None,
    patch_size: int = 384,
    vision_feature_height: int = 27,
    add_generation_prompt: bool = False,
) -> TestDataPair:
    # Use default grid_pinpoints if not provided
    if grid_pinpoints is None:
        grid_pinpoints = DEFAULT_GRID_PINPOINTS

    # Create processors using the same logic as prepare_test_data
    hf_processor, lev_batch_processor = _create_processors(
        model_name=model_name,
        grid_pinpoints=grid_pinpoints,
        max_length=max_length,
        max_num_patches=max_num_patches,
        patch_size=patch_size,
        vision_feature_height=vision_feature_height,
        add_generation_prompt=add_generation_prompt,
    )

    # --- HF Processing (NO padding - HF model uses dynamic shapes) ---
    hf_text = hf_processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
    is_multi_image = len(images) > 1

    hf_processed = hf_processor(
        images=images,
        text=hf_text,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=max_length,
        padding_mode=False,  # HF model doesn't need padding
    )

    # Handle multi-image: pixel_values may be (num_images, patches, C, H, W)
    hf_pixel_values = hf_processed["pixel_values"]
    hf_image_sizes = hf_processed["image_sizes"]

    # For single image, extract from batch dimension
    # Multi-image keeps 5D format: (num_images, patches, C, H, W)
    if not (isinstance(hf_pixel_values, np.ndarray) and hf_pixel_values.ndim == 5 and is_multi_image):
        hf_pixel_values = hf_pixel_values[0]
        hf_image_sizes = hf_image_sizes[0]

    hf_data = HFProcessedData(
        input_ids=hf_processed["input_ids"][0],
        pixel_values=hf_pixel_values,
        attention_mask=hf_processed["attention_mask"][0],
        image_sizes=hf_image_sizes,
    )

    # --- Levanter Processing (with padding + grid_mask) ---
    example = {"messages": messages, "images": images}
    lev_results = lev_batch_processor([example])
    lev_result = lev_results[0]

    lev_data = LevProcessedData(
        input_ids=lev_result["input_ids"],
        pixel_values=lev_result["pixel_values"],
        attention_mask=lev_result["attention_mask"],
        grid_mask=lev_result["grid_mask"],
        unpad_indices=lev_result.get("unpad_indices"),
        loss_mask=lev_result["loss_mask"],
    )

    return TestDataPair(
        hf=hf_data,
        lev=lev_data,
        raw_images=images,
        messages=messages,
    )


def create_grid_mask(actual_patches: int, total_patches: int) -> np.ndarray:
    grid_mask = np.zeros(total_patches, dtype=np.bool_)
    grid_mask[:actual_patches] = True
    return grid_mask


def pad_pixel_values(pixel_values: np.ndarray, total_patches: int) -> np.ndarray:
    actual_patches = pixel_values.shape[0]
    if actual_patches >= total_patches:
        return pixel_values[:total_patches]
    pad_shape = (total_patches - actual_patches,) + pixel_values.shape[1:]
    padding = np.zeros(pad_shape, dtype=pixel_values.dtype)
    return np.concatenate([pixel_values, padding], axis=0)


def get_actual_patches_from_grid_mask(grid_mask: np.ndarray) -> int:
    return int(grid_mask.sum())


@dataclass
class LogitsComparisonResult:

    overall_mean_diff: float
    overall_max_diff: float
    passed: bool
    # Detailed fields (only populated when detailed=True)
    pre_image_mean_diff: float = 0.0
    pre_image_max_diff: float = 0.0
    image_mean_diff: float = 0.0
    image_max_diff: float = 0.0
    post_image_mean_diff: float = 0.0
    post_image_max_diff: float = 0.0
    details: Optional[dict[str, Any]] = None


def compare_logits_by_region(
    hf_logits: np.ndarray,
    lev_logits: np.ndarray,
    input_ids: np.ndarray,
    image_token_id: int,
    tolerance: float = 1e-2,
    verbose: bool = True,
    detailed: bool = True,
    attention_mask: Optional[np.ndarray] = None,
) -> LogitsComparisonResult:
    # Ensure same sequence length
    seq_len = min(hf_logits.shape[0], lev_logits.shape[0])
    hf_logits = hf_logits[:seq_len]
    lev_logits = lev_logits[:seq_len]
    input_ids = input_ids[:seq_len]
    if attention_mask is not None:
        attention_mask = attention_mask[:seq_len]
        valid_mask = attention_mask.astype(bool)
        valid_count = valid_mask.sum()
        lev_logits_valid = lev_logits[valid_mask]
    else:
        valid_mask = np.ones(seq_len, dtype=bool)
        valid_count = seq_len
        lev_logits_valid = lev_logits
    # Simple mode: just compute overall diff for valid positions
    if not detailed:
        if attention_mask is not None:
            # Only compare valid (non-padding) positions
            diff = np.abs(hf_logits - lev_logits_valid)
            overall_mean_diff = float(np.mean(diff))
            overall_max_diff = float(np.max(diff))
            if verbose:
                print(
                    f"Overall ({valid_count} valid tokens): mean={overall_mean_diff:.6e}, max={overall_max_diff:.6e}"
                )
        else:
            diff = np.abs(hf_logits - lev_logits)
            overall_mean_diff = float(np.mean(diff))
            overall_max_diff = float(np.max(diff))
            if verbose:
                print(f"Overall ({seq_len} tokens): mean={overall_mean_diff:.6e}, max={overall_max_diff:.6e}")

        passed = overall_mean_diff < tolerance
        if verbose:
            print(f"{'PASS' if passed else 'FAIL'} (tol={tolerance})")

        return LogitsComparisonResult(
            overall_mean_diff=overall_mean_diff,
            overall_max_diff=overall_max_diff,
            passed=passed,
        )

    # Detailed mode: split by region
    image_mask = input_ids == image_token_id
    has_image = image_mask.any()

    if has_image:
        image_start = int(np.where(image_mask)[0][0])
        num_image_tokens = int(image_mask.sum())
        post_image_start = image_start + num_image_tokens
    else:
        image_start = seq_len
        num_image_tokens = 0
        post_image_start = seq_len

    if verbose:
        print(f"Image tokens: start={image_start}, count={num_image_tokens}")

    # 1. Pre-image text
    if image_start > 0:
        diff = np.abs(hf_logits[:image_start] - lev_logits[:image_start])
        pre_image_mean_diff = float(np.mean(diff))
        pre_image_max_diff = float(np.max(diff))
    else:
        pre_image_mean_diff = 0.0
        pre_image_max_diff = 0.0

    # 2. Image tokens
    if num_image_tokens > 0:
        diff = np.abs(hf_logits[image_start:post_image_start] - lev_logits[image_start:post_image_start])
        image_mean_diff = float(np.mean(diff))
        image_max_diff = float(np.max(diff))
    else:
        image_mean_diff = 0.0
        image_max_diff = 0.0

    # 3. Post-image text
    if post_image_start < seq_len:
        diff = np.abs(hf_logits[post_image_start:] - lev_logits[post_image_start:])
        post_image_mean_diff = float(np.mean(diff))
        post_image_max_diff = float(np.max(diff))
    else:
        post_image_mean_diff = 0.0
        post_image_max_diff = 0.0

    # Overall
    overall_mean_diff = float(np.mean(np.abs(hf_logits - lev_logits)))
    overall_max_diff = float(np.max(np.abs(hf_logits - lev_logits)))

    # Pass/fail per region
    passed = pre_image_mean_diff < tolerance and image_mean_diff < tolerance and post_image_mean_diff < tolerance

    if verbose:
        print(f"Pre-image ({image_start}): mean={pre_image_mean_diff:.6e}, max={pre_image_max_diff:.6e}")
        print(f"Image ({num_image_tokens}): mean={image_mean_diff:.6e}, max={image_max_diff:.6e}")
        print(
            f"Post-image ({seq_len - post_image_start}): mean={post_image_mean_diff:.6e}, max={post_image_max_diff:.6e}"
        )
        print(f"Overall: mean={overall_mean_diff:.6e}, max={overall_max_diff:.6e}")
        print(f"{'PASS' if passed else 'FAIL'} (tol={tolerance})")

    return LogitsComparisonResult(
        overall_mean_diff=overall_mean_diff,
        overall_max_diff=overall_max_diff,
        passed=passed,
        pre_image_mean_diff=pre_image_mean_diff,
        pre_image_max_diff=pre_image_max_diff,
        image_mean_diff=image_mean_diff,
        image_max_diff=image_max_diff,
        post_image_mean_diff=post_image_mean_diff,
        post_image_max_diff=post_image_max_diff,
        details={
            "image_start": image_start,
            "num_image_tokens": num_image_tokens,
            "post_image_start": post_image_start,
        },
    )


def verify_pixel_values_consistency(
    hf_pixel_values: np.ndarray,
    lev_pixel_values: np.ndarray,
    grid_mask: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    actual_patches = get_actual_patches_from_grid_mask(grid_mask)

    # Extract valid patches from Levanter output
    lev_valid = lev_pixel_values[:actual_patches]

    # Compare with HF output
    return np.allclose(hf_pixel_values, lev_valid, rtol=rtol, atol=atol)


@dataclass
class LevJaxTensors:

    input_ids: Any  # NamedArray (Batch, Position)
    pixel_values: Any  # NamedArray (Batch, NumPatches, Channels, Height, Width)
    grid_mask: Any  # NamedArray (Batch, GridMask)
    unpad_indices: Optional[Any] = None  # NamedArray (Batch, NumImageTokens) - None for multi-image
    loss_mask: Any = None  # NamedArray (Batch, Position) - mask for loss computation
    # Axes for reference
    Batch: Any = None
    Position: Any = None
    NumPatches: Any = None
    Channels: Any = None
    Height: Any = None
    Width: Any = None
    GridMaskAxis: Any = None
    NumImageTokens: Any = None


def create_lev_jax_tensors(
    lev_data: LevProcessedData,
    batch_size: int = 1,
) -> LevJaxTensors:

    seq_len = len(lev_data.input_ids)

    # Define axes
    Batch = Axis("batch", batch_size)
    Position = Axis("position", seq_len)

    # Create input_ids tensor - replicate single sample to batch_size
    input_ids_single = jnp.array(lev_data.input_ids, dtype=jnp.int32).reshape(1, -1)
    input_ids_batched = jnp.tile(input_ids_single, (batch_size, 1))
    input_ids = hax.named(input_ids_batched, (Batch, Position))

    # Pixel values - already padded by BatchImageProcessor
    total_patches = lev_data.pixel_values.shape[0]
    channels = lev_data.pixel_values.shape[1]
    height = lev_data.pixel_values.shape[2]
    width = lev_data.pixel_values.shape[3]

    NumPatches = Axis("num_patches", total_patches)
    Channels = Axis("channels", channels)
    Height = Axis("height", height)
    Width = Axis("width", width)
    GridMaskAxis = Axis("grid_mask", total_patches)

    # Pixel values - replicate to batch_size
    pv_single = jnp.array(lev_data.pixel_values, dtype=jnp.float32).reshape(1, total_patches, channels, height, width)
    pv_batched = jnp.tile(pv_single, (batch_size, 1, 1, 1, 1))
    pixel_values = hax.named(pv_batched, (Batch, NumPatches, Channels, Height, Width))

    # Grid mask - replicate to batch_size
    gm_single = jnp.array(lev_data.grid_mask).reshape(1, -1)
    gm_batched = jnp.tile(gm_single, (batch_size, 1))
    grid_mask = hax.named(gm_batched, (Batch, GridMaskAxis))

    # Unpad indices - replicate to batch_size (None for multi-image case)
    if lev_data.unpad_indices is not None:
        num_image_tokens = lev_data.unpad_indices.shape[0]
        NumImageTokens = Axis("num_image_tokens", num_image_tokens)
        ui_single = jnp.array(lev_data.unpad_indices, dtype=jnp.int32).reshape(1, -1)
        ui_batched = jnp.tile(ui_single, (batch_size, 1))
        unpad_indices = hax.named(ui_batched, (Batch, NumImageTokens))
    else:
        # Multi-image case: no unpad_indices needed
        unpad_indices = None
        NumImageTokens = None

    # Loss mask - replicate to batch_size
    loss_mask_single = jnp.array(lev_data.loss_mask, dtype=jnp.float32).reshape(1, -1)
    loss_mask_batched = jnp.tile(loss_mask_single, (batch_size, 1))
    loss_mask = hax.named(loss_mask_batched, (Batch, Position))

    return LevJaxTensors(
        input_ids=input_ids,
        pixel_values=pixel_values,
        grid_mask=grid_mask,
        unpad_indices=unpad_indices,
        loss_mask=loss_mask,
        Batch=Batch,
        Position=Position,
        NumPatches=NumPatches,
        Channels=Channels,
        Height=Height,
        Width=Width,
        GridMaskAxis=GridMaskAxis,
        NumImageTokens=NumImageTokens,
    )
