# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.11.0",
#     "transformers==5.12.1",
#     "pillow>=12.2.0",
#     "pyarrow==23.0.1",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import dataclasses
    import hashlib
    import json
    import platform
    import subprocess
    import tempfile
    from collections.abc import Iterator
    from contextlib import closing
    from dataclasses import dataclass
    from enum import StrEnum
    from importlib.metadata import version
    from io import BytesIO
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from huggingface_hub import HfApi, HfFileSystem
    from PIL import Image, ImageDraw
    from pyarrow.parquet import ParquetFile
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from torch import nn
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPImageProcessorPil,
        CLIPVisionConfig,
        CLIPVisionModel,
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Build a VLM: explain it, implement it, test it

    Tracking: [Marin #9524](https://github.com/marin-community/marin/issues/9524).

    A vision-language model (VLM) predicts text using images and text as context. You will connect a frozen CLIP
    vision encoder to a frozen Delphi 998M language model by training a small multilayer perceptron (MLP), called a
    projector. Hugging Face provides the transformer internals. You implement the boundaries between them.

    ## How to work through this notebook

    Use this Feynman-method cycle for **each of the eight exercises (0-7)**:

    1. **Explain:** fill in the Markdown answer cell (edit its `mo.md` source) in words you could use with a
    programmer who has never trained a VLM. Define technical terms instead of hiding behind them.
    2. **Predict:** write the expected shape, behavior, or result before running code.
    3. **Implement:** replace the marked `raise NotImplementedError(...)` statements. Most functions need one to
    ten lines. Setup, data creation, plotting, and saving are provided.
    4. **Test:** run the exercise cell and its checks. A passing assertion verifies behavior, not your explanation.
    5. **Teach back:** revise your original explanation, identify a mistaken prediction, and explain why the
    observed behavior follows from the implementation. If you cannot explain it, reduce the example and retry.

    An unfinished exercise blocks its dependent cells. Independent Markdown remains readable. **There is no
    automatic solution mode or complete answer key in this notebook.** Folded explanations are hints; attempt your
    own explanation before opening them. You may consult API documentation for syntax.

    Start with `RunMode.SMOKE`, which uses tiny random models without network access. After all exercises pass,
    restart the marimo session, select `RunMode.PRETRAINED`, and rerun with pinned Delphi/CLIP checkpoints. Smoke
    results check mechanics, not pretrained-model quality.

    The selected alignment dataset is **LLaVA 558K image-caption pairs**. Pretrained mode streams a bounded sample;
    offline smoke mode uses four colored squares as a mechanics check. Neither the small sample nor the smoke task
    establishes general visual question answering (VQA). This PyTorch prototype does not launch Iris jobs or
    reproduce Marin's JAX/Levanter training stack. The base model needs no preceding text-only instruction-tuning
    phase for this exercise.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 1. Open this notebook in marimo

    On [molab](https://molab.marimo.io), open this Python notebook and enable a GPU using the notebook specs
    button in the app header. See [molab's GPU instructions](https://marimo.io/blog/reintroducing-molab).
    The Config cell defaults to `device="cuda"` and checks GPU access before you load models. Start with
    `RunMode.SMOKE`, then switch to `RunMode.PRETRAINED` after completing the exercises.

    For local use, set `device="cpu"` if needed and launch from the repository root:

    ```bash
    uv run --no-project --with 'marimo==0.25.1' marimo edit --sandbox experiments/multimodality/exploration.py
    ```

    The Python file declares its dependencies in a script metadata header. Marimo's sandbox uses an isolated
    environment, avoiding the full Marin training environment. Choose **lazy execution** in marimo's runtime
    settings while working through exercises: editing a cell marks dependent cells stale until you choose to run
    them.

    The **Load models**, **Load examples**, **Train projector**, **Generate answers**, and **Save projector**
    buttons gate expensive work. Opening the notebook does not download models or data. Independent exercise
    instructions remain readable even while an unfinished code cell raises `NotImplementedError`.

    Edit the Config cell to select `RunMode.PRETRAINED` for pinned Delphi/CLIP weights and the bounded LLaVA
    sample; this downloads gigabytes of weights on first use. CUDA requires a compatible PyTorch installation.
    CPU works but the pretrained exercise is slow. On molab, the fork checkout and saved weights live on the
    remote runtime. Export your notebook edits and saved projector before ending the session.

    Write teach-back answers by editing the corresponding Markdown cell's `mo.md` string; they persist in this
    Python file. Marimo runs cells by dependency, not visual position. Cell-local scratch variables begin with `_`.
    The baseline `projector` is never trained in place: training returns a separate `trained_projector`, and
    generation/saving consume that result. Click the relevant run buttons again after changing an upstream
    exercise.
    """
    )
    return


@app.cell
def _():
    clone_fork = mo.ui.run_button(label="Download Marin fork")
    mo.output.append(
        mo.vstack(
            [
                mo.md(
                    "### Download the Marin source for later exploration\n\n"
                    "Click to clone [bkolligs/marin](https://github.com/bkolligs/marin/tree/main), "
                    "using a shallow checkout of `main`. Git must be installed. "
                    "To choose another destination, edit `marin_fork_path` in the next cell before clicking. "
                    "Repeat clicks reuse the checkout without pulling or overwriting edits. "
                    "This downloads source code only; it does not install Marin or change the VLM backbones."
                ),
                clone_fork,
            ]
        )
    )
    return (clone_fork,)


@app.cell
def _(clone_fork):
    mo.stop(not clone_fork.value, mo.md("Click **Download Marin fork** when you need the source."))
    marin_fork_path = Path.home() / "marin-notebook-repos" / "bkolligs-marin"
    _fork_url = "https://github.com/bkolligs/marin.git"
    if marin_fork_path.exists():
        if not (marin_fork_path / ".git").exists():
            raise ValueError(f"Destination exists but is not a Git checkout: {marin_fork_path}")
        _origin = subprocess.run(
            ["git", "-C", str(marin_fork_path), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if _origin != _fork_url:
            raise ValueError(f"Expected origin {_fork_url}, found {_origin}. Choose another destination.")
    else:
        marin_fork_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", "--branch", "main", _fork_url, str(marin_fork_path)],
            check=True,
        )
    marin_fork_commit = subprocess.run(
        ["git", "-C", str(marin_fork_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"Marin source: {marin_fork_path}\nChecked-out commit: {marin_fork_commit}")
    return marin_fork_commit, marin_fork_path


@app.cell
def _():

    class RunMode(StrEnum):
        PRETRAINED = "pretrained"
        SMOKE = "smoke"

    @dataclass(frozen=True)
    class Config:
        mode: RunMode = RunMode.SMOKE
        device: str = "cuda"  # Set to "cpu" for local exploration without a GPU.
        lm_id: str = "marin-community/delphi-3e20-998Mparams-49.8Btokens"
        lm_revision: str = "9b913fac63d36a80424949775c63c728ac8fa9fc"
        vision_id: str = "openai/clip-vit-base-patch32"
        vision_revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        seed: int = 42
        learning_rate: float = 1e-3
        steps: int = 40
        max_new_tokens: int = 64
        dataset_id: str = "theblackcat102/llava-pretrain"
        dataset_revision: str = "57906fbd3e2ec529a202a0d66ce1a01c7e7ecf84"
        stream_scan_rows: int = 32
        train_examples: int = 8
        heldout_examples: int = 4

    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Enable a GPU in molab's notebook specs and rerun this cell. "
                "For local CPU exploration, set Config.device to 'cpu'."
            )
        # Run a kernel as well as detecting the device to catch incompatible CUDA builds.
        _gpu_probe = (torch.ones(1, device=device) + 1).item()
        assert _gpu_probe == 2
        print(
            {
                "gpu": torch.cuda.get_device_name(device),
                "vram_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
                "torch_cuda": torch.version.cuda,
            }
        )
    # BF16 reduces CUDA activation/weight memory. CPU uses FP32 for portability.
    lm_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(dataclasses.asdict(cfg))
    print({name: version(name) for name in ("torch", "transformers", "pillow", "pyarrow", "huggingface-hub")})
    return RunMode, cfg, device, lm_dtype


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 2. Load the parts and inspect their interfaces

    Delphi is a **base** model: it learned next-token prediction on text. The chosen checkpoint has about 998M
    parameters, hidden width 1536, and a Llama 3 vocabulary even though its architecture is Qwen3. Always load the
    tokenizer shipped with the checkpoint.

    CLIP ViT-B/32 resizes/crops to 224x224 pixels. Its patch stem makes a 7x7 grid of 32x32 patches. The vision
    transformer adds one classification (CLS) token, giving 50 vectors of width 768. We will keep the 49 patch
    vectors and remove CLS. These vectors are contextual: each has already attended to the other patches.

    The tiny branch is only setup for offline verification. Both branches feed the same implementation below. Exact
    model revisions are pinned so later runs use the same weights. Loading this base model does not require a chat
    template.
    """
    )
    return


@app.cell
def _():
    load_models = mo.ui.run_button(label="Load models")
    mo.output.append(load_models)
    return (load_models,)


@app.cell
def _(RunMode, cfg, device, lm_dtype, load_models):
    mo.stop(not load_models.value, mo.md("Click **Load models** when ready."))
    torch.manual_seed(cfg.seed)
    if cfg.mode == RunMode.PRETRAINED:
        tokenizer = AutoTokenizer.from_pretrained(cfg.lm_id, revision=cfg.lm_revision)
        lm = AutoModelForCausalLM.from_pretrained(
            cfg.lm_id, revision=cfg.lm_revision, dtype=lm_dtype, attn_implementation="eager"
        )
        processor = CLIPImageProcessorPil.from_pretrained(cfg.vision_id, revision=cfg.vision_revision)
        vision = CLIPVisionModel.from_pretrained(cfg.vision_id, revision=cfg.vision_revision)
    else:
        words = [
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
            "What",
            "color",
            "is",
            "the",
            "square",
            "?",
            "Answer",
            ":",
            "red",
            "green",
            "blue",
            "yellow",
        ]
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        lm = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(words),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=256,
                attention_dropout=0.0,
            )
        )
        vision = CLIPVisionModel(
            CLIPVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=32,
                patch_size=8,
            )
        )
        processor = CLIPImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
    lm = lm.to(device=device, dtype=lm_dtype).eval().requires_grad_(False)
    vision = vision.to(device).eval().requires_grad_(False)
    print("LM:", type(lm).__name__, "hidden width:", lm.config.hidden_size)
    print("Vision:", type(vision).__name__, "hidden width:", vision.config.hidden_size)
    print("LM parameters:", f"{sum(p.numel() for p in lm.parameters()):,}")
    print("BOS / EOS / PAD:", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    return lm, processor, tokenizer, vision


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 3. Use LLaVA 558K for caption alignment

    The source is [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), the 558K
    alignment set, not the 665K instruction mixture or the re-captioned ReCap set. Its original distribution
    separates image archives from conversation JSON.

    For streaming, we use
    [theblackcat102/llava-pretrain](https://huggingface.co/datasets/theblackcat102/llava-pretrain), a community
    Parquet copy with **558,128 rows**, embedded image bytes, and JSON-encoded `conversations`. Its revision is
    pinned above. We verified its row count/schema and bounded streaming, not full-corpus equivalence to the
    original release. Record this exact source in experiment results.

    Pretrained mode reads at most `stream_scan_rows` rows and retains 8 training images and 4 held-out images by
    default. The provided bounded Parquet iterator uses HTTP range reads and avoids materializing the full
    approximately 28 GB repository. Parquet range reads can fetch an entire row group; `stream_scan_rows=32` bounds
    rows, not network bytes. This is **streamed sampling followed by a small in-memory exercise**, not full-corpus
    streaming training.

    We split by SHA-256 of the encoded image bytes before selecting examples: about one fifth map to held-out. This
    keeps byte-identical images in one split, even if their captions differ. It does not detect visually identical
    images with different encodings. The first rows are not a representative shuffled validation set; results are
    learning diagnostics only.

    Each row contains a human request and a `gpt` caption. We supervise the caption. For this experiment we replace
    the varied requests with a fixed prompt, “Describe this image. Caption:”, so the only distinguishing input is
    the image. We represent images through embeddings; the literal `<image>` marker is not passed as a text token.

    Smoke mode remains offline and uses colored-square targets. To use real captions, restart with
    `RunMode.PRETRAINED`; the tiny smoke tokenizer has no useful natural-language vocabulary. The training and
    masking exercises are shared. Captions can be noisy and several descriptions can be valid: inspect examples and
    use caption loss plus wrong-image controls, not exact string match as the quality score.

    The provided loader uses Hugging Face file access plus PyArrow record batches. Threaded Parquet reads continued
    after early stopping during validation, so it uses synchronous reads (`use_threads=False`, `pre_buffer=False`)
    and closes the generator and file handles explicitly. This is supplied I/O boilerplate; the exercises focus on
    targets and the model. Revisit buffering when building full-corpus training.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 0: identify what the model should learn

    **Explain:** In an image-caption pair stored as a conversation, which text should the model predict? Why should
    `<image>` and the human request not become caption targets?

    **Predict:** Inspect the small fixture below. Write the exact output string your parser should return. Explain
    why lazily streaming rows does not imply zero buffering or zero image downloads.

    **Implement:** Complete `caption_target`: parse the JSON string, require exactly two turns in human → gpt
    order, and return the nonempty assistant caption with surrounding whitespace removed. Raise `ValueError` for
    wrong roles/turn count or an empty caption; do not silently drop malformed rows. Invalid JSON can propagate its
    parsing error.

    **Test and revise:** Run the checks, then explain why holding out rows independently could leak the same image
    into both splits. What duplicate images would the provided byte-hash split fail to catch?
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Teach the input/target distinction in plain language.]

    **Prediction:** [Write the fixture's caption and explain streaming versus caching.]

    **What actually happened:** [Complete after the checks and data preview.]

    **Gap I discovered:** [Name one assumption about labels, images, or splitting.]

    **Revised explanation:** [Explain the data path without using “streaming” as a substitute for its mechanics.]
    """
    )
    return


@app.cell
def _():
    def caption_target(conversations: str) -> str:
        """Extract the caption from one LLaVA human/gpt conversation JSON string."""
        raise NotImplementedError("Exercise 0: validate the two turns and extract the caption")

    fixture = json.dumps(
        [
            {"from": "human", "value": "<image>\nDescribe this image."},
            {"from": "gpt", "value": "  A red square on a white background.  "},
        ]
    )
    assert caption_target(fixture) == "A red square on a white background."
    for turns in (
        [{"from": "human", "value": "<image>"}],
        [{"from": "gpt", "value": "wrong first role"}, {"from": "human", "value": "wrong target"}],
        [{"from": "human", "value": "<image>"}, {"from": "gpt", "value": " "}],
    ):
        try:
            caption_target(json.dumps(turns))
        except ValueError:
            pass  # This check requires an explicit failure, not silent row loss.
        else:
            raise AssertionError("Malformed conversations must fail instead of becoming training targets.")
    print("Exercise 0 checks passed.")
    return (caption_target,)


@app.cell
def _():
    load_examples = mo.ui.run_button(label="Load examples")
    mo.output.append(load_examples)
    return (load_examples,)


@app.cell
def _(RunMode, caption_target, cfg, load_examples):
    mo.stop(not load_examples.value, mo.md("Click **Load examples** after completing Exercise 0."))

    @dataclass(frozen=True)
    class Example:
        image: Image.Image
        answer: str
        source_id: str

    COLORS = ("red", "green", "blue", "yellow")
    PROMPT = "What color is the square? Answer:" if cfg.mode == RunMode.SMOKE else "Describe this image. Caption:"

    def square_examples(box: tuple[int, int, int, int]) -> list[Example]:
        examples = []
        for color in COLORS:
            image = Image.new("RGB", (224, 224), "white")
            ImageDraw.Draw(image).rectangle(box, fill=color)
            examples.append(Example(image, color, hashlib.sha256(image.tobytes()).hexdigest()))
        return examples

    def stream_llava_rows(limit: int) -> Iterator[dict]:
        """Read a bounded sequence of Parquet rows using synchronous HTTP range reads."""
        files = HfApi().list_repo_files(cfg.dataset_id, repo_type="dataset", revision=cfg.dataset_revision)
        filesystem = HfFileSystem()
        remaining = limit
        for filename in sorted(name for name in files if name.endswith(".parquet")):
            path = f"datasets/{cfg.dataset_id}@{cfg.dataset_revision}/{filename}"
            with filesystem.open(path, "rb", block_size=1 << 20) as source:
                with ParquetFile(source, pre_buffer=False) as parquet:
                    for batch in parquet.iter_batches(
                        batch_size=32, columns=["image", "conversations"], use_threads=False
                    ):
                        for row in batch.to_pylist():
                            yield row
                            remaining -= 1
                            if remaining == 0:
                                return

    def llava_examples() -> tuple[list[Example], list[Example]]:
        train, heldout = ([], [])
        seen = set()
        with closing(stream_llava_rows(cfg.stream_scan_rows)) as rows:
            for row in rows:
                raw = row["image"]["bytes"]
                identity = hashlib.sha256(raw).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                is_heldout = int(identity[:8], 16) % 5 == 0
                destination = heldout if is_heldout else train
                limit = cfg.heldout_examples if is_heldout else cfg.train_examples
                if len(destination) == limit:
                    continue  # Keep byte-identical images once, irrespective of their captions.
                caption = caption_target(row["conversations"])
                with Image.open(BytesIO(raw)) as image:
                    destination.append(Example(image.convert("RGB"), caption, identity))
        if len(train) != cfg.train_examples or len(heldout) != cfg.heldout_examples:
            raise ValueError("Not enough unique examples in both splits; increase stream_scan_rows explicitly.")
        assert {item.source_id for item in train}.isdisjoint(item.source_id for item in heldout)
        return (train, heldout)

    if cfg.mode == RunMode.SMOKE:
        train_examples = square_examples((48, 48, 176, 176))
        heldout_examples = square_examples((22, 74, 118, 170))
    else:
        train_examples, heldout_examples = llava_examples()
    print("Source:", "offline colored squares" if cfg.mode == RunMode.SMOKE else cfg.dataset_id)
    print("Retained train / held-out:", len(train_examples), len(heldout_examples))
    _fig, _axes = plt.subplots(2, 4, figsize=(10, 5))
    for _row, examples in enumerate((train_examples, heldout_examples)):
        for _index, (ax, _example) in enumerate(zip(_axes[_row], examples[:4], strict=True)):
            ax.imshow(_example.image)
            ax.set_title(("train " if _row == 0 else "held out ") + str(_index))
            ax.axis("off")
            print("train" if _row == 0 else "held out", _index, repr(_example.answer))
    plt.tight_layout()
    mo.output.append(plt.gcf())
    plt.close(plt.gcf())
    return Example, PROMPT, heldout_examples, train_examples


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 1: Keep the image information

    **Explain first:** Explain how an image becomes a sequence of vectors. What does the extra classification token
    represent? Why could averaging all vectors hide information needed later?

    **Predict before running:** Predict the full vision output shape and selected patch shape for the current mode.
    Derive the patch count from image size and patch size.

    **Implement:** Complete `select_patch_features`. The processor and encoder call are supplied.

    **After the checks:** Describe what changed when you removed one sequence position, and what would change if
    you accidentally removed one batch item.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Write 3-5 plain-language sentences.]

    **Prediction:** [Write shapes or expected behavior, with your reasoning.]

    **What actually happened:** [Fill this in after running the checks.]

    **Gap I discovered:** [State what you misunderstood or what the checks did not establish.]

    **Revised explanation:** [Teach the idea again in simpler words, using a concrete example.]
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    <details>
    <summary>Explanation hint — open after your first attempt</summary>

    ## 4. Turn pixels into a sequence of visual features

    The processor converts PIL images to `[batch, channels, height, width]`, rescales pixel values, and normalizes
    channels using CLIP's preprocessing. The encoder returns `[batch, patches + 1, vision_width]`. Slicing `[:, 1:,
    :]` removes CLS; it does not average patches.

    We cache the **final hidden-state patch features** because the encoder is frozen and there is no random
    augmentation. This saves rerunning CLIP on every step. `no_grad()` is appropriate here: we do not need
    gradients into the encoder or pixels. If we later train the vision encoder, this cache must be removed and
    features recomputed with gradients.

    </details>
    """
    )
    return


@app.cell
def _(Example, device, heldout_examples, processor, train_examples, vision):
    def select_patch_features(states: torch.Tensor) -> torch.Tensor:
        """Return the contextual patch vectors, excluding the classification token."""
        raise NotImplementedError("Exercise 1: select patches without pooling or changing their order")

    def patch_features(examples: list[Example]) -> torch.Tensor:
        pixels = processor(images=[example.image for example in examples], return_tensors="pt").pixel_values
        with torch.no_grad():
            states = vision(pixel_values=pixels.to(device)).last_hidden_state
        print("pixels:", tuple(pixels.shape), "vision states:", tuple(states.shape))
        return select_patch_features(states).detach()

    train_features = patch_features(train_examples)
    heldout_features = patch_features(heldout_examples)
    print("Cached patches:", tuple(train_features.shape))
    print("Patch count:", (vision.config.image_size // vision.config.patch_size) ** 2)

    # Check that two different images retain distinct features, all patch positions, and no CLS.
    probe = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    selected = select_patch_features(probe)
    assert selected.shape == (2, 4, 3), "Keep batch and feature axes; remove exactly one sequence position."
    torch.testing.assert_close(selected[:, 0], probe[:, 1])
    torch.testing.assert_close(selected[:, -1], probe[:, -1])
    assert not torch.equal(train_features[0], train_features[1]), "Different images lost their distinction."
    print("Exercise 1 checks passed.")
    return heldout_features, train_features


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 2: Translate visual features into LM embeddings

    **Explain first:** Explain to a beginner why equal vector widths do not mean two models understand the vectors
    in the same way. What exactly must the projector learn?

    **Predict before running:** Write the input, intermediate, and output shapes. Derive the trainable parameter
    count including biases before printing it.

    **Implement:** Complete `build_projector` with Linear → GELU → Linear. Both linear outputs have LM width. Keep
    each patch independent.

    **After the checks:** Explain why reversing patch order before the projector reverses its outputs. Would the
    same statement hold for the full language model?
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Write 3-5 plain-language sentences.]

    **Prediction:** [Write shapes or expected behavior, with your reasoning.]

    **What actually happened:** [Fill this in after running the checks.]

    **Gap I discovered:** [State what you misunderstood or what the checks did not establish.]

    **Revised explanation:** [Teach the idea again in simpler words, using a concrete example.]
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    <details>
    <summary>Explanation hint — open after your first attempt</summary>

    ## 5. Learn the connector

    For each patch vector `v`, the projector computes `W₂ GELU(W₁v + b₁) + b₂`. Its output width equals the
    language model's hidden width, making each output a continuous input embedding. There are no discrete image
    token IDs and no vocabulary expansion in this prototype.

    For pretrained Delphi, the shapes are `[B, 49, 768] → [B, 49, 1536]`. This two-layer MLP has approximately
    3.54M parameters. Matching widths alone does not align semantics: training learns which embeddings help the
    frozen LM predict the answer.

    The projector stays FP32; its outputs are cast to the LM dtype at the boundary. Casting is differentiable. We
    retain the initial state for controlled reruns and a save/reload check.

    </details>
    """
    )
    return


@app.cell
def _(cfg, device, lm, train_features, vision):
    torch.manual_seed(cfg.seed)

    def build_projector(vision_width: int, lm_width: int) -> nn.Module:
        """Map every patch independently through Linear, GELU, Linear to LM width."""
        raise NotImplementedError("Exercise 2: build the two-layer projector")

    projector = build_projector(vision.config.hidden_size, lm.config.hidden_size).to(device)
    initial_projector = copy.deepcopy(projector.state_dict())
    print("Trainable parameters:", f"{sum(p.numel() for p in projector.parameters()):,}")
    print("Projected shape:", tuple(projector(train_features).shape))

    projected = projector(train_features)
    assert projected.shape == (*train_features.shape[:2], lm.config.hidden_size)
    reversed_patches = projector(train_features.flip(1))
    torch.testing.assert_close(reversed_patches, projected.flip(1))
    assert not torch.equal(projected[0], projected[1]), "The projector discarded image differences."
    del projected, reversed_patches
    print("Exercise 2 checks passed: patch order is preserved; images remain distinct.")
    return initial_projector, projector


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 3: Join images and text without leaking labels

    **Explain first:** Explain the difference between hiding a position from attention and excluding it from the
    training loss. Why must the first answer token still be visible as an input during teacher forcing?

    **Predict before running:** Draw the sequence with positions numbered. Mark the image span, first answer
    position, EOS, and padding. Predict which labels are -100.

    **Implement:** Complete `prefix_embeddings` for one image at a time: its `features` input is `[1, patches,
    vision_width]`. Complete `answer_targets` for a one-dimensional answer-ID tensor. Right-padding and batching
    are supplied. Preserve BOS → patches → prompt order.

    **After the checks:** Explain why using EOS as PAD does not require masking every EOS token. Describe the bug
    that would create.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Write 3-5 plain-language sentences.]

    **Prediction:** [Write shapes or expected behavior, with your reasoning.]

    **What actually happened:** [Fill this in after running the checks.]

    **Gap I discovered:** [State what you misunderstood or what the checks did not establish.]

    **Revised explanation:** [Teach the idea again in simpler words, using a concrete example.]
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    <details>
    <summary>Explanation hint — open after your first attempt</summary>

    ## 6. Assemble embeddings, labels, and padding

    Each training sequence is:

    ```text
    [BOS] [image patch embeddings] [fixed prompt] [target tokens] [EOS] [PAD...]
    ```

    Targets are captions in pretrained mode and color answers in smoke mode.

    `inputs_embeds` has shape `[B, S, D]`. `attention_mask` has shape `[B, S]`: 1 means a real position and 0 means
    padding. The causal LM combines this with its internal lower-triangular mask, so a position cannot read future
    answers. Vision patches have already interacted bidirectionally inside CLIP; their LM positions follow ordinary
    causal attention.

    `labels` has shape `[B, S]`. Prompt, image, BOS, and padding positions carry `-100`, PyTorch's ignored-loss
    marker. Answer and EOS positions carry vocabulary IDs. Padding may share EOS's ID: `attention_mask=0` hides
    padding from attention, while `labels=-100` excludes padding from cross entropy. Setting only the attention
    mask does not suppress loss.

    We tokenize the fixed prompt and the leading-space answer separately, defining an explicit token boundary. The
    same prompt encoding is used at inference. No automatic BOS insertion occurs; we add BOS once ourselves.

    One example is built at a time and then padded on the right. This makes variable answer lengths explicit and
    avoids hiding the alignment inside a data collator.

    The LM assigns sequence positions to the image embeddings just as it does to text embeddings. Qwen3 uses rotary
    position embeddings (RoPE) inside attention, so image patches consume context positions. We keep the entire
    sequence below the checkpoint's 4096-position training context; adding more image patches leaves fewer
    positions for text.

    </details>
    """
    )
    return


@app.cell
def _(
    PROMPT,
    device,
    lm,
    projector,
    tokenizer,
    train_examples,
    train_features,
):
    @dataclass(frozen=True)
    class Batch:
        inputs_embeds: torch.Tensor
        attention_mask: torch.Tensor
        labels: torch.Tensor
        answer_starts: tuple[int, ...]

    def prefix_embeddings(features: torch.Tensor, projector_model: nn.Module) -> torch.Tensor:
        embedding = lm.get_input_embeddings()
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)  # noqa: F841
        prompt_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(device)  # noqa: F841
        image_embeds = projector_model(features).to(embedding.weight.dtype)  # noqa: F841
        raise NotImplementedError("Exercise 3a: concatenate BOS, projected patches, and prompt along sequence")

    def answer_targets(prefix_length: int, answer_ids: torch.Tensor) -> torch.Tensor:
        """Return ignored prefix targets followed by answer/EOS targets on the same device."""
        raise NotImplementedError("Exercise 3b: mask the prefix while preserving every answer and EOS ID")

    def training_batch(features: torch.Tensor, answers: list[str], projector_model: nn.Module) -> Batch:
        rows, targets, starts = ([], [], [])
        for feature, answer in zip(features, answers, strict=True):
            prefix = prefix_embeddings(feature.unsqueeze(0), projector_model)[0]
            answer_ids = [*tokenizer(" " + answer, add_special_tokens=False).input_ids, tokenizer.eos_token_id]
            ids = torch.tensor(answer_ids, device=device)
            rows.append(torch.cat((prefix, lm.get_input_embeddings()(ids)), dim=0))
            targets.append(answer_targets(len(prefix), ids))
            starts.append(len(prefix))
        length = max(len(row) for row in rows)
        return Batch(
            inputs_embeds=torch.stack([F.pad(row, (0, 0, 0, length - len(row))) for row in rows]),
            attention_mask=torch.stack([torch.arange(length, device=device) < len(row) for row in rows]).long(),
            labels=torch.stack([F.pad(target, (0, length - len(target)), value=-100) for target in targets]),
            answer_starts=tuple(starts),
        )

    answers = [example.answer for example in train_examples]
    batch = training_batch(train_features, answers, projector)
    print("embeddings / attention mask / labels:")
    print(batch.inputs_embeds.shape, batch.attention_mask.shape, batch.labels.shape)
    for position in range(batch.answer_starts[0] - 2, batch.labels.shape[1]):
        label = batch.labels[0, position].item()
        print(position, "ignored" if label == -100 else repr(tokenizer.decode([label])))
    _prefix = prefix_embeddings(train_features[:1], projector)
    bos_expected = lm.get_input_embeddings()(torch.tensor([[tokenizer.bos_token_id]], device=device))
    torch.testing.assert_close(_prefix[:, :1], bos_expected)
    patch_count = train_features.shape[1]
    torch.testing.assert_close(_prefix[:, 1 : 1 + patch_count], projector(train_features[:1]).to(_prefix.dtype))
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    torch.testing.assert_close(_prefix[:, 1 + patch_count :], lm.get_input_embeddings()(prompt_ids))
    uneven = training_batch(train_features[:2], ["red", "green blue yellow"], projector)
    for _row, answer in enumerate(("red", "green blue yellow")):
        start = uneven.answer_starts[_row]
        valid_length = int(uneven.attention_mask[_row].sum())
        _expected = [*tokenizer(" " + answer, add_special_tokens=False).input_ids, tokenizer.eos_token_id]
        assert (uneven.labels[_row, :start] == -100).all(), "The prefix must not contribute to loss."
        assert uneven.labels[_row, start:valid_length].tolist() == _expected, "Preserve answer tokens and EOS."
        assert (uneven.labels[_row, valid_length:] == -100).all(), "Ignore padding independently of token ID."
    assert uneven.attention_mask[0].sum() < uneven.attention_mask[1].sum()
    del _prefix, uneven
    # Unequal answer lengths exercise right-padding, including when PAD and EOS share an ID.
    print("Exercise 3 checks passed: ordering, answer/EOS targets, and unequal-length padding.")
    return answers, batch, prefix_embeddings, training_batch


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 4: Derive next-token supervision

    **Explain first:** Explain why a logit at position t is compared with a label at t+1. Use a three-word
    sentence, without referring to an API.

    **Predict before running:** Identify which position predicts the first answer token and which predicts EOS.
    Predict whether changing ignored-position logits will change the loss.

    **Implement:** Complete `shifted_answer_loss` using cross entropy. Apply the one-token shift exactly once and
    exclude -100 labels.

    **After the checks:** Explain any mismatch with the library loss. Could an unshifted implementation produce a
    plausible-looking training curve? Why?
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Write 3-5 plain-language sentences.]

    **Prediction:** [Write shapes or expected behavior, with your reasoning.]

    **What actually happened:** [Fill this in after running the checks.]

    **Gap I discovered:** [State what you misunderstood or what the checks did not establish.]

    **Revised explanation:** [Teach the idea again in simpler words, using a concrete example.]
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    <details>
    <summary>Explanation hint — open after your first attempt</summary>

    ## 7. See the two masks and the one-token shift

    At position `t`, the model predicts the token at `t+1`. Hugging Face shifts logits and labels internally when
    given `labels`. The last prompt position predicts the first answer token; the last answer position predicts
    EOS. Do not pre-shift labels yourself.

    The training objective is the mean negative log-probability of the supervised answer/EOS tokens:

    `loss = -mean(log p(target[t+1] | embeddings[:t+1]))`.

    Teacher forcing supplies previous ground-truth answer tokens while predicting the next one. That is different
    from generation, where the model feeds back its own predictions. The heatmap below shows valid causal attention
    for the first example; the bars show which **logit positions** receive supervision after shifting.

    </details>
    """
    )
    return


@app.cell
def _(batch, device, lm):
    valid = batch.attention_mask[0].bool()
    length = len(valid)
    causal = torch.ones((length, length), device=device, dtype=torch.bool).tril()
    allowed = causal & valid[None, :] & valid[:, None]
    supervised_logits = batch.labels[0, 1:] != -100
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].imshow(allowed.cpu(), origin="upper", interpolation="nearest")
    _axes[0].set(xlabel="key position", ylabel="query position", title="Visible positions")
    _axes[1].step(range(length - 1), supervised_logits.cpu(), where="mid")
    _axes[1].set(xlabel="logit position (predicts next token)", title="Loss contributions", ylim=(-0.1, 1.1))
    plt.tight_layout()
    mo.output.append(plt.gcf())
    plt.close(plt.gcf())
    output = lm(
        inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask, labels=batch.labels, use_cache=False
    )

    def shifted_answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute next-token cross entropy over answer/EOS targets, ignoring -100."""
        raise NotImplementedError("Exercise 4: align predictions with next-token targets and compute loss")

    manual_loss = shifted_answer_loss(output.logits, batch.labels)
    print("model loss / manually shifted loss:", output.loss.item(), manual_loss.item())
    torch.testing.assert_close(output.loss.float(), manual_loss, rtol=0.0001, atol=1e-05)
    with torch.no_grad():
        altered = output.logits.detach().clone()
        ignored_predictions = batch.labels[:, 1:] == -100
        altered[:, :-1][ignored_predictions] = 0
        torch.testing.assert_close(shifted_answer_loss(altered, batch.labels), manual_loss.detach())
    del altered
    # Changing predictions that have no supervised next-token target cannot change the loss.
    print("Exercise 4 checks passed: loss parity and ignored-position invariance.")
    loss_verified = True
    return (loss_verified,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 5: Send learning signals through frozen weights

    **Explain first:** Explain how changing an input can change a function output even when its weights are fixed.
    Apply that reasoning to projector → frozen LM → loss.

    **Predict before running:** Predict which parameters should have gradients. Predict what wrapping the LM
    training forward pass in no_grad would do.

    **Implement:** Replace the unfinished line with one backward call on `_gradient_output.loss`. The provided code
    rebuilds the forward graph on each cell run, so rerunning this exercise does not reuse a consumed graph. Leave
    the model weights frozen.

    **After the checks:** Explain the difference among eval(), requires_grad_(False), and no_grad(). Why does the
    frozen LM still consume activation memory?
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Write 3-5 plain-language sentences.]

    **Prediction:** [Write shapes or expected behavior, with your reasoning.]

    **What actually happened:** [Fill this in after running the checks.]

    **Gap I discovered:** [State what you misunderstood or what the checks did not establish.]

    **Revised explanation:** [Teach the idea again in simpler words, using a concrete example.]
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    <details>
    <summary>Explanation hint — open after your first attempt</summary>

    ## 8. Frozen weights still transmit gradients

    `requires_grad_(False)` prevents parameter-gradient storage and updates for the LM. It does **not** disable
    differentiation with respect to the LM inputs. We need `∂loss/∂image_embeddings` to update the projector.

    Wrapping the LM training forward pass in `no_grad()` would break this path. Calling `eval()` is different: it
    changes behaviors such as dropout, while leaving autograd enabled. The frozen LM still needs activation memory
    for backpropagation; freezing 998M parameters does not make training free.

    The checks below verify that the loss reaches the projector and that neither pretrained component accumulates
    parameter gradients.

    </details>
    """
    )
    return


@app.cell
def _(
    answers,
    lm,
    loss_verified,
    projector,
    train_features,
    training_batch,
    vision,
):
    assert loss_verified
    projector.zero_grad(set_to_none=True)
    _gradient_batch = training_batch(train_features[:1], [answers[0]], projector)
    _gradient_output = lm(
        inputs_embeds=_gradient_batch.inputs_embeds,
        attention_mask=_gradient_batch.attention_mask,
        labels=_gradient_batch.labels,
        use_cache=False,
    )
    raise NotImplementedError("Exercise 5: propagate the scalar loss backward without unfreezing the LM")
    _gradient_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in projector.parameters()]))
    print("Projector gradient norm:", _gradient_norm.item())
    assert torch.isfinite(_gradient_norm) and _gradient_norm > 0
    assert all(p.grad is None for p in lm.parameters())
    assert all(p.grad is None for p in vision.parameters())
    projector.zero_grad(set_to_none=True)
    gradients_verified = True
    return (gradients_verified,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 9. Evaluate before training, then try to overfit

    We use AdamW with no weight decay and clip the projector gradient norm at 1.0. Each step trains on one of the
    retained examples to limit activation memory. Cycling gives equal exposure when the update count is divisible
    by the training-example count; otherwise exposure differs by at most one update. Forty steps is a starting
    budget, not a promised convergence threshold.

    Evaluation averages each example's answer/EOS loss. We compare correct images to a cyclic shift of the images:
    images are reassigned to different targets. A reassigned natural image may still partially fit its caption. If
    the correct-image loss becomes lower, that is evidence the predictions depend on the image. A reduced training
    loss alone cannot establish this.

    Each training-button click creates a separate `trained_projector` from the initial weights and a fresh
    optimizer to make the comparison reproducible. We log held-out loss before and after training; the plot
    contains individual training-example losses, which can oscillate across examples.
    """
    )
    return


@app.cell
def _():
    train_button = mo.ui.run_button(label="Train projector")
    mo.output.append(train_button)
    return (train_button,)


@app.cell
def _(
    Example,
    answers,
    cfg,
    gradients_verified,
    heldout_examples,
    heldout_features,
    initial_projector,
    lm,
    projector,
    train_button,
    train_examples,
    train_features,
    training_batch,
):
    mo.stop(not train_button.value, mo.md("Complete Exercises 0-5, then click **Train projector**."))
    assert gradients_verified

    @torch.no_grad()
    def answer_loss(features: torch.Tensor, examples: list[Example], projector_model: nn.Module) -> float:
        losses = []
        for feature, example in zip(features, examples, strict=True):
            evaluation = training_batch(feature.unsqueeze(0), [example.answer], projector_model)
            result = lm(
                inputs_embeds=evaluation.inputs_embeds,
                attention_mask=evaluation.attention_mask,
                labels=evaluation.labels,
                use_cache=False,
            )
            losses.append(result.loss.item())
        return sum(losses) / len(losses)

    trained_projector = copy.deepcopy(projector)
    trained_projector.load_state_dict(initial_projector)
    optimizer = torch.optim.AdamW(trained_projector.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
    before = {
        "train": answer_loss(train_features, train_examples, trained_projector),
        "heldout": answer_loss(heldout_features, heldout_examples, trained_projector),
        "heldout_wrong_image": answer_loss(heldout_features.roll(1, dims=0), heldout_examples, trained_projector),
    }
    history = []
    for step in range(cfg.steps):
        _index = step % len(train_examples)
        optimizer.zero_grad(set_to_none=True)
        training = training_batch(train_features[_index : _index + 1], [answers[_index]], trained_projector)
        result = lm(
            inputs_embeds=training.inputs_embeds,
            attention_mask=training.attention_mask,
            labels=training.labels,
            use_cache=False,
        )
        result.loss.backward()
        norm = nn.utils.clip_grad_norm_(trained_projector.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(result.loss.item())
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={history[-1]:.4f} gradient_norm={norm.item():.4f}")
        del result, training
    after = {
        "train": answer_loss(train_features, train_examples, trained_projector),
        "heldout": answer_loss(heldout_features, heldout_examples, trained_projector),
        "heldout_wrong_image": answer_loss(heldout_features.roll(1, dims=0), heldout_examples, trained_projector),
    }
    print("before:", before)
    print("after: ", after)
    plt.plot(range(1, cfg.steps + 1), history)
    plt.xlabel("Projector update")
    plt.ylabel("Answer + EOS cross entropy")
    mo.output.append(plt.gcf())
    plt.close(plt.gcf())
    return after, before, trained_projector


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 6: Generate an answer without teacher forcing

    **Explain first:** Explain how generation differs from training. What supplies the second answer token's
    context at inference?

    **Predict before running:** Predict how the embedding sequence length changes per iteration. Explain which
    logit row you need and what happens at EOS.

    **Implement:** Complete `generate_answer`. Use the explicit `max_new_tokens` budget in a bounded greedy loop,
    append predicted-token embeddings, and return only the generated text. All evaluation/reporting code is
    supplied.

    **After the checks:** Explain why low teacher-forced loss can coexist with poor free-running answers. Identify
    the work repeated by this cache-free decoder.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Your explanation and predictions — edit this cell

    **First explanation:** [Write 3-5 plain-language sentences.]

    **Prediction:** [Write shapes or expected behavior, with your reasoning.]

    **What actually happened:** [Fill this in after running the checks.]

    **Gap I discovered:** [State what you misunderstood or what the checks did not establish.]

    **Revised explanation:** [Teach the idea again in simpler words, using a concrete example.]
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    <details>
    <summary>Explanation hint — open after your first attempt</summary>

    ## 10. Generate without showing the answer

    Generation starts with BOS, image embeddings, and the prompt only. Each iteration takes the most likely next
    token, appends its embedding, and reruns the LM. EOS stops decoding. This explicit greedy loop makes the
    train/inference boundary visible.

    We disable the KV cache here to keep the code short; production decoding should cache prior attention keys and
    values (the KV cache). Greedy decoding is deterministic for a fixed model/device, but bitwise reproducibility
    across hardware is not guaranteed. Base models may emit extra text, and a random smoke model's strings have no
    semantic significance.

    In smoke mode we report strict color exact-match, so “red square” does not match “red”. With four examples the
    score moves in 25-point increments. In LLaVA mode, inspect generated captions alongside targets and compare
    caption loss for correct versus wrong images; valid captions need not match the target word for word.

    </details>
    """
    )
    return


@app.cell
def _():
    generate_button = mo.ui.run_button(label="Generate answers")
    mo.output.append(generate_button)
    return (generate_button,)


@app.cell
def _(
    RunMode,
    cfg,
    generate_button,
    heldout_examples,
    heldout_features,
    lm,
    prefix_embeddings,
    tokenizer,
    trained_projector,
):
    mo.stop(not generate_button.value, mo.md("Complete Exercise 6, then click **Generate answers**."))

    @torch.no_grad()
    def generate_answer(feature: torch.Tensor, max_new_tokens: int) -> str:
        """Greedily decode from image + prompt only; stop at EOS or the token budget."""
        embedded = prefix_embeddings(feature.unsqueeze(0), trained_projector)  # noqa: F841
        generated = []  # noqa: F841
        raise NotImplementedError(
            "Exercise 6: implement the greedy next-token loop"
        )  # Keep the LM cache disabled. Decode only generated IDs, excluding EOS.

    predictions = []
    for _example, correct, wrong in zip(
        heldout_examples, heldout_features, heldout_features.roll(1, dims=0), strict=True
    ):
        prediction = generate_answer(correct, cfg.max_new_tokens)
        wrong_prediction = generate_answer(wrong, cfg.max_new_tokens)
        predictions.append(prediction)
        print(f"target={_example.answer!r} correct_image={prediction!r} wrong_image={wrong_prediction!r}")
    accuracy = None
    if cfg.mode == RunMode.SMOKE:
        accuracy = sum(
            (
                prediction.lower() == example.answer
                for prediction, example in zip(predictions, heldout_examples, strict=True)
            )
        ) / len(heldout_examples)
        print("Held-out rendering exact match:", accuracy)
    else:
        print("Caption evaluation: use the correct/wrong-image losses above and inspect generations.")
    assert generate_answer(heldout_features[0], 0) == "", "Respect the output token budget."
    with torch.no_grad():
        for feature in heldout_features:
            _prefix = prefix_embeddings(feature.unsqueeze(0), trained_projector)
            next_id = lm(inputs_embeds=_prefix, use_cache=False).logits[0, -1].argmax().item()
            _expected = (
                ""
                if next_id == tokenizer.eos_token_id
                else tokenizer.decode([next_id], skip_special_tokens=True).strip()
            )
            assert generate_answer(feature, 1) == _expected, "Use final-position logits."
    assert generate_answer(heldout_features[0], cfg.max_new_tokens) == predictions[0]
    # At a zero-token budget, generation must not produce an answer.
    print(
        "Exercise 6 budget, first-token, and determinism checks passed. "
        "These do not establish answer quality or exhaustively test later EOS handling."
    )
    return (accuracy,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 11. Save what actually learned

    Only the projector changed. Its weights are usable only with the same backbone, encoder, feature selection,
    preprocessing, prompt, and sequence layout. We save those references alongside configuration and measured
    losses. In pretrained mode, these projector weights plus the pinned backbones and notebook architecture are
    enough to reconstruct inference. In smoke mode, reconstruction also requires replaying the seeded random
    backbone initialization; those random backbone weights are not saved. Resuming training exactly would also
    require optimizer and random-number-generator (RNG) state.

    Click the save button to write the trained projector. The output goes to a temporary directory, outside the
    repository. After saving the weights, use **Download projector** to export them. Clicking save again replaces this
    notebook's local output file. The reload check compares projected embeddings before and after a real
    serialization round trip.
    """
    )
    return


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save projector")
    mo.output.append(save_button)
    return (save_button,)


@app.cell
def _(
    PROMPT,
    RunMode,
    accuracy,
    after,
    before,
    cfg,
    device,
    heldout_examples,
    heldout_features,
    lm_dtype,
    processor,
    save_button,
    train_examples,
    trained_projector,
):
    mo.stop(not save_button.value, mo.md("Click **Save projector** to write the trained weights."))
    artifact_dir = Path(tempfile.gettempdir()) / "marin-vlm-exploration" / cfg.mode.value
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "projector.pt"
    metadata = {
        "config": dataclasses.asdict(cfg),
        "prompt": PROMPT,
        "layout": "BOS,image_patches,prompt,answer,EOS",
        "vision_features": "last_hidden_state[:,1:,:]",
        "processor": processor.to_dict(),
        "packages": {name: version(name) for name in ("torch", "transformers", "pillow", "pyarrow", "huggingface-hub")},
        "python": platform.python_version(),
        "lm_dtype": str(lm_dtype),
        "data_source": "synthetic" if cfg.mode == RunMode.SMOKE else cfg.dataset_id,
        "train_image_ids": [example.source_id for example in train_examples],
        "heldout_image_ids": [example.source_id for example in heldout_examples],
        "split_rule": (
            "synthetic rendering shift"
            if cfg.mode == RunMode.SMOKE
            else "sha256(encoded_image) first 8 hex digits modulo 5; heldout=0"
        ),
        "before": before,
        "after": after,
        "heldout_exact_match": accuracy,
        "issue": "https://github.com/marin-community/marin/issues/9524",
    }
    # Convert StrEnum and any processor metadata into JSON-safe primitives.
    metadata = json.loads(json.dumps(metadata))
    torch.save(
        {
            "projector": {name: value.cpu() for name, value in trained_projector.state_dict().items()},
            "metadata": metadata,
        },
        artifact_path,
    )
    restored = torch.load(artifact_path, map_location=device, weights_only=True)
    restored_projector = copy.deepcopy(trained_projector)
    restored_projector.load_state_dict(restored["projector"])
    with torch.no_grad():
        torch.testing.assert_close(restored_projector(heldout_features), trained_projector(heldout_features))
    print("Saved and reloaded:", artifact_path)
    mo.output.append(
        mo.download(
            data=artifact_path.read_bytes,
            filename=f"marin-vlm-{cfg.mode.value}-projector.pt",
            mimetype="application/octet-stream",
            label="Download projector",
        )
    )
    print(
        json.dumps({"mode": cfg.mode.value, "before": before, "after": after, "heldout_exact_match": accuracy}, indent=2)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise 7: challenge your explanation with an experiment

    **Explain:** If the training loss falls, have we shown the model uses images? Explain why or why not to someone
    who has never seen a loss curve.

    **Predict:** Choose one change: replace the MLP with a linear projector, or average patch vectors into one
    image token. Write a directional prediction for loss, token/parameter count, and image dependence. Give a
    reason that could be wrong.

    **Implement and compare:** Save the first run's measurements below, then make your change. For pooling,
    edit the existing feature-extraction cell after its Exercise 1 checks: average `train_features` and
    `heldout_features` along the patch axis with `keepdim=True`, assigning back to those names in that same cell.
    Keep `select_patch_features` unchanged; its job remains to preserve all patches. Each rerun extracts fresh
    features before pooling them. Run the stale dependent cells and click the training/generation buttons again.
    Reinitialize the projector and optimizer; hold backbone, examples, seed, and update budget fixed. Keep both
    correct-image and wrong-image evaluations. Do not compare smoke results against pretrained results.

    | Run | Mode | Change | Image tokens | Parameters | Train loss | Held-out loss | Wrong-image loss | Exact match |
    |---|---|---|---|---|---|---|---|---|
    | Baseline | | | | | | | | |
    | Your variant | | | | | | | | |

    **Your prediction and reasoning:** [Write before running the variant.]

    **Your implementation change:** [Name the function and describe what you changed.]

    **Evidence for or against your explanation:** [Use the measurements, including negative results. Use exact
    match only for the synthetic color task; mark it N/A for captions.]

    **Revised explanation:** [Explain what your experiment taught you, without jargon.]

    **What this cannot establish:** [Discuss the bounded caption sample or four-color dataset and the frozen/random
    versus pretrained backbone.]

    **Transfer question:** What would need to change to scale caption alignment and then train on natural-image
    questions, and what would stay the same? Name the data split, trainable components, and loss-mask decisions.

    Do not mark the exercise complete merely because code ran. You should be able to reconstruct the
    image-to-answer path from a blank page and explain each gradient and mask.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 12. Change one thing at a time

    1. **Linear versus MLP projector:** replace the `nn.Sequential` with one `nn.Linear(vision_width, lm_width)`.
    Rerun the projector cell and click the training and generation buttons again. Keep checkpoint revisions,
    examples, seed, and update budget fixed; compare correct/wrong-image loss, trainable parameter counts, and (for
    smoke mode only) exact match. This toy task cannot decide which connector is better for natural-image VQA.
    2. **Image-token count:** averaging all patches into one vector shortens the LM sequence, but loses the
    separate spatial positions. Apply the same transformation to training and held-out features. Compare quality
    and memory at a fixed training budget.
    3. **Natural images:** the pretrained path now uses a bounded LLaVA 558K sample. For a larger experiment,
    replace this first-row sample with a representative split by source image and train on fresh streamed batches.
    A suggested workflow is caption alignment (train the connector to predict image captions with both backbones
    frozen), followed by image-question-answer instruction tuning. Pretrained mode demonstrates caption alignment
    on a small sample; smoke mode demonstrates a synthetic color question. Full visual instruction tuning remains a
    later stage. Mask prompt/image positions exactly as above.
    4. **Adapt the LM:** after alignment, consider low-rank adaptation (LoRA), which learns small additional weight
    matrices, or selected unfrozen layers. Add those trainable parameters to the optimizer and budget for their
    gradients/state. The current optimizer updates only the projector. Freezing the LM here was a choice for
    learning and low-cost exploration, not a requirement of VLM training.
    5. **Marin 8B:** choose an immutable revision of `marin-community/marin-8b-base`, load its tokenizer/config,
    and recreate the projector. Its hidden width is 4096, so this Delphi projector cannot be reused unchanged.
    Budget for substantially more weight and activation memory.

    Record the exact mode, revisions, seed, data split, update count, before/after losses, generated answers, and
    limitations in [#9524](https://github.com/marin-community/marin/issues/9524). This notebook does not post to
    GitHub automatically.

    ### How this connects to Marin

    - [Model download handles](../models.py) identify pretrained checkpoints separately from architecture presets.
    - [Levanter Llama implementation](../../lib/levanter/src/levanter/models/llama.py) separates embeddings,
    transformer, and language-model head. Those are the same boundaries used here.
    - [Grug](../grug/README.md) is a template-based model/train/launch workflow with its own variants. This
    notebook does not load Delphi weights into Grug.
    - A native training experiment will also need multimodal data loading, checkpoint export, distributed
    placement, and evaluation wiring. The notebook establishes the tensor and loss contracts before that
    integration.

    ### Sources

    - [Delphi 998M model card](https://huggingface.co/marin-community/delphi-3e20-998Mparams-49.8Btokens)
    - [CLIP model and preprocessing API](https://huggingface.co/docs/transformers/model_doc/clip)
    - [Qwen3 model API](https://huggingface.co/docs/transformers/model_doc/qwen3)
    - [LLaVA architecture](https://huggingface.co/docs/transformers/model_doc/llava)

    No pretrained capability results are included in the saved notebook. Run the cells on your chosen hardware to
    produce them.
    """
    )
    return


if __name__ == "__main__":
    app.run()
