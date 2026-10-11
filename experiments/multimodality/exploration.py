# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "torch==2.11.0",
#     "transformers==5.12.1",
#     "pillow>=12.2.0",
#     "pyarrow==23.0.1",
#     "matplotlib",
#     "huggingface-hub==1.33.0",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import dataclasses
    import hashlib
    import json
    import platform
    import tempfile
    from collections.abc import Iterator
    from contextlib import closing
    from dataclasses import dataclass
    from importlib.metadata import version
    from io import BytesIO
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from huggingface_hub import HfApi, HfFileSystem
    from PIL import Image
    from pyarrow.parquet import ParquetFile
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Build a VLM with Delphi + SigLIP

    Connect a frozen SigLIP vision encoder to the frozen Delphi 998M language model. Train only a small
    projector that turns image features into LM input embeddings.

    Work through data loading, tokenization, embedding alignment, next-token prediction and loss, training,
    and further experiments. Each exercise introduces the background you need, then leaves the core code for you.
    Replace the marked `NotImplementedError` statements and inspect the supplied shapes and checks.

    Tracking: [Marin #9524](https://github.com/marin-community/marin/issues/9524).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup

    Open this file in [molab](https://molab.marimo.io) and enable a GPU, or run locally:

    ```bash
    uv run --no-project --with 'marimo==0.25.1' marimo edit --sandbox experiments/multimodality/exploration.py
    ```

    Use lazy execution while editing exercises. Set `Config.device` to `cuda` for a GPU or `cpu` locally.
    Model and data downloads start only when you click their buttons. The pretrained weights require gigabytes
    of downloads; CPU training will be slow. Export your notebook edits and projector before ending a remote session.
    """)
    return


@app.cell
def _():

    @dataclass(frozen=True)
    class Config:
        device: str = "cpu"  # Set to "cpu" for local exploration without a GPU.
        lm_id: str = "marin-community/delphi-3e20-998Mparams-49.8Btokens"
        lm_revision: str = "9b913fac63d36a80424949775c63c728ac8fa9fc"
        vision_id: str = "google/siglip-base-patch16-224"
        vision_revision: str = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
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
    print(cfg)
    return cfg, device, lm_dtype


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checkpoints

    Delphi is a base language model trained to predict the next text token. Load its own tokenizer.
    SigLIP converts a processed 224 x 224 image into a 14 x 14 grid of patch vectors: `[B, 196, 768]`.
    Here `B` is batch size. Its final hidden states contain patches without a classification (CLS) token.
    Keep all 196 positions; avoid the pooled output.

    Both backbones stay frozen. The projector will map the vision width to Delphi's hidden width.
    The checkpoint revisions below fix which weights and preprocessing this study uses.
    """)
    return


@app.cell
def _():
    load_models = mo.ui.run_button(label="Load models")
    mo.output.append(load_models)
    return (load_models,)


@app.cell
def _(cfg, device, lm_dtype, load_models):
    mo.stop(not load_models.value, mo.md("Click **Load models** when ready."))
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_id, revision=cfg.lm_revision)
    lm = AutoModelForCausalLM.from_pretrained(
        cfg.lm_id, revision=cfg.lm_revision, dtype=lm_dtype, attn_implementation="eager"
    )
    processor = SiglipImageProcessor.from_pretrained(cfg.vision_id, revision=cfg.vision_revision)
    vision = SiglipVisionModel.from_pretrained(cfg.vision_id, revision=cfg.vision_revision)
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


@app.cell
def _(tokenizer):
    print("Vocabulary size:", len(tokenizer))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Data loading

    We use a bounded sample of [LLaVA image-caption pairs](https://huggingface.co/datasets/theblackcat102/llava-pretrain).
    Each row contains encoded image bytes and a human/gpt conversation. The human turn supplies a request;
    the gpt turn supplies the caption target. Images become embeddings, so `<image>` is never a caption target.

    The supplied loader reads Parquet batches with HTTP range requests and retains a small in-memory sample.
    `stream_scan_rows` bounds inspected rows, but a read can fetch a whole row group. Image-byte hashes assign
    examples to training or held-out splits and keep identical bytes together. Different encodings of the same
    image can still cross splits. This small first-row sample is a learning diagnostic.

    Every image uses the same prompt, `Describe this image. Caption:`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: extract a caption

    Background: supervised learning needs an input and a target. Here the image and fixed prompt are inputs;
    the assistant caption is the target. The loader decodes JSON when needed before passing a list of turns.

    Code: implement `caption_target` to require two turns in human → gpt order and return the stripped,
    nonempty caption. Your existing implementation is kept below. Run the fixture checks and inspect the images.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Your data notes: The model should predict the caption text when provided the image.
    `<image>` and the human request are inputs, so they should not be caption targets.
    Fixture prediction: "A red square on a white background."
    """)
    return


@app.cell
def _():
    def caption_target(conversations: list[dict]) -> str:
        """Extract the caption from one LLaVA human/gpt conversation."""
        if len(conversations) != 2:
            raise ValueError("There need to be two turns in this conversation")
        if ["human", "gpt"] != [conv["from"] for conv in conversations]:
            raise ValueError("Turn must only contain human and GPT in that order")

        gpt_turn = conversations[1]["value"].strip()
        if not gpt_turn:
            raise ValueError("Empty turn")
        return gpt_turn

    fixture = [
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "  A red square on a white background.  "},
    ]
    assert caption_target(fixture) == "A red square on a white background."
    for turns in (
        [{"from": "human", "value": "<image>"}],
        [{"from": "gpt", "value": "wrong first role"}, {"from": "human", "value": "wrong target"}],
        [{"from": "human", "value": "<image>"}, {"from": "gpt", "value": " "}],
        [{"from": "human", "value": "<image>"}, {"from": "human", "value": " "}],
    ):
        try:
            caption_target(turns)
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
def _(caption_target, cfg, load_examples):
    mo.stop(not load_examples.value, mo.md("Click **Load examples** after completing Exercise 0."))

    @dataclass(frozen=True)
    class Example:
        image: Image.Image
        answer: str
        source_id: str

    PROMPT = "Describe this image. Caption:"

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
                turns = row["conversations"]
                caption = caption_target(json.loads(turns) if isinstance(turns, str) else turns)
                with Image.open(BytesIO(raw)) as image:
                    destination.append(Example(image.convert("RGB"), caption, identity))
        if len(train) != cfg.train_examples or len(heldout) != cfg.heldout_examples:
            raise ValueError("Not enough unique examples in both splits; increase stream_scan_rows explicitly.")
        assert {item.source_id for item in train}.isdisjoint(item.source_id for item in heldout)
        return (train, heldout)

    train_examples, heldout_examples = llava_examples()
    print("Source:", cfg.dataset_id)
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
    mo.md(r"""
    ## 2. Tokenization

    Background: the checkpoint tokenizer maps text to vocabulary IDs, which the LM embedding table maps
    to vectors. A word can span several tokens. We tokenize the prompt and caption separately to define
    their boundary, add beginning-of-sequence (BOS) ourselves, and append end-of-sequence (EOS) to each caption.
    EOS teaches the LM when to stop. The leading caption space separates text from the prompt ending `Caption:`.

    Code: complete `caption_ids` with the Delphi tokenizer, `add_special_tokens=False`, and a leading
    space before the caption. Append exactly one EOS ID. Inspect IDs and decoded text below.
    """)
    return


@app.cell
def _(train_examples):
    train_examples[0].answer
    return


@app.cell
def _(lm, tokenizer):
    text = "The pteradactyl is"
    tokenized_text = tokenizer(text=text, return_tensors="pt").to(lm.device)
    print(tokenized_text)
    prediction = lm.generate(**tokenized_text)
    print(prediction)
    print(tokenizer.batch_decode(prediction)[0])
    return (prediction,)


@app.cell
def _(prediction, tokenizer):
    tokenizer.decode(prediction[:, -4:])
    return


@app.cell
def _(tokenizer):
    tokenizer.decode(466)
    return


@app.cell
def _(tokenizer):
    tokenizer.vocab["EOS"]
    return


@app.cell
def _(tokenizer):
    tokenizer.bos_token_id
    return


@app.cell
def _(tokenizer, train_examples):
    def caption_ids(caption: str) -> list[int]:
        """Tokenize a leading-space caption and append EOS."""
        tokens =  [*tokenizer(" " + caption, add_special_tokens=False).input_ids, tokenizer.eos_token_id]
        return tokens
    _caption = train_examples[0].answer
    _ids = caption_ids(_caption)
    print(_ids, [*tokenizer(" " + _caption, add_special_tokens=False).input_ids, tokenizer.eos_token_id])
    assert _ids == [*tokenizer(" " + _caption, add_special_tokens=False).input_ids, tokenizer.eos_token_id]
    print("Caption:", _caption)
    print("IDs:", _ids)
    print("Decoded:", tokenizer.decode(_ids))
    return (caption_ids,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: retain image patches

    Background: the processor normalizes pixels and returns `[B, 3, 224, 224]`. SigLIP's vision transformer
    returns `[B, 196, 768]` contextual patch features. Each patch has already attended to the other patches.
    There is no CLS position to remove. Averaging patches loses their separate spatial positions.

    Code: complete `select_patch_features` to keep every patch in order. The processor call, frozen encoder,
    and feature cache are supplied. `no_grad()` is appropriate for this frozen feature extraction.
    """)
    return


@app.cell
def _(vision):
    print("Patch count: ", (vision.config.image_size // vision.config.patch_size)**2)
    return


@app.cell
def _(Example, device, heldout_examples, processor, train_examples, vision):
    def select_patch_features(states: torch.Tensor) -> torch.Tensor:
        """Return the contextual patch vectors, preserving every SigLIP patch position."""
        print(states.shape)
        return states

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

    # Check that two different images retain distinct features, all patch positions, and patch order.
    probe = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    selected = select_patch_features(probe)
    assert selected.shape == probe.shape, "SigLIP has no CLS token; keep every patch."
    torch.testing.assert_close(selected[:, 0], probe[:, 0])
    torch.testing.assert_close(selected[:, -1], probe[:, -1])
    assert not torch.equal(train_features[0], train_features[1]), "Different images lost their distinction."
    print("Exercise 1 checks passed.")
    return heldout_features, train_features


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. LM embedding alignment

    Background: an LM reads vectors of its hidden width. Text IDs become vectors through its embedding table;
    image features need a learned projection into that space. Matching widths makes the tensors compatible,
    and caption training learns useful semantics.

    Code: complete `build_projector` with Linear → GELU → Linear. Both linear outputs have LM width.
    Preserve batch and patch axes: `[B, 196, vision_width] → [B, 196, lm_width]`.
    The projector stays FP32; casting its output to the LM dtype preserves gradients.
    """)
    return


@app.cell
def _(cfg, device, lm, train_features, vision):
    torch.manual_seed(cfg.seed)

    def build_projector(vision_width: int, lm_width: int) -> nn.Module:
        """Map every patch independently through Linear, GELU, Linear to LM width."""
        print(vision_width, lm_width)
        return nn.Sequential(
            nn.Linear(vision_width, lm_width),
            nn.GELU(),
            nn.Linear(lm_width, lm_width)
        )

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
    mo.md(r"""
    ### Exercise: assemble the sequence and targets

    Background: teacher forcing supplies the correct previous caption tokens while predicting the next token.
    Build `[BOS] [image embeddings] [prompt embeddings] [caption embeddings] [EOS]`, then right-pad.
    `inputs_embeds` is `[B, S, D]`; attention masks and labels are `[B, S]`.
    Here `B` is batch size, `S` is sequence length, and `D` is the LM hidden width.

    The attention mask hides padding. The LM's causal mask hides future positions. Labels of `-100` exclude
    BOS, images, prompt, and padding from loss; caption and EOS labels keep their vocabulary IDs.
    EOS must remain supervised even if the tokenizer also uses EOS for padding.

    Code: complete `prefix_embeddings` and `answer_targets`. Concatenate along the sequence axis and keep
    all caption/EOS targets. Batching and right-padding are supplied. Image patches consume LM context positions.
    """)
    return


@app.cell
def _(tokenizer):
    tokenizer.vocab["PAD"], len(tokenizer.vocab)
    return


@app.cell
def _(
    PROMPT,
    caption_ids,
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
        bos_embedding = embedding(bos)
        prompt_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(device)  # noqa: F841
        image_embeds = projector_model(features).to(embedding.weight.dtype)  # noqa: F841
        prompt_embeds = embedding(prompt_ids)
        input_prefix = torch.cat([bos_embedding, image_embeds, prompt_embeds], axis=1)
        return input_prefix

    def answer_targets(prefix_length: int, answer_ids: torch.Tensor) -> torch.Tensor:
        """Return ignored prefix targets followed by answer/EOS targets on the same device."""
        raise NotImplementedError("Exercise 3b: mask the prefix while preserving every answer and EOS ID")

    def training_batch(features: torch.Tensor, answers: list[str], projector_model: nn.Module) -> Batch:
        rows, targets, starts = ([], [], [])
        for feature, answer in zip(features, answers, strict=True):
            prefix = prefix_embeddings(feature.unsqueeze(0), projector_model)[0]
            answer_ids = caption_ids(answer)
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
    mo.md(r"""
    ## 4. Next-token prediction and loss

    Background: logits at position `t` predict the token at `t+1`. The final prompt position predicts the first
    caption token, and the final caption position predicts EOS. Cross entropy is the mean negative log-probability
    of the supervised targets. Labels of `-100` are ignored.

    Code: complete `shifted_answer_loss`: compare `logits[:, :-1]` with `labels[:, 1:]` using cross entropy.
    Flatten batch and sequence axes, preserve the vocabulary axis, and compute in FP32.
    Hugging Face shifts internally when given unshifted labels; compare your loss with that result.
    The plot shows causal attention and the prediction positions that contribute to loss.
    """)
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
    mo.md(r"""
    ## 5. Training

    Background: freezing weights prevents parameter updates while still allowing derivatives with respect
    to their inputs. The loss must flow through Delphi into the projector. `eval()` controls behaviors such
    as dropout; it does not disable gradients. `no_grad()` around the LM training forward would break learning.

    Code: complete the backward call below. Check that the projector receives nonzero gradients and both
    backbones retain no parameter gradients.
    """)
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
    mo.md(r"""
    ### Exercise: update the projector

    Background: AdamW uses gradients to update parameters. Each step uses one caption to limit activation memory.
    Training starts from the same initial projector on each click. Falling training loss may reflect memorization;
    held-out loss and mismatched-image loss help check whether the image affects prediction.

    Code: complete `projector_step` with backward propagation, gradient clipping at norm 1.0, and an optimizer
    step. The surrounding loop, evaluation, and loss plot are supplied. Compare before/after losses for correct
    and cyclically reassigned images. Forty updates are only a starting budget.
    """)
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

    def projector_step(loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Backpropagate, clip gradients, update weights, and return the pre-clipping norm."""
        raise NotImplementedError("Training: backward, clip gradients at 1.0, optimizer.step, return norm")

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
        norm = projector_step(result.loss, trained_projector, optimizer)
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
    mo.md(r"""
    ## 6. Further experiments

    ### Exercise: generate a caption

    Background: generation begins with the image and prompt, then feeds back each predicted token.
    Greedy decoding chooses the largest final-position logit. Stop at EOS or the token budget.
    Unlike teacher forcing, generation never receives the reference caption.

    Code: complete `generate_answer`. Run the LM with `inputs_embeds` and `use_cache=False`, choose the next ID,
    append its embedding, and repeat. Decode only generated IDs, excluding EOS. Compare captions from correct
    and reassigned images; the supplied checks verify mechanics, not caption quality.
    """)
    return


@app.cell
def _():
    generate_button = mo.ui.run_button(label="Generate answers")
    mo.output.append(generate_button)
    return (generate_button,)


@app.cell
def _(
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
    print("Compare the correct/wrong-image losses and inspect generated captions.")
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
    return prediction, predictions


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Save your projector

    Only the projector learned. Save its weights together with the pinned checkpoints, processor, prompt,
    sequence layout, data split, and measured losses. Those references are required to reconstruct inference.
    Exact training resumption also requires optimizer and RNG state.

    The button writes to a temporary directory and checks a save/reload round trip. Download the file before
    ending a remote session.
    """)
    return


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save projector")
    mo.output.append(save_button)
    return (save_button,)


@app.cell
def _(
    PROMPT,
    after,
    before,
    cfg,
    device,
    heldout_examples,
    heldout_features,
    lm_dtype,
    predictions,
    processor,
    save_button,
    train_examples,
    trained_projector,
):
    mo.stop(not save_button.value, mo.md("Click **Save projector** to write the trained weights."))
    artifact_dir = Path(tempfile.gettempdir()) / "marin-vlm-exploration"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "projector.pt"
    metadata = {
        "config": dataclasses.asdict(cfg),
        "prompt": PROMPT,
        "layout": "BOS,image_patches,prompt,answer,EOS",
        "vision_features": "last_hidden_state",
        "processor": processor.to_dict(),
        "packages": {name: version(name) for name in ("torch", "transformers", "pillow", "pyarrow", "huggingface-hub")},
        "python": platform.python_version(),
        "lm_dtype": str(lm_dtype),
        "data_source": cfg.dataset_id,
        "train_image_ids": [example.source_id for example in train_examples],
        "heldout_image_ids": [example.source_id for example in heldout_examples],
        "split_rule": "sha256(encoded_image) first 8 hex digits modulo 5; heldout=0",
        "before": before,
        "after": after,
        "generated_captions": predictions,
        "issue": "https://github.com/marin-community/marin/issues/9524",
    }
    # Convert processor metadata into JSON-safe primitives.
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
            filename="delphi-siglip-projector.pt",
            mimetype="application/octet-stream",
            label="Download projector",
        )
    )
    print(json.dumps({"before": before, "after": after, "generated_captions": predictions}, indent=2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: compare one variant

    Background: a controlled comparison changes one factor while keeping checkpoints, examples, seed, and
    update budget fixed. A lower training loss alone cannot show improved image use.

    Code: choose a linear projector or average patches into one image vector. For pooling, transform both
    `train_features` and `heldout_features` after extraction using the patch axis and `keepdim=True`.
    Reinitialize the projector and optimizer, rerun dependent cells, and train again.

    | Run | Change | Image tokens | Parameters | Train loss | Held-out loss | Wrong-image loss |
    |---|---|---|---|---|---|---|
    | Baseline | MLP, all patches | | | | | |
    | Variant | | | | | | |

    Write a prediction before running, then explain whether the measured result supports it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Change one thing at a time

    1. **Projector:** compare one linear layer with the two-layer MLP. Keep Delphi and SigLIP fixed.
    2. **Image-token count:** average patches into one vector. Apply the same transformation to both splits;
       compare loss and memory at the same update budget.
    3. **Data size:** increase the bounded caption sample, then use a representative split by source image.
       Keep the caption objective fixed before trying image-question-answer data.
    4. **Learning rate or updates:** change one of these and compare the loss curves from the same initial weights.
    5. **Trainable components:** after alignment, try LoRA (small learned weight additions) or selected Delphi layers.
       Include their parameters in the optimizer and budget for additional memory.

    Record revisions, split, seed, update count, before/after losses, generated captions, and limitations.
    This small caption study does not establish general visual question answering.

    [Delphi checkpoint](https://huggingface.co/marin-community/delphi-3e20-998Mparams-49.8Btokens) ·
    [SigLIP checkpoint](https://huggingface.co/google/siglip-base-patch16-224) ·
    [SigLIP API](https://huggingface.co/docs/transformers/model_doc/siglip)
    """)
    return


if __name__ == "__main__":
    app.run()
