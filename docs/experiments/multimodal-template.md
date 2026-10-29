# SigLIP–Delphi multimodal template

`experiments/multimodality/base/` trains an image-caption VLM using the
[SigLIP encoder](https://github.com/bkolligs/marin/pull/6)
and [Delphi 998M](https://huggingface.co/marin-community/delphi-3e19-998Mparams-5Btokens).
`VlmConfig` owns both pinned Hugging Face checkpoint paths.
`VlmConfig.from_checkpoints()` resolves their model dimensions; its default vision
checkpoint is `google/siglip-so400m-patch14-384`.

A single image is resized to 384×384 and encoded into 729 SigLIP patch features.
The VLM omits SigLIP's pooled contrastive head and uses the patch features directly.
A learned linear projection maps each feature to Delphi's 1536-dimensional
embedding space. These features precede the caption tokens in Delphi's causal
transformer. Training predicts caption tokens and EOS from BOS and preceding
caption tokens; image positions and right padding do not contribute to the loss.
All parameters are trainable. The projection starts with random weights; both
backbones load pretrained weights.

## Copy and edit

Copy the base directory before editing, following the layout of `experiments/grug/base/`:

```bash
cp -R experiments/multimodality/base experiments/multimodality/my_variant
```

- `model.py` owns the concrete model, projection, and backbone loading.
- `train.py` owns caption preprocessing, loss, batching, and training.
- `launch.py` owns checkpoint choices and run configuration.

Update the absolute imports in a copied variant to reference its own modules,
such as `experiments.multimodality.my_variant.model`. Components reuse the
Levanter SigLIP and Qwen3 implementations and their named arrays; this
template does not copy Grug's text transformer implementation. Add variant tests
beside the variant.

## Run

Provide JSONL, JSON, or Parquet records with `image` and `text` fields. Images use
the shared image source loader and can be paths, URLs, or supported byte records.
For example, a JSONL record with an absolute image path is:

```json
{"image": "/data/example.png", "text": "A red boat on the water."}
```

Inspect the selected model configurations:

```bash
uv run python -m experiments.multimodality.base.launch \
  --train-data /data/captions.jsonl --output-path ./checkpoints/siglip-delphi \
  --steps 1000 --batch-size 8 --learning-rate 1e-5 --text-length 512 --dry-run
```

The dry run fetches model configuration metadata but does not read training data,
download model weights, or submit a job. Remove `--dry-run` to train in the current
process. To run a copy, replace `base` with its directory name in the module path.

`--text-length` includes the EOS prediction. The loader truncates captions to
reserve EOS, right-pads with EOS, and masks padding. Image-prefix length plus
text length must fit Delphi's 4096-position context.

The streaming loader repeats records in source order, carries partial batches
across epochs, and supports one JAX process. It does not shuffle or partition
records across hosts. Keep inputs and batch size unchanged when resuming: the
loader replays and skips batches up to the restored optimizer step. Levanter
checkpoints include the full VLM and optimizer state under `--output-path`.
Reusing the same command and output path automatically restores the latest
checkpoint. Increase `--steps` to extend the run. Use a different output path
for each experiment.

This base supports single-image caption training. Chat formatting, multiple
images, distributed input loading, generation, and full-VLM Hugging Face export
are not implemented. Architecture changes belong in copied variants.
