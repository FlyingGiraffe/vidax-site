---
sidebar_position: 2
title: Installation
---

# Installation

vidax targets Python ≥3.10 and follows the standard `src`-layout, installed
editable straight from a clone (no PyPI package yet — `pip install vidax`
will be added here once `v0.1.0-alpha` ships).

```bash
git clone https://github.com/FlyingGiraffe/vidax.git
cd vidax
pip install -e ".[tpu,torch,text]"
```

## Extras

vidax's own model implementations never depend on `torch`/`transformers` —
those libraries are used solely to deserialize checkpoints and tokenize
text, so which extras you need depends on which model family you're
running:

| Extra | Purpose | Needed by |
| --- | --- | --- |
| `tpu` | Correct TPU `jaxlib` wheel | Every model, on a TPU VM |
| `torch` | Deserialize `.pth`/`.safetensors` checkpoints | Wan2.1, Wan2.2, Cosmos-Predict2.5, LTX-Video, LTX-2.5, HunyuanVideo-1.5 — **not** Cosmos3, which ships `.safetensors` loaded directly without `torch` |
| `text` | Tokenizers (UMT5-XXL for Wan, Qwen2.5-VL/Qwen2Tokenizer for Cosmos, T5-XXL for LTX-Video, Gemma-4 for LTX-2.5, Qwen2.5-VL+byT5 for HunyuanVideo-1.5) | Every model |
| `i2v` | `pillow`, for image/video conditioning frames | Every model's I2V/image2world path |
| `dev` | `pytest` + `pytest-xdist` | Running the test suite |

So a Wan-only install is `pip install -e ".[tpu,torch,text]"`; a
Cosmos-Predict2.5 install additionally needs `i2v`
(`".[tpu,torch,text,i2v]"`); a Cosmos3-only install can skip `torch`
entirely (`".[tpu,text,i2v]"`). Core dependencies (`jax`, `flax`, `numpy`,
`safetensors`, `imageio`, `imageio-ffmpeg`) install automatically with the
base package regardless.

## Checkpoints

vidax loads the **original PyTorch/safetensors checkpoints** released by
each model's authors (Wan-AI, NVIDIA) and translates them into Flax
pytrees at load time — there are no separate vidax-hosted weights. See each
[model guide](./models/wan2_1.md) for the exact HuggingFace repo to
download from, the expected checkpoint file layout, and which files come
from which repo (some models, like Cosmos-Predict2.5, assemble their DiT,
VAE, and text encoder from three separate HuggingFace repos).
