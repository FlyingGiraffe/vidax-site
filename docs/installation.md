---
sidebar_position: 2
title: Installation
---

# Installation

vidax targets **Python 3.10–3.12** and follows the standard `src`-layout. It's
installed editable straight from a clone — there's no PyPI release yet
(`pip install vidax` will be added here once one ships).

```bash
git clone https://github.com/FlyingGiraffe/vidax.git
cd vidax
pip install -e .                # everything needed to run a model
pip install -e ".[tpu]"         # on a Cloud TPU VM — pulls the right jax[tpu] wheel
```

That's the whole story for running models — there are **no per-model-family
extras to juggle**. `torch`, `transformers`, `sentencepiece`, and `pillow`
are ordinary (non-optional) dependencies: every model uses them to
*deserialize* the released `.pth`/`.safetensors` checkpoints and to tokenize
text / load the I2V conditioning image. vidax's own model code (everything
under `vidax.models`) never imports them — Cosmos3, for instance, doesn't
touch `torch` at runtime (its checkpoint is `.safetensors`), but `torch` is
still installed as a base dependency.

## Extras

Only two optional groups remain:

| Extra | Contents | When you need it |
| --- | --- | --- |
| `tpu` | `jax[tpu]` | On a Cloud TPU VM, to get the correct `jaxlib` wheel. Off-TPU (CPU/GPU, for translation or CPU smoke tests) you don't need it. |
| `dev` | `pytest`, `pytest-xdist`, `ruff`, `build`, `twine` | Running the test suite, linting, or building the package. See [Contributing](https://github.com/FlyingGiraffe/vidax/blob/main/CONTRIBUTING.md). |

```bash
pip install -e ".[dev]"         # contributors — add ".[tpu]" too on a TPU VM
```

The base dependency set (`jax`, `flax`, `numpy`, `safetensors`, `imageio`,
`imageio-ffmpeg`, `torch`, `transformers`, `sentencepiece`, `pillow`)
installs automatically with `pip install -e .`.

## Checkpoints

vidax loads the **original PyTorch/safetensors checkpoints** released by
each model's authors (Wan-AI, NVIDIA, Lightricks, Tencent, THUDM/ZhipuAI)
and translates them into Flax pytrees at load time — there are no separate
vidax-hosted weights. See each
[model guide](./models/wan2_1.md) for the exact HuggingFace repo to
download from, the expected checkpoint file layout, and which files come
from which repo (some models, like Cosmos-Predict2.5 and HunyuanVideo,
assemble their DiT, VAE, and text encoder from several separate HuggingFace
repos).
