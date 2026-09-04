---
sidebar_position: 1
title: Loading PyTorch Weights
---

# Loading PyTorch weights

vidax's model implementations never depend on `torch`/`transformers` for
inference — those libraries are used solely to deserialize the original
authors' `.safetensors`/`.pth` checkpoints and to tokenize text. The
translation from a PyTorch state dict into a Flax pytree happens once, at
load time, via two pieces:

- [`vidax.translator.converter`](https://github.com/FlyingGiraffe/vidax/blob/main/src/vidax/translator/converter.py) —
  host-side (numpy) tensor layout conversion.
- [`vidax.translator.mappings`](https://github.com/FlyingGiraffe/vidax/blob/main/src/vidax/translator/mappings/) —
  one module per model family, mapping PyTorch state-dict keys to their
  Flax pytree equivalents. Structure mirrors `vidax.models` one-for-one:
  `wan2_1.py`, `wan2_2.py`, `wan2_2_diffusers.py` (Cosmos3's VAE, shipped in
  `diffusers`' `AutoencoderKLWan` layout — a different key naming than
  Wan2.2's own release, same underlying architecture), `cosmos2_5.py`,
  `cosmos3.py`, `reason1.py` (Qwen2.5-VL-7B text encoder — reused unmodified
  by HunyuanVideo-1.5's MLLM tower), `ltx_video.py`, `ltx2_5.py`,
  `hunyuan_video.py` (1.0) / `hunyuan_video1_5.py`, `cogvideox.py`, plus a
  shared `common.py` for mappers reused across every Wan version. The
  [API reference for `vidax.translator`](../api/translator.md) has the full
  `model_type` → mapper table and the per-leaf layout rules.

## Layout conversion

JAX/XLA prefers channels-last layouts, so every tensor is transposed on the
way in:

| PyTorch layout | JAX/Flax layout |
| --- | --- |
| Conv3D `[Batch, Channels, Time, Height, Width]` | Conv3D `[Batch, Time, Height, Width, Channels]` |
| Linear `[Out_Features, In_Features]` | Dense kernel `[In_Features, Out_Features]` |

The converter keys this off each tensor's PyTorch state-dict name and rank
— it doesn't need per-model special-casing to know a `.weight` tensor is a
conv kernel vs. a linear kernel, since rank alone (5D vs. 2D) disambiguates
them. See [Hardware & Sharding](./hardware-and-sharding.md) for why
checkpoints are converted to plain numpy arrays (not `jnp.array`s) at this
stage, rather than JAX arrays — it's what keeps loading a 5B+ param tree
from silently OOM-ing onto a single device before sharding ever runs.

## Verification: exact 1:1 parameter-tree matches

Every translated checkpoint is checked against the model's own initialized
Flax parameter tree — every key present, every shape matching, before any
forward pass is trusted. This has caught real bugs on its own (a swapped
mask-channel concatenation order, a missing DiT-internal timestep rescale)
and is the first verification step for every model in vidax:

| Model | Component | Match |
| --- | --- | --- |
| Cosmos-Predict2.5 | DiT | 569/569 keys |
| Cosmos-Predict2.5 | Reason1 text encoder | 338/338 keys |
| Cosmos3 (Nano/Edge) | DiT | 542/542 tensors, byte-exact against raw `.safetensors` |
| HunyuanVideo-1.5 | DiT / VAE / byT5 / SigLIP | 8.33B / 1.26B / 219M / 413M params, exact shape match |
| HunyuanVideo-1.5 | Qwen2.5-VL MLLM | translated by Cosmos's `map_reason1_text_encoder_keys` unchanged (7.07B params) |
| LTX-2.5 | DiT + connector / VAE / Gemma-4 | every hyperparameter read from the checkpoint's own embedded metadata |

An exact key/shape match is necessary but not sufficient — it confirms the
translator found a home for every weight, not that the *semantics* of the
mapping are right (see the blog's
[Engineering Notes](/blog/tags/engineering-notes) for real bugs that passed
this check and were only caught by comparing generated output against the
reference).

:::info Source
This page is adapted from `vidax/README.md`, the `vidax.translator` module,
and the verification numbers reported in each model's own guide. `vidax`
doesn't yet have a dedicated `docs/weight_bridge.md` with a worked
key-mapping example — this page will be expanded if/when one lands.
:::

---

See [Hardware & Sharding](./hardware-and-sharding.md) for what happens to a
checkpoint after translation (sharding, dtype casting), or each
[Model Family Guide](../models/wan2_1.md) for per-model checkpoint sources
and file layouts.
