---
sidebar_position: 1
title: Introduction
---

# vidax

**vidax** is a lightweight JAX/Flax inference engine and PyTorch-to-JAX
weight translator for modern Video Diffusion Transformers (DiTs) and beyond.
Built for **Google Cloud TPUs (v4, v5e, v6e)**, it eliminates framework
overhead with clean, explicit PyTree architectures and native multi-chip
parallelism for eight model families spanning several genuinely different
architectures — cross-attention DiTs (**Wan 2.1 / 2.2**,
**Cosmos-Predict2.5**, **LTX-Video**, **LTX-2.5**), an omnimodal
Mixture-of-Transformers (**Cosmos 3**), dual-stream/single-stream MMDiTs
(**HunyuanVideo 1.0** and **HunyuanVideo-1.5**), and a single
joint-attention DiT (**CogVideoX / CogVideoX1.5**).

:::info Source of truth
This site's docs are adapted from [`vidax/README.md`](https://github.com/FlyingGiraffe/vidax/blob/main/README.md)
and every file under [`vidax/docs/`](https://github.com/FlyingGiraffe/vidax/tree/main/docs).
The `vidax` engine repo is the canonical source for all model/feature
claims; this site never edits that repo, only reads from it.
:::

## Key features

- **Native TPU performance** — `jax.sharding` device meshes and a real
  Pallas flash-attention kernel, not `jax.nn`'s materializing default.
- **Universal weight translator** — loads PyTorch `.safetensors`/`.pth`
  checkpoints straight into Flax pytrees, verified against every model via
  exact 1:1 parameter-tree matches. See
  [Loading PyTorch Weights](./sharding/loading-pytorch-weights.md).
- **Two parallelism strategies** — Megatron-style tensor parallelism and
  DeepSpeed-Ulysses sequence parallelism, composable on the same mesh,
  picked per model/resolution depending on whether weight or activation
  memory is the bottleneck. See
  [Hardware & Sharding](./sharding/hardware-and-sharding.md).
- **Per-layer weight offloading** — `--offload_dit_weights` keeps a DiT's
  weights host-resident and streams one block-group at a time into HBM,
  extending every model's reach to resolutions/frame counts that don't fit
  fully device-resident. See [Weight Offloading](./sharding/weight-offloading.md).
- **Flow matching engine** — a Rectified Flow Euler sampler and a
  from-scratch UniPC multistep predictor-corrector port, covering every
  supported model's native scheduler (including Cosmos 3's Karras-sigma
  variant).
- **Faithful image/video conditioning** — each model family's own
  conditioning mechanism ported exactly, not approximated — CLIP
  cross-attention, per-token/per-frame latent substitution, and
  conditioning-mask channels all show up where the reference actually uses
  them.
- **Broad, growing model coverage** — cross-attention DiTs (Wan 2.1/2.2,
  Cosmos-Predict2.5, LTX-Video, LTX-2.5), Cosmos 3's architecturally
  distinct omnimodal Mixture-of-Transformers, the HunyuanVideo 1.0 / 1.5
  dual-stream/single-stream MMDiTs, and CogVideoX's single joint-attention
  DiT.
- **Reusable building blocks** — the flash-attention kernel, schedulers,
  and PyTorch→JAX translator are usable standalone. See the
  [API Reference](./api/index.md).

## Model support

Rows are merged across tasks when one checkpoint handles all of them (e.g.
Wan2.2 TI2V-5B, Cosmos-Predict2.5); kept separate when the reference ships
genuinely distinct checkpoints (e.g. Wan2.1's T2V vs. I2V, Wan2.2's A14B).

| Model Family | Variant | Task | Guide |
| --- | --- | --- | --- |
| Wan2.1 | 1.3B / 14B | T2V | [wan2_1](./models/wan2_1.md) |
| Wan2.1 | 14B (480P/720P) | I2V | [wan2_1](./models/wan2_1.md) |
| Wan2.2 | A14B (MoE) | T2V / I2V | [wan2_2](./models/wan2_2.md) |
| Wan2.2 | TI2V-5B | T2V / I2V | [wan2_2](./models/wan2_2.md) |
| Cosmos-Predict2.5 | 2B / 14B | T2V / I2V / V2V | [cosmos2_5](./models/cosmos2_5.md) |
| Cosmos3 | Nano (16B) / Edge (4B) | T2V / I2V | [cosmos3](./models/cosmos3.md) |
| LTX-Video (0.9.8) | 2B / 13B (dev/distilled) | T2V / I2V | [ltx_video](./models/ltx_video.md) |
| LTX-2.5 | 22B (dev/distilled) | T2V / I2V | [ltx2_5](./models/ltx2_5.md) |
| HunyuanVideo-1.5 | 8.3B (480p/720p) | T2V / I2V | [hunyuan_video_1_5](./models/hunyuan_video_1_5.md) |
| HunyuanVideo (1.0) | 13B | T2V / I2V | [hunyuan_video](./models/hunyuan_video.md) |
| CogVideoX / CogVideoX1.5 | 2B / 5B | T2V / I2V | [cogvideox](./models/cogvideox.md) |

See the
[main vidax README](https://github.com/FlyingGiraffe/vidax#-model-support)
for the up-to-date table, and the [Benchmark Explorer](/benchmarks) for
measured latency/memory numbers across every implemented row above.

Next: [Quickstart](./quickstart.md).
