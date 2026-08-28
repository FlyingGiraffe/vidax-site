---
sidebar_position: 2
title: Hardware & Sharding
---

# Hardware & Sharding

Practical reference for configuring vidax's parallelism flags
(`--tensor_parallel_size`, `--sequence_parallel_size`) and understanding
TPU topology conventions.

:::info Looking for the background?
This page is deliberately just the actionable rules. For the "why" —
what tensor and sequence parallelism actually do, and how `jax.jit`
compiles a loop (including the VAE-decode incident this taught us) — see
[Sharding, parallelism, and JIT compilation on TPUs, with JAX](/blog/hardware-and-sharding)
on the blog.
:::

## TPU topology

:::warning `vN-M` names TensorCores, not chips
A v4 chip has 2 TensorCores, so `v4-8` is a 4-**chip** slice
(`jax.device_count() == 4`), not 8 — vidax's benchmark machine is a v4-8.
`--tensor_parallel_size`/`--sequence_parallel_size` values throughout this
site are chip counts, so "full width on a v4-8" means **4**, not 8. v5e and
v6e chips have 1 TensorCore each, so `vN-M` there *does* equal chip count —
don't assume the ÷2 rule carries over once those are benchmarked.
:::

## Tensor parallelism (Megatron-style)

`--tensor_parallel_size` shards attention heads and FFN channels across
devices, Megatron-1D style — this is what lets full-resolution DiT
self-attention (tens of thousands of patches) fit in HBM at all.

- Must divide both `num_devices` and the model's `num_heads` (12 for
  Wan2.1's 1.3B DiT, 40 for its 14B DiT, 64 for T5, 24 for Wan2.2's 5B DiT,
  16/40 for Cosmos-Predict2.5's 2B/14B, 32-or-16 with an 8-head GQA
  constraint for Cosmos3 Nano/Edge — see each
  [model guide](../models/wan2_1.md) for its exact divisibility rule).
- Shards weight memory, not activation memory — doesn't help once
  per-token activation memory (not weights) is the binding constraint (see
  sequence parallelism below).
- `4` (full width on a v4-8) is a reasonable starting point for most
  14B-class models; raise chip count if available and still HBM-bound.

## Sequence parallelism (DeepSpeed-Ulysses)

`--sequence_parallel_size` shards the **token sequence itself** across
devices between blocks, instead of sharding weights — the
[DeepSpeed-Ulysses scheme](https://arxiv.org/abs/2309.14509). Reach for
this when a model's per-token activation memory (not its weights) is what
doesn't fit — e.g. Wan2.2's per-token AdaLN modulation at high token
counts.

`--tensor_parallel_size` and `--sequence_parallel_size` compose freely on
the same mesh — their **product** is the real divisibility constraint
against `num_heads`, not either one individually. `--sequence_parallel_size 1`
(the default) is a complete no-op.

:::note i2v + TP + SP isn't supported together for Wan2.1
Wan2.1's CLIP image-cross-attention branch isn't threaded through
Megatron's column-parallel sharding — combining `--tensor_parallel_size > 1`
with `--sequence_parallel_size > 1` for i2v raises `NotImplementedError`
rather than silently computing something wrong. Use one scheme or the
other for i2v with CLIP conditioning.
:::

## Choosing a config, per model family

See each model's own guide for verified, working combinations:
[Wan2.1](../models/wan2_1.md), [Wan2.2](../models/wan2_2.md)
(A14B in particular composes both axes),
[Cosmos-Predict2.5](../models/cosmos2_5.md), [Cosmos3](../models/cosmos3.md).
The [Benchmark Explorer](/benchmarks) records the exact TP/SP split used
for every measured row.

## Flash attention requires a mesh

`vidax.core.attention.dot_product_attention` dispatches to a Pallas
flash-attention kernel on TPU whenever no `bias`/`mask` is given. Any
multi-device call **must pass a `mesh` argument**, or it silently falls
back to the correct-but-much-slower XLA path that fully materializes the
attention logits matrix — every model script in vidax already does this
for you, but worth knowing if you're extending the code.

## JIT-safety rules of thumb

- Keep sequence lengths, frame counts, and spatial dimensions static or
  explicitly padded.
- Don't `jax.jit` a Python loop over many repeated forward passes
  (sampling steps, VAE decode chunks) — jit only the per-iteration
  function and call it from a plain Python loop.

---

See [Weight Offloading](./weight-offloading.md) for extending memory reach
beyond what sharding alone provides, or
[Loading PyTorch Weights](./loading-pytorch-weights.md) for the
checkpoint translation layer this all builds on.
