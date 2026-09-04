---
slug: weight-offloading
title: "Weight offloading: trading bandwidth for memory"
authors: [vidax-team]
tags: [engineering-notes, infrastructure]
date: 2026-09-04
description: >-
  What weight offloading is, why it exists (ZeRO-Offload), how vidax streams
  DiT weights from host RAM into HBM one chunk at a time, and what it
  actually costs — across five different model families, including one
  case where the same tool turned out to be the wrong one.
---

Some of vidax's models don't fit fully on-chip at the resolutions their
reference implementations were trained at, even after sharding weights and
activations across every available chip
([Sharding, parallelism, and JIT on TPUs](/blog/hardware-and-sharding)).
Weight offloading is the technique that closes that last gap: instead of
sharding, don't keep the weights resident on the accelerator at all — leave
them in host RAM and stream them in just in time.

{/* truncate */}

For the practical reference (which flag, which chunk size, per-model
guidance), see the [Weight Offloading docs](/docs/sharding/weight-offloading).
This post covers the idea, the implementation, and the real cost of using
it.

## Background: what weight offloading is

An accelerator's HBM has to hold everything needed for the current
computation at once: weights, and everything the forward/backward pass
produces along the way (activations — for a diffusion transformer, mostly
the per-token modulation and attention tensors; the closest analogy for
readers coming from LLM serving is the KV cache, though diffusion models
don't have one). When the weights alone are a large fraction of that
budget, there's less room left for everything else.

Weight offloading trades HBM for host-to-device bandwidth: keep the full
weight tree in ordinary host (CPU) RAM — which is far more abundant and far
cheaper than HBM — and copy only the small slice currently needed onto the
accelerator, just before it's used, then let that HBM be reclaimed once
that slice's computation is done. It isn't a new idea:
[ZeRO-Offload](https://arxiv.org/abs/2101.06840) (and, closer to this exact
use case, HuggingFace `diffusers`'
[`enable_sequential_cpu_offload()`](https://huggingface.co/docs/diffusers/optimization/memory))
both do a version of the same trade. What makes it worth doing at all is
that host-to-device transfer bandwidth is normally much higher than the
compute time of the layer you're about to run — if the two overlap well,
streaming weights in "for free" while the previous layer computes costs
close to nothing.

## How it's built in vidax

Every DiT in vidax repeats one block shape `num_layers` times — the
architectural property (`nn.scan`-style layer stacking) that makes
transformers scale by depth in the first place. That means a single
compiled program for "run one block/chunk of blocks" is reusable, verbatim,
for every chunk in the model — the compiler never needs to know *which*
chunk of weights it's holding, only that they match the shape/dtype it was
compiled for. The loop then looks like a direct application of the "jit the
unit, not the loop" rule from the
[sharding/JIT post](/blog/hardware-and-sharding#jit-compilation-what-jaxjit-actually-does-to-a-loop):

```python
chunk_apply = jax.jit(apply_one_chunk, donate_argnums=(0,))

for i in range(num_layers // chunk_size):
    chunk_params = jax.device_put(host_params[i], chunk_sharding)  # stream in
    x = chunk_apply(chunk_params, x)                               # compute
    # chunk_params' HBM is reclaimed once the next device_put reuses it
    # (donate_argnums lets the next chunk overwrite the same physical buffer)
```

`chunk_apply` is compiled exactly once, since every chunk has an identical
signature. `donate_argnums` tells JAX it's safe to reuse `chunk_params`'s
physical HBM for the *next* chunk's `device_put`, so the resident footprint
stays at one chunk's worth throughout a whole forward pass, not one chunk's
worth accumulating on top of the last. `--offload_chunk_size` controls how
many layers get grouped into one such chunk — a knob between "smallest
possible HBM footprint" (`chunk_size=1`) and "fewest, largest transfers"
(`chunk_size=num_layers`, i.e. the whole model).

## What it costs

TPU matmuls accumulate in float32 internally regardless of input dtype, so
this technique's cost isn't numerical — it's a wall-clock race between
transfer time and compute time. On paper this should be close to free: a
14B-class model's weights, sharded across 4 chips at bf16, are only a few
GB per chip — well under a second to transfer at typical TPU host-to-device
bandwidth, against tens of seconds of compute per diffusion step.

**Measured, it wasn't free.** Splitting that one transfer into 40 separate
per-layer transfers (one per layer of a 40-layer DiT) means whether they
overlap with the *previous* layer's compute — rather than serializing with
it — now matters a great deal. On real hardware, offloading measured
roughly **5x slower per step** than keeping everything resident, at the
chunk size that uses the least memory. Larger chunks (more layers streamed
together per transfer, at the cost of a bigger resident buffer) close some
but not all of that gap:

| `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| ---: | ---: | ---: |
| 1 (smallest buffer) | 141.7 | 15.2 |
| 8 | 131.3 | 15.3 |
| 20 | 123.7 | 23.0 |
| 40 (whole model, largest buffer) | 111.3 | 26.1 |

**The takeaway: treat offloading as a correctness/memory-fit tool for
configs that don't fit any other way, not a free option to reach for by
default.** It's the last lever to pull, after sharding, not the first.

## The same idea, five different forms

### Wan2.1: fixing an OOM that wasn't where it looked

Wan2.1 14B needed offloading at native 720p not because the DiT's own
per-step compute didn't fit — a fully-sharded fp32 DiT comfortably fits
*while the sampling loop runs* — but because the DiT's weights were staying
HBM-resident for the entire script, competing with an unrelated phase's
memory: VAE decode right after sampling ends, VAE encode right before an
I2V sampling loop starts. Offloading fixes this by only ever making the
weights HBM-resident for the brief window each chunk actually needs them,
freeing that memory back up for the next phase.

### A14B (Wan2.2): composing with sequence parallelism

Wan2.2's larger A14B model computes its timestep modulation **per token**
instead of per sample, which makes that one tensor alone multiple GB at
high resolution — pure *activation* memory that offloading, which only
targets weight memory, can't shrink no matter the chunk size. Fitting A14B
at real resolutions needed offloading *and* sequence parallelism at once,
which meant extending offloading to compose with a sharded token sequence
rather than assuming a full replica.

That combination surfaced a genuinely interesting bug, worth sharing as an
example of how subtle a sharding bug can look: outputs came out
blocky and spatially discontinuous, reproducible even at tiny resolutions.
The cause traced back to how a *row-parallel* layer (one where each chip
holds a slice of the input dimension, computes a partial output, and sums
partial outputs across chips) handles its bias. Each chip's local matmul
call adds the (replicated) bias *before* the cross-chip sum runs — so
summing `N` chips' outputs summed `N` copies of the bias instead of one.
Invisible with an all-zero bias (every synthetic test used one), but a real
per-layer error with trained biases, compounding across 40 layers into a
large enough numerical drift to visibly corrupt output. The fix subtracts
the bias before the cross-chip sum and adds exactly one copy back after —
a pattern worth remembering any time a row-parallel layer has a bias term
at all.

### Cosmos-Predict2.5: the straightforward case

Cosmos-Predict2.5's 14B DiT needed the same technique to reach its
reference's full 93-frame default (previously reduced to 45 frames without
it) — a single DiT with no MoE-style expert switching to compose with,
so the implementation is close to the offloading loop shown above,
essentially unmodified. With offloading, the full 93 frames run
comfortably at 14.7GB peak HBM/chip.

### HunyuanVideo 1.0: two chunk pools, one model

HunyuanVideo 1.0's 13B DiT needs offloading to fit the reference's real
129-frame / 720p default in HBM at all — a plain fit-the-sampling-loop case
like Wan2.1 and Cosmos above. The wrinkle is architectural: this DiT is a
dual-stream / single-stream MMDiT, 20 "double-stream" blocks followed by 40
"single-stream" blocks, and the two block types have **different parameter
shapes**. One chunk pool can't stream both. So `--offload_dit_weights`
runs *two* independent chunk loops — `--offload_chunk_size_double` (default
20, i.e. all double blocks in one chunk) and `--offload_chunk_size_single`
(default 40) — each its own separately-compiled `chunk_apply`, streamed and
freed in sequence. It's the same "jit the unit, stream one chunk at a time"
machinery as every other model here, just instantiated once per block
family instead of once for the whole stack. The per-step cost is dominated
by this offloading tax, not the implementation — HunyuanVideo-1.5, the same
block family with *zero* offloading, is already the slowest non-offloaded
row in vidax's benchmark table.

### LTX-2.5: offloading for a reason no other model needed it

LTX-2.5's 22B DiT needed `--offload_dit_weights` to reach its reference
resolution too, but for a genuinely different reason than every case
above: **its weights were never the bottleneck.** Fully resident, they're
only ~6.6GB/chip — comfortable headroom. What actually didn't fit was
activation memory from tracing the entire 48-block forward pass as one
fused `jax.jit` program, which — confirmed by measurement, not assumed —
scaled almost linearly with block count instead of staying roughly
constant, meaning per-block intermediates weren't being freed across
blocks. Offloading fixes this as a side effect: splitting the block loop
into `--offload_chunk_size`-sized separately-compiled chunks bounds peak
activation memory to one chunk's worth, by construction, independent of
whether weight streaming itself was ever needed. `--offload_chunk_size 8`
(the largest divisor of 48 that fits) got the reference's full
1216×704×121 resolution running at both checkpoints.

**A useful contrast, though: LTX-2.5's second, optional VAE decoder — a
transformer that runs local windowed attention instead of an ordinary
conv decode — hit a real memory ceiling of its own that offloading
*didn't* fix.** Isolated single-block profiling suggested each block
needed only ~12GB, comfortably under budget, but the real multi-device
pipeline measured over 32GB for the same block — a genuine, still
unexplained gap between isolated and in-context profiling. Chunking
further didn't move the number at all. What did was reaching for a tool
this repo already had a working pattern for: Megatron-style tensor
parallelism across the same mesh the DiT and text encoder already use,
which dropped peak memory to a comfortable 14.76GB/chip. **The lesson
worth generalizing: not every "doesn't fit" problem is a streaming
problem.** Offloading trades HBM for bandwidth by keeping less resident at
once; sharding trades HBM for communication by splitting the work itself
across chips. When one doesn't move the number, it's worth checking
whether the other, already-built machinery applies instead of continuing
to tune the first.

---

See [Sharding, parallelism, and JIT on TPUs](/blog/hardware-and-sharding)
for the mesh/JIT background this builds on, and the
[Weight Offloading docs](/docs/sharding/weight-offloading) for the
practical, per-model reference.
