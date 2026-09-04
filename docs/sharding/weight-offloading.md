---
sidebar_position: 3
title: Weight Offloading
---

# Weight Offloading

Practical reference for `--offload_dit_weights`/`--offload_chunk_size` —
per-layer weight offloading, for configs that don't fit fully
device-resident even with tensor/sequence parallelism.

## What it does

Keeps a DiT's weight tree in **host RAM** instead of HBM, streaming
`--offload_chunk_size` consecutive blocks' worth into a small fixed-shape
HBM buffer at a time during the sampling loop — the same idea as
DeepSpeed's ZeRO-Offload or `diffusers`' `enable_sequential_cpu_offload()`,
applied per-layer. Implemented for Wan2.1's `WanDiT`, Wan2.2 A14B's
`WanDiT`, Cosmos-Predict2.5's `CosmosDiT`, LTX-2.5's 22B DiT, and
HunyuanVideo 1.0's dual-stream / single-stream DiT (two independent chunk
pools, `--offload_chunk_size_double` / `--offload_chunk_size_single`, since
the two block types have different parameter shapes).

:::warning Not a free option — real throughput cost
Measured on Wan2.1 14B T2V at native 720P: **130.0s/step offloaded vs.
26.1s/step non-offloaded at 480P.** Treat this as a correctness/memory-fit
tool for configs that don't fit any other way, not something to enable by
default.
:::

## Chunk size: HBM vs. throughput tradeoff

`--offload_chunk_size N` groups `N` consecutive blocks per offloaded HBM
buffer/compile (must divide the model's layer count). Larger chunks are
somewhat faster but use more HBM. Measured on Wan2.1 14B T2V at native
720P:

| `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| ---: | ---: | ---: |
| 1 | 141.7 | 15.2 |
| 8 | 131.3 | 15.3 |
| 20 | 123.7 | 23.0 |
| 40 (whole model) | 111.3 | 26.1 |

`--offload_chunk_size 1` is the safer default for a first attempt on
unfamiliar hardware or when combining with anything else that needs HBM
headroom. Screen candidate chunk sizes cheaply with
`benchmarks/sweep_offload_chunks.py --num_runs 1 --num_steps 5` before
committing to a full run.

## Per-model guidance

| Model | When needed | Composes with SP? |
| --- | --- | --- |
| Wan2.1 (T2V/I2V) | Native 720P only — not needed at 480P | No |
| Wan2.2 A14B | All native-720P and most 480P configs | Yes — required at native 720P |
| Cosmos-Predict2.5 14B | To reach the full 93-frame reference default | Yes |
| LTX-2.5 22B | Always, at the reference resolution — but to bound *activation* memory (per-block temporaries weren't freed across the fused 48-block trace), not weight residency | No (composes with TP) |
| HunyuanVideo 1.0 13B | Always, to fit the reference's 129-frame / 720p default in HBM at all | No (composes with TP) |

A14B specifically needs offloading composed with
`--sequence_parallel_size` because its AdaLN modulation is per-*token*, not
per-sample — offloading alone reduces weight residency but can't shrink
that per-token activation memory. LTX-2.5 is the odd case: its weights fit
comfortably (~6.6 GB/chip at `tp=4`), and offloading is used only for its
side effect of splitting the block loop into separately-compiled chunks
that bound peak activation memory. See each
[model guide](../models/wan2_1.md) for verified working flag combinations
and measured numbers, and the [Benchmark Explorer](/benchmarks) for every
row.

## Correctness note

Offloaded output is not bit-identical to non-offloaded output (each block
compiles as a separate `jax.jit` program, so XLA makes slightly different
fusion/precision choices — a ~1-3% difference in output magnitude per
forward pass), but this has been verified to not be corruption: decoded
video stays visually and statistically coherent (matching frame mean/std)
in every comparison against the non-offloaded reference.

---

See [Hardware & Sharding](./hardware-and-sharding.md) for the parallelism
flags this composes with.
