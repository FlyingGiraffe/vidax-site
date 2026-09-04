---
slug: hardware-and-sharding
title: "Sharding, parallelism, and JIT compilation on TPUs, with JAX"
authors: [vidax-team]
tags: [engineering-notes, infrastructure]
date: 2026-09-04
description: >-
  A gentle background on Megatron-style tensor parallelism, DeepSpeed-Ulysses
  sequence parallelism, and how jax.jit really compiles a loop — illustrated
  with the "VAE decode hang" that turned out not to be a hang at all.
---

Video DiTs are big in two different ways at once: the weights can be too
large for one chip, and — because self-attention runs over every patch in
every frame — a single forward pass's activations can be too large for one
chip even when the weights fit fine. This post is background on the two
sharding strategies vidax uses to deal with that (tensor parallelism,
sequence parallelism), and on a JAX-specific gotcha around `jax.jit` and
Python loops that's caused more real bugs, in more of vidax's models, than
anything else in this codebase.

{/* truncate */}

For the practical, day-to-day reference (which flags to pass, divisibility
rules, when to reach for which scheme), see the
[Hardware & Sharding docs](/docs/sharding/hardware-and-sharding). This post
is the "why" and the underlying mental model.

## Why shard a video DiT at all

A text-to-video DiT patchifies a video into a sequence of tokens — one per
small spatio-temporal patch — and runs ordinary transformer self-attention
over that whole sequence. For a short image, that sequence might be a few
hundred tokens. For a few seconds of video at real resolution, it's tens of
thousands. Self-attention costs `O(sequence_length² × num_heads)`, so both
the weights *and* the activations of a video DiT can each, independently,
be too big for a single TPU chip's HBM (high-bandwidth memory — the
on-chip memory an accelerator computes against, analogous to a GPU's VRAM).
Two different sharding strategies fix two different halves of that
problem.

## Tensor parallelism (Megatron-style): shard the weights

The idea: split each layer's weight matrix across `N` chips along its
output dimension, so each chip only ever holds and computes with `1/N` of
the parameters. A minimal example, sharding a single dense layer's kernel
across a device mesh axis called `"tp"`:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices(), axis_names=("tp",))

# Split the *output* features across chips: each chip holds
# (in_features, out_features / tp_size) of the kernel.
kernel_sharding = NamedSharding(mesh, P(None, "tp"))
kernel = jax.device_put(kernel, kernel_sharding)

def dense(x, kernel):
    return x @ kernel  # each chip computes its own output slice

y = jax.jit(dense)(x, kernel)  # GSPMD inserts any needed communication
```

This is "column-parallel": each chip computes a different slice of the
output, and no communication is needed until something needs the full
result back (a "row-parallel" layer right after it, which shards its
*input* dimension instead and sums the partial results with an
all-reduce). Chaining column-parallel → row-parallel layers, which is
exactly what an attention block's QKV projection → output projection and
an MLP's up-projection → down-projection look like, means the
all-reduce only has to happen once per block, not once per layer. This is
the [Megatron-LM](https://arxiv.org/abs/1909.08053) scheme, and it's what
`vidax.core.sharding.build_tpu_mesh`/`shard_wan_params` set up for every
DiT and text encoder in vidax.

Tensor parallelism shards weights (and whatever activations are direct
projections of them), but every chip still holds the **full token
sequence**. That's fine until the sequence itself — not the weights — is
what doesn't fit.

## Sequence parallelism (DeepSpeed-Ulysses): shard the tokens

The idea here is the opposite: keep the full weights on every chip, but
give each chip only `1/N` of the *sequence*. The tricky part is
self-attention, which needs every token to see every other token — you
can't just compute attention locally on each chip's slice.
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)'s answer is to
reshuffle data instead of tokens or weights: before attention, an
`all_to_all` redistributes so each chip ends up holding *every* token but
only *some* of the attention heads; attention runs locally per-head; a
second `all_to_all` reshuffles back to "every chip, its own token slice,
every head" for the rest of the block. Sketched out:

```python
# Before: each chip holds (local_seq_chunk, all_heads)
# After all_to_all: each chip holds (all_seq, local_head_chunk)
kv_by_head = jax.lax.all_to_all(kv_by_seq, axis_name="sp",
                                 split_axis=0, concat_axis=1)
out_by_head = local_attention(kv_by_head)  # full sequence, few heads
out_by_seq = jax.lax.all_to_all(out_by_head, axis_name="sp",
                                 split_axis=1, concat_axis=0)
```

No chip ever recomputes another chip's tokens — it's a pure data
reshuffle around one attention op, with ordinary per-chip compute for
everything else (FFN, norms, modulation) running on that chip's own
sequence slice the whole time. This is what actually shrinks the
per-token activation memory that tensor parallelism can't touch.

The two compose: `build_tpu_mesh` builds a 3-axis `(dp, tp, sp)` mesh, and
vidax's larger models (Wan2.2's A14B) need both at once, since one axis
alone doesn't shard enough of *either* weights or activations to fit.

CogVideoX-1.5 is a second, trickier user of sequence parallelism. Its
block has no separate cross-attention: text and video tokens sit in **one
joint self-attention sequence**, so you can't just `all_to_all` the whole
thing — the 226 replicated text tokens would be reshuffled too. The port
(`sequence_parallel_joint_self_attention`) instead sends only the visual
q/k/v through the head↔sequence all-to-all and slices the replicated text
q/k/v down to each device's local head range before one local attention
call. That's what runs CogVideoX-1.5 at its native 1360×768 (~45k visual
tokens), whose per-block activations don't fit a v4 chip otherwise — and
for CogVideoX the two schemes are mutually exclusive, because the 5B DiT's
weights fit replicated per chip and the port never threads weight-sharding
through the sequence-parallel path.

## JIT compilation: what `jax.jit` actually does to a loop

`jax.jit` doesn't compile your Python function — it **traces** it once
(running it with abstract placeholder values to record what array
operations happen) and compiles *that trace* into a single fused program.
This matters a lot for any function containing a Python `for` loop: the
loop itself runs at trace time, in Python, so a 20-iteration loop doesn't
produce a 20-iteration compiled program — it produces one compiled program
containing 20 iterations' worth of operations, fully unrolled, all of
whose intermediate results the compiler may need to keep alive
simultaneously.

### The VAE decode "hang" that wasn't a hang

Video VAE decoders in vidax decode a video's latent frames in chunks (a
handful of frames at a time), in a loop, because decoding the whole video
in one shot doesn't fit in memory. The very first version of this loop
looked entirely reasonable:

```python
@jax.jit
def decode_video(vae_params, latent_chunks):
    frames = []
    for chunk in latent_chunks:          # traced, i.e. unrolled!
        frames.append(vae.apply(vae_params, chunk))
    return jnp.concatenate(frames, axis=1)
```

Run against a real ~20-chunk video, this appeared to hang for **45+
minutes**. It wasn't a deadlock — attaching `py-spy` to the running process
showed it stuck deep inside XLA's compiler. `@jax.jit` on the *outer*
function traces straight through the Python loop, so XLA was being asked
to compile one enormous program containing 20 chunks' worth of decoder
operations back to back, with no ability to free one chunk's intermediates
before the next chunk's began — a compile-time and memory problem
disguised as a runtime one.

The fix is to jit only the per-chunk function, and call it from an
ordinary (untraced) Python loop:

```python
decode_chunk = jax.jit(vae.apply)          # compiled once, for one shape

frames = []
for chunk in latent_chunks:                # a normal Python loop
    frames.append(decode_chunk(vae_params, chunk))
frames = jnp.concatenate(frames, axis=1)
```

Every chunk has the same shape, so `decode_chunk` is compiled exactly
once — 20 calls to an already-compiled program, not one enormous
from-scratch compilation. Just as importantly, each iteration's
intermediate buffers can now be freed once that iteration's call returns,
instead of all needing to coexist in one giant program's memory footprint.
The rule of thumb this generalizes to, used everywhere in vidax that has a
"do the same thing N times" loop (diffusion sampling steps,
[per-layer weight offloading](/blog/weight-offloading)'s chunk loop): **jit
the repeated unit of work, not the loop around it.**

### A cross-model theme

This isn't a Wan-specific fix — it's a pattern, and it keeps showing up
wherever a model has some form of chunked, repeated decode step.

LTX-2.5's optional "diffusion" VAE decoder (a transformer that runs local,
windowed attention over small 3D neighborhoods, rather than a single
deterministic conv decode) hit the exact same shape of problem, one level
deeper: a naive version that gathered every attention window at once
before running any math didn't just compile slowly, it OOM'd outright —
147GB of intermediate memory for a *tiny* test shape, since the local
window's volume multiplies memory by the product of all three of its
axes. Processing one slice of the window at a time in an ordinary Python
loop fixed the OOM but reintroduced the *other* half of this post's
lesson: at real resolution, the loop unrolled into a genuine 45+
minute-class compile, once per block, 24 blocks deep. The fix was
`jax.lax.scan` — JAX's built-in "repeat this step N times" primitive,
which (unlike a Python `for` loop inside `jax.jit`) compiles the
per-step body exactly once regardless of how many steps run, closing the
compile-time problem the same way jitting only the per-chunk function did
above. What remained after that was an ordinary memory-fit problem, closed
with the same tensor-parallel sharding already built for every other
model in vidax.

HunyuanVideo-1.5's VAE decoder ran into a version of this too — its
decode OOM'd on a full-resolution run with peak memory that didn't add up
to any single layer's own compute, the same "one fused trace doesn't free
each stage's intermediates before the next starts" signature. The fix that
landed is the same shape as all the others: on top of the spatial tiling
the reference VAE already does (latent-space H/W tiles with a linear
cross-fade of the overlaps), each tile is now decoded through a
*per-decoder-block staged pipeline* — `decode_stage_level_block` /
`decode_stage_level_upsample`, each its own `jax.jit` call — so one
stage's temporaries are freed before the next stage runs, instead of the
whole decode being one fused trace. That got the reference's real
121-frame/480p default decoding within budget.

## Summary

| Question | Tool |
| --- | --- |
| Weights too big for one chip? | Tensor parallelism (shard weights, replicate sequence) |
| Activations too big for one chip, weights fit fine? | Sequence parallelism (shard sequence, replicate weights) |
| Both? | Compose both — separate mesh axes |
| A repeated per-chunk/per-step operation inside a loop? | `jax.jit` the per-iteration function, call it from a plain Python loop — never `jax.jit` the loop itself |

---

See the [Hardware & Sharding docs](/docs/sharding/hardware-and-sharding) for
the practical, flag-by-flag reference, and
[Weight Offloading](/blog/weight-offloading) for the next technique built on
top of this same "jit the unit, loop in Python" idea.
