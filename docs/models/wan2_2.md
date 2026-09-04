---
sidebar_position: 2
title: Wan2.2
---

# Wan2.2

Three standalone TPU inference scripts live in `examples/`. They share the
same building blocks (`vidax.core`, `vidax.schedulers`, `vidax.translator`)
but differ in checkpoint format, resolution defaults, and parallelism
strategy — see [Hardware & Sharding](../sharding/hardware-and-sharding.md)
for the engineering reasoning (Megatron vs. sequence parallelism, flash
attention, JIT safety, the dtype-casting/decode-speed bugs found getting
Wan2.2 working).

| Script | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- |
| `generate_wan2_2_ti2v.py` | 5B | Text-to-Video **and** Image-to-Video | `Wan2.2-TI2V-5B` |
| `generate_wan2_2_t2v_a14b.py` | 14B (MoE, two experts) | Text-to-Video | `Wan2.2-T2V-A14B` |
| `generate_wan2_2_i2v_a14b.py` | 14B (MoE, two experts) | Image-to-Video | `Wan2.2-I2V-A14B` |

All three build `vidax.models.wan.wan2_2.dit.WanDiT` (fully config-driven,
per-token AdaLN modulation) from a named preset
(`TI2V_5B_CONFIG`/`T2V_A14B_CONFIG`/`I2V_A14B_CONFIG`) — the architecture is
identical across all three; only size and (for I2V) `in_dim` differ.
Everything they need (`torch`, `transformers`) is installed by default; on a
Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

## TI2V (5B) — `generate_wan2_2_ti2v.py`

TI2V-5B is a single checkpoint supporting **both** text-to-video and
image-conditioned generation: pass `--image_path` for i2v, omit it for t2v.
Image conditioning works by substituting the known conditioning frame's
latent back into `x` between sampling steps (a per-token timestep of 0 for
that frame's tokens, re-applied after every step) — not an extra model
input, unlike A14B's i2v.

Uses `Wan2.2_VAE.pth` — a different file *and architecture* from Wan2.1's
`Wan2.1_VAE.pth` (48-channel latent space, 2×2 pixel-patchify wrapping, 16x
spatial / 4x temporal compression).

### Text-to-video

```bash
python examples/generate_wan2_2_ti2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ti2v.mp4"
```

### Image-to-video

```bash
python examples/generate_wan2_2_ti2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
  --image_path "./checkpoints/Wan2.2-TI2V-5B/examples/i2v_input.JPG" \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --tensor_parallel_size 4 \
  --output_path "out/output_ti2v_i2v.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | `.safetensors.index.json` manifest — the 5B DiT ships sharded; a single non-sharded `.safetensors` also works. |
| `--vae_checkpoint_path` | *required* | `Wan2.2_VAE.pth`. |
| `--t5_checkpoint_path` | *required* | Same file format/filename as Wan2.1 — the text encoder is byte-identical across versions. |
| `--image_path` | `None` | Omit for t2v. When given, output resolution is derived from the image's aspect ratio + `--max_area` (`--height`/`--width` ignored). |
| `--max_area` | `704*1280` | i2v only: target output pixel area. |
| `--prompt` | *required*, 1+ values | Same broadcast semantics as Wan2.1 t2v. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--tensor_parallel_size` | `1` | See the parallelism note below — means something different here than in the Wan2.1 scripts. |
| `--dtype` | `bfloat16` | Same choices/caveats as Wan2.1. |
| `--num_steps` | reference per-mode default | `None` resolves to `50` for t2v, `40` for i2v. |
| `--shift` | `5.0` | Reference default for TI2V-5B. |
| `--height` / `--width` | `704` / `1280` | Ignored if `--image_path` is given. Must be a multiple of 16. |
| `--num_frames` | `121` | Reference default for TI2V-5B (vs. 81 for Wan2.1/A14B). |
| `--fps` | `24` | vs. 16 for Wan2.1/A14B. |

:::info Parallelism note
At TI2V-5B's only supported resolution (704×1280, 121 frames), the
patch-token sequence is ~27k long, and Wan2.2's per-token AdaLN modulation
scales with that directly. Which mesh axis actually matters depends on
`--dit_dtype`: with the correct `float32` default, the ~5B DiT's fp32
weights (~20GB) dominate this 4-chip machine's HBM more than per-token
activation memory does, so **`--tensor_parallel_size 4
--sequence_parallel_size 1`** (full weight-sharding, no sequence-parallel)
is what actually fits end-to-end — confirmed via direct probing (`sp=4/tp=1`
OOMs during T5 encoding before the DiT even runs; `tp=2/sp=2` gets past T5
but OOMs inside the DiT sampling step). See
[Hardware & Sharding](../sharding/hardware-and-sharding.md) for why.
:::

**Status:** verified end-to-end for **both** t2v and i2v, at the full
reference resolution/frame count (704×1280, 121 frames) and full step count
(50 t2v / 40 i2v), 5 full benchmark runs each, no OOM, ~18.3GB peak
HBM/chip — see the [Benchmark Explorer](/benchmarks).

## T2V (A14B) — `generate_wan2_2_t2v_a14b.py`

A14B is a **Mixture-of-Experts** model: two separately-checkpointed 14B
DiTs (`high_noise_model`/`low_noise_model`), each the same `WanDiT`
architecture, switched per sampling step by comparing that step's timestep
against `--boundary * num_train_timesteps` — `high_noise_model` handles the
noisier early steps, `low_noise_model` the later ones. This is a plain
Python-level choice of which params pytree feeds the same jitted
`single_step` each iteration, not a traced/data-dependent branch.

Only **one expert is ever device-resident at a time** — both experts' cast
weights stay in host RAM; the currently-needed one is `device_put` onto the
mesh only when the schedule crosses the boundary, then dropped before the
other is placed. Keeping both experts resident at once (the first thing
tried) reliably ran out of HBM at `--tensor_parallel_size 4`, since a
single sharded 14B expert already leaves little headroom and Wan2.2's
per-token modulation means the forward pass's own working-set scales with
the full token count under Megatron TP alone.

`--tensor_parallel_size` and `--sequence_parallel_size` compose freely — the
currently-resident expert can have both its weights *and* the token
sequence sharded at once, e.g. `--tensor_parallel_size 2
--sequence_parallel_size 2`. This is what lets A14B run at a noticeably
larger resolution than either trick alone on 4 chips, though still short of
the reference's full 1280×720×81.

Unlike TI2V-5B, A14B reuses **Wan2.1's causal VAE** (`Wan2.1_VAE.pth`) and
Wan2.1's default resolution/frame count (1280×720, 81 frames).

```bash
python examples/generate_wan2_2_t2v_a14b.py \
  --high_noise_dit_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --low_noise_dit_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --output_path "out/output_t2v_a14b.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--high_noise_dit_checkpoint_path` / `--low_noise_dit_checkpoint_path` | *required* | The two experts' DiT checkpoints (or sharded manifests). |
| `--vae_checkpoint_path` | *required* | `Wan2.1_VAE.pth` (bundled in the A14B repo). |
| `--boundary` | `0.875` | Fraction of `num_train_timesteps` above which `high_noise_model` is used. |
| `--tensor_parallel_size` | `1` | Only one expert is ever device-resident at a time, so this behaves like sharding a single 14B model. Composes with `--sequence_parallel_size` — their product is the real head-divisibility constraint (40 per expert, 64 for T5). |
| `--sequence_parallel_size` | `1` | Worth trying together with `--tensor_parallel_size` at resolutions where even one device-resident expert doesn't fit alone. |
| `--shift` | `12.0` | Reference default for A14B T2V. |
| `--height` / `--width` | `720` / `1280` | At native resolution, the full reference frame count doesn't fit this 4-chip machine even with offloading + sequence parallelism — reduce `--num_frames`. |
| `--num_frames` | `81` | Reduce to `33` at native 720×1280 (the largest that fits — see [Weight Offloading](/blog/weight-offloading)). |
| `--offload_dit_weights` / `--offload_chunk_size` | off / `1` | Composed with the two-expert switch **and** with `--sequence_parallel_size > 1`. At native 720×1280, only `1` fits; at 480×832, `10` fits. |

**Status:** verified end-to-end against real checkpoints (both experts,
weight shapes/keys confirmed to exactly match `T2V_A14B_CONFIG`). Full
native resolution (1280×720) is reachable with `--tensor_parallel_size 2
--sequence_parallel_size 2 --offload_dit_weights --offload_chunk_size 1`,
reduced to `--num_frames 33` — 5 full benchmark runs: 33.7s compile, 2321.9s
generation, 46.4s/step, 18.1GB peak HBM/chip. At 480P
(`--height 480 --width 832`), the same TP/SP config fits the full 81 frames
with `--offload_chunk_size 10` — 65.8s compile, 2159.1s generation,
43.2s/step, 28.4GB peak HBM/chip. See the [Benchmark Explorer](/benchmarks)
for both rows.

## I2V (A14B) — `generate_wan2_2_i2v_a14b.py`

Same two-expert MoE switching as T2V-A14B. Unlike Wan2.1's I2V-14B, A14B has
**no CLIP vision cross-attention branch** — image conditioning instead
concatenates a mask+VAE-latent `y` (built from the same Wan2.1 causal VAE)
directly onto the noisy latent's channel axis before the DiT call, which is
why `I2V_A14B_CONFIG` sets `in_dim=36` (16 noise channels + 20 conditioning
channels).

```bash
python examples/generate_wan2_2_i2v_a14b.py \
  --high_noise_dit_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --low_noise_dit_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --image_path "./checkpoints/Wan2.2-I2V-A14B/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --output_path "out/output_i2v_a14b.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--boundary` | `0.900` | Reference default for I2V (vs. 0.875 for T2V). |
| `--shift` | `5.0` | Reference default for A14B I2V (vs. 12.0 for T2V). |
| `--max_area` | `720*1280` | Same `compute_latent_grid` as Wan2.1 I2V. At 480P the full 81 frames fits; at native 720P, only a reduced 33 frames does. |
| `--num_frames` | `81` | At 480P, the full 81 fits with `--offload_dit_weights --offload_chunk_size 10 --tensor_parallel_size 2 --sequence_parallel_size 2`. At native 720P, reduce to `33`. |
| `--offload_chunk_size` | `1` | Swept at 480P/81 frames: `1`, `2`, `4`, `10` all fit, `20` OOMs — `10` is the largest that fits. At native 720P/33 frames, only `1` fits. |

**Status:** verified end-to-end against real checkpoints, both experts
engaging correctly. Two full-scale configurations measured: **480P, full 81
frames** — 146.1s compile, 1780.0s generation, 44.5s/step, 28.3GB peak
HBM/chip; and **native 720P, reduced to 33 frames** — 102.9s compile,
1962.5s generation, 49.1s/step, 20.5GB peak HBM/chip. Despite the smaller
chunk size, native 720P's peak HBM is *lower* than 480P's — chunk size, not
resolution, is the dominant HBM cost at this scale (see
[Weight Offloading](/blog/weight-offloading)). A
pre-existing row-parallel bias-double-counting bug (affecting Wan2.1 too)
was found and fixed while verifying `--sequence_parallel_size 2` here — see
that same page for the story.

---

See [Wan2.1](./wan2_1.md), [Cosmos-Predict2.5](./cosmos2_5.md), and
[Cosmos3](./cosmos3.md) for the other model families, or the
[Benchmark Explorer](/benchmarks) for measured numbers across every row
above.
