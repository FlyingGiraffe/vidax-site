---
sidebar_position: 6
title: LTX-2.5
---

# LTX-2.5

:::info Architecturally unrelated to LTX-Video
Despite the shared "LTX" name and causal-conv VAE family, LTX-2.5 is a
much larger 22B DiT with cross-attention AdaLN and gated attention, an
8-layer "embeddings connector" bridging a Gemma-4 12B text encoder (not
T5) into the DiT, and an ancestral (SDE) Euler sampler instead of a plain
deterministic step. See [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
for the comparison.
:::

One standalone TPU inference script, `generate_ltx2_5.py`, covers both T2V
and I2V for the two released 22B checkpoints (dev, distilled). Scope is
deliberately narrower than LTX-2.5's full reference: video-only (no audio
generation), single-stage (no `LatentUpsampler` two-pass refinement).

| Script | Model | Params | Task | Checkpoint file example |
| --- | --- | --- | --- | --- |
| `generate_ltx2_5.py` | LTX-2.5 | 22B (dev) | T2V, I2V | `ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| `generate_ltx2_5.py` | LTX-2.5 | 22B (distilled) | T2V, I2V | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` |

Requires the `torch`, `text`, and `i2v` extras:

```bash
pip install -e ".[tpu,torch,text,i2v]"
```

## Checkpoints

Three separate checkpoint files from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5):

- `--dit_checkpoint_path`: `diffusion_models/ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors` — bundles the DiT *and* the embeddings connector.
- `--vae_checkpoint_path`: `vae/ltx-2.5-video-vae-conv-bf16.safetensors` (default `--vae_variant conv`) or `vae/ltx-2.5-video-vae-bf16.safetensors` (`--vae_variant diffusion`, a transformer/neighborhood-attention decoder — the official demos' decoder).
- `--text_encoder_checkpoint_path`: `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` — bundles Gemma-4, its embedded tokenizer, and the feature-extraction projection.

### Text-to-video

```bash
python examples/generate_ltx2_5.py \
  --dit_checkpoint_path "./checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --vae_checkpoint_path "./checkpoints/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --text_encoder_checkpoint_path "./checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --sampler dev \
  --tensor_parallel_size 4 \
  --offload_dit_weights --offload_chunk_size 8 \
  --output_path "out/output_ltx2_5_t2v.mp4"
```

`--sampler distilled` (the default) runs 8-step ancestral-Euler sampling
with no CFG; `--sampler dev` runs 30-step plain-Euler with real CFG
(`--guidance_scale 3.0`) plus a guidance-rescale correction
(`--guidance_rescale 0.7`) that avoids CFG's characteristic washed-out
over-saturation.

### Image-to-video

```bash
python examples/generate_ltx2_5.py \
  --dit_checkpoint_path "./checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --vae_checkpoint_path "./checkpoints/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --text_encoder_checkpoint_path "./checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "The cat looks up at the camera" \
  --sampler dev \
  --tensor_parallel_size 4 \
  --offload_dit_weights --offload_chunk_size 8 \
  --output_path "out/output_ltx2_5_i2v.mp4"
```

Conditioning works differently from LTX-Video's I2V: instead of clamping a
per-token timestep, LTX-2.5 threads an explicit per-token "denoise mask"
through the DiT's own timestep input, the sampler's x0 estimate, and its
stepped output — `--conditioning_strength` (default `1.0`) controls how
strongly the conditioning frame is enforced.

### Tensor parallelism

Unlike LTX-Video (where `tp=1`'s weights fit and only activations force
higher `tp`), `--tensor_parallel_size 4` (the default) is required here
just for the 22B DiT's and 12B Gemma-4's own weights to fit at all.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--vae_variant` | `conv` | `conv` (ResNet + pixel-shuffle-upsample) or `diffusion` (NATTEN-based, the official demos' decoder — noticeably slower, see [Status](#status)). |
| `--tensor_parallel_size` | `4` | Must divide `num_devices`, the DiT's `num_attention_heads` (32), and Gemma-4's `num_attention_heads` (16). |
| `--conditioning_strength` | `1.0` | I2V only. |
| `--dit_dtype` | `bfloat16` | Every released checkpoint's `scale_shift_table`/`prompt_scale_shift_table` (AdaLN modulation tables) ship in float32 and are always kept at float32 regardless of this flag — downcasting them was a measurable quality bug. |
| `--sampler` | `distilled` | `distilled` (8-step ancestral-Euler, no CFG) or `dev` (30-step plain-Euler, real CFG + guidance-rescale). |
| `--guidance_scale` / `--guidance_rescale` | sampler default | Defaults to `1.0`/`0.0` for distilled, `3.0`/`0.7` for dev. |
| `--height` / `--width` | `704` / `1216` | Must be divisible by 32. |
| `--num_frames` | `121` | Wants `1 + 8k` for an exact VAE round-trip. |
| `--offload_dit_weights` | off | Needed at the reference's own resolution — not because the DiT's weights don't fit (they do, comfortably), but because a fully fused 48-block forward pass doesn't free per-block activations across blocks. See [Weight offloading](/blog/weight-offloading). |
| `--offload_chunk_size` | `1` | `8` is the largest value confirmed to fit at the reference resolution (`tp=4`, both checkpoints). |

### Status

Verified end-to-end against real checkpoints, both `dev`/`distilled`
recipes and both VAE decoder variants, at the reference's own
1216×704/121-frame resolution:

| Variant | VAE decoder | Steps | Compile (s) | Generation (s) | Peak HBM/chip (GB) |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | conv | 30 | 87.7 | 217.9 | 16.7 |
| distilled | conv | 8 | 87.9 | 37.3 | 15.3 |
| dev | diffusion | 30 | 479.5 | 2859.6 | 16.1 |
| distilled | diffusion | 8 | 478.2 | 2680.8 | 14.8 |

The `diffusion` VAE variant is a genuinely different, slower decoder (its
own internal denoising pass, tensor-parallel-sharded across the same
mesh) — reach for it only when matching the official demos' exact output
characteristics matters more than speed; `conv` is the faster default for
most use. See the [Benchmark Explorer](/benchmarks) for the full, current
set of measured rows.

---

See [LTX-Video](./ltx_video.md) for the earlier, smaller, unrelated LTX
model, or [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
and [Weight offloading](/blog/weight-offloading) on the blog for background.
