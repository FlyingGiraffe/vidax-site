---
sidebar_position: 5
title: LTX-Video
---

# LTX-Video (0.9.8)

One standalone TPU inference script, `generate_ltx_video.py`, covers both
T2V and I2V (pass `--image_path` for I2V, omit it for T2V) across all three
released 0.9.8 sizes. Architecturally unrelated to the Wan/Cosmos scripts —
a different VAE, a different (plain T5-XXL) text encoder, and a different
RoPE convention — see [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
for how it compares to the rest of vidax's model families.

| Script | Model | Params | Task | Checkpoint file example |
| --- | --- | --- | --- | --- |
| `generate_ltx_video.py` | LTX-Video 0.9.8 | 2B (distilled) | T2V, I2V | `ltxv-2b-0.9.8-distilled.safetensors` |
| `generate_ltx_video.py` | LTX-Video 0.9.8 | 13B (dev) | T2V, I2V | `ltxv-13b-0.9.8-dev.safetensors` |
| `generate_ltx_video.py` | LTX-Video 0.9.8 | 13B (distilled) | T2V, I2V | `ltxv-13b-0.9.8-distilled.safetensors` |

Everything it needs (`torch`, `transformers`, `pillow`) is installed by
default. On a Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

## Checkpoints

Each variant's DiT + VAE ship together as **one flat `.safetensors` file**
from [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) —
pass its path as `--checkpoint_path`. The text encoder is a separate
download, [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)'s
`text_encoder`/`tokenizer` subfolders (a plain T5-XXL, shared by all three
sizes) — pass `text_encoder/model.safetensors.index.json`'s path as
`--t5_checkpoint_path`.

### Text-to-video

```bash
python examples/generate_ltx_video.py \
  --checkpoint_path "./checkpoints/LTX-Video-0.9.8-2B-distilled/ltxv-2b-0.9.8-distilled.safetensors" \
  --t5_checkpoint_path "./checkpoints/PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 8 --guidance_scale 1.0 \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx_t2v.mp4"
```

`--num_steps 8 --guidance_scale 1.0` matches the distilled checkpoints' own
recipe (few steps, no classifier-free guidance). For `13B-dev`, use real CFG
and more steps instead: `--num_steps 30 --guidance_scale 3.0`.

### Image-to-video

```bash
python examples/generate_ltx_video.py \
  --checkpoint_path "./checkpoints/LTX-Video-0.9.8-2B-distilled/ltxv-2b-0.9.8-distilled.safetensors" \
  --t5_checkpoint_path "./checkpoints/PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --num_steps 8 --guidance_scale 1.0 \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx_i2v.mp4"
```

The conditioning image is resized to exactly `--height`x`--width` (no
aspect-ratio-preserving crop), VAE-encoded, and blended into the first
latent frame at `--conditioning_strength` (default `1.0` — the first frame
*is* the encoded image).

### Tensor parallelism

`--tensor_parallel_size` (default `1`) Megatron-shards both the DiT's and
the T5 encoder's weights. Needed for the 13B checkpoints regardless of
resolution (their bf16 weights alone don't fit replicated on a single TPU
v4 chip's HBM) — and, less obviously, needed for **2B too** at the
reference's full 704x1216/121-frame resolution, where the self-attention
activations alone don't fit at `tp=1`. There is no `--sequence_parallel_size`
yet for this model family.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_path` | *required* | The flat `.safetensors` file bundling both the DiT and VAE. |
| `--t5_checkpoint_path` | *required* | `PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json`. |
| `--image_path` | `None` | Conditioning image, for I2V. Omit for T2V. |
| `--conditioning_strength` | `1.0` | I2V only: how strongly the conditioning image is enforced. |
| `--guidance_scale` | `3.0` | CFG scale. Use `1.0` for the distilled checkpoints. |
| `--dit_dtype` | `bfloat16` | Every released checkpoint ships natively as bf16 — no fp32-weights requirement here, unlike Wan2.1. |
| `--tensor_parallel_size` | `1` | See [Tensor parallelism](#tensor-parallelism) above. Must divide `num_devices`, `LTXDiT.num_attention_heads` (32), and the T5 encoder's `num_heads` (64). |
| `--num_steps` | `30` | Use far fewer (e.g. `8`) for distilled checkpoints. |
| `--sampler` | `LinearQuadratic` | `Uniform` \| `LinearQuadratic` \| `Constant`. Every released checkpoint's own embedded config uses `LinearQuadratic`. |
| `--height` / `--width` | `512` / `768` | Must be divisible by 32 (the VAE's spatial downscale factor). |
| `--num_frames` | `97` | Wants `1 + 8k` for an exact VAE round-trip; the reference's own default is `121`. |
| `--fps` | `24` | Output video frame rate. |

### Scope

**Not implemented in this first port**: multi-scale two-pass generation
(the reference's default pipeline runs a low-res pass, an upsampler, then
a high-res pass — this port always runs single-scale, single-pass), STG
(spatio-temporal guidance) and `cfg_star_rescale` (plain CFG only),
sequence parallelism, and V2V.

### Status

Verified end-to-end against real checkpoints, all three sizes, T2V:

| Variant | Resolution | Frames | Steps | Compile (s) | Generation (s) | Peak HBM/chip (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 13B (dev) | 1216×704 | 121 | 30 | 134.7 | 156.0 | 15.3 |
| 13B (distilled) | 1216×704 | 121 | 8 | 136.4 | 104.2 | 15.3 |
| 2B (distilled) | 1216×704 | 121 | 8 | 83.5 | 47.3 | 8.8 |

See the [Benchmark Explorer](/benchmarks) for the full, current set of
measured rows.

---

See [LTX-2.5](./ltx2_5.md) for the newer, larger, architecturally-unrelated
LTX model, or [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog for background on how these models differ.
