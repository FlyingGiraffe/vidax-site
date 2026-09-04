---
sidebar_position: 7
title: HunyuanVideo-1.5
---

# HunyuanVideo-1.5

One standalone TPU inference script, `generate_hunyuan_video_1_5.py`,
covers both T2V and I2V for a given resolution (pass `--image_path` for
I2V, omit it for T2V). Covers the 4 core (non-distilled,
non-sparse-attention, non-super-resolution) checkpoint variants: 480p/720p
× T2V/I2V.

| Script | Model | Params | Task | Checkpoint variant |
| --- | --- | --- | --- | --- |
| `generate_hunyuan_video_1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | T2V | `transformer/480p_t2v/` |
| `generate_hunyuan_video_1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | I2V | `transformer/480p_i2v/` |
| `generate_hunyuan_video_1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | T2V | `transformer/720p_t2v/` |
| `generate_hunyuan_video_1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | I2V | `transformer/720p_i2v/` |

## Checkpoints

Conditioning requires three separate text/vision towers on top of the DiT
itself: a Qwen2.5-VL-7B-Instruct text-only tower, a byT5-small
(Glyph-SDXL-v2) glyph/color encoder, and (I2V only) a SigLIP vision
encoder. `--checkpoint_dir` should point at
[`tencent/HunyuanVideo-1.5`](https://huggingface.co/tencent/HunyuanVideo-1.5)'s
downloaded root (`transformer/`, `vae/`, `text_encoder/{llm,byt5-small,Glyph-SDXL-v2}/`);
`--siglip_checkpoint_dir` (I2V only) at
[`black-forest-labs/FLUX.1-Redux-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)'s
downloaded root — a **gated** repo, request access first.

Everything it needs (`torch`, `transformers`, `pillow`) is installed by
default. On a Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

### Text-to-video

```bash
python examples/generate_hunyuan_video_1_5.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo-1.5" \
  --resolution 480p \
  --prompt "A golden retriever running on a beach at sunset, cinematic, high detail" \
  --height 480 --width 832 --num_frames 121 --num_steps 50 \
  --output_path "out/output_hunyuan_1_5_t2v.mp4"
```

### Image-to-video

```bash
python examples/generate_hunyuan_video_1_5.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo-1.5" \
  --siglip_checkpoint_dir "./checkpoints/FLUX.1-Redux-dev" \
  --resolution 480p \
  --image_path "./assets/dog.jpg" \
  --prompt "The dog starts running toward the camera" \
  --num_frames 121 --num_steps 50 \
  --tensor_parallel_size 4 \
  --output_path "out/output_hunyuan_1_5_i2v.mp4"
```

Omit `--height`/`--width` for I2V and the output resolution is derived from
the conditioning image's own aspect ratio — a portrait image produces a
portrait video, not a squished landscape one.

### Tensor parallelism

`--tensor_parallel_size` (default: every local device) Megatron-shards the
DiT's double/single-stream Q/K/V/output/FFN Dense layers across chips; the
rest (VAE, Qwen2.5-VL, byT5, SigLIP) is simply replicated across the same
mesh. Must divide `heads_num` (16). **Required in practice** — the 8.3B
DiT's bf16 weights (~16.6 GB) don't fit replicated alongside the other
components on a single TPU v4 chip. There is no `--sequence_parallel_size`
or weight offloading for this family yet; VAE decode has its own spatial
tiling (`--vae_tile_latent_size`, independent of TP).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_dir` | *required* | `tencent/HunyuanVideo-1.5`'s downloaded root. |
| `--siglip_checkpoint_dir` | `None` | `black-forest-labs/FLUX.1-Redux-dev`'s downloaded root. Required for I2V. |
| `--resolution` | `480p` | `480p` or `720p` — selects the checkpoint variant and the default `--shift`. |
| `--image_path` | `None` | Conditioning image, for I2V. Omit for T2V. |
| `--height` / `--width` | `480` / `832` | Must be divisible by 16. |
| `--num_frames` | `121` | Wants `1 + 4k` for an exact VAE round-trip. |
| `--num_steps` | `50` | Flow-matching Euler sampling steps. |
| `--shift` | resolution default | 480p: `5.0` both tasks; 720p: `9.0` T2V, `7.0` I2V. |
| `--guidance_scale` | `6.0` | Real CFG — these checkpoints have no embedded/distilled guidance path. |
| `--dit_dtype` | `bfloat16` | Cast target for the DiT's weights (checkpoints ship as float32). |
| `--tensor_parallel_size` | every local device | Megatron-shards the DiT's double/single-stream Dense layers; the other components stay replicated. Must divide `heads_num` (16). See [Tensor parallelism](#tensor-parallelism) above. |
| `--vae_tile_latent_size` | `16` | Latent-space spatial tile size for the tiled VAE decode. Shrink (e.g. `8`) if VAE decode OOMs — more likely at `tp > 1`, where the replicated components leave less per-chip headroom. |
| `--fps` | `24` | Output video frame rate. |

### Scope

- **No `--sequence_parallel_size` or weight offloading yet.** Tensor
  parallelism (above) is enough to fit the 4 core checkpoints at 480p and
  720p on a v4-8; a larger frame count or batch would need one of those
  additions. See [Sharding, parallelism, and JIT on TPUs](/blog/hardware-and-sharding).
- No distilled/step-distilled/sparse-attention or super-resolution
  checkpoint variants.
- VAE decode is spatially tiled but never temporally tiled (matching the
  reference VAE, which has no temporal-tiling path).

### Status

Verified end-to-end against real checkpoints, all 4 core variants. 5-run
averages on a TPU v4-8 (see the [Benchmark Explorer](/benchmarks) for the
full, current set):

| Variant | Task | Resolution | Frames | Steps | s/step | Peak HBM/chip (GB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 8.3B (480p) | T2V | 832×480 | 121 | 30 | 119.5 | 29.1 |
| 8.3B (480p) | I2V | 544×720 | 121 | 30 | 112.9 | 31.4 |
| 8.3B (720p) | T2V | 1280×720 | 121 | 30 | 221.0 | 30.3 |
| 8.3B (720p) | I2V | 832×1104 | 121 | 30 | 219.2 | 32.0 |

All rows use `tp=4`. This architecture's joint global self-attention over
the full image+text token sequence (no windowing) makes it the slowest
per-step model in the benchmark table.

---

See [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog for how HunyuanVideo-1.5's token-concatenation conditioning
(rather than cross-attention) compares to the other model families. See
[HunyuanVideo (1.0)](./hunyuan_video.md) for the 13B model in the same
block family, and the [Benchmark Explorer](/benchmarks) for measured
numbers.
