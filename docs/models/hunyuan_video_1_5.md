---
sidebar_position: 7
title: HunyuanVideo-1.5
---

# HunyuanVideo-1.5

:::warning Early support
This model family landed most recently and doesn't yet have tensor/sequence
parallelism or weight offloading — see [Scope](#scope) below. Generation
works and has been checked against real checkpoints, but isn't yet
benchmarked or tuned the way the other model families are.
:::

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

Requires the `torch`, `text`, and `i2v` extras:

```bash
pip install -e ".[tpu,torch,text,i2v]"
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
  --height 480 --width 832 --num_frames 121 --num_steps 50 \
  --output_path "out/output_hunyuan_1_5_i2v.mp4"
```

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
| `--fps` | `24` | Output video frame rate. |

### Scope

- **No `--tensor_parallel_size`/`--sequence_parallel_size`/weight
  offloading yet.** The script instead places each component (DiT,
  Qwen/byT5 text towers, VAE, SigLIP) on its own single TPU chip — works
  on a small dev box, but isn't real per-component sharding and won't
  scale to fewer/larger chips. See [Sharding, parallelism, and JIT on TPUs](/blog/hardware-and-sharding)
  for the pattern a future pass would extend.
- No distilled/step-distilled/sparse-attention or super-resolution
  checkpoint variants.
- No VAE tiling — a real limit at high resolution/frame count, since
  there's no chunked-decode escape hatch yet for this model's VAE.

---

See [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog for how HunyuanVideo-1.5's token-concatenation conditioning
(rather than cross-attention) compares to the other model families, or the
[Benchmark Explorer](/benchmarks) for measured numbers as they land.
