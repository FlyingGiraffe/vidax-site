---
sidebar_position: 3
title: Cosmos-Predict2.5
---

# Cosmos-Predict2.5

One standalone TPU inference script, `generate_cosmos2_5.py`, covers both
released sizes (2B and 14B, via `--model_size`) and all three tasks the
checkpoints support — text2world, image2world, video2world — selected by
which conditioning flag you pass. Both sizes share one architecture
(`vidax.models.cosmos2_5.dit.CosmosDiT`) and only differ in
`dim`/`ffn_dim`/`num_heads`/`num_layers`. It shares vidax's common building
blocks with the Wan scripts but differs in checkpoint layout, sampler
(UniPC instead of Euler), and text encoder (a 7B-parameter VLM instead of
T5) — see [Architecture, in brief](#architecture-in-brief) below.

| Script | Model | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- | --- |
| `generate_cosmos2_5.py` | Cosmos-Predict2.5 | 2B | Text2World, Image2World, Video2World | `Cosmos-Predict2.5-2B/base/pre-trained` |
| `generate_cosmos2_5.py` | Cosmos-Predict2.5 | 14B | Text2World, Image2World, Video2World | `Cosmos-Predict2.5-14B/base/pre-trained` |

`torch` (to deserialize the `.pt` DiT/VAE checkpoints), `transformers` (the
Reason1/Qwen2.5-VL-7B tokenizer), and `pillow` (image2world/video2world's
conditioning frames) are all core dependencies, installed by default. On a
Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

## Checkpoints

Checkpoints come from **three separate repos**:

- DiT + VAE: [nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
  or [nvidia/Cosmos-Predict2.5-14B](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B) —
  the DiT as a flat PyTorch state dict
  (`base/pre-trained/<uuid>/model_ema_bf16.pt` for 2B,
  `base/pre-trained/<uuid>_ema_bf16.pt` for 14B); the VAE (Wan2.1's causal
  VAE, reused verbatim — see [Architecture, in brief](#architecture-in-brief)) as
  `tokenizer.pth`, shipped only in the 2B repo and shared by both sizes.
- Text encoder (shared by both sizes): [nvidia/Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) —
  a standard HuggingFace-format repo, sharded
  `model-NNNNN-of-NNNNN.safetensors` + `model.safetensors.index.json`. Pass
  the `.index.json` manifest's path as `--reason1_checkpoint_path`;
  `--tokenizer_path` then defaults to that same directory.

Pass `--model_size 2B` (default) or `--model_size 14B` to match whichever
`--dit_checkpoint_path` you point at.

Unlike Wan2.1's I2V (a separate 14B model, CLIP cross-attention) but *like*
Wan2.2 TI2V-5B's I2V, image2world/video2world conditioning works by
substituting known frames' latents back into `x` between sampling steps —
plus a concatenated mask channel and a tiny per-frame timestep for
conditioning frames, not frame substitution alone. See
[Architecture, in brief](#architecture-in-brief) for the full mechanism.

### Text2World

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos2_5_t2v.mp4"
```

Unlike the Wan scripts (Euler sampling, ~50 steps), this uses
`FlowUniPCMultistepScheduler` — a higher-order predictor-corrector solver
that reaches comparable quality in far fewer steps (35 vs. 50), at the cost
of a small rolling history of previous model outputs.

### Image2World

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos2_5_i2v.mp4"
```

The conditioning image is resized/center-cropped to the largest resolution
under `--max_area` divisible by VAE stride × DiT patch size, then
VAE-encoded into a single conditioning latent frame — output resolution is
derived from the image, ignoring `--height`/`--width`.

### Video2World

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --video_path "./checkpoints/Cosmos-Predict2.5-2B/assets/example_clip.mp4" \
  --num_conditional_latent_frames 2 \
  --prompt "The scene continues, the camera slowly pans right" \
  --num_steps 35 \
  --output_path "out/output_cosmos2_5_v2v.mp4"
```

Conditions on the input video's first `1 + 4 * (num_conditional_latent_frames - 1)`
pixel frames (1 frame for 1 conditioning latent frame, 5 for 2) — matching
the reference's training distribution over zero (t2v), one, or two
conditioning latent frames.

### Tensor + sequence parallelism

`--tensor_parallel_size` (default `1`) Megatron-shards both the DiT's
attention heads/FFN channels *and* Reason1's weights — Reason1's 7B params
are by far the largest of the three checkpoints, so this is where TP
matters most for fitting on fewer chips. `--sequence_parallel_size`
(default `1`) shards the DiT's token sequence itself (DeepSpeed-Ulysses),
independent of TP — the two compose freely, and SP alone is what matters
for pushing to much higher resolution/frame counts (`--sequence_parallel_size`
only affects the DiT; Reason1's 512-token sequence is far too short for
sequence parallelism to help).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_size` | `2B` | `2B` \| `14B`. Must match `--dit_checkpoint_path`'s actual size. |
| `--dit_checkpoint_path` | *required* | `model_ema_bf16.pt`/`model_ema_fp32.pt` — the flat EMA-only state dict, **not** the raw `model.pt` training checkpoint. |
| `--vae_checkpoint_path` | *required* | `tokenizer.pth` — Wan2.1's own VAE checkpoint format. |
| `--reason1_checkpoint_path` | *required* | Reason1's `model.safetensors.index.json`. |
| `--image_path` / `--video_path` | `None` | Mutually exclusive; select image2world/video2world. |
| `--num_conditional_latent_frames` | `1` | `--video_path` only: 1 or 2 (forced to 1 for `--image_path`). |
| `--max_area` | `704*1280` | image2world/video2world only. |
| `--guide_scale` | `7.0` | CFG scale — matches the reference default. |
| `--tensor_parallel_size` | `1` | Must divide `num_devices`, `CosmosDiT.num_heads` (16 for 2B, 40 for 14B), and Reason1's GQA `num_key_value_heads` (4 — the binding constraint, so `tp` in `{1,2,4}` in practice). |
| `--sequence_parallel_size` | `1` | Requires the latent frame count to divide evenly by this value. |
| `--offload_dit_weights` | off | Needed for 14B to reach the reference's full 93-frame default on a 4-chip machine (without it, reduced to 45 frames). Composes with `--sequence_parallel_size`. |
| `--offload_chunk_size` | `1` | Must divide `num_layers` (28 for 2B, 36 for 14B). |
| `--num_steps` | `35` | UniPC sampling steps. |
| `--solver_order` | `2` | UniPC solver order. |
| `--shift` | `5.0` | Flow-matching noise-schedule shift. |
| `--height` / `--width` | `704` / `1280` | Ignored if `--image_path`/`--video_path` is given. Must be divisible by 16. |
| `--num_frames` | `93` | The reference trains primarily around 93-frame (~5.8s @ 16fps) clips at 720p. |
| `--fps` | `16` | Output frame rate. |

### Status

**Verified end-to-end on real weights, both text2world and image2world** —
output is coherent, prompt-matching video. Getting there took six real
bugs, found only once run against real checkpoints — dominated by a
wrongly-added EDM-style preconditioning wrapper borrowed from a reference
class this checkpoint's training config never actually uses. See the
blog's [Engineering Notes](/blog/tags/engineering-notes) for more
correctness bugs found porting vidax's models.

- DiT: exact 1:1 parameter match (569/569 keys) against `CosmosDiT`'s
  initialized tree.
- Reason1: exact 1:1 match (338/338 keys), plus a real tokenizer + real
  forward pass producing a `(B, 512, 100352)` embedding with sane
  statistics.
- Parallelism: both `--tensor_parallel_size 4` and
  `--sequence_parallel_size 2` complete cleanly with correct output shapes.

**14B**: same translator mapping, DiT architecture, and sampling loop as
2B, just wider/deeper — verified end-to-end at full resolution and step
count (704×1280, 93 frames, 35 steps) with `--tensor_parallel_size 4
--offload_dit_weights --offload_chunk_size 1`: 5 full benchmark runs, 48.5s
compile, 4479.6s generation, 128.0s/step, **14.7GB** peak HBM/chip.

### Quick testing

Full-resolution, full-step runs are slow to iterate with — use a smaller
config while making changes:

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path ... --vae_checkpoint_path ... --reason1_checkpoint_path ... \
  --prompt "..." \
  --height 256 --width 256 --num_frames 9 --num_steps 20 \
  --output_path out/quick_test.mp4
```

This still exercises the full pipeline (DiT, VAE, Reason1, UniPC) end to
end — enough to tell "did this crash / does the output look structurally
different" apart from a full quality judgment.

## Architecture, in brief

Cosmos-Predict2.5 reuses Wan2.1's causal VAE unchanged, but its DiT departs
from Wan's in several ways (per-head QK-RMSNorm, rotate-half RoPE,
AdaLN-LoRA modulation), conditions on a 7B Reason1 (Qwen2.5-VL) text tower
instead of T5, and samples with UniPC instead of Euler. For background on
DiTs, conditioning mechanisms, and how each vidax model family compares,
see [Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog.

## Coming later

**Cosmos 3** is a separate, architecturally unrelated model family (a
Mixture-of-Transformers, not a DiT continuation) — see [Cosmos3](./cosmos3.md).

---

See [Wan2.1](./wan2_1.md), [Wan2.2](./wan2_2.md), and [Cosmos3](./cosmos3.md)
for the other model families, or the [Benchmark Explorer](/benchmarks) for
measured numbers across every row above.
