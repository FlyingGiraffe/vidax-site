---
sidebar_position: 1
title: Wan2.1
---

# Wan2.1

Two standalone TPU inference scripts live in `examples/`, one per task. They
share vidax's common building blocks (`vidax.core`, `vidax.schedulers`,
`vidax.translator`) — see [Hardware & Sharding](../sharding/hardware-and-sharding.md)
for the engineering behind the parallelism strategy (Megatron tensor
parallelism, flash attention, JIT safety).

| Script | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- |
| `generate_wan2_1_t2v.py --model_size 1.3B` | 1.3B | Text-to-Video | `Wan2.1-T2V-1.3B` |
| `generate_wan2_1_t2v.py --model_size 14B` | 14B | Text-to-Video | `Wan2.1-T2V-14B` |
| `generate_wan2_1_i2v.py` | 14B | Image-to-Video | `Wan2.1-I2V-14B-480P` or `Wan2.1-I2V-14B-720P` |

Both T2V sizes share one architecture/script
(`vidax.models.wan.wan2_1.dit.WanDiT`, fully config-driven); `--model_size`
just selects which hyperparameter preset
(`T2V_1_3B_CONFIG`/`T2V_14B_CONFIG`) to build it with. I2V only ships as
14B (no 1.3B I2V checkpoint exists), so its script has no `--model_size`
flag — it always builds `I2V_14B_CONFIG`.

Both scripts use `torch` (to deserialize `.pth`/`.safetensors` checkpoints)
and `transformers`/`sentencepiece` (tokenization) — all core dependencies,
installed by default. On a Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

`--tokenizer_path` defaults to `<t5_checkpoint_dir>/google/umt5-xxl` for
both scripts, matching the official HuggingFace repo layout — pass it
explicitly if yours differs.

## T2V (1.3B / 14B) — `generate_wan2_1_t2v.py`

Checkpoints (DiT `.safetensors`, VAE `.pth`, T5 `.pth` + its
`google/umt5-xxl` tokenizer folder) come from the official
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) or
[Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) repos — pass
`--model_size` to match whichever you downloaded.

### Basic generation (1.3B)

```bash
python examples/generate_wan2_1_t2v.py \
  --model_size 1.3B \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 50 \
  --output_path "out/output.mp4"
```

### 14B generation

The 14B DiT ships sharded across multiple `.safetensors` files with a
`.safetensors.index.json` manifest — pass that manifest's path, not a
single `.safetensors` file:

```bash
python examples/generate_wan2_1_t2v.py \
  --model_size 14B \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-14B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --num_steps 50 \
  --output_path "out/output_14b.mp4"
```

:::tip CFG isn't optional
`--shift` (noise-schedule shift, default `5.0`) and classifier-free guidance
(`--guide_scale`, default `5.0`, `--negative_prompt` defaulting to the
reference's quality-negative-prompt) match the reference pipeline's own
defaults and aren't optional extras — skipping CFG produces washed-out,
low-contrast output, since the model's raw conditional prediction alone
regresses hard toward an "average video."
:::

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_size` | `1.3B` | `1.3B` or `14B`. Must match `--dit_checkpoint_path`'s actual checkpoint. |
| `--dit_checkpoint_path` | *required* | DiT `.safetensors` (1.3B) or `.safetensors.index.json` manifest (14B, sharded). |
| `--vae_checkpoint_path` | *required* | VAE `.pth` checkpoint. |
| `--t5_checkpoint_path` | *required* | T5 (UMT5-XXL encoder) `.pth` checkpoint. |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | UMT5-XXL tokenizer directory. |
| `--prompt` | *required*, 1+ values | One prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica. |
| `--negative_prompt` | reference's `sample_neg_prompt` | Negative prompt for CFG. |
| `--guide_scale` | `5.0` | CFG scale: `velocity = uncond + guide_scale * (cond - uncond)`. |
| `--tensor_parallel_size` | `1` | Megatron-style weight sharding — must divide `num_devices` and `num_heads` (12 for 1.3B, 40 for 14B, 64 for T5). `4` is a reasonable start on a v4-8 for either size; raise it if you hit HBM OOM. See [Hardware & Sharding](../sharding/hardware-and-sharding.md). |
| `--sequence_parallel_size` | `1` | DeepSpeed-Ulysses token-sequence sharding, independent of `--tensor_parallel_size` — the two compose freely. Not needed at 1.3B; may help 14B at higher resolutions. |
| `--dtype` | `bfloat16` | Compute dtype for T5/VAE and the DiT's *activations* (not weights — see `--dit_dtype`). `float16` fails at runtime — TPU's XLA backend has no `float16` matmul. |
| `--dit_dtype` | `float32` | Cast target for the DiT's *weights*, independent of `--dtype`. Defaults to `float32` — see [Precision: fp32 DiT weights](#precision-fp32-dit-weights) below for why. |
| `--offload_dit_weights` | off | Per-layer weight offloading — keeps the DiT host-resident, streaming `--offload_chunk_size` blocks into HBM at a time. Needed at native 720P; not needed at 480P. Real throughput cost — see [Weight Offloading](../sharding/weight-offloading.md). Not combinable with `--sequence_parallel_size > 1` for this model. |
| `--offload_chunk_size` | `1` | Blocks grouped per offloaded HBM buffer/compile. Must divide `num_layers` (30 for 1.3B, 40 for 14B). Native-720P benchmark rows use `20`. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `50` | Sampling steps. |
| `--shift` | `5.0` | Flow-matching noise-schedule shift. |
| `--height` / `--width` | `480` / `832` | Output resolution. |
| `--num_frames` | `81` | Output frame count. |
| `--output_path` | `output_video.mp4` | With multiple prompts, each saved as `<output_path>_<i>.mp4`. |

**Status:** fully verified end-to-end against real checkpoints for both
1.3B and 14B — output confirmed coherent.

## I2V (14B) — `generate_wan2_1_i2v.py`

I2V only ships as a **14B** model, as **two separate checkpoints** tuned
for different resolution ranges —
[Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) and
[Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)
(no 1.3B I2V checkpoint exists). Both share an identical
architecture/`config.json` — only the weights and resolution they were
tuned at differ — so one script drives both; point
`--dit_checkpoint_path` at whichever you downloaded and pass a matching
`--max_area`/`--shift` (below). Both additionally need a **CLIP vision
encoder** checkpoint
(`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, bundled in each
repo) to extract image features from the conditioning frame.

### 480P checkpoint

```bash
python examples/generate_wan2_1_i2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --clip_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --image_path "./checkpoints/Wan2.1-I2V-14B-480P/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --max_area $((480 * 832)) \
  --output_path "out/output_i2v_480p.mp4"
```

`--max_area` (aligned to the checkpoint's own trained resolution range)
leaves `--shift` on its default auto-selection, which resolves to `3.0` at
this scale.

### 720P checkpoint

```bash
python examples/generate_wan2_1_i2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
  --clip_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --image_path "./checkpoints/Wan2.1-I2V-14B-720P/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --output_path "out/output_i2v_720p.mp4"
```

`--max_area` defaults to 720×1280 already (left unset here); `--shift`
auto-resolves to `5.0` at this scale.

Conditioning is built two ways:
[`WanVAEEncoder`](https://github.com/FlyingGiraffe/vidax/blob/main/src/vidax/models/wan/wan2_1/vae.py)
encodes the image (one real frame followed by zero frames) into the DiT's
mask+latent conditioning channels, and
[`ClipVisionTransformer`](https://github.com/FlyingGiraffe/vidax/blob/main/src/vidax/models/wan/wan2_1/clip_vision.py)
extracts CLIP features the DiT cross-attends onto through a second,
image-only K/V projection.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | 480P or 720P checkpoint's `.safetensors.index.json` manifest. |
| `--vae_checkpoint_path` / `--t5_checkpoint_path` / `--clip_checkpoint_path` | *required* | VAE, T5, and CLIP vision checkpoints. |
| `--image_path` | *required* | Conditioning image — output resolution is derived from it, not set directly. |
| `--prompt` | *required* | Single string (unlike the t2v script's list). |
| `--tensor_parallel_size` | `1` | Must divide `num_heads` (40 for the 14B DiT, 64 for T5). `4` (full width on a v4-8) is the typical starting point. |
| `--sequence_parallel_size` | `1` | Independent of `--tensor_parallel_size`. Verified working with the CLIP cross-attention branch only when `--tensor_parallel_size 1` — combining both with i2v's CLIP branch isn't supported yet. |
| `--dtype` | `bfloat16` | Same as T2V. |
| `--dit_dtype` | `float32` | Also applies to the conditioning tensor `y` and the sampling loop's `latents`, since both concatenate directly into the DiT's input. |
| `--offload_dit_weights` | off | Needed for the 720P checkpoint at native resolution; not needed for 480P. |
| `--offload_chunk_size` | `1` | Same semantics as T2V. |
| `--num_steps` | `40` | Reference i2v default (vs. 50 for t2v). |
| `--shift` | auto | `3.0` if `--max_area <= 832*480` (480P scale), `5.0` otherwise (720P scale). Pass explicitly to override. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--max_area` | `720*1280` | Bounds output pixel count, preserving the input image's aspect ratio, aligned to VAE stride × DiT patch size. Set to `480*832` for the 480P checkpoint. |
| `--num_frames` | `81` | Output frame count. |

**Status:** verified end-to-end against real checkpoints for **both** 480P
and 720P, output confirmed coherent for each at its own resolution. 480P
runs fully device-resident with no special handling. Native 720P needs
`--offload_dit_weights` — a fully-resident fp32 DiT otherwise leaves no HBM
headroom for the conditioning image's VAE encode right before generation
starts.

## Precision: fp32 DiT weights

Wan2.1's DiT weights default to `--dit_dtype float32`, not `bfloat16` —
different from every other model in vidax. The reference implementation's
checkpoints ship natively as float32 on disk, and its attention block wraps
residual updates in `float32` autocast without ever casting back down, so
the residual stream stays float32 for virtually the entire 40-layer
network — bf16 autocast applies only transiently to individual matmul/conv
ops, never to the stored weights or the accumulating residual.

:::warning Why this matters
Rounding the checkpoint to bf16 at load (vidax's original, incorrect
default) looks fine at most scales, but compounds into real, visually
obvious output corruption — hazy, flat, low-detail — specifically at large
token counts (native 720P, 81 frames), where 40 layers' worth of bf16
rounding error in a long residual chain finally becomes significant.
:::

`--dit_dtype bfloat16` remains fully supported as an explicit opt-in
(useful for memory-constrained runs at smaller/safer token counts). The
memory cost of the fp32 default is real — roughly double the DiT's
resident weight memory versus bf16 — which is the direct cause of
I2V-14B-720P needing `--offload_dit_weights` at native resolution. See the
blog's [Engineering Notes](/blog/tags/engineering-notes) for more numerics
and correctness gotchas found porting vidax's models.

---

See [Wan2.2](./wan2_2.md) (TI2V-5B, A14B), [Cosmos-Predict2.5](./cosmos2_5.md),
and [Cosmos3](./cosmos3.md) for the other model families, or the
[Benchmark Explorer](/benchmarks) for measured numbers across every row
above.
