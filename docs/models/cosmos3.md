---
sidebar_position: 4
title: Cosmos3
---

# Cosmos3

One standalone TPU inference script, `generate_cosmos3.py`, covers both
released checkpoint sizes' text-to-video and image-to-video generation via
`--model_size {nano,edge}` — the only two of Cosmos3's several surfaces
this port covers (see [Scope](#scope) for what's deliberately left out).
See [Architecture, in brief](#architecture-in-brief) for the full picture and
[Hardware & Sharding](../sharding/hardware-and-sharding.md) for the shared
TPU/JAX engineering background (sharding, flash attention, dtype
conventions).

| Script | Model | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- | --- |
| `generate_cosmos3.py --model_size nano` | Cosmos3-Nano | 16B | Text2Video, Image2Video | `Cosmos3-Nano/` |
| `generate_cosmos3.py --model_size edge` | Cosmos3-Edge | 4B | Text2Video, Image2Video | `Cosmos3-Edge/` |

This model never touches `torch` at runtime (checkpoints ship as
`.safetensors`, loaded directly), but `transformers` (the tokenizer + chat
template) and `pillow` (the conditioning frame) are used — all core
dependencies, installed by default. On a Cloud TPU VM also add the `tpu`
extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

Both sizes share the exact same weight layout and DiT code
(`Cosmos3Transformer`); `--model_size` selects which preset
(`NANO_CONFIG`/`EDGE_CONFIG`) to build it with. Edge additionally uses a
squared-ReLU MLP instead of Nano's SwiGLU, and a different Q/K-norm scheme —
both handled transparently by the shared code (see
[Architecture, in brief](#architecture-in-brief)).

## Cosmos3-Nano (16B) — `--model_size nano`

Ships as one self-contained HuggingFace `diffusers`-format repo, with the
components this port uses at
`transformer/diffusion_pytorch_model.safetensors.index.json` (the DiT,
sharded), `vae/diffusion_pytorch_model.safetensors` (Wan2.2-TI2V-5B's own
VAE, reused verbatim — see [Architecture, in brief](#architecture-in-brief)), and
`text_tokenizer/` (pass as-is to `--tokenizer_path`).

:::info Memory note
At 16B parameters (~29GB in bf16), Nano is close to or larger than a single
TPU v4 chip's ~30GB HBM budget on its own, before any activations.
`--tensor_parallel_size` is **not optional** the way it is for
Cosmos-Predict2.5's 2B — use at least `--tensor_parallel_size 4` (all
devices on a v4-8) unless running on a pod slice with proportionally more
HBM per chip.
:::

### Text2Video

```bash
python examples/generate_cosmos3.py \
  --model_size nano \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Nano/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Nano/text_tokenizer" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --max_text_len 3072 \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos3_nano_t2v.mp4"
```

A short plain-text prompt works for Nano, but see [Prompting](#prompting) —
checkpoints are documented to expect richer, JSON-structured prompts, and
Edge in particular needs this for good quality.

### Image2Video

```bash
python examples/generate_cosmos3.py \
  --model_size nano \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Nano/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Nano/text_tokenizer" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "A cat wearing sunglasses on a boat in the ocean, waves splashing" \
  --negative_prompt "Low quality, blurry, oversaturated, static, distorted." \
  --max_text_len 128 \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos3_nano_i2v.mp4"
```

`--image_path` anchors latent frame 0 to the VAE-encoded conditioning image
(resized, not cropped) and denoises the remaining frames — the same
frame-substitution mechanism as Cosmos-Predict2.5's image2world.

## Cosmos3-Edge (4B) — `--model_size edge`

Same `diffusers`-format layout as Nano, smaller (`hidden_size=2048`, 28
layers, 16/8 attention/KV heads vs. Nano's 4096/36/32/8) and comfortably
fits a single TPU v4 chip, so `--tensor_parallel_size 1` works. Uses its
own tokenizer (different vocab size than Nano's — don't mix checkpoints
across sizes).

:::danger Edge needs its own resolution/frame count/scheduler
Resolution, frame count, and scheduler are **not** interchangeable with
Nano's, and T2V/I2V use *different* scheduler values from each other.
Always pass Edge's explicit `--height`/`--width`/`--num_frames`/
`--num_steps`/`--use_karras_sigmas`/`--shift` as shown below. Getting this
wrong doesn't error — it just produces badly degraded output. See
[Status](#status) for how these values were determined.
:::

### Text2Video

```bash
python examples/generate_cosmos3.py \
  --model_size edge \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Edge/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Edge/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Edge/text_tokenizer" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --max_text_len 3072 \
  --tensor_parallel_size 4 \
  --height 480 --width 832 --num_frames 121 \
  --num_steps 35 --use_karras_sigmas false --shift 10.0 \
  --add_duration_template false --add_resolution_template false \
  --output_path "out/output_cosmos3_edge_t2v.mp4"
```

### Image2Video

```bash
python examples/generate_cosmos3.py \
  --model_size edge \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Edge/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Edge/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Edge/text_tokenizer" \
  --image_path "./checkpoints/Cosmos3-Edge/assets/example_i2v_input.jpg" \
  --prompt "A car driving along a coastal mountain road" \
  --negative_prompt "Low quality, blurry, oversaturated, static, distorted." \
  --max_text_len 128 \
  --tensor_parallel_size 4 \
  --height 480 --width 832 --num_frames 121 \
  --num_steps 20 --use_karras_sigmas false --shift 12.0 \
  --add_duration_template false --add_resolution_template false \
  --output_path "out/output_cosmos3_edge_i2v.mp4"
```

Note `--shift 10.0` for T2V vs. `--shift 12.0` for I2V — don't reuse one
task's value for the other.

### Quick testing

```bash
python examples/generate_cosmos3.py \
  --model_size nano \
  --dit_checkpoint_path ... --vae_checkpoint_path ... --tokenizer_path ... \
  --prompt "..." \
  --tensor_parallel_size 4 \
  --height 256 --width 256 --num_frames 9 --num_steps 10 --max_text_len 3072 \
  --output_path out/quick_test.mp4
```

This config is valid **only for Nano.** Swapping only `--model_size` to
`edge` (keeping 256×256/9 frames, Karras sigmas) reliably produces
degraded, incoherent output — it doesn't error, it just looks badly wrong.
For a genuine Edge smoke test, use its real recipe's scheduler
(`--use_karras_sigmas false --shift 10.0`) even at a reduced frame count,
e.g. `--height 480 --width 832 --num_frames 50 --num_steps 35`.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_size` | `nano` | `nano` or `edge`. Must match `--dit_checkpoint_path`'s actual checkpoint. |
| `--dit_checkpoint_path` | *required* | `transformer/diffusion_pytorch_model.safetensors.index.json` — a flat-layout state dict, unlike Cosmos-Predict2.5's nested layout. |
| `--vae_checkpoint_path` | *required* | Wan2.2-TI2V-5B's VAE in `diffusers`' `AutoencoderKLWan` layout — identical for Nano and Edge. |
| `--tokenizer_path` | *required* | The checkpoint's own `text_tokenizer/` directory — don't mix Nano's and Edge's. |
| `--add_duration_template` / `--add_resolution_template` | `true` | Whether to append duration/FPS and resolution metadata sentences to the prompt. Every real usage example passes `false` for both — see [Prompting](#prompting). |
| `--max_text_len` | `3072` | Fixed padded text-token length (JAX needs a static shape). Must comfortably fit the negative prompt too — the default negative prompt tokenizes to ~2800 tokens. |
| `--guide_scale` | `6.0` | CFG scale. |
| `--tensor_parallel_size` | `1` | Must divide `num_devices`, attention heads, and KV heads (32/8 for Nano, 16/8 for Edge — GQA's KV-head count is the binding constraint, so `tp` in `{1,2,4,8}`). Effectively required for Nano; optional but useful for Edge. |
| `--num_steps` | `35` | UniPC sampling steps. Nano default; also Edge T2V's real value — Edge I2V's real recipe uses `20`. |
| `--use_karras_sigmas` | `true` | Nano's schedule. Edge's real recipe (both tasks) uses a plain shift-warped (non-Karras) schedule — pass `false` for Edge. |
| `--shift` | `5.0` | Only read when `--use_karras_sigmas false`. Edge: `10.0` for T2V, `12.0` for I2V. |
| `--karras_sigma_min` / `--karras_sigma_max` | `0.147` / `200.0` | Only read when `--use_karras_sigmas true`. |
| `--height` / `--width` | `704` / `1280` | Must be divisible by 32. Targets **Nano's** spec — Edge needs an explicit override (e.g. `480`/`832`). |
| `--num_frames` | `93` | Nano default — Edge's real recipe uses `121`. |
| `--fps` | `24.0` | Also injected into the mRoPE temporal modulation and the prompt's duration-metadata sentence. |

### Status

**Both sizes verified: clean, stable, high-quality output** — Nano at its
full resolution (1280×704, 93 frames, 35 steps), Edge at its real per-task
recipe (480×832, 121 frames, non-Karras; T2V: 35 steps/`shift=10.0`, I2V:
20 steps/`shift=12.0`), **with a properly JSON-structured prompt** (see
[Prompting](#prompting) — this matters far more for Edge than for Nano).

Three real, distinct bugs were found and fixed to get here — a
vision-segment mRoPE offset computed from the padded text length instead
of each prompt's real token count, Edge silently running at Nano's
resolution/scheduler defaults, and both models needing a JSON-structured
prompt rather than a short plain-text one. See the blog's
[Engineering Notes](/blog/tags/engineering-notes) for more correctness bugs
found porting vidax's models.

### Prompting

Cosmos3's checkpoints document that prompts should be **"upsampled into a
specific JSON structure"** for optimal quality, not passed as a short
plain-text sentence. The JSON structure covers subjects,
background/setting, lighting, aesthetics, cinematography, and a temporal
caption.

:::tip This matters more than it looks
A short prompt like *"A red panda climbing a bamboo tree"* still produces
recognizable output on Nano (16B), but on Edge (4B) the same short prompt
produces flat, oversaturated, mostly featureless output — swapping in a
real JSON-upsampled prompt (same everything else) produces fully
photorealistic, detailed output instead.
:::

`--prompt` still accepts a plain string, but for Edge specifically use a
JSON-structured one for real work. `generate_cosmos3.py`'s default negative
prompt is itself a real, JSON-structured negative prompt — pair it with a
comparably detailed positive prompt: pairing a short positive prompt with a
much longer negative prompt inflates the vision-segment mRoPE offset gap
between the cond/uncond passes and can produce badly corrupted output.

## Scope

This port covers **text2video and image2video only** — a deliberate scope
decision. Cosmos3 also supports several surfaces this repo doesn't
implement:

- **Video2video, action-conditioned generation, sound-conditioned
  generation** — the reference pipeline supports all three, none are wired
  up here.
- **The "Reasoner" surface** (causal-LM-only, text/vision-in, text-out) —
  this port never loads `lm_head` or `vision_encoder/` at all, since
  neither is needed for pure generation.

## Architecture, in brief

Cosmos3 is an omnimodal Mixture-of-Transformers: each decoder layer carries
two full parallel weight sets (a causal "und"/text pathway and a
full-attention "gen"/diffusion pathway) sharing one packed token sequence,
with no AdaLN modulation anywhere and a genuinely different interleaved 3D
mRoPE scheme from either Wan's or Cosmos-Predict2.5's RoPE. Nano and Edge
share the exact same code, differing only by config (SwiGLU vs.
squared-ReLU MLP, different Q/K-norm handling, different `rope_theta`). For
background on Mixture-of-Transformers and how it compares to the DiT
families above, see
[Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog.

## Coming later

- **Cosmos3-Super** (the 64B sibling), if useful.
- **Video2video, action-conditioned, and sound-conditioned generation**, and
  the **Reasoner** surface — see [Scope](#scope) for why these are out for
  now.

---

See [Wan2.1](./wan2_1.md), [Wan2.2](./wan2_2.md), and
[Cosmos-Predict2.5](./cosmos2_5.md) for the other model families, or the
[Benchmark Explorer](/benchmarks) for measured numbers across every row
above.
