---
sidebar_position: 9
title: CogVideoX
---

# CogVideoX / CogVideoX1.5

One standalone TPU inference script, `generate_cogvideox.py`, covers every
released CogVideoX checkpoint (THUDM / ZhipuAI) — both generations, both
tasks (pass `--image_path` for I2V, omit it for T2V) — selected with
`--variant`.

| `--variant` | HF repo | Task | Notable |
| --- | --- | --- | --- |
| `2b` | [`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b) | T2V | no RoPE — fixed 3D sincos positional embedding; `snr_shift_scale=3` |
| `5b` | [`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b) | T2V | 3D RoPE, v-prediction (the canonical model) |
| `5b-i2v` | [`THUDM/CogVideoX-5b-I2V`](https://huggingface.co/THUDM/CogVideoX-5b-I2V) | I2V | image latent concatenated on channels; **learned** positional-embedding buffer locks it to 720×480 |
| `1.5-5b` | [`THUDM/CogVideoX1.5-5B`](https://huggingface.co/THUDM/CogVideoX1.5-5B) | T2V | `patch_size_t=2` (temporal patchify), `"slice"`-grid RoPE, 81/161 frames, 1360×768 |
| `1.5-5b-i2v` | [`THUDM/CogVideoX1.5-5B-I2V`](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) | I2V | + `ofs` (offset) embedding added to the timestep embedding |

Every variant shares one DiT class, one causal 3D-conv VAE
(spatial ÷8, temporal ÷4, `latent_channels=16`), the T5-v1.1-XXL text
encoder (reused verbatim from the LTX-Video port), and the
`vidax.schedulers.cogvideox` schedulers (`CogVideoXDDIMScheduler` /
`CogVideoXDPMScheduler` — v-prediction + zero-terminal-SNR + SD3-style SNR
shift + `trailing` spacing).

Everything it needs (`torch`, `transformers`, `pillow`) is installed by
default. On a Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

## Checkpoints

Download a diffusers-format repo
(`huggingface-cli download THUDM/CogVideoX-5b --local-dir ./checkpoints/CogVideoX-5b`);
the script expects the standard `transformer/`, `vae/`, `text_encoder/`,
`tokenizer/` subdirectories under `--model_dir`. The `t5-v1.1-xxl` weights +
tokenizer are **byte-identical across every CogVideoX repo** (only the
stored dtype differs, and bf16→f32 is exact), so `--t5_dir` / `--tokenizer_dir`
can point at any one downloaded copy — handy when disk is tight (download
the other repos with `--exclude "text_encoder/*" "tokenizer/*"`).

### Text-to-video

```bash
python examples/generate_cogvideox.py \
  --model_dir ./checkpoints/CogVideoX-5b --variant 5b \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_inference_steps 50 --guidance_scale 6.0 --scheduler dpm \
  --output_path out/cogvideox_5b_t2v.mp4
```

`--scheduler dpm` + `--use_dynamic_cfg` (default) matches the reference
`cli_demo.py` recipe for the 5B models; `--scheduler ddim --no_dynamic_cfg`
matches the recommended recipe for CogVideoX-2b.

### Image-to-video

```bash
python examples/generate_cogvideox.py \
  --model_dir ./checkpoints/CogVideoX-5b-I2V --variant 5b-i2v \
  --t5_dir ./checkpoints/CogVideoX-5b/text_encoder \
  --tokenizer_dir ./checkpoints/CogVideoX-5b/tokenizer \
  --image_path examples/assets/cat.jpg \
  --prompt "the cat looks around, gentle camera push-in" \
  --output_path out/cogvideox_5b_i2v.mp4
```

`5b-i2v` and `1.5-5b-i2v` use different default resolutions (720×480 vs
1360×768) and frame counts (49 vs 81) — the script picks the right default
per `--variant`. `5b-i2v` is **locked** to 720×480 by a learned positional
embedding: its conditioning frame is resized to that box and, by default
(`--match_image_aspect`), the output video is rescaled back to the
conditioning image's aspect ratio afterwards (preserving the generated pixel
budget, snapped to /16). Add `--sequence_parallel_size 4` for `1.5-5b-i2v`
at its native 1360×768 (see [Tensor & sequence parallelism](#tensor--sequence-parallelism)).

### Tensor & sequence parallelism

```bash
python examples/generate_cogvideox.py \
  --model_dir ./checkpoints/CogVideoX1.5-5B --variant 1.5-5b \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --sequence_parallel_size 4 \
  --output_path out/cogvideox_1_5_5b_t2v.mp4
```

`--tensor_parallel_size` (default: every device) Megatron-shards the DiT's
and T5's attention heads; the FFN stays replicated, which is fine since the
5B checkpoint's bf16 weights fit replicated on a single TPU v4 chip. It is
capped to a divisor of the head count (30 for 2b, 48 for the 5B models), so
2b runs `tp=2` / `dp=2` on a v4-8.

`--sequence_parallel_size` (default `1`) shards the DiT's **visual token
sequence** itself across devices (DeepSpeed-Ulysses), leaving the 226 text
tokens replicated. This is what runs CogVideoX-1.5 at its native 1360×768
(~45k visual tokens after `patch_size_t=2`, whose per-block activations
don't fit a v4 chip otherwise) — e.g. `--sequence_parallel_size 4` on a
v4-8. It must divide the device count, the head count, and the visual token
count.

For CogVideoX the two are **mutually exclusive** (`--tensor_parallel_size`
must be `1` or unset whenever `--sequence_parallel_size > 1`) — the 5B DiT
fits replicated per chip, so the port doesn't thread column/row-parallel
weight-sharding through the sequence-parallel path. See
[Hardware & Sharding](../sharding/hardware-and-sharding.md).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_dir` | *required* | Downloaded diffusers CogVideoX repo (with `transformer/ vae/ text_encoder/ tokenizer/`). |
| `--variant` | `5b` | One of `2b`, `5b`, `5b-i2v`, `1.5-5b`, `1.5-5b-i2v`. |
| `--t5_dir` / `--tokenizer_dir` | `<model_dir>/{text_encoder,tokenizer}` | Override — the `t5-v1.1-xxl` weights are byte-identical across every CogVideoX repo. |
| `--prompt` | *required*, 1+ values | One prompt (broadcast) or exactly `batch_size` prompts. |
| `--negative_prompt` | `""` | CFG negative prompt (diffusers' own default). |
| `--image_path` | `None` | Conditioning image, for the `*-i2v` variants. Omit for T2V. |
| `--match_image_aspect` / `--no_match_image_aspect` | on | I2V: rescale the output to the conditioning image's aspect ratio. |
| `--num_frames` | `49` (1.0) / `81` (1.5) | Output frame count. |
| `--height` / `--width` | per-variant reference (720×480 / 1360×768) | Must be divisible by 16 (VAE ÷8 × `patch_size=2`). Not overridable for `5b-i2v`. |
| `--num_inference_steps` | `50` | Sampling steps. |
| `--guidance_scale` | `6.0` | CFG scale. |
| `--scheduler` | `dpm` | `ddim` \| `dpm`. |
| `--use_dynamic_cfg` / `--no_dynamic_cfg` | on | diffusers' cosine CFG-scale schedule. |
| `--tensor_parallel_size` | every device | Capped to a divisor of the head count (30 / 48). Must be `1` alongside `--sequence_parallel_size > 1`. |
| `--sequence_parallel_size` | `1` | Must divide `num_devices`, the head count, and the visual token count. |
| `--dtype` / `--dit_dtype` | `bfloat16` | Activation / DiT-weight dtype. The T5 encode always runs in float32 regardless (see [Architecture notes](#architecture-notes)). |
| `--seed` | `42` | Initial noise seed. |
| `--fps` | `16` | Output video frame rate. |
| `--output_path` | `output_cogvideox.mp4` | With multiple prompts, each video is saved as `<output_path>_<i>.mp4`. |

## Architecture notes

- **DiT.** A structural port of `CogVideoXTransformer3DModel`. Each block
  runs **one joint self-attention** over the concatenated `[text(226);
  visual]` sequence — there is no cross-attention. `CogVideoXLayerNormZero`
  modulates the text and visual halves with separate shift/scale/gate
  triples sharing one `LayerNorm`. AdaLN modulation is **per-sample, not
  per-token**. RoPE is applied to the **visual slice of q/k only** — text
  tokens are never rotated. 2b has no RoPE (fixed 3D sincos instead);
  `5b-I2V` adds a learned `pos_embedding` buffer locking it to 720×480; 1.5
  uses `patch_size_t=2` temporal patchifying with a linear patch projection.
- **Text encoder.** CogVideoX conditions on `t5-v1.1-xxl` and — unlike
  LTX — passes the encoder **no attention mask**, so the full 226-token
  padded sequence is attended. T5-XXL's intermediate activations reach
  ~1e5, so a bf16 encode over that sequence loses 16–37% relative accuracy;
  the encode therefore runs in **float32** regardless of `--dtype`, with
  only the output embeddings cast to bf16 and the ~19 GB fp32 T5 params
  freed right after the one-time prompt encode.
- **VAE.** A causal 3D-conv VAE. `encode`/`decode` walk the clip in fixed
  temporal chunks carrying a causal-conv cache between chunks, exactly as
  diffusers' `_encode`/`_decode`; decode also runs in overlapping spatial
  tiles (`_tiled_decode`) — the un-tiled 512-channel 3D-conv feature maps
  OOM a v4 chip at the reference resolution.
- **I2V conditioning** follows `CogVideoXImageToVideoPipeline.prepare_latents`:
  encode the single conditioning frame, scale it, zero-pad to
  `latent_frames` → `[image_latent, zeros × (N-1)]`, and concatenate it onto
  the channel axis (→ 32) every denoising step.
- **Sequence parallelism.** CogVideoX's single joint `[text; visual]`
  attention means DeepSpeed-Ulysses can't be applied as-is —
  `sequence_parallel_joint_self_attention` sends only the visual q/k/v
  through the head↔sequence all-to-all and slices the replicated text q/k/v
  to the local head range before one local attention call. See the
  [porting postmortem](https://github.com/FlyingGiraffe/vidax/blob/main/docs/lessons/cogvideox_debugging.md).

## Status

Verified end-to-end against real checkpoints, every variant. Selected
5-run-average benchmark rows on a TPU v4-8 (see the
[Benchmark Explorer](/benchmarks) for the full, current set):

| Variant | Task | Resolution | Frames | Steps | Parallelism | s/step | Peak HBM/chip (GB) |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 2B | T2V | 720×480 | 49 | 50 | `tp=2`, `dp=2` | 4.2 | 17.2 |
| 5B | T2V | 720×480 | 49 | 50 | `tp=4` | 9.4 | 23.2 |
| 5B | I2V | 512×672 | 49 | 50 | `tp=4` | 9.4 | 23.3 |
| 1.5-5B | T2V | 1360×768 | 81 | 50 | `sp=4` | 52.8 | 31.5 |
| 1.5-5B | I2V | 896×1184 | 81 | 50 | `sp=4` | 52.8 | 31.5 |

The 1.5 rows sit right at the v4's ~31.5 GB HBM ceiling — a larger frame
count would also need weight offloading.

---

See [LTX-Video](./ltx_video.md) (whose T5-XXL tower this port reuses), or
[Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog for how CogVideoX's single joint-attention block compares to the
cross-attention DiTs.
