---
sidebar_position: 8
title: HunyuanVideo (1.0)
---

# HunyuanVideo (1.0)

HunyuanVideo 1.0 ships T2V and I2V as **genuinely separate** checkpoints
(`hunyuan-video-t2v-720p` vs `hunyuan-video-i2v-720p`) with different text
encoders, so `examples/` has one standalone TPU script per task:

| Script | Params | Task | Checkpoint | Text encoder |
| --- | --- | --- | --- | --- |
| `generate_hunyuan_video.py` | 13B DiT | T2V | `hunyuan-video-t2v-720p/` | Llama3-8B (`.language_model` only) + CLIP-L |
| `generate_hunyuan_video_i2v.py` | 13B DiT | I2V | `hunyuan-video-i2v-720p/` | full multimodal LLaVA-Llama3-8B + CLIP-L |

Both are the `"HYVideo-T/2-cfgdistill"` preset. I2V is implemented in the
reference's `token_replace` mode (the shipped checkpoint's default);
`latent_concat`, its other I2V mode, is not ported.

The DiT is the **same dual-stream (×20) + single-stream (×40) MMDiT family
as [HunyuanVideo-1.5](./hunyuan_video_1_5.md)** — every block, RoPE, and
AdaLN building block is reused unmodified from that port. What differs: a
real `(1,2,2)` spatial patchify (1.5 is `(1,1,1)`), a single-LLM +
pooled-CLIP-L text stack (no byT5 / SigLIP token-concatenation), a
GroupNorm causal-conv VAE, and fused-QKV checkpoint Linears that the
translator splits on load.

Everything it needs (`torch`, `transformers`, `pillow`) is installed by
default. On a Cloud TPU VM also add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

## Text-to-video — `generate_hunyuan_video.py`

`--checkpoint_dir` points at
[`tencent/HunyuanVideo`](https://huggingface.co/tencent/HunyuanVideo)'s
downloaded root; `--clip_checkpoint_dir` at
[`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14).
The Llama text encoder must be **extracted** from
`xtuner/llava-llama-3-8b-v1_1-transformers` first (see below).

```bash
python examples/generate_hunyuan_video.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo" \
  --text_encoder_dir "./checkpoints/HunyuanVideo/text_encoder" \
  --clip_checkpoint_dir "./checkpoints/HunyuanVideo/clip-vit-large-patch14" \
  --prompt "A golden retriever running on a beach at sunset, cinematic, high detail" \
  --height 720 --width 1280 --num_frames 129 --num_steps 50 \
  --tensor_parallel_size 4 \
  --output_path "out/output_hunyuan_1_t2v.mp4"
```

### Preparing the Llama text encoder

Unlike HunyuanVideo-1.5's Qwen2.5-VL tower (downloaded pre-packaged),
HunyuanVideo 1.0's LLM text encoder must be extracted from the full
`xtuner/llava-llama-3-8b-v1_1-transformers` checkpoint — only the
`.language_model` sub-module (a plain `LlamaModel`, no vision tower) is
ever used:

```bash
python -c "
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
m = LlavaForConditionalGeneration.from_pretrained('xtuner/llava-llama-3-8b-v1_1-transformers', dtype=torch.float32, low_cpu_mem_usage=True)
p = AutoProcessor.from_pretrained('xtuner/llava-llama-3-8b-v1_1-transformers')
m.language_model.save_pretrained('./checkpoints/HunyuanVideo/text_encoder')
p.tokenizer.save_pretrained('./checkpoints/HunyuanVideo/text_encoder')
"
```

(The reference's own preprocessing script does the same thing but
unconditionally calls `.to(0)` — a GPU-only assumption; the snippet above is
the CPU-portable equivalent.) The full `llava-llama-3-8b-v1_1-transformers`
download (~17 GB) can be deleted once the extracted `text_encoder/`
directory is confirmed to load — only the extracted tower (~16 GB) is needed
at inference time.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_dir` | *required* | `tencent/HunyuanVideo`'s downloaded root. |
| `--text_encoder_dir` | *required* | The extracted Llama text-decoder tower (see above). |
| `--clip_checkpoint_dir` | *required* | `openai/clip-vit-large-patch14`'s downloaded root. |
| `--model` | `HYVideo-T/2-cfgdistill` | Named hyperparameter preset — the released checkpoint is the `guidance_embed=True` cfgdistill variant. |
| `--prompt` | *required* | Text prompt. |
| `--negative_prompt` | `None` | Defaults to the reference's own `NEGATIVE_PROMPT` when `--guidance_scale != 1.0`, else empty. |
| `--height` / `--width` | `720` / `1280` | Output resolution in pixels (aligned to 16). |
| `--num_frames` | `129` | Must be `1 + 4k` (VAE temporal compression 4). |
| `--num_steps` | `50` | Flow-matching Euler sampling steps. |
| `--shift` | `7.0` | Flow-match schedule shift. |
| `--guidance_scale` | `1.0` | Real CFG. Default `1.0` (off) matches the reference — the embedded/distilled guidance is the primary mechanism. |
| `--embedded_guidance_scale` | `6.0` | Embedded/distilled guidance fed to `guidance_in`. |
| `--dtype` / `--dit_dtype` | `bfloat16` | Activation / DiT-weight dtype (checkpoint ships float32/bf16-mixed). |
| `--tensor_parallel_size` | every local device | Megatron-shards the DiT's double/single-stream Q/K/V/output/FFN Dense layers. Must divide `heads_num` (24). Required in practice — the 13B DiT doesn't fit replicated on one TPU v4 chip. |
| `--offload_dit_weights` / `--offload_chunk_size_{double,single}` | off / `20` / `40` | Per-layer weight offloading — required to fit the reference's real 129-frame/720p default in HBM. See [Weight Offloading](../sharding/weight-offloading.md). |
| `--vae_tile_latent_size` | reference default | Shrink (e.g. `8`) if VAE decode OOMs. |
| `--fps` | `24` | Output video frame rate. |
| `--output_path` | `output.mp4` | Output video path. |

## Image-to-video — `generate_hunyuan_video_i2v.py`

A **separate script** (separate checkpoint, separate text encoder).
Conditioning works by `token_replace`: the reference image's own clean VAE
latent literally replaces the first latent frame before every sampling step,
and the DiT's AdaLN modulation uses a second "as-if-t=0" vector for that
first frame's tokens. Text conditioning is the **full multimodal LLaVA
model** — the reference image goes through a CLIP ViT-L/14-336 vision tower
+ a 2-layer projector and is spliced into the Llama decoder's input
embeddings at the `<image>` placeholder positions.

```bash
python examples/generate_hunyuan_video_i2v.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo-I2V" \
  --llava_checkpoint_dir "./checkpoints/llava-llama-3-8b-v1_1-transformers" \
  --clip_checkpoint_dir "./checkpoints/clip-vit-large-patch14" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "The cat stretches and walks across the windowsill, cinematic" \
  --i2v_resolution 720p --num_frames 129 --num_steps 50 \
  --tensor_parallel_size 4 \
  --output_path "out/output_hunyuan_1_i2v.mp4"
```

`--llava_checkpoint_dir` points at the **full** (not `.language_model`-only)
`xtuner/llava-llama-3-8b-v1_1-transformers` download.

### CLI reference (I2V-specific)

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_dir` | *required* | `tencent/HunyuanVideo-I2V`'s downloaded root. |
| `--vae_checkpoint_dir` | `--checkpoint_dir` | The I2V VAE checkpoint is byte-identical to T2V's. |
| `--llava_checkpoint_dir` | *required* | The **full** LLaVA root (vision tower + projector + language model). |
| `--clip_checkpoint_dir` | *required* | `openai/clip-vit-large-patch14` — pooled CLIP-L, used unconditionally. |
| `--image_path` | *required* | Reference/conditioning image. |
| `--i2v_resolution` | `720p` | Resolution bucket (`720p`/`540p`/`360p`); output H/W are the reference image's aspect ratio snapped to the bucket's candidate sizes. |
| `--num_frames` | `129` | Must be `1 + 4k`. |
| `--shift` | `17.0` | I2V's real flow-match shift — **not** T2V's 7.0. |
| `--guidance_scale` | `1.0` | Real CFG, off by default. |
| `--embedded_guidance_scale` | `6.0` | Fed to `guidance_in`. |
| `--image_embed_interleave` | `4` | `token_replace`'s real value — subsamples every Nth projected image-patch row before splicing into the text-state sequence. |
| `--tensor_parallel_size` | every local device | Must divide `heads_num` (24). |
| `--offload_dit_weights` / `--offload_chunk_size_{double,single}` | off / `20` / `40` | Same per-layer offloading as the T2V script. |
| `--vae_tile_latent_size` | reference default | Shrink if VAE decode OOMs. |
| `--output_path` | `output.mp4` | Output video path. |

## Architecture notes

- **DiT.** The same `MMDoubleStreamBlock` ×20 + `MMSingleStreamBlock` ×40
  MMDiT as HunyuanVideo-1.5 (`hidden_size=3072`, `heads_num=24`,
  `mlp_width_ratio=4`, `gelu_tanh` MLP, `qk_norm=True`, zero-init AdaLN,
  RoPE `theta=256`) — reused **unmodified** from the 1.5 port. The one
  behavioural difference is a real `patch_size=(1,2,2)` spatial downsample
  (1.5 uses `(1,1,1)`), a reshape-then-`Dense` that is mathematically exact
  because the reference's `Conv3d`(kernel==stride) patches don't overlap.
- **Text conditioning is a single LLM + a separate pooled CLIP-L vector.**
  The Llama tower's hidden states (`text_states_dim=4096`,
  `hidden_state_skip_layer=2`, `crop_start=95`) are refined through a
  `SingleTokenRefiner` and become the DiT's *entire* `txt` sequence — no
  byT5, no SigLIP, no token-stream reordering. A separate pooled CLIP-L
  vector feeds `vector_in` into the AdaLN modulation vector alongside the
  timestep embedding.
- **Text encoders.** The Llama3-8B decoder tower (`hidden_size=4096`, 32
  layers, 32 query / 8 KV heads, `rope_theta=500000`, `vocab_size=128320`)
  is a fresh small port — not a reuse of Cosmos's Qwen2 tower, whose
  hardcoded q/k/v-proj bias convention this checkpoint doesn't share. CLIP-L
  is a fresh standard pre-LN CLIP *text* tower (`hidden_size=768`, 12
  layers), pooled at the EOS position.
- **VAE (`"884-16c-hy"`).** diffusers' standard `AutoencoderKLCausal3D` —
  GroupNorm (not RMSNorm), plain strided `CausalConv3d` down/upsample (no
  pixel-(un)shuffle, unlike 1.5's VAE), a single-head causal `Attention`
  mid-block. `block_out_channels=[128,256,512,512]`, `latent_channels=16`,
  `scaling_factor=0.476986`, spatial ÷8 / temporal ÷4. Only spatial tiling
  is implemented (matching the reference).
- **Guidance embedding.** The released checkpoint is the cfgdistill
  variant (`guidance_embed=True`); `guidance_in` is always exercised, fed
  `embedded_guidance_scale * 1000`. Real CFG is supported on top of this but
  defaults off.
- **Checkpoint translator.** The DiT's one structural difference from 1.5's
  mapper: 1.0's checkpoint stores **fused** QKV Linears
  (`img_attn_qkv` / `txt_attn_qkv` / `single_blocks.N.linear1`) — split into
  contiguous per-projection chunks before writing to the Flax tree. See the
  [porting postmortem](https://github.com/FlyingGiraffe/vidax/blob/main/docs/lessons/hunyuan_video_debugging.md).

## Status

Verified end-to-end against real checkpoints, T2V and I2V. 5-run-average
benchmark rows on a TPU v4-8 (see the [Benchmark Explorer](/benchmarks) for
the full, current set):

| Task | Resolution | Frames | Steps | Parallelism | s/step | Peak HBM/chip (GB) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| T2V | 1280×720 | 129 | 50 | `tp=4` + offload (`chunk 20/40`) | 299.6 | 18.4 |
| I2V | 832×1088 | 129 | 50 | `tp=4` + offload (`chunk 20/40`) | 306.4 | 18.4 |

The per-step cost is dominated by the offloading tax plus this
architecture's joint global self-attention over the full image+text token
sequence (no windowing) — HunyuanVideo-1.5, the same block family with
*zero* offloading, is already the slowest non-offloaded row in the
benchmark table.

---

See [HunyuanVideo-1.5](./hunyuan_video_1_5.md) for the newer 8.3B model in
the same block family, or
[Understanding video diffusion architectures](/blog/video-diffusion-architectures)
on the blog for background on the dual-stream / single-stream MMDiT design.
