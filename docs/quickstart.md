---
sidebar_position: 3
title: Quickstart
---

# Quickstart

Generate your first video with Wan2.1 T2V (1.3B) on a TPU VM.

```bash
# Clone and install (editable, with TPU / torch-checkpoint-loading / tokenizer extras)
git clone https://github.com/FlyingGiraffe/vidax.git
cd vidax
pip install -e ".[tpu,torch,text]"

# Generate a video (Wan2.1 T2V, 1.3B)
python examples/generate_wan2_1_t2v.py \
  --model_size 1.3B \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 50 \
  --output_path "out/output.mp4"
```

Checkpoints (DiT `.safetensors`, VAE `.pth`, T5 `.pth`) come from the
official [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
HuggingFace repo — vidax translates them into Flax pytrees at load time, it
does not host its own weights.

## Next steps

- Pick your model family in [Model Family Guides](./models/wan2_1.md) for
  the full CLI reference (all sizes/tasks, sharding flags, checkpoint
  layouts).
- Read [Sharding & Topology Configs](./sharding/hardware-and-sharding.md)
  before deploying on anything larger than a single-host `v4-8`/`v5e-8` —
  tensor-parallel vs. sequence-parallel choice depends on model size and
  target resolution.
- Check the [Benchmark Explorer](/benchmarks) for measured latency/memory
  numbers per model, resolution, and TPU generation before picking a config.
- Browse the [blog's Engineering Notes](/blog/tags/engineering-notes) for
  real bugs found porting each model against real checkpoints — useful
  background if your own output looks subtly wrong and you want to rule out
  a known failure mode first.
