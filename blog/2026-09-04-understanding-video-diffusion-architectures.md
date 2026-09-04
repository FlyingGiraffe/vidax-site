---
slug: video-diffusion-architectures
title: "Understanding video diffusion architectures: DiT, MoT, and conditioning"
authors: [vidax-team]
tags: [engineering-notes, modeling]
date: 2026-09-04
description: >-
  A background primer on diffusion transformers, text/image conditioning,
  and Mixture-of-Transformers — and a tour of what each model family in
  vidax actually uses.
---

vidax now ports nine model families, and no two of them make exactly the
same architectural choices. Rather than write a separate deep-dive per
model, this is a tour of the handful of ideas that, combined differently,
account for most of the differences: the Diffusion Transformer (DiT) itself,
how text and image conditioning get into it, and Mixture-of-Transformers —
an architecture that isn't a DiT variant at all.

{/* truncate */}

## Diffusion Transformers (DiT)

A video diffusion model's job is to predict, given a noisy video and a
timestep, either the noise or the "velocity" needed to denoise it a step
further. A DiT does this by patchifying the (spatio-temporal) video into a
sequence of tokens and running an ordinary transformer over them, with the
timestep injected into every block via **adaptive layer norm (AdaLN)**:
the timestep gets embedded and passed through a small MLP that predicts a
per-block scale/shift/gate, applied around that block's attention and MLP
sublayers. A minimal sketch:

```python
def adaln_block(x, timestep_emb, attn, mlp, norm):
    # One small MLP per block predicts how this block should modulate
    # its own attention and MLP outputs, conditioned on the timestep.
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
        modulation_mlp(timestep_emb)
    )
    h = norm(x) * (1 + scale_attn) + shift_attn
    x = x + gate_attn * attn(h)

    h = norm(x) * (1 + scale_mlp) + shift_mlp
    x = x + gate_mlp * mlp(h)
    return x
```

This is the shape every DiT in vidax follows for its diffusion timestep —
Wan's, Cosmos-Predict2.5's — though the details (how many separate
modulation vectors, whether it's per-sample or per-token, the exact
RoPE convention for positions) all differ. One family, Cosmos3, skips AdaLN
entirely — more on that below.

## Text conditioning: cross-attention

Text almost always enters through ordinary **cross-attention**: video
tokens as the query, text-encoder tokens as the key/value, inserted as an
extra sublayer per block alongside self-attention:

```python
def cross_attend_to_text(video_tokens, text_tokens, cross_attn):
    q = project_q(video_tokens)
    k, v = project_kv(text_tokens)
    return cross_attn(q, k, v)  # video tokens attend over text tokens
```

What differs across models is *which* text encoder produces `text_tokens`
— a small T5 variant, or a multi-billion-parameter vision-language model —
and, for Cosmos-Predict2.5, *how many* of that encoder's layers get used
(concatenating hidden states from every decoder layer, not just the last).

### The alternative: put conditioning tokens in the sequence, not a side branch

Not every model uses cross-attention for text at all. HunyuanVideo 1.0 and
1.5 instead **concatenate** their text tokens directly into the same
self-attention sequence the video tokens live in, tagged by a learned
per-source embedding so the model can tell which tokens came from where. In
1.5 that's two separate towers — a 7B vision-language model and a small
glyph/color encoder — plus, for I2V, SigLIP vision tokens; 1.0 is simpler,
a single Llama-3-8B tower's hidden states (with a separate pooled CLIP-L
vector folded into the AdaLN modulation, not the sequence). Either way
there's no separate cross-attention sublayer — conditioning happens because
every video token's self-attention already sees the text tokens sitting
right next to it in the sequence.

CogVideoX takes the same idea to its minimal form: **one** joint
self-attention per block over `[text (226 tokens); video]`, no
cross-attention sublayer and no second tower — just T5-XXL text embeddings
prepended to the video sequence, with RoPE applied to the video slice only.
Cosmos3's MoT pathway (below) is another variant: text and video tokens
share one sequence, just processed by different weight sets depending which
pathway they belong to.

## Image/video conditioning (I2V)

Conditioning on a starting image or clip (rather than pure text) shows up
in vidax in two genuinely different forms:

**Cross-attention to a separate image embedding.** Wan2.1's I2V model
adds a *second* cross-attention branch per block, attending to a CLIP
embedding of the conditioning image — structurally identical to the text
cross-attention above, just with a different key/value source.

**Latent substitution.** Wan2.2's TI2V-5B and Cosmos-Predict2.5 instead
VAE-encode the conditioning frames, then substitute those known latents
directly into the noisy input at the corresponding positions, at every
sampling step — paired with an extra mask channel marking which frames are
"given" versus "to be generated":

```python
def substitute_known_frames(x, known_latents, is_conditioning_frame):
    # is_conditioning_frame: per-frame boolean mask
    return jnp.where(is_conditioning_frame, known_latents, x)
```

No extra cross-attention branch needed — conditioning is expressed purely
through what's already in the input, plus the mask channel telling the
model which parts of that input to trust as ground truth.

## Mixture-of-Transformers: not sparse MoE

Cosmos3 is the one model family in vidax that isn't a DiT at all. It's an
omnimodal **Mixture-of-Transformers (MoT)** — a term that's easy to
confuse with sparse Mixture-of-Experts (MoE), but describes something
different. Sparse MoE routes individual tokens to a subset of expert
weights via a learned gate. MoT instead gives each **modality** its own
full weight set, used by every token of that modality, all the time — no
routing, no gate:

```python
def mot_layer(tokens, is_text, text_weights, video_weights):
    # every text token always uses text_weights; every video token
    # always uses video_weights -- decided by modality, not learned
    text_out = transformer_block(tokens, text_weights) 
    video_out = transformer_block(tokens, video_weights)
    return jnp.where(is_text, text_out, video_out)
```

Cosmos3 pairs this with a one-directional attention pattern: a causal
text ("understanding") pathway that self-attends normally, and a
full-attention "generation" (diffusion) pathway that attends over *both*
its own tokens and the text pathway's — so text conditions video, but
video never feeds back into text. And unlike every DiT above, there's no
AdaLN anywhere: the timestep is embedded and added directly into the noisy
video tokens once, then flows through ordinary pre-norm transformer blocks.

## What each model in vidax actually uses

| Model | Architecture | Text conditioning | Image/video conditioning | Sampler |
| --- | --- | --- | --- | --- |
| Wan2.1 | DiT | T5 (UMT5) cross-attention | Separate CLIP cross-attention branch | Euler |
| Wan2.2 | DiT (A14B: 2-expert MoE by timestep) | T5 (UMT5) cross-attention | Latent substitution + mask (TI2V-5B) | Euler |
| Cosmos-Predict2.5 | DiT, AdaLN-LoRA | Reason1 (7B VLM), all-layer concat | Latent substitution + mask + per-frame timestep | UniPC |
| Cosmos3 | Mixture-of-Transformers, no AdaLN | Shared packed sequence (own MoT pathway) | Same packed-sequence mechanism as text | UniPC (Karras) |
| LTX-Video | DiT | T5-XXL cross-attention | Latent lerp + per-token effective-timestep clamp | Rectified Flow (LinearQuadratic Euler) |
| LTX-2.5 | DiT, cross-attention AdaLN + gated attention | Gemma-4 (12B), via a learned connector | Per-token "denoise mask" (a generalized form of LTX-Video's clamp) | Ancestral (SDE) or plain Euler, by checkpoint |
| HunyuanVideo-1.5 | Dual-stream/single-stream MMDiT | Qwen2.5-VL (7B) + byT5 glyph encoder, token concatenation | Channel-concat latent + SigLIP tokens, concatenated into the sequence | Rectified Flow (Euler) |
| HunyuanVideo (1.0) | Dual-stream/single-stream MMDiT (same family as 1.5) | Llama-3-8B token concatenation + pooled CLIP-L into AdaLN | `token_replace`: latent substitution + a second "as-if-t=0" AdaLN vector for the first frame | Rectified Flow (Euler) |
| CogVideoX / 1.5 | DiT, one joint `[text; video]` self-attention, per-sample AdaLN | T5-XXL, prepended to the sequence (no cross-attention) | Channel-concat image latent (→ 32 ch), every step | DDIM / DPM-Solver++ (v-prediction, zero-terminal-SNR) |

LTX-2.5 also has a second, genuinely different VAE decoder option: instead
of the usual deterministic convolutional decoder, `--vae_variant diffusion`
runs a small transformer that samples its own internal denoising step
directly on pixels — a different design from every other model's VAE here.

## A few things worth knowing

A handful of debugging lessons from getting these models running turned
out to be useful *diagnostic technique*, not just "here's a bug we found" —
worth keeping in mind for any diffusion model, not just the one that
surfaced them.

**A low-noise reconstruction probe separates "wrong noise-level handling"
from "wrong weights."** When Cosmos-Predict2.5's early output came out as
a rigid grid of scrambled color blobs, the diagnostic that actually found
the cause wasn't staring at attention or RoPE code — it was encoding a
real photo with the VAE, adding a *small* amount of noise, running one
forward pass, and decoding. That reconstructed the photo almost perfectly
at low noise and degraded into the same grid-like garbage at high noise,
which pointed straight at *noise-level conditioning* (the model was being
told the wrong timestep, via a mistakenly-added preconditioning
transform) rather than the network's weights or architecture. Any
diffusion model producing texture-like or grid-like garbage is worth
testing this way before assuming the bug is somewhere more exotic.

**Generically-plausible-but-prompt-disconnected output usually means the
conditioning signal is weak, not that CFG/precision is wrong.** LTX-2.5
once produced video that looked structurally fine but kept drifting
toward an unrelated recurring motif regardless of prompt or seed. The
cause was a rescale bug in the text-feature extractor that fed the DiT a
conditioning signal roughly 7x weaker than intended — not NaN, not
wrong-shaped, just quietly under-scaled, so the model's own generative
prior dominated. When output is "fine but ignores the prompt," it's worth
checking the conditioning path itself before spending time on CFG scale,
guidance rescale, or precision — a weak signal produces exactly the kind
of plausible-but-generic output that's easy to misread as a model-quality
ceiling instead of a bug.

**A bf16 text encoder is only safe if the padding is masked.** CogVideoX
feeds its T5-XXL encoder no attention mask (the reference doesn't either),
so the full 226-token padded sequence is attended. T5-XXL's residual
stream reaches magnitudes around 1e5 — about two significant digits in
bfloat16 — and over 24 layers × 226 unmasked positions the error compounds
to 16–37% relative, with JAX-bf16 and torch-bf16 diverging from *each
other* by that much too. Per-block parity against an fp32 input stays at
~1e-7; it's purely the bf16 residual stream that blows up. LTX-Video never
hit this because it always passes the mask, so the pad positions get a
`-inf` bias and never enter the sum. The fix isn't bf16 parity — the
reference is unstable in bf16 too — it's running that one-time prompt
encode in float32 and casting only the output embeddings down.

**Prompt formatting requirements matter more for smaller models.** Both
Cosmos3 checkpoints document that prompts should be "upsampled" into a
structured format (subjects, setting, lighting, cinematography, a
temporal caption) rather than passed as a short sentence. Nano (16B)
tolerates a short prompt reasonably well; Edge (4B) does not — the same
short prompt that's fine on Nano produces flat, oversaturated,
featureless output on Edge, with nothing in the pipeline (no error, no
broken shape) to flag that the prompt itself is the limiting factor.
Documented prompt-format requirements are easy to treat as polish; for a
smaller model they can be the difference between unusable and excellent
output.

---

See [Sharding, parallelism, and JIT on TPUs](/blog/hardware-and-sharding)
and [Weight offloading](/blog/weight-offloading) for how these
architectures actually get run at scale, or the model guides under
[Model Family Guides](/docs/models/wan2_1) for checkpoints and CLI usage.
