export type VideoTask = 'T2V' | 'I2V';

export interface GalleryVideo {
  id: string;
  family: string;
  model: string;
  task: VideoTask;
  resolution: string;
  /** Encoded frame width/height, measured directly from the .mp4 (not the
   * benchmark config value). Kept for reference / the resolution label;
   * the gallery now crops every card to one fixed aspect ratio per task
   * (16:9 for T2V, 3:4 for I2V) via `object-fit: cover`. */
  width: number;
  height: number;
  src: string;
  poster: string;
  samplingTimeS: number;
  /** Brief description of the *model* that produced this clip (not the
   * input prompt/image -- every clip uses the same conditioning inputs,
   * see examples/assets/ in vidax). */
  description: string;
}

// Demo clips copied from vidax/out/**/*.mp4 (run 1 of each combo's 5-run
// benchmark set) into static/videos/. Sampling time is each combo's
// `avg.wall_s` from src/data/benchmarks.json — regenerate both together via
// `npm run gen-benchmarks` if vidax's benchmark results change.
// width/height/resolution are measured directly from the encoded video
// (via ffmpeg), not copied from the benchmark config, since I2V output
// resolution is only known at runtime from the conditioning image.
// Posters are first-frame JPEGs in static/img/posters/, used so the
// gallery loads instantly without fetching any video bytes until a clip
// is actually played (see VideoGallery's preload="none").
// TODO: add Cosmos2.5 I2V/V2V and Cosmos3 I2V clips once those runs exist.

// Each ~2-3 lines at a gallery card's width; the card clamps to 3 lines and
// reserves that height so every card aligns whether the text runs 2 or 3.
const WAN2_1_DESC =
  'A cross-attention DiT conditioned by a UMT5-XXL text encoder and sampled with a Rectified-Flow Euler scheduler. The I2V variant adds a separate CLIP-image cross-attention branch per block.';
const WAN2_2_DESC =
  "Wan's DiT rebuilt as a two-expert Mixture-of-Experts routed by timestep, with per-token (not per-sample) modulation. TI2V-5B is instead one dense model, sharing a single checkpoint across T2V and I2V.";
const COSMOS2_5_DESC =
  'A DiT with AdaLN-LoRA modulation, conditioned by a 7B Qwen2.5-VL ("Reason1") text tower with every decoder layer concatenated, and sampled with a flow-matching UniPC multistep solver.';
const COSMOS3_DESC =
  'An omnimodal Mixture-of-Transformers — a full weight set per modality — with dual causal / full-attention pathways, interleaved 3D mRoPE, no AdaLN, and a Karras-sigma UniPC sampler.';
const COGVIDEOX_DESC =
  'A DiT running one joint self-attention over [text; video] with no cross-attention, per-sample AdaLN, 3D RoPE (fixed 3D sincos on 2B), a causal 3D-conv VAE, and a T5-v1.1-XXL text encoder. v-prediction with zero-terminal-SNR.';
const COGVIDEOX1_5_DESC =
  'CogVideoX with temporal patchification (patch_size_t=2), a "slice"-grid 3D RoPE, and a native 1360×768 / 81-frame recipe run under DeepSpeed-Ulysses sequence parallelism.';
const HUNYUAN_VIDEO_DESC =
  '13B dual-stream + single-stream MMDiT with joint full-sequence self-attention, conditioned by a Llama-3-8B tower plus a pooled CLIP-L vector folded into AdaLN. Flow-matching Euler; token_replace for I2V.';
const HUNYUAN_VIDEO_1_5_DESC =
  '8.3B MMDiT that conditions by concatenating Qwen2.5-VL and byT5 text tokens directly onto the video sequence — no cross-attention. SigLIP adds the reference frame for I2V. Rectified-Flow Euler.';
const LTX_VIDEO_DESC =
  'A from-scratch LTX port: a pixel-unshuffle-patchify VAE with a noise-conditioned decoder, a plain T5-XXL text encoder, and a Rectified-Flow Euler sampler on a linear-quadratic sigma schedule.';
const LTX2_5_DESC =
  'A 22B LTX DiT with cross-attention AdaLN and per-head gated attention, an 8-layer embeddings connector, a Gemma-4 12B text encoder, and an ancestral (SDE) Euler sampler.';
const LTX2_5_DIFFVAE_DESC =
  'A 22B LTX DiT (cross-attention AdaLN, per-head gated attention, Gemma-4 text encoder), here decoded with the slower transformer / neighborhood-attention diffusion VAE rather than the conv decoder.';

export const GALLERY_VIDEOS: GalleryVideo[] = [
  {
    id: 'wan2_1_1.3b_t2v',
    family: 'Wan2.1',
    model: '1.3B',
    task: 'T2V',
    resolution: '832x480',
    width: 832,
    height: 480,
    src: '/videos/wan2_1_1.3b_t2v_1.mp4',
    poster: '/img/posters/wan2_1_1.3b_t2v_1.jpg',
    samplingTimeS: 456.39,
    description: WAN2_1_DESC,
  },
  {
    id: 'wan2_1_14b_480p_t2v',
    family: 'Wan2.1',
    model: '14B',
    task: 'T2V',
    resolution: '832x480',
    width: 832,
    height: 480,
    src: '/videos/wan2_1_14b_480p_t2v_1.mp4',
    poster: '/img/posters/wan2_1_14b_480p_t2v_1.jpg',
    samplingTimeS: 1552.89,
    description: WAN2_1_DESC,
  },
  {
    id: 'wan2_1_14b_720p_t2v',
    family: 'Wan2.1',
    model: '14B',
    task: 'T2V',
    resolution: '1280x720',
    width: 1280,
    height: 720,
    src: '/videos/wan2_1_14b_720p_t2v_1.mp4',
    poster: '/img/posters/wan2_1_14b_720p_t2v_1.jpg',
    samplingTimeS: 6562.93,
    description: WAN2_1_DESC,
  },
  {
    id: 'wan2_1_14b_480p_i2v',
    family: 'Wan2.1',
    model: '14B',
    task: 'I2V',
    resolution: '544x720',
    width: 544,
    height: 720,
    src: '/videos/wan2_1_14b_480p_i2v_1.mp4',
    poster: '/img/posters/wan2_1_14b_480p_i2v_1.jpg',
    samplingTimeS: 1372.13,
    description: WAN2_1_DESC,
  },
  {
    id: 'wan2_1_14b_720p_i2v',
    family: 'Wan2.1',
    model: '14B',
    task: 'I2V',
    resolution: '832x1104',
    width: 832,
    height: 1104,
    src: '/videos/wan2_1_14b_720p_i2v_1.mp4',
    poster: '/img/posters/wan2_1_14b_720p_i2v_1.jpg',
    samplingTimeS: 5651.55,
    description: WAN2_1_DESC,
  },
  {
    id: 'wan2_2_5b-ti2v_t2v',
    family: 'Wan2.2',
    model: 'TI2V-5B',
    task: 'T2V',
    resolution: '1280x704',
    width: 1280,
    height: 704,
    src: '/videos/wan2_2_5b-ti2v_t2v_1.mp4',
    poster: '/img/posters/wan2_2_5b-ti2v_t2v_1.jpg',
    samplingTimeS: 649.66,
    description: WAN2_2_DESC,
  },
  {
    id: 'wan2_2_5b-ti2v_i2v',
    family: 'Wan2.2',
    model: 'TI2V-5B',
    task: 'I2V',
    resolution: '800x1088',
    width: 800,
    height: 1088,
    src: '/videos/wan2_2_5b-ti2v_i2v_1.mp4',
    poster: '/img/posters/wan2_2_5b-ti2v_i2v_1.jpg',
    samplingTimeS: 665.44,
    description: WAN2_2_DESC,
  },
  {
    id: 'wan2_2_a14b_t2v',
    family: 'Wan2.2',
    model: 'A14B',
    task: 'T2V',
    resolution: '832x480',
    width: 832,
    height: 480,
    src: '/videos/wan2_2_a14b_t2v_1.mp4',
    poster: '/img/posters/wan2_2_a14b_t2v_1.jpg',
    samplingTimeS: 2408.18,
    description: WAN2_2_DESC,
  },
  {
    id: 'wan2_2_a14b_i2v',
    family: 'Wan2.2',
    model: 'A14B',
    task: 'I2V',
    resolution: '544x720',
    width: 544,
    height: 720,
    src: '/videos/wan2_2_a14b_i2v_1.mp4',
    poster: '/img/posters/wan2_2_a14b_i2v_1.jpg',
    samplingTimeS: 2063.17,
    description: WAN2_2_DESC,
  },
  {
    id: 'wan2_2_a14b_720p_i2v',
    family: 'Wan2.2',
    model: 'A14B',
    task: 'I2V',
    resolution: '832x1104',
    width: 832,
    height: 1104,
    src: '/videos/wan2_2_a14b_720p_i2v_1.mp4',
    poster: '/img/posters/wan2_2_a14b_720p_i2v_1.jpg',
    samplingTimeS: 2163.74,
    description: WAN2_2_DESC,
  },
  {
    id: 'cosmos2_5_2b_t2v',
    family: 'Cosmos-Predict2.5',
    model: '2B',
    task: 'T2V',
    resolution: '1280x704',
    width: 1280,
    height: 704,
    src: '/videos/cosmos2_5_2b_t2v_1.mp4',
    poster: '/img/posters/cosmos2_5_2b_t2v_1.jpg',
    samplingTimeS: 1497.28,
    description: COSMOS2_5_DESC,
  },
  {
    id: 'cosmos2_5_14b_t2v',
    family: 'Cosmos-Predict2.5',
    model: '14B',
    task: 'T2V',
    resolution: '1280x704',
    width: 1280,
    height: 704,
    src: '/videos/cosmos2_5_14b_t2v_1.mp4',
    poster: '/img/posters/cosmos2_5_14b_t2v_1.jpg',
    samplingTimeS: 4577.11,
    description: COSMOS2_5_DESC,
  },
  {
    id: 'cosmos3_nano_t2v',
    family: 'Cosmos3',
    model: 'Nano (16B)',
    task: 'T2V',
    resolution: '1280x704',
    width: 1280,
    height: 704,
    src: '/videos/cosmos3_nano_t2v_1.mp4',
    poster: '/img/posters/cosmos3_nano_t2v_1.jpg',
    samplingTimeS: 437.51,
    description: COSMOS3_DESC,
  },
  {
    id: 'cosmos3_edge_t2v',
    family: 'Cosmos3',
    model: 'Edge (4B)',
    task: 'T2V',
    resolution: '832x480',
    width: 832,
    height: 480,
    src: '/videos/cosmos3_edge_t2v_1.mp4',
    poster: '/img/posters/cosmos3_edge_t2v_1.jpg',
    samplingTimeS: 274.81,
    description: COSMOS3_DESC,
  },
  {
    id: 'ltx_video0_9_8_2b_distilled_t2v',
    family: 'LTX-Video',
    model: '2B distilled',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx_video0_9_8_2b_distilled_t2v_1.mp4',
    poster: '/img/posters/ltx_video0_9_8_2b_distilled_t2v_1.jpg',
    samplingTimeS: 162.62,
    description: LTX_VIDEO_DESC,
  },
  {
    id: 'ltx_video0_9_8_13b_dev_t2v',
    family: 'LTX-Video',
    model: '13B dev',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx_video0_9_8_13b_dev_t2v_1.mp4',
    poster: '/img/posters/ltx_video0_9_8_13b_dev_t2v_1.jpg',
    samplingTimeS: 347.59,
    description: LTX_VIDEO_DESC,
  },
  {
    id: 'ltx_video0_9_8_13b_distilled_t2v',
    family: 'LTX-Video',
    model: '13B distilled',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx_video0_9_8_13b_distilled_t2v_1.mp4',
    poster: '/img/posters/ltx_video0_9_8_13b_distilled_t2v_1.jpg',
    samplingTimeS: 300.57,
    description: LTX_VIDEO_DESC,
  },
  {
    id: 'ltx2_5_22b_dev_t2v',
    family: 'LTX-2.5',
    model: '22B dev',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx2_5_22b_dev_t2v_1.mp4',
    poster: '/img/posters/ltx2_5_22b_dev_t2v_1.jpg',
    samplingTimeS: 409.69,
    description: LTX2_5_DESC,
  },
  {
    id: 'ltx2_5_22b_distilled_t2v',
    family: 'LTX-2.5',
    model: '22B distilled',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx2_5_22b_distilled_t2v_1.mp4',
    poster: '/img/posters/ltx2_5_22b_distilled_t2v_1.jpg',
    samplingTimeS: 214.98,
    description: LTX2_5_DESC,
  },
  {
    id: 'ltx2_5_22b_dev_diffvae_t2v',
    family: 'LTX-2.5',
    model: '22B dev, diffusion VAE',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx2_5_22b_dev_diffvae_t2v_1.mp4',
    poster: '/img/posters/ltx2_5_22b_dev_diffvae_t2v_1.jpg',
    samplingTimeS: 3444.33,
    description: LTX2_5_DIFFVAE_DESC,
  },
  {
    id: 'ltx2_5_22b_distilled_diffvae_t2v',
    family: 'LTX-2.5',
    model: '22B distilled, diffusion VAE',
    task: 'T2V',
    resolution: '1216x704',
    width: 1216,
    height: 704,
    src: '/videos/ltx2_5_22b_distilled_diffvae_t2v_1.mp4',
    poster: '/img/posters/ltx2_5_22b_distilled_diffvae_t2v_1.jpg',
    samplingTimeS: 3247.36,
    description: LTX2_5_DIFFVAE_DESC,
  },
  {
    id: 'hunyuan_video1_5_480p_t2v',
    family: 'HunyuanVideo-1.5',
    model: '8.3B (480p)',
    task: 'T2V',
    resolution: '832x480',
    width: 832,
    height: 480,
    src: '/videos/hunyuan_video1_5_480p_t2v_1.mp4',
    poster: '/img/posters/hunyuan_video1_5_480p_t2v_1.jpg',
    samplingTimeS: 4015.23,
    description: HUNYUAN_VIDEO_1_5_DESC,
  },
  {
    id: 'hunyuan_video1_5_720p_t2v',
    family: 'HunyuanVideo-1.5',
    model: '8.3B (720p)',
    task: 'T2V',
    resolution: '1280x720',
    width: 1280,
    height: 720,
    src: '/videos/hunyuan_video1_5_720p_t2v_1.mp4',
    poster: '/img/posters/hunyuan_video1_5_720p_t2v_1.jpg',
    samplingTimeS: 7133.48,
    description: HUNYUAN_VIDEO_1_5_DESC,
  },
  {
    id: 'hunyuan_video1_5_480p_i2v',
    family: 'HunyuanVideo-1.5',
    model: '8.3B (480p)',
    task: 'I2V',
    resolution: '544x720',
    width: 544,
    height: 720,
    src: '/videos/hunyuan_video1_5_480p_i2v_1.mp4',
    poster: '/img/posters/hunyuan_video1_5_480p_i2v_1.jpg',
    samplingTimeS: 3838.14,
    description: HUNYUAN_VIDEO_1_5_DESC,
  },
  {
    id: 'hunyuan_video1_5_720p_i2v',
    family: 'HunyuanVideo-1.5',
    model: '8.3B (720p)',
    task: 'I2V',
    resolution: '832x1104',
    width: 832,
    height: 1104,
    src: '/videos/hunyuan_video1_5_720p_i2v_1.mp4',
    poster: '/img/posters/hunyuan_video1_5_720p_i2v_1.jpg',
    samplingTimeS: 7125.03,
    description: HUNYUAN_VIDEO_1_5_DESC,
  },
  {
    id: 'hunyuan_video_720p_t2v',
    family: 'HunyuanVideo',
    model: '13B',
    task: 'T2V',
    resolution: '1280x720',
    width: 1280,
    height: 720,
    src: '/videos/hunyuan_video_720p_t2v_1.mp4',
    poster: '/img/posters/hunyuan_video_720p_t2v_1.jpg',
    samplingTimeS: 15620.67,
    description: HUNYUAN_VIDEO_DESC,
  },
  {
    id: 'hunyuan_video_720p_i2v',
    family: 'HunyuanVideo',
    model: '13B',
    task: 'I2V',
    resolution: '832x1088',
    width: 832,
    height: 1088,
    src: '/videos/hunyuan_video_720p_i2v_1.mp4',
    poster: '/img/posters/hunyuan_video_720p_i2v_1.jpg',
    samplingTimeS: 16007.44,
    description: HUNYUAN_VIDEO_DESC,
  },
  {
    id: 'cogvideox_2b_t2v',
    family: 'CogVideoX',
    model: '2B',
    task: 'T2V',
    resolution: '720x480',
    width: 720,
    height: 480,
    src: '/videos/cogvideox_2b_t2v_1.mp4',
    poster: '/img/posters/cogvideox_2b_t2v_1.jpg',
    samplingTimeS: 519.18,
    description: COGVIDEOX_DESC,
  },
  {
    id: 'cogvideox_5b_t2v',
    family: 'CogVideoX',
    model: '5B',
    task: 'T2V',
    resolution: '720x480',
    width: 720,
    height: 480,
    src: '/videos/cogvideox_5b_t2v_1.mp4',
    poster: '/img/posters/cogvideox_5b_t2v_1.jpg',
    samplingTimeS: 835.27,
    description: COGVIDEOX_DESC,
  },
  {
    id: 'cogvideox_5b_i2v',
    family: 'CogVideoX',
    model: '5B',
    task: 'I2V',
    resolution: '512x672',
    width: 512,
    height: 672,
    src: '/videos/cogvideox_5b_i2v_1.mp4',
    poster: '/img/posters/cogvideox_5b_i2v_1.jpg',
    samplingTimeS: 851.15,
    description: COGVIDEOX_DESC,
  },
  {
    id: 'cogvideox_1_5_5b_t2v',
    family: 'CogVideoX1.5',
    model: '5B',
    task: 'T2V',
    resolution: '1360x768',
    width: 1360,
    height: 768,
    src: '/videos/cogvideox_1_5_5b_t2v_1.mp4',
    poster: '/img/posters/cogvideox_1_5_5b_t2v_1.jpg',
    samplingTimeS: 3380.16,
    description: COGVIDEOX1_5_DESC,
  },
  {
    id: 'cogvideox_1_5_5b_i2v',
    family: 'CogVideoX1.5',
    model: '5B',
    task: 'I2V',
    resolution: '896x1184',
    width: 896,
    height: 1184,
    src: '/videos/cogvideox_1_5_5b_i2v_1.mp4',
    poster: '/img/posters/cogvideox_1_5_5b_i2v_1.jpg',
    samplingTimeS: 3400.79,
    description: COGVIDEOX1_5_DESC,
  },
];

// Curated subset shown on the homepage: the largest / best config per model
// family+version, one each, so the 3×3 (T2V) and 2×4 (I2V) grids cover the
// whole model zoo without repeats. The full set lives on /gallery.
export const HOMEPAGE_T2V_IDS: string[] = [
  'wan2_1_14b_720p_t2v', // Wan2.1
  'wan2_2_a14b_t2v', // Wan2.2 (MoE A14B)
  'cosmos2_5_14b_t2v', // Cosmos-Predict2.5
  'cosmos3_nano_t2v', // Cosmos3
  'ltx_video0_9_8_13b_dev_t2v', // LTX-Video 0.9.8
  'ltx2_5_22b_dev_t2v', // LTX-2.5
  'hunyuan_video1_5_720p_t2v', // HunyuanVideo-1.5
  'hunyuan_video_720p_t2v', // HunyuanVideo 1.0
  'cogvideox_1_5_5b_t2v', // CogVideoX1.5
];

export const HOMEPAGE_I2V_IDS: string[] = [
  'wan2_1_14b_720p_i2v', // Wan2.1
  'wan2_2_a14b_720p_i2v', // Wan2.2 A14B
  'wan2_2_5b-ti2v_i2v', // Wan2.2 TI2V-5B
  'hunyuan_video1_5_720p_i2v', // HunyuanVideo-1.5
  'hunyuan_video_720p_i2v', // HunyuanVideo 1.0
  'cogvideox_1_5_5b_i2v', // CogVideoX1.5
  'cogvideox_5b_i2v', // CogVideoX
  'wan2_1_14b_480p_i2v', // Wan2.1 (480P recipe)
];
