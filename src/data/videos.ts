export type VideoTask = 'T2V' | 'I2V';

export interface GalleryVideo {
  id: string;
  family: string;
  model: string;
  task: VideoTask;
  resolution: string;
  /** Encoded frame width/height, measured directly from the .mp4 (not the
   * benchmark config value) -- used to size each card to its real aspect
   * ratio instead of forcing a fixed box. */
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

const WAN2_1_DESC =
  'DiT conditioned by a UMT5-XXL text encoder, sampled with a Rectified-Flow Euler scheduler.';
const WAN2_2_DESC =
  "DiT redesigned as a two-expert Mixture-of-Experts with per-token timestep conditioning.";
const COSMOS2_5_DESC =
  "DiT conditioned by a Qwen2.5-VL-7B ('Reason1') text tower, sampled with a flow-matching UniPC multistep solver.";
const COSMOS3_DESC =
  'An omnimodal Mixture-of-Transformers with dual causal/full-attention pathways and interleaved 3D mRoPE.';

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
    samplingTimeS: 1393.27,
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
];
