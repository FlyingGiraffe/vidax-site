#!/usr/bin/env python3
"""Regenerate src/data/benchmarks.json from vidax/benchmarks/results/*.json.

Read-only with respect to the vidax/ repo -- this script only reads files
under ../vidax/benchmarks/results/ and writes into this site's own
src/data/benchmarks.json. Run it again any time vidax's benchmark results
change (e.g. after the Cosmos2.5 14B re-run finishes, or once v5e/v6e/fp8
rows are added).

Usage:
    python3 scripts/gen_benchmarks_data.py [--vidax-repo PATH]
"""

import argparse
import glob
import json
import os

# Legacy result filenames that predate a later naming convention change in
# the vidax repo -- mapped to the slug this site displays instead. Update
# this if vidax renames/adds result files.
SLUG_OVERRIDES = {
    "wan2_1_14b_t2v": "wan2_1_14b_480p_t2v",
}

# Result files that aren't a single benchmark-table row (e.g. multi-point
# sweeps) and are intentionally excluded from the explorer table.
EXCLUDE_SUFFIXES = ("_offload_sweep",)

# I2V runs record resolution as the literal string "NonexNone" in vidax's
# own results JSON -- it's derived from the conditioning image at runtime,
# not a fixed config value, so the harness never wrote a real one. These
# overrides are measured directly (via ffprobe/ffmpeg) from each row's own
# run-1 output video in vidax/out/<slug>/<slug>_1.mp4, so the table always
# shows the *actual* pixel resolution of that run rather than a guess.
# Re-measure and update this if vidax reruns any I2V benchmark with a
# different conditioning image.
I2V_RESOLUTION_OVERRIDES = {
    "wan2_1_14b_480p_i2v": "544x720",
    "wan2_1_14b_720p_i2v": "832x1104",
    "wan2_2_5b-ti2v_i2v": "800x1088",
    "wan2_2_a14b_i2v": "544x720",
    "wan2_2_a14b_720p_i2v": "832x1104",
    # CogVideoX I2V derives its output size from the conditioning image, so
    # the recorded `resolution` (the T2V config value) is not what was
    # rendered -- these are measured from vidax/out/<slug>/<slug>_1.mp4.
    "cogvideox_5b_i2v": "512x672",
    "cogvideox_1_5_5b_i2v": "896x1184",
    # HunyuanVideo (1.0 & 1.5) I2V records "NonexNone"; measured from the
    # run-1 output video.
    "hunyuan_video1_5_480p_i2v": "544x720",
    "hunyuan_video1_5_720p_i2v": "832x1104",
    "hunyuan_video_720p_i2v": "832x1088",
}

# Wan2.1 and Wan2.2 are architecturally distinct model generations (not
# variants of one "Wan" family) -- fold the version into the family label
# so the table/filters never collapse them together. Same for HunyuanVideo
# 1.0 vs 1.5 and the two LTX generations. CogVideoX vs CogVideoX1.5 share a
# `model`/`version` and are split by `size` in code below instead.
FAMILY_LABELS = {
    ("wan", "2.1"): "Wan2.1",
    ("wan", "2.2"): "Wan2.2",
    ("cosmos", "2.5"): "Cosmos-Predict2.5",
    ("cosmos", "3"): "Cosmos3",
    ("hunyuan_video", "1.5"): "HunyuanVideo-1.5",
    ("hunyuan_video", ""): "HunyuanVideo",
    ("ltx2_5", ""): "LTX-2.5",
    ("ltx_video", "0.9.8"): "LTX-Video",
}

# (ioDtype, weightDtype) per raw result slug, sourced directly from vidax's
# own benchmarking writeup (vidax/docs/benchmarking.md's results table),
# which is the only place this split is recorded -- the raw
# benchmarks/results/*.json files carry no precision field at all. I/O dtype
# is the compute dtype for activations/latents/VAE/text encoder (--dtype);
# Weight dtype is specifically the DiT's own weight dtype (--dit_dtype where
# a model exposes that flag separately, --dtype otherwise). Wan2.1/Wan2.2
# keep their DiT weights in fp32 by default even under a bf16 I/O dtype
# (see docs/lessons/wan2_1_precision_debugging.md) -- Cosmos2.5/Cosmos3 do
# not need this split, both dtypes are bf16. Cross-checked against each raw
# JSON's own compile_s/generation_s against the doc table's values before
# trusting a match. Update this (and re-verify against the doc) if vidax
# changes a model's default dtype or adds an fp8 sweep.
DTYPE_OVERRIDES = {
    "cosmos2_5_14b_t2v": ("bf16", "bf16"),
    "cosmos2_5_2b_t2v": ("bf16", "bf16"),
    "cosmos3_edge_t2v": ("bf16", "bf16"),
    "cosmos3_nano_t2v": ("bf16", "bf16"),
    "wan2_1_1.3b_t2v": ("bf16", "bf16"),
    "wan2_1_14b_480p_i2v": ("bf16", "bf16"),
    "wan2_1_14b_720p_i2v": ("bf16", "fp32"),
    "wan2_1_14b_t2v": ("bf16", "bf16"),  # legacy filename for wan2_1_14b_480p_t2v
    "wan2_1_14b_720p_t2v": ("bf16", "fp32"),
    "wan2_2_5b-ti2v_i2v": ("bf16", "fp32"),
    "wan2_2_5b-ti2v_t2v": ("bf16", "fp32"),
    "wan2_2_a14b_720p_i2v": ("bf16", "fp32"),
    "wan2_2_a14b_i2v": ("bf16", "fp32"),
    "wan2_2_a14b_t2v": ("bf16", "fp32"),
    # CogVideoX: all bf16. CogVideoX-2b's checkpoint actually ships float16
    # and is cast to bf16 here -- the doc's Weight-dtype column still reads
    # bf16 (with a footnote), so match that.
    "cogvideox_2b_t2v": ("bf16", "bf16"),
    "cogvideox_5b_t2v": ("bf16", "bf16"),
    "cogvideox_5b_i2v": ("bf16", "bf16"),
    "cogvideox_1_5_5b_t2v": ("bf16", "bf16"),
    "cogvideox_1_5_5b_i2v": ("bf16", "bf16"),
    # HunyuanVideo 1.0 & 1.5: DiT weights cast to bf16 (checkpoints ship
    # fp32); activations bf16.
    "hunyuan_video_720p_t2v": ("bf16", "bf16"),
    "hunyuan_video_720p_i2v": ("bf16", "bf16"),
    "hunyuan_video1_5_480p_t2v": ("bf16", "bf16"),
    "hunyuan_video1_5_480p_i2v": ("bf16", "bf16"),
    "hunyuan_video1_5_720p_t2v": ("bf16", "bf16"),
    "hunyuan_video1_5_720p_i2v": ("bf16", "bf16"),
    # LTX-Video 0.9.8: fully bf16. LTX-2.5: bf16 bulk, except the fp32
    # AdaLN scale/shift tables the checkpoint itself ships -- the doc's
    # Weight-dtype column reports the dominant dtype (bf16) with a footnote.
    "ltx_video0_9_8_2b_distilled_t2v": ("bf16", "bf16"),
    "ltx_video0_9_8_13b_dev_t2v": ("bf16", "bf16"),
    "ltx_video0_9_8_13b_distilled_t2v": ("bf16", "bf16"),
    "ltx2_5_22b_dev_t2v": ("bf16", "bf16"),
    "ltx2_5_22b_distilled_t2v": ("bf16", "bf16"),
    "ltx2_5_22b_dev_diffvae_t2v": ("bf16", "bf16"),
    "ltx2_5_22b_distilled_diffvae_t2v": ("bf16", "bf16"),
}

# Human-readable size/variant label per raw result slug. Resolution is
# already its own column, so 480p/720p checkpoint suffixes are dropped here
# to avoid repeating it.
SIZE_LABELS = {
    "cosmos2_5_14b_t2v": "14B",
    "cosmos2_5_2b_t2v": "2B",
    "cosmos3_edge_t2v": "Edge (4B)",
    "cosmos3_nano_t2v": "Nano (16B)",
    "wan2_1_1.3b_t2v": "1.3B",
    "wan2_1_14b_480p_i2v": "14B",
    "wan2_1_14b_480p_t2v": "14B",
    "wan2_1_14b_t2v": "14B",  # legacy filename for wan2_1_14b_480p_t2v, see SLUG_OVERRIDES
    "wan2_1_14b_720p_i2v": "14B",
    "wan2_1_14b_720p_t2v": "14B",
    "wan2_2_5b-ti2v_i2v": "TI2V-5B",
    "wan2_2_5b-ti2v_t2v": "TI2V-5B",
    "wan2_2_a14b_720p_i2v": "A14B",
    "wan2_2_a14b_i2v": "A14B",
    "wan2_2_a14b_t2v": "A14B",
    "cogvideox_2b_t2v": "2B",
    "cogvideox_5b_t2v": "5B",
    "cogvideox_5b_i2v": "5B",
    "cogvideox_1_5_5b_t2v": "5B",
    "cogvideox_1_5_5b_i2v": "5B",
    # The 480p/720p suffix here is the checkpoint variant (each is trained
    # for its own resolution band + default --shift), not just the output
    # size -- keep it, matching vidax/docs/benchmarking.md.
    "hunyuan_video1_5_480p_t2v": "8.3B (480p)",
    "hunyuan_video1_5_480p_i2v": "8.3B (480p)",
    "hunyuan_video1_5_720p_t2v": "8.3B (720p)",
    "hunyuan_video1_5_720p_i2v": "8.3B (720p)",
    "hunyuan_video_720p_t2v": "13B",
    "hunyuan_video_720p_i2v": "13B",
    "ltx_video0_9_8_2b_distilled_t2v": "2B distilled",
    "ltx_video0_9_8_13b_dev_t2v": "13B dev",
    "ltx_video0_9_8_13b_distilled_t2v": "13B distilled",
    "ltx2_5_22b_dev_t2v": "22B dev",
    "ltx2_5_22b_distilled_t2v": "22B distilled",
    "ltx2_5_22b_dev_diffvae_t2v": "22B dev, diffusion VAE",
    "ltx2_5_22b_distilled_diffvae_t2v": "22B distilled, diffusion VAE",
}


def r(v, nd=3):
    return round(v, nd) if isinstance(v, (int, float)) else None


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_vidax = os.path.normpath(os.path.join(here, "..", "..", "vidax"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--vidax-repo", default=default_vidax, help="Path to the vidax repo checkout")
    args = parser.parse_args()

    results_dir = os.path.join(args.vidax_repo, "benchmarks", "results")
    out_path = os.path.join(here, "..", "src", "data", "benchmarks.json")

    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        raw_slug = os.path.splitext(os.path.basename(path))[0]
        if raw_slug.endswith(EXCLUDE_SUFFIXES):
            continue
        slug = SLUG_OVERRIDES.get(raw_slug, raw_slug)
        with open(path) as f:
            d = json.load(f)
        avg = d.get("avg", {})
        wall_s = avg.get("wall_s")
        num_frames = d.get("num_frames")

        resolution = d.get("resolution")
        task = (d.get("task") or "").upper()
        if task == "I2V":
            # vidax's recorded `resolution` field for I2V rows is either the
            # literal "NonexNone" placeholder or (for wan2_2_5b-ti2v_i2v) a
            # stale copy of the T2V config value -- neither is the real
            # output size, so I2V rows always use the measured override.
            resolution = I2V_RESOLUTION_OVERRIDES.get(raw_slug)
            if resolution is None:
                raise ValueError(
                    f"{raw_slug}: I2V row has no measured override in "
                    "I2V_RESOLUTION_OVERRIDES -- measure it from "
                    "vidax/out/<slug>/<slug>_1.mp4 and add it"
                )

        model, version = d.get("model"), d.get("version")
        if model == "cogvideox":
            # CogVideoX and CogVideoX1.5 share model="cogvideox"/version="";
            # the 1.5 checkpoints carry a "1_5" size prefix.
            family = "CogVideoX1.5" if (d.get("size") or "").startswith("1_5") else "CogVideoX"
        else:
            family = FAMILY_LABELS.get((model, version))
        if family is None:
            raise ValueError(f"{raw_slug}: no FAMILY_LABELS entry for (model={model!r}, version={version!r})")
        size_label = SIZE_LABELS.get(raw_slug)
        if size_label is None:
            raise ValueError(f"{raw_slug}: no SIZE_LABELS entry -- add one")
        dtypes = DTYPE_OVERRIDES.get(raw_slug)
        if dtypes is None:
            raise ValueError(f"{raw_slug}: no DTYPE_OVERRIDES entry -- add one, sourced from vidax/docs/benchmarking.md")
        io_dtype, weight_dtype = dtypes

        rows.append(
            {
                "slug": slug,
                "family": family,
                "version": version,
                "size": d.get("size"),
                "sizeLabel": size_label,
                "task": (d.get("task") or "").upper(),
                "resolution": resolution,
                "numFrames": num_frames,
                "numSteps": d.get("num_steps"),
                "jaxVersion": d.get("jax_version"),
                "deviceKind": d.get("device_kind"),
                "deviceCount": d.get("device_count"),
                "tensorParallelSize": d.get("tensor_parallel_size"),
                "sequenceParallelSize": d.get("sequence_parallel_size"),
                "ioDtype": io_dtype,
                "weightDtype": weight_dtype,
                "numRuns": d.get("num_runs"),
                "compileS": r(avg.get("compile_s"), 2),
                "perStepS": r(avg.get("per_step_s"), 3),
                "wallS": r(wall_s, 2),
                "peakHbmGb": r(avg.get("peak_hbm_gb"), 2),
                "fps": r(num_frames / wall_s, 3) if (num_frames and wall_s) else None,
            }
        )

    rows.sort(key=lambda x: (x["family"], x["version"] or "", x["slug"]))

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
