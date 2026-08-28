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
}

# Wan2.1 and Wan2.2 are architecturally distinct model generations (not
# variants of one "Wan" family) -- fold the version into the family label
# so the table/filters never collapse them together.
FAMILY_LABELS = {
    ("wan", "2.1"): "Wan2.1",
    ("wan", "2.2"): "Wan2.2",
    ("cosmos", "2.5"): "Cosmos-Predict2.5",
    ("cosmos", "3"): "Cosmos3",
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
