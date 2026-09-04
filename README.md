# vidax-site

Docs & showcase website for [`vidax`](https://github.com/FlyingGiraffe/vidax),
a JAX/Flax inference engine and PyTorch→JAX weight translator for video
diffusion models on TPU. Built with
[Docusaurus 3](https://docusaurus.io/) (TypeScript + React), deployed to
GitHub Pages at
**<https://flyinggiraffe.github.io/vidax-site/>**.

**This repo never edits `vidax/`.** It only *reads* from a sibling checkout
of `vidax/` — model docs, benchmark result JSON, and sample output videos —
and copies/transforms what it needs into `vidax-site/`. `vidax`'s first
release (v0.1.0) covers ten model families, but its benchmark results and
sample clips keep changing, so this site's data snapshots need manual
re-sync — see [Keeping data in sync](#keeping-data-in-sync).

## Structure

The homepage (`src/pages/index.tsx`) stacks:

| Section | Component / page |
| --- | --- |
| Hero & launch bar | [`src/components/HeroBanner`](src/components/HeroBanner) |
| Showcase (curated 3×3 T2V / 2×4 I2V preview + "Browse all" button) | [`src/components/VideoGallery`](src/components/VideoGallery) — full grid at `/gallery` |
| Benchmark Explorer (compact scroll-capped embed) | [`src/components/BenchmarkExplorer`](src/components/BenchmarkExplorer) — full table at `/benchmarks` |
| Acknowledgments footer (BibTeX block commented out until the arXiv report is up) | [`src/components/CitationFooter`](src/components/CitationFooter) |

Docs live under [`docs/`](docs) (Docusaurus doc plugin, sidebar in
`sidebars.ts`): **Getting Started**, **Model Family Guides** (one page per
family), **Sharding & Topology**, and an **API Reference** for the reusable
`vidax.core` / `vidax.schedulers` / `vidax.translator` building blocks.

The **Blog** ([`blog/`](blog), Docusaurus blog plugin at `/blog`) carries
the long-form engineering writeups, with a custom tag-split sidebar
(`src/theme/BlogListPage`) that separates "type" tags (Engineering Notes /
Research) from "content" tags (Infrastructure / Modeling).

```text
vidax-site/
├── README.md
├── package.json                 # npm scripts: start, build, serve, typecheck, deploy, gen-benchmarks
├── docusaurus.config.ts         # site metadata, navbar, footer, GH Pages target, navbar SVG strings
├── sidebars.ts                  # doc sidebar: Getting Started / Model Family Guides / Sharding & Topology / API Reference
├── scripts/
│   └── gen_benchmarks_data.py   # regenerates src/data/benchmarks.json from vidax/benchmarks/results/*.json
├── .github/workflows/deploy.yml # build + publish to GitHub Pages on push to main (actions/deploy-pages)
├── blog/                        # Docusaurus blog — long-form engineering notes
│   ├── authors.yml               # TODO: placeholder identity — replace before publishing
│   ├── tags.yml                  # canonical tag taxonomy (two independent groups)
│   ├── 2026-09-04-sharding-parallelism-and-jit-on-tpus.md
│   ├── 2026-09-04-weight-offloading.md
│   └── 2026-09-04-understanding-video-diffusion-architectures.md
├── docs/
│   ├── intro.md, quickstart.md, installation.md
│   ├── models/                  # one page per family: wan2_1, wan2_2, cosmos2_5, cosmos3,
│   │                            #   ltx_video, ltx2_5, hunyuan_video_1_5, hunyuan_video, cogvideox
│   ├── sharding/                # loading-pytorch-weights, hardware-and-sharding, weight-offloading
│   └── api/                     # index + attention, rope, sharding, schedulers, translator
│                                #   (ported from vidax/docs/api/*)
├── src/
│   ├── components/
│   │   ├── HeroBanner/          # tagline + Quickstart/GitHub/arXiv/Blog buttons (pip row commented out)
│   │   ├── VideoGallery/        # T2V/I2V tabs; uniform per-task crop; ?tab= deep-link; preview mode for the homepage
│   │   ├── BenchmarkExplorer/   # filterable + sortable table; sticky header; height="full|page|compact"
│   │   ├── CitationFooter/      # Acknowledgments (BibTeX "Cite vidax" block commented out — no arXiv yet)
│   │   └── icons/               # GitHubIcon, PaperIcon (also inlined as raw SVG in docusaurus.config.ts's navbar)
│   ├── data/
│   │   ├── benchmarks.json      # GENERATED — do not hand-edit; see scripts/gen_benchmarks_data.py
│   │   └── videos.ts            # gallery clip metadata + HOMEPAGE_{T2V,I2V}_IDS (the curated homepage subset)
│   ├── pages/                   # index.tsx (home), gallery.tsx, benchmarks.tsx
│   ├── theme/BlogListPage/      # swizzled: two-group tag filter + JSON-LD structured data
│   └── css/custom.css
└── static/
    ├── img/
    │   ├── logo*.svg, favicon.ico  # placeholder branding — swap for real marks
    │   └── posters/                # 32 first-frame JPEGs, one per gallery clip (~1.2 MB total)
    └── videos/                     # 32 demo .mp4s (run 1 of each combo), re-encoded H.264 (~28 MB total)
```

## Data provenance

Nothing on this site is fabricated — everything traces back to a real file
in `vidax/`, either linked directly or copied/transformed in:

- **Benchmark Explorer** (`src/data/benchmarks.json`, 32 rows) is generated
  from [`vidax/benchmarks/results/*.json`](../vidax/benchmarks/results) by
  `scripts/gen_benchmarks_data.py`. All rows are **TPU v4, bfloat16 I/O**
  (some DiTs keep fp32 weights) — vidax hasn't benchmarked v5e/v6e or fp8,
  and there's no GPU/PyTorch baseline collected. The generator carries
  per-slug override tables (family label, size label, I/O vs. weight dtype,
  and measured I2V output resolution) — read its docstring before editing.
  - `family` is version-qualified (`Wan2.1` vs. `Wan2.2`, `Cosmos3` vs.
    `Cosmos-Predict2.5`, `HunyuanVideo` vs. `HunyuanVideo-1.5`,
    `CogVideoX` vs. `CogVideoX1.5`) — these are architecturally distinct
    generations, and the family filter must not collapse them.
  - I2V rows' `resolution` is **not** taken from vidax's benchmark JSON
    (it records `"NonexNone"` or a stale T2V value). It's measured with
    `ffprobe` from each row's own run-1 output video in `vidax/out/` — see
    `I2V_RESOLUTION_OVERRIDES` in the generator. Re-measure if vidax reruns
    an I2V benchmark with a different conditioning image.
- **Video Gallery** (`static/videos/*.mp4`, `static/img/posters/*.jpg`,
  `src/data/videos.ts`) — 32 clips (run 1 of each model/task/resolution
  combo's 5-run benchmark set), re-encoded to web-sized H.264 from
  `vidax/out/`. `width`/`height`/`resolution` are measured directly from
  the encoded `.mp4`. Every card is **cropped to one aspect ratio per
  task** (16:9 T2V, 3:4 I2V) via `object-fit: cover`, and the description
  is clamped to three lines *and* reserves that height, so cards align in a
  grid regardless of blurb length. Each `description` is a short note about
  *the model architecture* (shared across a family), not the prompt — every
  clip uses the same conditioning inputs (`vidax/examples/assets/`).
  `HOMEPAGE_T2V_IDS` / `HOMEPAGE_I2V_IDS` pick the curated homepage subset
  (largest/best config per family+version, one each).
- **Docs** (`docs/**/*.md`) — adapted from
  [`vidax/docs/`](../vidax/docs) and `vidax/README.md`. Model guides and
  the API reference track the corresponding `vidax/docs/models/*.md` and
  `vidax/docs/api/*.md`; the blog posts are refreshed from
  `vidax/docs/lessons/` + the sharding/offloading docs. Source links point
  into the `vidax` repo on GitHub.

### Video loading performance

The gallery mounts a `<video>` per clip but fetches no video bytes until
the viewer presses play: `preload="none"` + a `poster` (per-clip first-frame
JPEG, ~1.2 MB for all 32 vs. ~28 MB for the clips) means the initial load
only pulls the small posters. Only the active tab's clips are mounted.
Regenerate a poster with (needs `ffmpeg` — e.g. from the
`vidax-report-fig` conda env):

```bash
ffmpeg -y -i static/videos/<id>_1.mp4 -vf "select=eq(n\,0),scale=480:-2" -frames:v 1 -q:v 4 static/img/posters/<id>_1.jpg
```

### Known placeholders (fill in before / after the arXiv report)

- `docusaurus.config.ts`: arXiv URL is still `arxiv.org/abs/TODO` (hero
  button, navbar, footer). Org/repo (`FlyingGiraffe/vidax-site`) and the
  Pages `url`/`baseUrl` are correct and live.
- `src/components/HeroBanner`: the `pip install` launch row is **commented
  out** (no PyPI release yet) — restore with the real package/version when
  it ships.
- `src/components/CitationFooter`: the "Cite vidax" heading + BibTeX block
  are **commented out** — restore with the real entry once the report is up.
- `static/img/logo*.svg`, `favicon.ico`: placeholder marks. No OG social
  card yet (`docusaurus.config.ts` marks where to add one).
- `blog/authors.yml`: placeholder `vidax-team` identity.
- `src/components/icons/PaperIcon.tsx`: generic document glyph, not arXiv's
  mark (deliberately generic to avoid misusing a trademark).
- Benchmark Explorer has no v5e/v6e, fp8, or GPU-baseline rows — the
  filters already handle those dimensions once
  `vidax/benchmarks/results/*.json` grows them.

## Development

Requires Node.js ≥18. `npm run typecheck` and `npm run build` both pass
clean (the CI workflow runs `npm ci && npm run build` on Node 20).

```bash
npm install
npm start         # dev server with hot reload
npm run build     # static build to build/
npm run serve     # preview the production build locally
```

If your system Node is too old (Ubuntu's stock `apt` package is often
Node 10), the simplest fix without `sudo` is a dedicated conda env:

```bash
conda create -n vidax-site-node -c conda-forge nodejs=20
conda activate vidax-site-node
```

## Keeping data in sync

`vidax-site` is a separate checkout from `vidax`; nothing here updates
automatically. Re-run when relevant:

```bash
# after vidax/benchmarks/results/*.json changes (new runs, TPU gens, fp8, baselines)
python3 scripts/gen_benchmarks_data.py --vidax-repo ../vidax
# then update src/data/videos.ts sampling times / add rows to match

# after new/updated demo clips land in vidax/out/
ffmpeg -y -i ../vidax/out/<combo>/<combo>_1.mp4 -an -c:v libx264 -crf 30 -preset slow \
  -pix_fmt yuv420p -movflags +faststart static/videos/<combo>_1.mp4
# ...regenerate its poster (above) and add/update the entry in src/data/videos.ts
```

Model-doc, API-reference, and blog content is adapted by hand from
`vidax/docs/` — re-read the corresponding source file when a model or
subsystem changes.

## Deployment

Live on GitHub Pages via `.github/workflows/deploy.yml` (build + publish on
push to `main`, plus manual `workflow_dispatch`). Requires repo
**Settings → Pages → Source: GitHub Actions**. The workflow uses
`actions/deploy-pages` — no `gh-pages` branch, no deploy key. `url` /
`baseUrl` / `organizationName` / `projectName` in `docusaurus.config.ts`
already match `FlyingGiraffe/vidax-site`.
