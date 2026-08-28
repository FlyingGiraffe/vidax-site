# vidax-site

Marketing/docs website for [`vidax`](https://github.com/FlyingGiraffe/vidax), a
JAX/Flax inference engine for video diffusion models. Built with
[Docusaurus](https://docusaurus.io/) (TypeScript + React), deployed to GitHub
Pages.

**This repo never edits `vidax/`.** It only *reads* from a sibling checkout
of `vidax/` — model docs, benchmark result JSON, and sample output videos —
and copies/transforms what it needs into `vidax-site/`. `vidax` is still
under active development (Cosmos2.5/Wan2.1/Wan2.2 nearly done, Cosmos2.5
benchmarking in progress, more model families coming), so this site's data
snapshots will go stale — see [Keeping data in sync](#keeping-data-in-sync).

## Structure

The homepage (`src/pages/index.tsx`) composes the five sections from the
site spec:

| Section | Component / page |
| --- | --- |
| Hero & Launch Bar | [`src/components/HeroBanner`](src/components/HeroBanner) |
| Interactive Video Gallery | [`src/components/VideoGallery`](src/components/VideoGallery) (also at `/gallery`) |
| Interactive Benchmark Explorer | [`src/components/BenchmarkExplorer`](src/components/BenchmarkExplorer) (also at `/benchmarks`) |
| Structured API Documentation | [`docs/`](docs) (Docusaurus doc sidebar, see `sidebars.ts`) |
| Community & Citation Footer | [`src/components/CitationFooter`](src/components/CitationFooter) |

Plus a **Blog** ([`blog/`](blog), Docusaurus's built-in blog plugin at `/blog`)
for lessons-learned writeups (model architectures, running them efficiently,
future papers) — linked from the hero buttons, the top navbar (between
Benchmarks and Gallery), and the footer.

```text
vidax-site/
├── README.md
├── package.json                 # npm scripts: start, build, deploy, gen-benchmarks
├── docusaurus.config.ts         # site metadata, navbar, footer, GH Pages target
├── sidebars.ts                  # doc sidebar: Getting Started / Model Guides / Sharding / Weight Bridge
├── scripts/
│   └── gen_benchmarks_data.py   # regenerates src/data/benchmarks.json from vidax/benchmarks/results/*.json
├── blog/                        # Docusaurus blog (lessons learned, architectures, papers)
│   ├── authors.yml               # TODO: replace placeholder author with real name/handle
│   └── 2026-08-22-welcome-to-the-vidax-blog.md
├── docs/                        # Structured API Documentation (Docusaurus doc sidebar)
│   ├── intro.md, quickstart.md, installation.md
│   ├── models/                  # one page per model family (wan2_1, wan2_2, cosmos2_5, cosmos3)
│   ├── sharding/                # hardware & sharding, weight offloading
│   └── weight-bridge/           # PyTorch -> Flax weight translator overview
├── src/
│   ├── components/
│   │   ├── HeroBanner/          # tagline + Quickstart/GitHub/arXiv/Blog/pip buttons
│   │   ├── VideoGallery/        # T2V/I2V tabs, per-clip model description/resolution/sampling-time
│   │   ├── BenchmarkExplorer/   # filterable + sortable benchmark table
│   │   ├── CitationFooter/      # BibTeX + GitHub issues/discussions + TRC acknowledgments
│   │   └── icons/                # GitHubIcon, PaperIcon (also duplicated as raw SVG strings in
│   │                              # docusaurus.config.ts for the navbar's non-React "html" items)
│   ├── data/
│   │   ├── benchmarks.json      # generated -- do not hand-edit, see scripts/gen_benchmarks_data.py
│   │   └── videos.ts            # gallery clip metadata (model description, exact resolution, sampling time, poster/src paths)
│   ├── pages/                   # index (home), gallery.tsx, benchmarks.tsx (standalone full-page views)
│   └── css/custom.css
└── static/
    ├── img/
    │   ├── logo.svg, favicon.ico  # placeholders -- swap for real branding
    │   └── posters/                # first-frame JPEGs per gallery clip (see "Video loading performance")
    └── videos/                  # demo .mp4 clips copied from vidax/out/**/*.mp4 (run 1 of each combo)
```

## Data provenance

Nothing on this site is fabricated data — everything traces back to a real
file in `vidax/`, either linked to directly or copied/transformed in:

- **Benchmark Explorer** (`src/data/benchmarks.json`) is generated from
  [`vidax/benchmarks/results/*.json`](../vidax/benchmarks/results) by
  `scripts/gen_benchmarks_data.py`. All current rows are **TPU v4,
  bfloat16** — vidax hasn't benchmarked v5e/v6e or fp8 yet, and there's no
  PyTorch/GPU baseline comparison collected yet either (the explorer shows
  a visible notice about this rather than inventing numbers). The
  `Cosmos-Predict2.5` 14B row reflects the last completed run at the time
  this scaffold was built; a fresh benchmarking pass is in progress in
  `vidax` — rerun the generator once it lands.
  - `family` is version-qualified (`Wan2.1` vs. `Wan2.2`, `Cosmos-Predict2.5`
    vs. `Cosmos3`) rather than a bare `Wan`/`Cosmos` — these are
    architecturally distinct model generations within vidax, not size
    variants of one model, and the family filter/table must not collapse
    them.
  - I2V rows' `resolution` is **not** taken from vidax's benchmark JSON
    (which records either the literal string `"NonexNone"` or, for one row,
    a stale copy of the T2V config value — neither is the real output
    size). It's measured directly from each row's own run-1 output video
    in `vidax/out/` via ffmpeg — see `I2V_RESOLUTION_OVERRIDES` in
    `scripts/gen_benchmarks_data.py`. Re-measure and update that dict if
    vidax reruns an I2V benchmark with a different conditioning image.
- **Video Gallery** (`static/videos/*.mp4`, `static/img/posters/*.jpg`,
  `src/data/videos.ts`) — 14 representative clips (run 1 of each
  model/task/resolution combo's 5-run benchmark set) copied from
  `vidax/out/`. Each entry's `width`/`height`/`resolution` are measured
  directly from the encoded `.mp4` (same ffmpeg measurements as the
  benchmark table's I2V overrides above), not copied from config, and each
  card is sized to that exact aspect ratio via inline `aspect-ratio` CSS —
  no more assuming 16:9 or labeling I2V clips "image-derived". Each video's
  `description` is a short blurb about *the model architecture* (shared
  across all clips from that model family/size), not the input prompt —
  every clip uses the same conditioning inputs (see `examples/assets/` in
  `vidax`), so there was nothing prompt-specific worth surfacing per card.
- **Docs** (`docs/models/*.md`, `docs/sharding/*.md`,
  `docs/weight-bridge/overview.md`) — summarized from
  [`vidax/docs/`](../vidax/docs) and `vidax/README.md`. Each page has a
  `:::info Source` callout pointing at the source file(s) and a TODO for
  porting the full guide (these are currently condensed, not verbatim
  copies — the full docs run 300–580 lines each).
- **Model support table**, **install instructions**, **quickstart command**
  — copied verbatim from `vidax/README.md`.

### Video loading performance

The gallery mounts a `<video>` per clip but never fetches video bytes until
the viewer actually presses play: `preload="none"` plus a `poster` (a
per-clip first-frame JPEG in `static/img/posters/`, ~15–45KB each vs.
300KB–2.4MB for the full clip — about 472KB total vs. 16MB for all 14
videos) means the initial page load only downloads the small poster images.
Only the active tab's ~5–9 clips are mounted at all (switching T2V/I2V tabs
unmounts the previous tab's `<video>` elements). Regenerate posters with:

```bash
ffmpeg -y -i static/videos/<id>.mp4 -ss 00:00:00.5 -vframes 1 -vf "scale='min(480,iw)':-2" -q:v 4 static/img/posters/<id>.jpg
```

### Known placeholders (fill in before publishing)

- `docusaurus.config.ts`: arXiv URL (`arxiv.org/abs/TODO`), GitHub org/repo
  assumed as `FlyingGiraffe/vidax` — confirm before deploy.
- `src/components/HeroBanner`: pip install command assumes package name
  `vidax` — there's no PyPI release yet, only editable installs from source.
- `src/components/CitationFooter`: BibTeX entry has placeholder author
  list/arXiv id.
- `static/img/logo.svg`, `static/img/favicon.ico`: minimal placeholder
  marks, not real vidax branding.
- No social-card OG image yet (`docusaurus.config.ts` notes where to add
  one).
- `blog/authors.yml`: placeholder `vidax-team` author (name/handle/avatar)
  — replace with a real identity before publishing.
- `src/components/icons/PaperIcon.tsx` is a generic document glyph, not
  arXiv's actual logo/wordmark (kept generic deliberately to avoid
  misrepresenting a trademark) — swap it for something more specific if
  desired.
- Benchmark Explorer has no TPU v5e/v6e, fp8, or GPU-baseline columns
  populated — the filters exist and will work as soon as
  `vidax/benchmarks/results/*.json` grows those dimensions.

## Development

Requires Node.js ≥18 (the system Node may be too old — see below). Verified
working: `npm install`, `npm run typecheck`, and `npm run build` all pass
clean with no warnings.

```bash
npm install
npm start        # local dev server with hot reload
npm run build     # static build to build/
npm run serve     # preview the production build locally
```

If your system Node is too old (Ubuntu's stock `apt` package is often
Node 10, well below Docusaurus 3's Node ≥18 requirement), the simplest fix
without touching system packages or `sudo` is a dedicated conda env:

```bash
conda create -n vidax-site-node -c conda-forge nodejs=20
conda activate vidax-site-node
```

## Keeping data in sync

`vidax-site` is a separate repo/checkout from `vidax`, so nothing here
updates automatically when `vidax` changes. Re-run when relevant:

```bash
# after vidax/benchmarks/results/*.json changes (new runs, new TPU gens, fp8, GPU baselines)
python3 scripts/gen_benchmarks_data.py --vidax-repo ../vidax

# after new/updated demo clips land in vidax/out/
cp ../vidax/out/<combo>/<combo>_1.mp4 static/videos/
# ...and add a matching entry to src/data/videos.ts
```

## Deployment

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on push
to `main` via `actions/deploy-pages`. Requires GitHub Pages set to "GitHub
Actions" as the source in repo settings, and `organizationName`/
`projectName`/`url` in `docusaurus.config.ts` to match the real repo once
this is pushed to GitHub.
