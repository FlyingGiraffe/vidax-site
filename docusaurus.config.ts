import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// TODO: confirm GitHub org/repo and update `organizationName`/`projectName`/`url` below
// once vidax-site is actually pushed to GitHub Pages. Currently assumes the same
// FlyingGiraffe org that hosts github.com/FlyingGiraffe/vidax.
const GITHUB_ORG = 'FlyingGiraffe';
const VIDAX_REPO = 'vidax';
const SITE_REPO = 'vidax-site';

// Kept as raw markup (not JSX) since navbar "html" items render outside
// React — see src/components/icons/ for the React versions used elsewhere
// (HeroBanner). Keep both in sync if you change one.
const GITHUB_ICON_SVG =
  '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>';
const PAPER_ICON_SVG =
  '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.5 1.5h6l3 3v10a.5.5 0 0 1-.5.5h-8.5a.5.5 0 0 1-.5-.5v-12a.5.5 0 0 1 .5-.5Z"/><path d="M9.5 1.5v3h3"/><path d="M5.25 8.25h5.5M5.25 10.25h5.5M5.25 12.25h3.5"/></svg>';

const config: Config = {
  title: 'vidax',
  tagline: 'A Unified JAX Framework for Video Generative Models on Accelerator Meshes',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // TODO: replace with the real GitHub Pages URL once published, e.g.
  // https://flyinggiraffe.github.io
  url: `https://${GITHUB_ORG.toLowerCase()}.github.io`,
  baseUrl: `/${SITE_REPO}/`,

  organizationName: GITHUB_ORG,
  projectName: SITE_REPO,
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  markdown: {
    // `future.v4: true` (below) disables MDX1-compat admonition syntax by
    // default, which silently breaks every `:::type Title` block sitewide
    // (renders as literal text instead of a styled callout) in favor of
    // the newer `:::type[Title]` bracket syntax. Opt back into the old,
    // simpler syntax explicitly rather than rewriting every admonition.
    mdx1Compat: {
      admonitions: true,
    },
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: `https://github.com/${GITHUB_ORG}/${SITE_REPO}/tree/main/`,
        },
        blog: {
          blogTitle: 'vidax Blog',
          blogDescription:
            'Lessons learned building vidax: video model architectures, running them efficiently on TPU, and papers built on top of it.',
          postsPerPage: 10,
          showReadingTime: true,
          editUrl: `https://github.com/${GITHUB_ORG}/${SITE_REPO}/tree/main/`,
          feedOptions: {
            type: ['rss', 'atom'],
            copyright: `Copyright © ${new Date().getFullYear()} the vidax authors.`,
          },
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // TODO: add static/img/vidax-social-card.jpg (1200x630 OG image) and
    // uncomment once real artwork exists.
    // image: 'img/vidax-social-card.jpg',
    // respectPrefersColorScheme is deliberately off: combined with a fixed
    // defaultMode, it's a known Docusaurus footgun where the toggle needs
    // two clicks the first time a visitor's OS preference disagrees with
    // defaultMode (the client "corrects" to the system preference on
    // mount, but the toggle's first click still flips from the
    // pre-correction defaultMode-based state, undoing the correction
    // instead of changing the visible theme). Always starting from a fixed
    // defaultMode regardless of OS preference makes the toggle deterministic.
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'vidax',
      logo: {
        alt: 'vidax logo',
        src: 'img/logo-light.svg',
        srcDark: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        { to: '/benchmarks', label: 'Benchmark', position: 'left' },
        { to: '/blog', label: 'Blog', position: 'left' },
        { to: '/gallery', label: 'Gallery', position: 'left' },
        {
          type: 'html',
          position: 'right',
          // TODO: replace with the real arXiv link once the report is posted
          value: `<a href="https://arxiv.org/abs/TODO" class="navbar__link navbarIconLink" target="_blank" rel="noopener noreferrer">${PAPER_ICON_SVG}arXiv</a>`,
        },
        {
          type: 'html',
          position: 'right',
          value: `<a href="https://github.com/${GITHUB_ORG}/${VIDAX_REPO}" class="navbar__link navbarIconLink" target="_blank" rel="noopener noreferrer">${GITHUB_ICON_SVG}GitHub</a>`,
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            { label: 'Getting Started', to: '/docs/intro' },
            { label: 'Model Family Guides', to: '/docs/models/wan2_1' },
            { label: 'Sharding & Topology', to: '/docs/sharding/loading-pytorch-weights' },
            { label: 'API Reference', to: '/docs/api/' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'GitHub Issues', href: `https://github.com/${GITHUB_ORG}/${VIDAX_REPO}/issues` },
            { label: 'Discussions', href: `https://github.com/${GITHUB_ORG}/${VIDAX_REPO}/discussions` },
            { label: 'arXiv Paper', href: 'https://arxiv.org/abs/TODO' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Benchmark Explorer', to: '/benchmarks' },
            { label: 'Video Gallery', to: '/gallery' },
            { label: 'Blog', to: '/blog' },
            { label: 'GitHub Repository', href: `https://github.com/${GITHUB_ORG}/${VIDAX_REPO}` },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} the vidax authors. Built with Docusaurus. Supported by the GoogleTPU Research Cloud (TRC) program.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'latex'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
