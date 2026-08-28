import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// Docs are user guides only (install, checkpoints, CLI flags, practical
// reference). Developer notes, architecture deep-dives, design writeups,
// and debugging postmortems live on the blog instead (/blog) -- see its
// "Engineering Notes"/"Research" and "Infrastructure"/"Modeling" tags.
const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: ['intro', 'quickstart', 'installation'],
    },
    {
      type: 'category',
      label: 'Model Family Guides',
      items: [
        'models/wan2_1',
        'models/wan2_2',
        'models/cosmos2_5',
        'models/cosmos3',
        'models/ltx_video',
        'models/ltx2_5',
        'models/hunyuan_video_1_5',
      ],
    },
    {
      type: 'category',
      label: 'Sharding & Topology',
      items: [
        'sharding/loading-pytorch-weights',
        'sharding/hardware-and-sharding',
        'sharding/weight-offloading',
      ],
    },
  ],
};

export default sidebars;
