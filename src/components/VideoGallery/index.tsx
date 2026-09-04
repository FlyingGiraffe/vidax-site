import React, { useEffect, useMemo, useState } from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {
  GALLERY_VIDEOS,
  HOMEPAGE_T2V_IDS,
  HOMEPAGE_I2V_IDS,
  type VideoTask,
} from '@site/src/data/videos';
import styles from './styles.module.css';

const TABS: Array<{ key: VideoTask; label: string }> = [
  { key: 'T2V', label: 'Text-to-Video' },
  { key: 'I2V', label: 'Image-to-Video' },
];

// One fixed aspect ratio per task -- every card is cropped to this box with
// `object-fit: cover`, so a row of clips reads as a clean grid regardless of
// each clip's own encoded aspect ratio.
const ASPECT: Record<VideoTask, string> = {
  T2V: '16 / 9',
  I2V: '3 / 4',
};

// `/gallery?tab=t2v` / `?tab=i2v` deep-links: the homepage's per-tab "Browse
// all" buttons point here so each opens the matching tab. A query param
// (not a `#hash`) so Docusaurus's build-time anchor check stays happy.
function tabFromQuery(): VideoTask | null {
  if (typeof window === 'undefined') return null;
  const t = new URLSearchParams(window.location.search).get('tab')?.toLowerCase();
  if (t === 'i2v') return 'I2V';
  if (t === 't2v') return 'T2V';
  return null;
}

function VideoCard({ video }: { video: (typeof GALLERY_VIDEOS)[number] }): React.ReactElement {
  const src = useBaseUrl(video.src);
  const poster = useBaseUrl(video.poster);
  return (
    <div className={styles.card} title={`${video.samplingTimeS.toFixed(1)}s sampling time (5-run avg, TPU v4)`}>
      <video
        className={styles.video}
        style={{ aspectRatio: ASPECT[video.task] }}
        src={src}
        poster={poster}
        controls
        loop
        muted
        playsInline
        preload="none"
      />
      <div className={styles.meta}>
        <div className={styles.metaTop}>
          <span className={styles.modelName}>
            {video.family} {video.model}
          </span>
          <span className={styles.resolution}>{video.resolution}</span>
        </div>
        <div className={styles.description}>{video.description}</div>
      </div>
    </div>
  );
}

interface VideoGalleryProps {
  /** Homepage mode: show only the curated subset per task (3×3 T2V, 2×4
   * I2V) and render a full-width link to the full gallery underneath. */
  preview?: boolean;
  /** Base path the preview-mode "Browse all" button points at (a
   * `?tab=t2v` / `?tab=i2v` query for the active tab is appended). */
  moreHref?: string;
}

export default function VideoGallery({
  preview = false,
  moreHref = '/gallery',
}: VideoGalleryProps): React.ReactElement {
  const [activeTab, setActiveTab] = useState<VideoTask>('T2V');

  // On the full gallery page, honor the ?tab= query on load (and on
  // back/forward) so a deep link opens the right tab.
  useEffect(() => {
    if (preview) return;
    const sync = () => {
      const t = tabFromQuery();
      if (t) setActiveTab(t);
    };
    sync();
    window.addEventListener('popstate', sync);
    return () => window.removeEventListener('popstate', sync);
  }, [preview]);

  const selectTab = (t: VideoTask) => {
    setActiveTab(t);
    if (!preview && typeof window !== 'undefined') {
      window.history.replaceState(null, '', `?tab=${t.toLowerCase()}${window.location.hash}`);
    }
  };

  const filtered = useMemo(() => {
    const forTask = GALLERY_VIDEOS.filter((v) => v.task === activeTab);
    if (!preview) return forTask;
    const order = activeTab === 'T2V' ? HOMEPAGE_T2V_IDS : HOMEPAGE_I2V_IDS;
    return order
      .map((id) => forTask.find((v) => v.id === id))
      .filter((v): v is (typeof GALLERY_VIDEOS)[number] => v !== undefined);
  }, [activeTab, preview]);

  const gridClass =
    activeTab === 'I2V'
      ? preview
        ? styles.gridI2vPreview
        : styles.gridPortrait
      : preview
        ? styles.gridT2vPreview
        : styles.grid;

  return (
    <section className={styles.section}>
      <div className={styles.tabs} role="tablist" aria-label="Video task">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            className={activeTab === tab.key ? styles.tabActive : styles.tab}
            onClick={() => selectTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className={gridClass}>
        {filtered.map((video) => (
          <VideoCard key={video.id} video={video} />
        ))}
      </div>
      {preview && (
        <Link
          className={styles.moreButton}
          to={`${moreHref}?tab=${activeTab.toLowerCase()}`}
        >
          Browse all samples in the Gallery →
        </Link>
      )}
    </section>
  );
}
