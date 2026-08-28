import React, { useMemo, useState } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { GALLERY_VIDEOS, type VideoTask } from '@site/src/data/videos';
import styles from './styles.module.css';

const TABS: Array<{ key: VideoTask; label: string }> = [
  { key: 'T2V', label: 'Text-to-Video' },
  { key: 'I2V', label: 'Image-to-Video' },
];

function VideoCard({ video }: { video: (typeof GALLERY_VIDEOS)[number] }): React.ReactElement {
  const src = useBaseUrl(video.src);
  const poster = useBaseUrl(video.poster);
  return (
    <div className={styles.card} title={`${video.samplingTimeS.toFixed(1)}s sampling time (5-run avg, TPU v4)`}>
      <video
        className={styles.video}
        style={{ aspectRatio: `${video.width} / ${video.height}` }}
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

export default function VideoGallery(): React.ReactElement {
  const [activeTab, setActiveTab] = useState<VideoTask>('T2V');

  const filtered = useMemo(
    () => GALLERY_VIDEOS.filter((v) => v.task === activeTab),
    [activeTab],
  );

  return (
    <section className={styles.section}>
      <div className={styles.tabs} role="tablist" aria-label="Video task">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            className={activeTab === tab.key ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className={activeTab === 'I2V' ? styles.gridPortrait : styles.grid}>
        {filtered.map((video) => (
          <VideoCard key={video.id} video={video} />
        ))}
      </div>
    </section>
  );
}
