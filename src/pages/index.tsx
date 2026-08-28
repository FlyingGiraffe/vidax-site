import React from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import HeroBanner from '@site/src/components/HeroBanner';
import VideoGallery from '@site/src/components/VideoGallery';
import BenchmarkExplorer from '@site/src/components/BenchmarkExplorer';
import CitationFooter from '@site/src/components/CitationFooter';
import styles from './index.module.css';

export default function Home(): React.ReactElement {
  return (
    <Layout
      title="vidax"
      description="A Unified JAX Framework for Video Generative Models on Accelerator Meshes"
    >
      <HeroBanner />
      <main>
        <section className={styles.section}>
          <Heading as="h2">Showcase</Heading>
          <VideoGallery />
        </section>

        <section className={styles.section}>
          <Heading as="h2">Benchmark Explorer</Heading>
          <BenchmarkExplorer />
        </section>

        <CitationFooter />
      </main>
    </Layout>
  );
}
