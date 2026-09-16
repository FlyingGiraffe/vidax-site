import React from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import BenchmarkExplorer from '@site/src/components/BenchmarkExplorer';
import styles from './index.module.css';

export default function BenchmarksPage(): React.ReactElement {
  return (
    <Layout
      title="Benchmarks"
      description="vidax benchmark explorer — latency, HBM usage, and FPS across models, tasks, and TPU generations"
    >
      <main>
        <section className={styles.section}>
          <Heading as="h1">Benchmark Explorer</Heading>
          <BenchmarkExplorer height="page" />
          <p className={styles.sectionSub}>
            Resolution is always <code>width×height</code>. T2V rows are
            landscape (width &gt; height); I2V rows are portrait (height &gt;
            width) — output resolution is derived from the standardized
            conditioning image's own aspect ratio, not a fixed size. The
            exception is CogVideoX-5b-I2V and CogVideoX1.5-5B-I2V, which are
            locked to one fixed generation resolution (same as their T2V
            sibling) and rescale the rendered video to the conditioning
            image's aspect ratio afterward — their rows show that fixed
            generation resolution, not the final rescaled output size.
          </p>
        </section>
      </main>
    </Layout>
  );
}
