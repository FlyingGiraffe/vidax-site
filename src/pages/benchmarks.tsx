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
          <BenchmarkExplorer />
        </section>
      </main>
    </Layout>
  );
}
