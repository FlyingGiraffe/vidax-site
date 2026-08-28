import React from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import VideoGallery from '@site/src/components/VideoGallery';
import styles from './index.module.css';

export default function GalleryPage(): React.ReactElement {
  return (
    <Layout title="Gallery" description="vidax video gallery — T2V and I2V samples per model">
      <main>
        <section className={styles.section}>
          <Heading as="h1">Video Gallery</Heading>
          <VideoGallery />
        </section>
      </main>
    </Layout>
  );
}
