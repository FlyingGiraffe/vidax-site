import React from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import BenchmarkExplorer from '@site/src/components/BenchmarkExplorer';
import styles from './index.module.css';

// Deep-links straight to the Issue Forms template (vidax repo's
// .github/ISSUE_TEMPLATE/benchmark_submission.yml) so "Submit a result"
// lands on a pre-filled form, not a blank issue.
const SUBMIT_ISSUE_URL =
  'https://github.com/FlyingGiraffe/vidax/issues/new?template=benchmark_submission.yml';

// Mirrors one row of vidax-site's own scripts/community_benchmarks.json
// (minus slug/fps/contributor/contributorUrl, which are filled in when a
// submission is merged) -- keep this in sync if that shape changes.
const SCHEMA_EXAMPLE = `[
  {
    "family": "Wan2.1",
    "version": "2.1",
    "size": "1.3b",
    "sizeLabel": "1.3B",
    "task": "T2V",
    "resolution": "832x480",
    "numFrames": 81,
    "numSteps": 50,
    "jaxVersion": "0.11.0",
    "deviceKind": "TPU v7",
    "deviceCount": 2,
    "tensorParallelSize": 2,
    "sequenceParallelSize": 1,
    "ioDtype": "bf16",
    "weightDtype": "bf16",
    "numRuns": 1,
    "compileS": 62.1,
    "perStepS": 2.41,
    "wallS": 168.3,
    "peakHbmGb": 9.8
  }
]`;

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

        <section className={`${styles.section} ${styles.submitSection}`}>
          <Heading as="h2">Submit Your Own Results</Heading>
          <p className={styles.submitIntro}>
            Ran vidax on hardware not covered above, or want to add a newer
            model/config? Submit the following form to add your results. Every
            row shows its contributor linked to their GitHub profile.
          </p>
          <div className={styles.submitCard}>
            <div className={styles.submitActions}>
              <Link className="button button--primary button--lg" href={SUBMIT_ISSUE_URL}>
                Submit a benchmark result on GitHub
              </Link>
            </div>
            <p className={styles.schemaIntro}>
              The form covers one result at a time. Submitting several at
              once (or you already have them in this shape)? Skip the
              individual fields and paste a JSON array into the form's{' '}
              <strong>Bulk JSON</strong> field instead — each object:
            </p>
            <pre className={styles.schemaBlock}>
              <code>{SCHEMA_EXAMPLE}</code>
            </pre>
          </div>
        </section>
      </main>
    </Layout>
  );
}
