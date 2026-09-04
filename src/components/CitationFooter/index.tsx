import React from 'react';
// import CodeBlock from '@theme/CodeBlock';
import styles from './styles.module.css';

// TODO: the arXiv report isn't out yet, so the "Cite vidax" block is
// commented out (not deleted) -- restore it, with the real BibTeX entry
// (author list, arXiv id, year), once the report is posted.
//
// const BIBTEX = `@misc{vidax2026,
//   title        = {vidax: A Unified JAX Framework for Video Generative Models on Accelerator Meshes},
//   author       = {TODO: author list},
//   year         = {2026},
//   eprint       = {TODO.arXiv-id},
//   archivePrefix= {arXiv},
//   url          = {https://arxiv.org/abs/TODO}
// }`;

export default function CitationFooter(): React.ReactElement {
  return (
    <section className={styles.section}>
      <div className={styles.col}>
        {/* TODO: restore once the arXiv report is posted.
        <h3>Cite vidax</h3>
        <CodeBlock language="latex">{BIBTEX}</CodeBlock>
        */}
        <h3>Acknowledgments</h3>
        <p className={styles.ack}>
          Development supported by the Google{' '}
          <a href="https://sites.research.google/trc/about/">TPU Research Cloud (TRC)</a> program.
        </p>
      </div>
    </section>
  );
}
