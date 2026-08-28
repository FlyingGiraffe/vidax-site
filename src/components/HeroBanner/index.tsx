import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import GitHubIcon from '@site/src/components/icons/GitHubIcon';
import PaperIcon from '@site/src/components/icons/PaperIcon';
import styles from './styles.module.css';

// TODO: fill in the real arXiv link once the report is posted, and confirm
// the pip package name/version once v0.1.0-alpha is actually published to PyPI.
const ARXIV_URL = 'https://arxiv.org/abs/TODO';
const PIP_INSTALL_CMD = 'pip install vidax==0.1.0a0';

export default function HeroBanner(): React.ReactElement {
  const { siteConfig } = useDocusaurusContext();

  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <h1 className={styles.title}>{siteConfig.title}</h1>
        <p className={styles.tagline}>{siteConfig.tagline}</p>
        <div className={styles.buttonRow}>
          <Link className="button button--primary button--lg" to="/docs/quickstart">
            Quickstart
          </Link>
          <Link
            className={`button button--secondary button--lg ${styles.iconButton}`}
            href="https://github.com/FlyingGiraffe/vidax"
          >
            <GitHubIcon />
            GitHub
          </Link>
          <Link
            className={`button button--secondary button--lg ${styles.iconButton}`}
            href={ARXIV_URL}
          >
            <PaperIcon />
            arXiv
          </Link>
          <Link className="button button--secondary button--lg" to="/blog">
            Blog
          </Link>
        </div>
        <div className={styles.pipRow}>
          <span className={styles.pipLabel}>v0.1.0-alpha</span>
          <code className={styles.pipCode}>{PIP_INSTALL_CMD}</code>
        </div>
      </div>
    </header>
  );
}
