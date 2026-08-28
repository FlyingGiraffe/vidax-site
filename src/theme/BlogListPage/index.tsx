import React, { type ReactNode, useMemo, useState } from 'react';
import clsx from 'clsx';

import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {
  PageMetadata,
  HtmlClassNameProvider,
  ThemeClassNames,
} from '@docusaurus/theme-common';
import Layout from '@theme/Layout';
import BlogListPaginator from '@theme/BlogListPaginator';
import SearchMetadata from '@theme/SearchMetadata';
import type { Props } from '@theme/BlogListPage';
import type { Content } from '@theme/BlogPostPage';
import BlogListPageStructuredData from '@theme/BlogListPage/StructuredData';
import styles from './styles.module.css';

type PostMetadata = Content['metadata'];

// Tag taxonomy defined in blog/tags.yml, grouped into two independent
// filter axes for the sidebar (rather than one flat "Tags" list) --
// "Type" (what kind of writing) and "Content" (what subsystem it's
// about). Any tag not in either list (shouldn't happen once tags.yml is
// kept in sync, but e.g. a stray inline tag) falls into a generic
// leftover "Other" group instead of being silently dropped.
const TYPE_TAG_LABELS = ['Engineering Notes', 'Research'];
const CONTENT_TAG_LABELS = ['Infrastructure', 'Modeling'];

// Ejected (via `docusaurus swizzle @docusaurus/theme-classic BlogListPage
// --eject --danger`) and rewritten as an SGLang/lmsys-blog-style card grid
// with search/tag filters in the left column, replacing the default
// BlogLayout + BlogSidebar (which just listed every post title as nav
// links) -- see also styles.module.css. Only this list page is customized;
// individual post pages still use the default theme.

function BlogListPageMetadata(props: Props): ReactNode {
  const { metadata } = props;
  const {
    siteConfig: { title: siteTitle },
  } = useDocusaurusContext();
  const { blogDescription, blogTitle, permalink } = metadata;
  const isBlogOnlyMode = permalink === '/';
  const title = isBlogOnlyMode ? siteTitle : blogTitle;
  return (
    <>
      <PageMetadata title={title} description={blogDescription} />
      <SearchMetadata tag="blog_posts_list" />
    </>
  );
}

function formatDate(date: PostMetadata['date']): string {
  return new Date(date).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

function BlogCard({ post }: { post: PostMetadata }): ReactNode {
  return (
    <Link to={post.permalink} className={styles.card}>
      <h2 className={styles.cardTitle}>{post.title}</h2>
      <div className={styles.cardMeta}>
        <time dateTime={String(post.date)}>{formatDate(post.date)}</time>
        {typeof post.readingTime === 'number' && (
          <span> · {Math.max(1, Math.ceil(post.readingTime))} min read</span>
        )}
      </div>
      {post.description && <p className={styles.cardDescription}>{post.description}</p>}
      {post.tags.length > 0 && (
        <div className={styles.cardTags}>
          {post.tags.map((tag) => (
            <span key={tag.label} className={styles.cardTag}>
              {tag.label}
            </span>
          ))}
        </div>
      )}
    </Link>
  );
}

function TagFilterGroup({
  label,
  tags,
  activeTags,
  onToggle,
}: {
  label: string;
  tags: string[];
  activeTags: Set<string>;
  onToggle: (label: string) => void;
}): ReactNode {
  if (tags.length === 0) return null;
  return (
    <div className={styles.sidebarSection}>
      <span className={styles.sidebarLabel}>{label}</span>
      <div className={styles.tagList}>
        {tags.map((tag) => (
          <label key={tag} className={styles.tagCheckbox}>
            <input type="checkbox" checked={activeTags.has(tag)} onChange={() => onToggle(tag)} />
            {tag}
          </label>
        ))}
      </div>
    </div>
  );
}

function BlogListPageContent(props: Props): ReactNode {
  const { items, metadata } = props;

  const posts = useMemo(() => items.map(({ content }) => content.metadata), [items]);

  const allTags = useMemo(() => {
    const labels = new Set<string>();
    posts.forEach((post) => post.tags.forEach((tag) => labels.add(tag.label)));
    return labels;
  }, [posts]);

  const typeTags = useMemo(
    () => TYPE_TAG_LABELS.filter((label) => allTags.has(label)),
    [allTags],
  );
  const contentTags = useMemo(
    () => CONTENT_TAG_LABELS.filter((label) => allTags.has(label)),
    [allTags],
  );
  const otherTags = useMemo(
    () =>
      Array.from(allTags)
        .filter((label) => !TYPE_TAG_LABELS.includes(label) && !CONTENT_TAG_LABELS.includes(label))
        .sort(),
    [allTags],
  );

  const [search, setSearch] = useState('');
  const [activeTags, setActiveTags] = useState<Set<string>>(new Set());

  const filteredPosts = useMemo(() => {
    const query = search.trim().toLowerCase();
    return posts.filter((post) => {
      if (activeTags.size > 0 && !post.tags.some((tag) => activeTags.has(tag.label))) {
        return false;
      }
      if (!query) return true;
      return (
        post.title.toLowerCase().includes(query) ||
        (post.description ?? '').toLowerCase().includes(query)
      );
    });
  }, [posts, search, activeTags]);

  const toggleTag = (label: string) => {
    setActiveTags((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  const isDefault = search === '' && activeTags.size === 0;
  const resetAll = () => {
    setSearch('');
    setActiveTags(new Set());
  };

  return (
    <Layout>
      <div className={clsx('container', 'margin-vert--lg', styles.page)}>
        <aside className={styles.sidebar}>
          <div className={styles.sidebarSection}>
            <label htmlFor="blog-search" className={styles.sidebarLabel}>
              Search
            </label>
            <input
              id="blog-search"
              type="search"
              className={styles.searchInput}
              placeholder="Search posts…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <TagFilterGroup label="Type" tags={typeTags} activeTags={activeTags} onToggle={toggleTag} />
          <TagFilterGroup
            label="Content"
            tags={contentTags}
            activeTags={activeTags}
            onToggle={toggleTag}
          />
          <TagFilterGroup label="Other" tags={otherTags} activeTags={activeTags} onToggle={toggleTag} />

          <button type="button" className={styles.resetBtn} onClick={resetAll} disabled={isDefault}>
            Reset
          </button>
        </aside>

        <main className={styles.main}>
          {filteredPosts.length === 0 ? (
            <p className={styles.empty}>No posts match your search/filters.</p>
          ) : (
            <div className={styles.cardGrid}>
              {filteredPosts.map((post) => (
                <BlogCard key={post.permalink} post={post} />
              ))}
            </div>
          )}
          <BlogListPaginator metadata={metadata} />
        </main>
      </div>
    </Layout>
  );
}

export default function BlogListPage(props: Props): ReactNode {
  return (
    <HtmlClassNameProvider
      className={clsx(ThemeClassNames.wrapper.blogPages, ThemeClassNames.page.blogListPage)}>
      <BlogListPageMetadata {...props} />
      <BlogListPageStructuredData {...props} />
      <BlogListPageContent {...props} />
    </HtmlClassNameProvider>
  );
}
