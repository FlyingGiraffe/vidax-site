import React, { useMemo, useState } from 'react';
import benchmarksData from '@site/src/data/benchmarks.json';
import styles from './styles.module.css';

interface BenchmarkRow {
  slug: string;
  family: string;
  version: string | null;
  size: string | null;
  sizeLabel: string;
  task: string;
  resolution: string | null;
  numFrames: number | null;
  numSteps: number | null;
  jaxVersion: string | null;
  deviceKind: string | null;
  deviceCount: number | null;
  tensorParallelSize: number | null;
  sequenceParallelSize: number | null;
  // I/O dtype is the compute dtype for activations/latents/VAE/text
  // encoder; Weight dtype is specifically the DiT's own weight dtype --
  // vidax's Wan2.1/2.2 rows keep fp32 DiT weights under a bf16 I/O dtype,
  // so these must stay two separate columns rather than one "precision".
  ioDtype: string;
  weightDtype: string;
  numRuns: number | null;
  compileS: number | null;
  perStepS: number | null;
  wallS: number | null;
  peakHbmGb: number | null;
  fps: number | null;
  // Community-submitted rows aren't run to the same standard as vidax's own
  // (5-run-averaged) numbers -- `numRuns` says how many runs a row's
  // averaged from so readers can judge noise, rather than the table
  // silently implying every row meets the same bar.
  contributor: string | null;
  contributorUrl: string | null;
}

const ROWS = benchmarksData as BenchmarkRow[];

type SortValue = string | number | null;

interface ColumnDef {
  key: string;
  label: string[];
  sortable: boolean;
  // Left-aligned text in a column wider than its content leaves a big gap
  // trailing the text but not leading it, reading as an unbalanced margin
  // -- true for every short/uniform value here (a task code, a dtype, a
  // number) except the two name-like columns (Model, Contributor), which
  // stay left-aligned since they're read as text, not a value to scan.
  align?: 'left' | 'center';
  getValue?: (r: BenchmarkRow) => SortValue;
  render: (r: BenchmarkRow) => React.ReactNode;
}

function fmt(v: number | null, digits = 2): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits);
}

// Column order/labels loosely follow vidax's own docs/benchmarking.md
// results table. Labels are split across lines (rendered with <br/>) so
// headers stay narrow instead of forcing wide columns. `width` is a percent
// of the table's own width (sums to 100) so the table always exactly fits
// its container instead of overflowing horizontally.
const COLUMNS: (ColumnDef & { width: number })[] = [
  {
    key: 'model',
    label: ['Model'],
    // Widest column: the longest label ("LTX-2.5 22B distilled, diffusion
    // VAE") needs the room so it doesn't wrap to three lines.
    width: 21,
    sortable: true,
    getValue: (r) => `${r.family} ${r.sizeLabel}`,
    render: (r) => `${r.family} ${r.sizeLabel}`,
  },
  {
    key: 'task',
    label: ['Task'],
    width: 5,
    sortable: true,
    align: 'center',
    getValue: (r) => r.task,
    render: (r) => r.task,
  },
  {
    key: 'resolution',
    label: ['Resolution'],
    width: 9,
    sortable: true,
    align: 'center',
    getValue: (r) => r.resolution ?? '',
    render: (r) => r.resolution ?? '—',
  },
  {
    key: 'deviceKind',
    label: ['TPU'],
    width: 7,
    sortable: false,
    align: 'center',
    render: (r) => r.deviceKind ?? '—',
  },
  {
    key: 'weightDtype',
    label: ['Weight', 'dtype'],
    width: 7,
    sortable: false,
    align: 'center',
    render: (r) => r.weightDtype,
  },
  {
    key: 'ioDtype',
    label: ['I/O', 'dtype'],
    width: 6,
    sortable: false,
    align: 'center',
    render: (r) => r.ioDtype,
  },
  {
    key: 'perStepS',
    // Shorter than the old "Denoising Latency (s/step)" so the column can
    // be narrow; matches vidax's own docs/benchmarking.md wording.
    label: ['Per-step', '(s)'],
    width: 8,
    sortable: true,
    align: 'center',
    getValue: (r) => r.perStepS,
    render: (r) => fmt(r.perStepS, 3),
  },
  {
    key: 'wallS',
    label: ['Wall', '(s)'],
    width: 8,
    sortable: true,
    align: 'center',
    getValue: (r) => r.wallS,
    render: (r) => fmt(r.wallS, 1),
  },
  {
    key: 'fps',
    label: ['FPS'],
    width: 7,
    sortable: true,
    align: 'center',
    getValue: (r) => r.fps,
    render: (r) => fmt(r.fps, 3),
  },
  {
    key: 'peakHbmGb',
    label: ['Peak HBM', '(GB)'],
    width: 8,
    sortable: true,
    align: 'center',
    getValue: (r) => r.peakHbmGb,
    render: (r) => fmt(r.peakHbmGb, 2),
  },
  {
    key: 'numRuns',
    // Community rows aren't held to vidax's own 5-run-averaged standard, so
    // this is what lets readers judge how noisy a given number might be
    // instead of assuming every row was measured the same way.
    label: ['Runs'],
    width: 5,
    sortable: true,
    align: 'center',
    getValue: (r) => r.numRuns,
    render: (r) => (r.numRuns === null || r.numRuns === undefined ? '—' : String(r.numRuns)),
  },
  {
    key: 'contributor',
    label: ['Contributor'],
    width: 9,
    sortable: true,
    getValue: (r) => r.contributor ?? '',
    render: (r) =>
      r.contributor ? (
        r.contributorUrl ? (
          <a href={r.contributorUrl} target="_blank" rel="noopener noreferrer">
            {r.contributor}
          </a>
        ) : (
          r.contributor
        )
      ) : (
        '—'
      ),
  },
];

const DEFAULT_SORT_KEY = 'perStepS';
const DEFAULT_SORT_ASC = true;

interface BenchmarkExplorerProps {
  /** Height of the scroll window around the table body (the sticky header
   * stays pinned):
   *  - `'full'`    — no cap (default)
   *  - `'page'`    — tall cap, for the dedicated /benchmarks page
   *  - `'compact'` — short cap, for the homepage embed */
  height?: 'full' | 'page' | 'compact';
}

export default function BenchmarkExplorer({
  height = 'full',
}: BenchmarkExplorerProps): React.ReactElement {
  const families = useMemo(() => Array.from(new Set(ROWS.map((r) => r.family))).sort(), []);
  const devices = useMemo(
    () => Array.from(new Set(ROWS.map((r) => r.deviceKind).filter(Boolean))).sort() as string[],
    [],
  );
  const tasks = useMemo(() => Array.from(new Set(ROWS.map((r) => r.task))).sort(), []);

  const [familyFilter, setFamilyFilter] = useState<string>('all');
  const [deviceFilter, setDeviceFilter] = useState<string>('all');
  const [taskFilter, setTaskFilter] = useState<string>('all');
  const [sortKey, setSortKey] = useState<string>(DEFAULT_SORT_KEY);
  const [sortAsc, setSortAsc] = useState(DEFAULT_SORT_ASC);

  const isDefault =
    familyFilter === 'all' &&
    deviceFilter === 'all' &&
    taskFilter === 'all' &&
    sortKey === DEFAULT_SORT_KEY &&
    sortAsc === DEFAULT_SORT_ASC;

  const resetAll = () => {
    setFamilyFilter('all');
    setDeviceFilter('all');
    setTaskFilter('all');
    setSortKey(DEFAULT_SORT_KEY);
    setSortAsc(DEFAULT_SORT_ASC);
  };

  const rows = useMemo(() => {
    let out = ROWS.filter((r) => {
      if (familyFilter !== 'all' && r.family !== familyFilter) return false;
      if (deviceFilter !== 'all' && r.deviceKind !== deviceFilter) return false;
      if (taskFilter !== 'all' && r.task !== taskFilter) return false;
      return true;
    });
    const column = COLUMNS.find((c) => c.key === sortKey);
    if (column?.getValue) {
      const getValue = column.getValue;
      out = [...out].sort((a, b) => {
        const av = getValue(a);
        const bv = getValue(b);
        if (av === null) return 1;
        if (bv === null) return -1;
        const cmp = typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv));
        return sortAsc ? cmp : -cmp;
      });
    }
    return out;
  }, [familyFilter, deviceFilter, taskFilter, sortKey, sortAsc]);

  const toggleSort = (key: string) => {
    if (key === sortKey) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  return (
    <section className={styles.section}>
      <div className={styles.filters}>
        <label className={styles.filter}>
          Model Family
          <select value={familyFilter} onChange={(e) => setFamilyFilter(e.target.value)}>
            <option value="all">All</option>
            {families.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.filter}>
          TPU Generation
          <select value={deviceFilter} onChange={(e) => setDeviceFilter(e.target.value)}>
            <option value="all">All</option>
            {devices.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.filter}>
          Task
          <select value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)}>
            <option value="all">All</option>
            {tasks.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className={styles.resetBtn}
          onClick={resetAll}
          disabled={isDefault}
        >
          Reset
        </button>
      </div>

      <div
        className={[
          styles.tableWrap,
          height === 'compact' && styles.tableWrapCompact,
          height === 'page' && styles.tableWrapPage,
        ]
          .filter(Boolean)
          .join(' ')}
      >
        <table className={styles.table}>
          <colgroup>
            {COLUMNS.map((col) => (
              <col key={col.key} style={{ width: `${col.width}%` }} />
            ))}
          </colgroup>
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th key={col.key} style={col.align ? { textAlign: col.align } : undefined}>
                  {col.sortable ? (
                    <button
                      className={styles.sortBtn}
                      style={col.align ? { textAlign: col.align } : undefined}
                      onClick={() => toggleSort(col.key)}
                      title={
                        sortKey === col.key
                          ? `Sorted by ${col.label.join(' ')} (${sortAsc ? 'ascending' : 'descending'}) — click to reverse`
                          : `Click to sort by ${col.label.join(' ')}`
                      }
                    >
                      {col.label.map((line, i) => (
                        <React.Fragment key={line}>
                          {i > 0 && <br />}
                          {line}
                        </React.Fragment>
                      ))}
                      {/* Every sortable header carries a triangle: a dim one
                          on the columns you *can* sort by, a solid
                          directional one on the column that's actually
                          active. Without the dim marker there's no cue that
                          the other headers are clickable at all. */}
                      <span
                        className={
                          sortKey === col.key ? styles.sortActive : styles.sortInactive
                        }
                        aria-hidden="true"
                      >
                        {sortKey === col.key ? (sortAsc ? '▲' : '▼') : '▲'}
                      </span>
                    </button>
                  ) : (
                    <span
                      className={styles.headerLabel}
                      style={col.align ? { textAlign: col.align } : undefined}
                    >
                      {col.label.map((line, i) => (
                        <React.Fragment key={line}>
                          {i > 0 && <br />}
                          {line}
                        </React.Fragment>
                      ))}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.slug}>
                {COLUMNS.map((col) => (
                  <td key={col.key} style={col.align ? { textAlign: col.align } : undefined}>
                    {col.render(r)}
                  </td>
                ))}
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={COLUMNS.length} className={styles.empty}>
                  No rows match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
