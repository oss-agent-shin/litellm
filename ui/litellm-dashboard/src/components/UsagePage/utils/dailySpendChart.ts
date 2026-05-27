import { DateRangePickerValue } from "@tremor/react";

import { BreakdownMetrics, DailyData, SpendMetrics } from "../types";

/**
 * Format a Date as a local-time YYYY-MM-DD string.
 *
 * Mirrors `formatDate` in `networking.tsx` so that comparisons between the
 * picker range and the API-returned `date` strings use the same calendar.
 * Using local-time components matches what the daily activity calls send to
 * the backend in `start_date`/`end_date`.
 */
export function formatLocalDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
 * X-axis label used in place of a calendar date when the picker range is
 * a single calendar day. The Daily Spend chart only has day-resolution data,
 * so a time-of-day style label is the cleanest signal to the user that
 * they are looking at one day rather than the (otherwise identical) date.
 */
export const SINGLE_DAY_TIME_LABEL = "12 AM";

/**
 * True if both `from` and `to` are set and fall on the same local calendar
 * day. Tolerant of nullish inputs.
 */
export function isSingleDayRange(
  dateValue: DateRangePickerValue | undefined | null,
): boolean {
  if (!dateValue || !dateValue.from || !dateValue.to) return false;
  return formatLocalDate(dateValue.from) === formatLocalDate(dateValue.to);
}

const SPEND_METRIC_KEYS: (keyof SpendMetrics)[] = [
  "spend",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "api_requests",
  "successful_requests",
  "failed_requests",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
];

function zeroMetrics(): SpendMetrics {
  return {
    spend: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    api_requests: 0,
    successful_requests: 0,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
}

function addMetrics(a: SpendMetrics, b: SpendMetrics): SpendMetrics {
  const out = { ...a };
  for (const key of SPEND_METRIC_KEYS) {
    out[key] = (a[key] || 0) + (b[key] || 0);
  }
  return out;
}

function emptyBreakdown(): BreakdownMetrics {
  return {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
    endpoints: {},
  };
}

type BreakdownObjectKey = Exclude<keyof BreakdownMetrics, "api_keys">;
const BREAKDOWN_OBJECT_KEYS: BreakdownObjectKey[] = [
  "models",
  "model_groups",
  "mcp_servers",
  "providers",
  "entities",
  "endpoints",
];

function mergeApiKeyBreakdown(
  a: { [key: string]: any } | undefined,
  b: { [key: string]: any } | undefined,
): { [key: string]: any } {
  const out: { [key: string]: any } = { ...(a || {}) };
  if (!b) return out;
  for (const [id, value] of Object.entries(b)) {
    if (out[id]) {
      out[id] = {
        ...out[id],
        ...value,
        metrics: addMetrics(out[id].metrics, value.metrics),
        metadata: { ...(out[id].metadata || {}), ...(value.metadata || {}) },
      };
    } else {
      out[id] = value;
    }
  }
  return out;
}

function mergeMetricWithMetadata<
  T extends {
    metrics: SpendMetrics;
    metadata?: object;
    api_key_breakdown?: { [key: string]: any };
  },
>(a: T, b: T): T {
  return {
    ...a,
    ...b,
    metrics: addMetrics(a.metrics, b.metrics),
    metadata: { ...(a.metadata || {}), ...(b.metadata || {}) },
    api_key_breakdown: mergeApiKeyBreakdown(a.api_key_breakdown, b.api_key_breakdown),
  };
}

function mergeBreakdown(
  a: BreakdownMetrics | undefined,
  b: BreakdownMetrics | undefined,
): BreakdownMetrics {
  const merged = emptyBreakdown();
  for (const src of [a, b]) {
    if (!src) continue;
    for (const k of BREAKDOWN_OBJECT_KEYS) {
      const fromSrc = src[k] || {};
      const target = merged[k] as { [key: string]: any };
      for (const [id, value] of Object.entries(fromSrc)) {
        if (target[id]) {
          target[id] = mergeMetricWithMetadata(target[id] as any, value as any);
        } else {
          target[id] = value;
        }
      }
    }
    // api_keys uses KeyMetricWithMetadata — no nested api_key_breakdown.
    const fromSrcKeys = src.api_keys || {};
    for (const [id, value] of Object.entries(fromSrcKeys)) {
      const existing = merged.api_keys[id];
      if (existing) {
        merged.api_keys[id] = {
          ...existing,
          ...value,
          metrics: addMetrics(existing.metrics, value.metrics),
          metadata: { ...existing.metadata, ...value.metadata },
        };
      } else {
        merged.api_keys[id] = value;
      }
    }
  }
  return merged;
}

/**
 * Collapse a list of `DailyData` rows into a single row labelled with the
 * supplied `label`, summing every `SpendMetrics` numeric field and merging
 * every `breakdown` sub-object across the rows. Used for the single-day
 * case where the chart should render one bar regardless of how many
 * paginated or timezone-ghost rows the API returned.
 */
export function collapseDailyResults(rows: DailyData[], label: string): DailyData {
  let metrics = zeroMetrics();
  let breakdown = emptyBreakdown();
  for (const r of rows) {
    if (!r) continue;
    if (r.metrics) metrics = addMetrics(metrics, r.metrics);
    if (r.breakdown) breakdown = mergeBreakdown(breakdown, r.breakdown);
  }
  return { date: label, metrics, breakdown };
}

function dateInRange(date: string, fromYmd: string | null, toYmd: string | null): boolean {
  if (fromYmd && date < fromYmd) return false;
  if (toYmd && date > toYmd) return false;
  return true;
}

/**
 * Shape the paginated daily activity `results` for the Daily Spend bar
 * chart so it matches the user's selected `dateValue`:
 *
 * - **Single-day range** (the "Today" preset, or any custom same-day pick):
 *   collapse every row into a single bar labelled `SINGLE_DAY_TIME_LABEL`.
 *   This dedupes the paginated rows (one bar per page is the LIT-3383
 *   symptom) and silently absorbs the backend timezone ghost-row for
 *   tomorrow UTC. Spend = sum across all returned rows.
 *
 * - **Multi-day range** (or no range): clamp rows to `[from, to]` using
 *   local-time YYYY-MM-DD comparison (same calendar the daily activity
 *   API calls use when populating `start_date`/`end_date`), then sort
 *   ascending without mutating the input.
 *
 * Fixes LIT-3383 ("Daily Spend chart uses date labels for Today").
 */
export function getDailySpendChartData(
  results: DailyData[],
  dateValue?: DateRangePickerValue,
): DailyData[] {
  if (!results) return [];

  if (isSingleDayRange(dateValue)) {
    if (results.length === 0) return [];
    return [collapseDailyResults(results, SINGLE_DAY_TIME_LABEL)];
  }

  const fromYmd = dateValue?.from ? formatLocalDate(dateValue.from) : null;
  const toYmd = dateValue?.to ? formatLocalDate(dateValue.to) : null;

  const filtered: DailyData[] = [];
  for (const row of results) {
    if (!row || !row.date) continue;
    if (!dateInRange(row.date, fromYmd, toYmd)) continue;
    filtered.push(row);
  }

  // Sort a copy so we never mutate the caller's array.
  return filtered
    .slice()
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}
