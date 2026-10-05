# ADR 0008: Audited similarity groups and immutable dataset freeze

- Status: Accepted
- Date: 2026-09-07
- Scope: Slice 8 duplicate analysis, speaker consistency, grouped splits, dataset artifacts

## Context

Slices 1–7 produced 277 verified raw assets, 277 reproducible 48 kHz mono PCM24 artifacts,
objective quality results, a frozen 40-item ASR benchmark and 40 approved human review decisions.
The corpus is one parsed speaker with 29.33 minutes of standardized audio. Quality has 261 pass,
16 review and no reject; all 16 review assets are included in the human-approved set.

Training needs a versioned train/validation/test snapshot without exact or near duplicates crossing
splits. The snapshot must not erase the distinction between 40 listened decisions and 237
filename-derived candidates, must preserve the original weak emotion labels, and must remain
rebuildable after working reports or model caches change.

The repository's official trainer consumes line-delimited JSON with minimum fields `fid`, `audio`
and `text`. That compact format is not rich enough to be the provenance authority, so it must be a
derived training view rather than the canonical dataset record.

## Decision

### Freeze all eligible assets while preserving review scope

Dataset `fuxuan@1` may contain all 277 assets if they pass the strict candidate resolver. Forty
assets use the latest approved review text and labels. The other 237 use their non-empty filename
candidate and weak label with explicit sources `filename_candidate_unreviewed` and
`weak_label_unreviewed`; they are never represented as human reviewed.

A latest rejected or pending review is exclusionary and cannot fall back to the filename. Quality
pass is eligible. Quality review is eligible only with a latest approved human decision; quality
reject is excluded. The current 16 review assets satisfy that exception, but the rule remains fail
closed for future data.

Every item records raw asset identity, source-location and metadata-profile provenance,
standardization identity and output hash, quality run/config/decision/reasons, final text and its
source, original weak label, final emotion fields, label source, and review decision/round/batch when
present.

### Treat duplicate and speaker evidence separately

Exact audio uses a freshly computed SHA-256 of each standardized file and verifies the catalog hash.
Exact and near text use a versioned Unicode/punctuation normalization plus symmetric scores. Acoustic
near-duplicate evidence uses a deterministic, gain/silence-tolerant 16 kHz log-mel fingerprint.

Speaker consistency uses the project's own frozen CAM++ encoder rather than a second speaker model:

- artifact `dots-studio/dots.tts-mf-1step`, revision
  `4e872aa8f47ee67887468e7494e36aad30908b1e`;
- `speaker_encoder.safetensors` SHA-256
  `1cf3861c9dee79e4db34bd0b8a4155e68bed27a7c6274e168bb6ee4fed191c85`;
- full-audio CPU float32 inference, 512 dimensions, L2-normalized before comparison.

Fingerprint and speaker thresholds come from a versioned calibration report over deterministic
synthetic transformations and the real 277-item pair distribution. Scores create review candidates;
they do not silently delete audio or rewrite `speaker_id`. Exact hash identity is the only automatic
duplicate conclusion. Details are frozen in `docs/reports/slice8-feature-selection.md`.

### Split connected similarity groups, not individual rows

Accepted duplicate/similarity edges form stable connected components. A component is the minimum
split unit, and its id is derived from the sorted member asset hashes. Singletons form their own
groups. No group may cross train/validation/test.

`fuxuan@1` targets 80/10/10 with seed `20260907`. A deterministic group-aware optimizer balances
item count, final primary emotion, duration and group count. Current 277-item integer totals target
221/28/28. Low-resource sad is reported separately; when at least three sad groups exist, validation
and test each receive at least one. Groups are never split and samples are never copied or
oversampled to manufacture a distribution. The complete algorithm is specified in
`docs/reports/slice8-split-strategy.md`.

### Make Parquet canonical and JSONL derived

The frozen directory is `datasets/fuxuan/v1/` and contains:

- canonical Parquet with full item lineage and split;
- exactly three trainer JSONL files whose records contain only `fid`, absolute `audio`, and `text`;
- `manifest.json`, `stats.json`, the exact build config, analysis references, and `checksums.txt`.

SQLite remains the live workflow authority. A registered dataset version binds the candidate
snapshot, analysis config, accepted edges, dataset config, implementation version and canonical
manifest hashes. Parquet is the portable frozen table; JSONL is regenerated from that same ordered
item set, never maintained by hand.

Freeze uses same-parent staging, writes and validates all artifacts, then publishes atomically.
Re-running identical inputs is an idempotent verification. Reusing `fuxuan@1` with any changed input,
decision, configuration, threshold, grouping or split is an error; changes require v2.

### Rebuild is part of acceptance

A clean rebuild into an independent temporary directory must reproduce all semantic artifact bytes
and checksums. Run timestamps and temporary paths are excluded from canonical files. Every JSONL path
must exist and match its standardized output hash, text must be non-empty, and Parquet/JSONL/catalog
asset sets and splits must agree.

## Alternatives considered

### Freeze only the 40 reviewed benchmark items

Rejected for v1. The benchmark was deliberately representative and validates the correction flow,
but 4.6 minutes is not the intended 29-minute training corpus. Keeping all eligible items is
acceptable only because the remaining filename provenance stays explicit and versioned.

### Mark all registered transcript ground truth as human reviewed

Rejected. Earlier ASR preparation registered filename-derived records for reproducibility, but only
40 assets were actually listened in Slice 7. Dataset lineage must preserve that distinction.

### Use speaker embedding as a duplicate detector

Rejected. Speaker embeddings intentionally suppress content differences, so same-speaker utterances
can be close without being duplicate recordings. Fingerprint, text and speaker evidence remain
separate.

### Random row-level stratified split

Rejected because near-identical recordings or texts can leak across splits. A fixed random seed does
not solve leakage when related rows are split independently.

### Store only JSONL manifests

Rejected because the three training fields cannot explain source, quality, corrections, grouping or
rebuild inputs. JSONL remains a minimal trainer adapter artifact.

## Consequences and exit conditions

- v1 can use the full corpus without overstating review coverage.
- Duplicate and speaker heuristics remain auditable suggestions until reviewed.
- Group constraints may prevent exact target ratios; actual deviations are reported rather than
  hidden by splitting groups.
- A pinned Parquet writer must be added before freeze output is implemented.
- Slice 8 completes only when real-data edge review is resolved, no similarity group crosses a split,
  all frozen artifacts validate and rebuild byte-identically, full tests pass, the acceptance report
  records final distributions and hashes, and `PLAN.md` is updated.
