# ADR 0009: Slice 9 reference-pool selection contract

- Status: Accepted
- Date: 2026-09-08
- Scope: `fuxuan@1` reference candidate features, pool-local ranking and diversity selection
- Configuration: `configs/lab/datasets/fuxuan_slice9_v1.yaml`

## Context

Slice 9 must produce a small, explainable and replayable reference set for dots.tts. The frozen
dataset has one speaker but several weak emotion labels, a mixture of reviewed and filename-derived
text, and quality/speaker reports from earlier slices. A mutable or mixed-input ranking would make a
later human decision impossible to reproduce.

## Decision

### Bind one immutable input

The selection is bound to dataset `fuxuan@1`, tree SHA-256
`77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`, and
`catalog.dataset_item` rows with `split=train` and speaker `崩铁符玄`. The runtime must create a
candidate snapshot from those rows and record its SHA-256 in every analysis run. Validation/test are
never candidate backfill. Standardized 48 kHz mono PCM24 is the canonical audio; the dots.tts
`top_db=30` replayed duration is the effective duration and must be `>0 && <=10s`.

The config pins the existing standardization, quality, speaker and threshold-calibration identities.
Raw quality values remain provenance; standardized quality values used for ranking must be recomputed
under the pinned analysis identity. Any missing, changed or mismatched artifact fails closed.

### Separate gates from ranking evidence

Identity, path/hash integrity, finite/non-silent audio, valid effective duration, target speaker and
non-empty exact text are hard gates. Quality-policy and speaker-threshold crossings generate review
evidence and do not silently delete an asset; analysis/decode/SHA errors are hard failures. An ASR
agreement value or phonological feature that is unavailable is stored as `null` with an explicit
status and its weight is renormalized, never imputed as zero.

The fixed v1 score weights are objective quality 0.30, speaker similarity 0.25, text provenance
0.15, silence 0.10, duration 0.05 and phonological coverage 0.15. Quantile clipping uses the
5th/95th percentiles independently inside each pool. These are selection-policy defaults, not a
claim that static scores replace Slice 10 model-loop evaluation; changing them requires a new
selection version and snapshot.

### Isolate emotion pools

`emotion_weak_label` and `emotion_primary` route a candidate to a pool only; they are not score
features. Each pool independently applies gates, normalization, score and diversity rerank. Rank
identity is `(pool_id, rank)`; global Top-K, cross-emotion backfill, cross-pool normalization and a
cross-emotion total order are forbidden. `main_neutral` and `emotion_neutral` have separate purposes
and score namespaces even though their route label is the same; their top-k intersection is capped at
three. A low-resource pool reports `shortfall` instead of borrowing another emotion.

## Alternatives rejected

- **Rank all 219 train rows once, then slice by label:** rejected because the largest pool would set
  the normalization and consume all high-scoring rows, making small pools incomparable and hiding
  low-resource shortfalls.
- **Use weak emotion as a numeric quality/emotion-strength feature:** rejected because directory and
  filename labels are routing metadata, not calibrated affect measurements.
- **Auto-reject every policy threshold crossing:** rejected because existing quality policy defines
  these crossings as review evidence and the reference decision still needs human listening.
- **Use the raw 277-item candidate snapshot:** rejected because it bypasses the frozen dataset tree,
  split assignment and Slice 8 exclusions.

## Consequences and exit conditions

- Every report can explain a candidate through fixed provenance, per-feature values, pool-local score,
  and deterministic tie-breaks.
- Feature/ranking snapshots and outputs are immutable by selection identity; a changed input, report,
  threshold or weight requires a new version.
- Static selection ends at a human-review window. The accepted references remain candidates for Slice
  10, where fixed-text/seed model-loop scores are measured separately and may reorder them.

