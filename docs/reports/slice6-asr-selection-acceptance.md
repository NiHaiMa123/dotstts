# Slice 6 acceptance: Transcript verification benchmark and ASR selection

- Date: 2026-08-28
- Result: Passed
- Benchmark: `fuxuan_asr@2`, 40 samples, 273.91 s
- Benchmark config SHA-256: `994918a3420e2826f91b4853077b2671d7a00000873fabc174b0db390e4448f0`
- Manifest SHA-256: `63df6618bff0e7387b8237ce41a004e3ebb48ccd3fd734caac7e94136e889d05`
- Comparison: `fuxuan_asr_selection@2`, lexicon `fuxuan_domain@1`
- Selected backend: `qwen3_asr` (Qwen/Qwen3-ASR-0.6B, Apache-2.0)

## Delivered behavior

Three new commands complete the verification loop:

- `dots.tts.lab asr-prepare` registers the 277 user-confirmed filename
  transcripts as ground truth and freezes a reason-tagged 40-sample benchmark.
- `dots.tts.lab asr-evaluate` validates a backend's raw output against the frozen
  benchmark, computes character-level CER, and stores the run.
- `dots.tts.lab asr-compare` ranks the newest successful run per backend, applies
  selection gates, and emits a priority-ordered human review queue.

Backends stay outside the library. `scripts/run_asr_backend.py` drives each one
from its own venv under `data/work/asr_envs/` against a revision-pinned snapshot
under `data/work/models/asr/`, using a versioned config in
`configs/lab/asr/backends/`. Catalog schema v7 adds run-level runtime metadata,
model license, and exact-match count to `asr_run`.

The v2 evaluator verifies that its config hash equals the catalog's frozen
config, that the backend result names the exact frozen manifest hash, and that
the runner hashed every derived audio file before inference. Same-version drift
is rejected before canonical benchmark files are touched. v2 explicitly inherits
v1's 40 asset IDs, so the normalization correction cannot silently change the
representative set.

ASR output is stored as hypothesis text and diffed. Nothing in this slice writes
to `text_candidate`.

## Benchmark construction

Selection is deterministic, and every sample records why it was chosen:

- all 16 assets the Slice 4 policy routed to `review`;
- all 6 `难过_sad` assets;
- per-emotion fill by linguistic difficulty (text length plus summed inverse
  character frequency), so rare characters outrank merely long lines.

Result: 10 neutral, 10 happy, 14 angry, 6 sad. Because selection targets
difficulty, the reported CER is pessimistic relative to the full 277 assets.

Scoring normalizes with NFKC, casefold, removal of Unicode categories `P` and
`Z`, and explicit removal of ASCII `~`. The explicit rule is needed because NFKC
maps the decorative full-width `～` to `~`, whose Unicode category is `Sm` rather
than punctuation. Traditional/simplified and numeral conversion stay off, so
those variants remain real errors. Re-preparing the benchmark reproduced the
identical manifest SHA-256; config drift under an existing version is blocked
without altering the frozen files.

## Backend comparison

| Rank | Backend | License | CER | Exact | Entity recall | Archaic recall | Punctuation | RTF | Peak torch |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | qwen3_asr 0.0.6 | Apache-2.0 | 0.0712 | 12/40 | 0.385 | 0.538 | full-width, 139 | 0.115 | 1,876,073,984 B |
| 2 | sensevoice (funasr 1.4.4) | FunASR OSS 1.1 | 0.0968 | 10/40 | 0.256 | 0.308 | none | 0.008 | 994,019,840 B |
| 3 | faster_whisper 1.2.1 | MIT | 0.1063 | 6/40 | 0.179 | 0.231 | mixed width, 126 | 0.027 | n/a |

CER by weak emotion label:

| Backend | neutral | happy | angry | sad |
|---|---:|---:|---:|---:|
| qwen3_asr | 0.107 | 0.055 | 0.057 | 0.069 |
| sensevoice | 0.113 | 0.089 | 0.084 | 0.118 |
| faster_whisper | 0.124 | 0.095 | 0.078 | 0.186 |

All three produced 40/40 `ok` results with no empty hypotheses. Gates
(`selection_max_aggregate_cer: 0.10`, `selection_max_realtime_factor: 0.5`)
qualified `qwen3_asr` and `sensevoice`; `faster_whisper` failed on CER.
faster_whisper's peak memory is `n/a` because CTranslate2 bypasses the torch
allocator — recorded with a note rather than filled with a fabricated figure.

Timestamp capability is now explicit in the comparison: faster-whisper supports
word timestamps, qwen3_asr supports optional forced alignment, and SenseVoice's
current runner exposes none. All remain disabled because this slice has no
human-aligned timestamp truth; no timestamp quality score is fabricated.

Only qwen3_asr holds steady across all four emotion labels. faster_whisper
degrades sharply on `sad` (0.186), the label with only 6 assets, and
misrecognized the speaker's self-reference `本座` as `本作` in 5 samples and 6
term occurrences.

## Four methodology defects found and corrected

**SenseVoice emoji markers inflated its CER by 27%.** Its rich-transcription
postprocessing emits emoji emotion tags, which are Unicode category `So` and
therefore survive normalization. 29 markers across 29 of 40 hypotheses were each
charged as an insertion: 131 instead of 103 edit operations, CER 0.1242 instead
of 0.0976, 2 exact matches instead of 10. Uncorrected, this would have ranked
SenseVoice last rather than second. The backend now strips non-linguistic symbols
and keeps the rich text in `metadata.backend_text_rich`, and `asr-evaluate`
refuses any run whose normalized hypotheses still carry non-letter, non-number
characters. The polluted run and the subsequent refusal are both retained in
`asr_run` as history.

**faster_whisper depended on an undeclared cuBLAS library.** Its first pass
succeeded only because a torch-enabled venv happened to be on PATH; from a clean
shell it failed with `cublas64_12.dll is not found`. Backend configs now declare
`native_library_dirs` and `required_native_libraries`, verified and prepended to
PATH before the backend imports and recorded in the run's `environment`. After
the fix the re-run reproduced CER 0.10711 exactly.

**Frozen identity was enforced too late.** Benchmark reports used to be written
before same-version drift was rejected, and evaluation accepted any local config
with the same benchmark ID/version. Preparation now performs a catalog preflight
before touching canonical files; evaluation requires the registered config hash
and the exact source manifest hash. The runner also verifies every derived audio
SHA-256 and refuses a manifest that changes during inference.

**Two counters treated formatting and occurrences incorrectly.** NFKC preserved
the decorative `～` as ASCII `~`, charging all three backends one deletion, while
the entity metric counted only one hit per term per sample. v2 removes `~`
explicitly and counts repeated terms. The 40 asset IDs are inherited from v1 so
this correction cannot change sample composition. Entity denominators are now
39 actual occurrences rather than 37 sample-term presences.

## Determinism

The v2 inference rerun produced byte-identical hypothesis strings to the latest
clean v1 run for all 40 items and all three backends. Score changes are therefore
entirely explained by the normalization and occurrence fixes. Historical v1
runs had already reproduced identical aggregate CER per backend:

| Backend | Successful runs | Distinct CER values |
|---|---:|---:|
| faster_whisper | 3 | 1 (0.10711) |
| qwen3_asr | 2 | 1 (0.07204) |
| sensevoice | 4 | 2 (0.12417 polluted, then 0.09763 three times) |

`asr-compare` reads only the newest successful run per backend within the selected
benchmark version, so superseded runs never compete with replacements. Re-running
the v2 comparison reproduces an identical summary apart from its timestamp.

Current catalog state: schema v8, 2 immutable benchmark versions, 80 benchmark
items, 277 ground-truth transcripts, 13 runs (12 succeeded, 1 correctly failed),
and 480 stored results. The schema-v8 addition belongs to Slice 7; Slice 6's
tables remain backward-compatible with v1.

## The binding finding

Named-entity recall peaks at 15/39 (0.3846). The selected backend loses about
61.5% of the
domain terms it encounters, and these unanimous failures are typical:

| Reference | faster_whisper | qwen3_asr | sensevoice |
|---|---|---|---|
| 符玄 | 扶悬 | 浮玄 | 浮悬 |
| 建木 | 剑木 | 剑目 | 剑幕 |
| 鳞渊境 | 凌渊境 | 灵渊禁 | 陵渊禁 |
| 太卜司 | 泰普斯 | 泰普斯 | 泰普斯 |
| 穷观阵 | 穷关阵 | 穷关镇 | 琼关镇 |
| 狩原毛峰 | 兽源毛风 | 寿元毛峰 | 寿元毛峰 |

These are the terms that make this corpus worth training on. This is the evidence
that ASR may only verify, never overwrite.

## Ground truth defects surfaced

Cross-backend consensus also tests the reference. 11 samples carry
`consensus_conflicts_reference` — backends agreeing within 0.10 pairwise while
all differ from the reference. That flag deliberately does not distinguish a
wrong reference from a shared model blind spot; both are present:

- Reference omits a leading interjection all three backends hear: #29, #31, #36,
  #40.
- Reference reads `哼哼` where all three hear a single `哼`: #5.
- Homophone spellings the audio cannot disambiguate: `不只`/`不止` (#12),
  `汇合`/`会合` (#15), `寸功不竟`/`不尽` (#26), `事务`/`事物` (#18),
  `当作`/`当做` (#31), `唔`/`嗯` (#39).
- Shared blind spots where the reference is right: #14, #16, #17, #23, #39.

Consequence: this benchmark's CER has a floor no better model can reach. The
`user_confirmed_filename` provenance stands as recorded, and Slice 7 resolves
these against the audio — with homophone cases resolved in favor of the
reference, since the audio carries no information about intended spelling.

## Review queue

31 of 40 samples are queued: 18 high, 9 medium, 4 low.

| Flag | Samples |
|---|---:|
| `all_backends_wrong` | 25 |
| `consensus_conflicts_reference` | 11 |
| `unanimous_named_entity_miss` | 11 |
| `named_entity_miss` | 11 |
| `high_cer_delta` | 4 |

The 9 unqueued samples each had at least one backend transcribe the line exactly
with no lexicon term lost; 4 of them (#22, #25, #33, #34) were exact for all
three backends. Reports: `data/reports/asr/comparison/` (JSON, two CSVs, HTML)
and `data/reports/asr/<backend>/evaluation.*` per backend.

## Automated verification

The Slice 6-focused suite now contains 21 passing tests. Coverage includes
lexicon validation, normalization collisions, explicit `～` removal, punctuation
profiling, residual-symbol detection, repeated term occurrences, pairwise spread,
backend ranking, gate failure, flag and priority assignment, superseded-run
exclusion, single-backend refusal, config/manifest drift rejection without file
overwrite, and rejection of rich-transcription markers with no partial catalog
write. `pip check` reports no broken requirements.

## Acceptance conclusion

Slice 6 meets its acceptance gates: backend selection rests on 40 locally scored
samples with recorded model revisions and inference hashes rather than on public
leaderboards, and high-divergence samples route into an explicit human review
queue. Four methodology defects are now corrected and recorded in benchmark v2.
The project may proceed to Slice 7, review and emotion-label
correction, which starts with the 18 high-priority samples and must also settle
the reference defects listed above.
