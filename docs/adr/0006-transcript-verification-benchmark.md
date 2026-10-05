# ADR 0006: Transcript verification benchmark and ASR backend selection

- Status: Accepted
- Date: 2026-08-27
- Last amended: 2026-08-28 (benchmark/comparison v2 integrity corrections)
- Scope: Slice 6 benchmark construction, ASR backend comparison, review routing

## Context

All 277 assets carry a filename-derived `text_candidate`. The user confirmed on
2026-08-27 that these filename bodies are accurate verbatim transcripts, so they
are registered as `transcript_ground_truth` with provenance
`user_confirmed_filename`. That makes a local benchmark possible without new
manual transcription, but it also means the ground truth inherits whatever the
filenames omit.

An ASR backend is needed to verify those transcripts, not to replace them. The
question this slice answers is narrow: which backend produces a diff report
worth a human's attention on this specific corpus? Public leaderboards do not
answer it, because this speaker's lines are dense with game-specific proper nouns
and classical Chinese wording that general-domain benchmarks do not contain.

## Benchmark

The current benchmark is frozen at `data/benchmarks/asr/fuxuan_v2/`, configured
by `configs/lab/asr/benchmark_v2.yaml`, with the same 40 asset IDs and 273.91
seconds as v1. v2 explicitly inherits v1's selection so a normalization fix
cannot silently replace a representative sample.

Selection is deterministic and reason-tagged, in this priority order:

1. all 16 assets the Slice 4 policy routed to `review`;
2. all 6 `难过_sad` assets, the low-resource emotion;
3. per-emotion fill by a linguistic-difficulty score, which is text length plus
   the sum of inverse character frequencies across the corpus, so rare
   characters outweigh long but ordinary lines.

That yields 10 neutral, 10 happy, 14 angry, 6 sad. Selecting for difficulty
means the resulting CER is deliberately pessimistic relative to the full 277.

Normalization before scoring: NFKC, casefold, removal of Unicode categories `P`
and `Z`, and explicit removal of ASCII `~`. The last rule removes the decorative
full-width `～` after NFKC maps it to the `Sm` category rather than punctuation.
Traditional/simplified and numeral-form conversion stay off, so script and
numeral variants remain real errors in v2. CER is character-level
Levenshtein distance over the summed reference length, not a per-line mean, so
short lines cannot dominate.

## Backends compared

Each backend runs in its own venv under `data/work/asr_envs/` against a
revision-pinned snapshot under `data/work/models/asr/`, driven by
`scripts/run_asr_backend.py` and a versioned config in
`configs/lab/asr/backends/`. Inference parameters are hashed into the run record,
so a parameter change cannot be mistaken for a model difference.

| Backend | Model | License | CER | Exact | Entity recall | Archaic recall | RTF | Peak torch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_asr 0.0.6 | Qwen/Qwen3-ASR-0.6B | Apache-2.0 | 0.0712 | 12/40 | 15/39 (0.385) | 7/13 (0.538) | 0.115 | 1.88 GB |
| sensevoice (funasr 1.4.4) | FunAudioLLM/SenseVoiceSmall | FunASR OSS 1.1 | 0.0968 | 10/40 | 10/39 (0.256) | 4/13 (0.308) | 0.008 | 0.99 GB |
| faster_whisper 1.2.1 | faster-whisper-large-v3-turbo | MIT | 0.1063 | 6/40 | 7/39 (0.179) | 3/13 (0.231) | 0.027 | n/a |

CER by weak emotion label:

| Backend | neutral | happy | angry | sad |
|---|---:|---:|---:|---:|
| qwen3_asr | 0.107 | 0.055 | 0.057 | 0.069 |
| sensevoice | 0.113 | 0.089 | 0.084 | 0.118 |
| faster_whisper | 0.124 | 0.095 | 0.078 | 0.186 |

Punctuation policy differs in kind, not degree: qwen3_asr emits 139 consistently
full-width Chinese marks, faster_whisper emits 126 marks that mix half-width
`, . ? !` with full-width `。 、`, and SenseVoice emits none at all under
`use_itn: false`. Since CER is computed punctuation-free, this is invisible in the
score and is reported separately.

Timestamp capability and activation are explicit report columns: faster-whisper
offers word timestamps, qwen3_asr offers optional forced alignment, and the
current SenseVoice runner exposes none. All are disabled because this slice has
no human-aligned timestamp truth; a timestamp quality score would be fabricated.

Peak GPU memory is measured through the torch allocator, which faster_whisper's
CTranslate2 runtime bypasses entirely. Its cell is `n/a` with a stored note
rather than a fabricated number.

## Four methodology defects found and fixed

**SenseVoice rich-transcription markers.** `rich_transcription_postprocess`
converts the model's `<|HAPPY|>`-style tags into emoji. Emoji are Unicode
category `So`, which benchmark normalization does not remove, so 29 markers
across 29 of 40 hypotheses were each charged as an insertion. That inflated
SenseVoice from 103 to 131 edit operations, CER from 0.0976 to 0.1242, and cut
exact matches from 10 to 2 — enough to rank it last instead of second. The
backend now strips non-linguistic symbols and keeps the rich text in
`metadata.backend_text_rich`, and `asr-evaluate` refuses any run whose normalized
hypotheses still contain non-letter, non-number characters. Both the polluted run
and the refusal remain in `asr_run` as history.

**Undeclared cuBLAS dependency.** faster_whisper's first pass only succeeded
because a torch-enabled venv happened to be on PATH; run from a clean shell it
failed with `cublas64_12.dll is not found`. Backend configs now declare
`native_library_dirs` and `required_native_libraries`, which are verified and
prepended to PATH before the backend imports, and are recorded in the run's
`environment`. This backend is therefore not self-contained on Windows — counted
against it, though not decisive, since a ~400 MB `nvidia-cublas-cu12` copy per
venv is the alternative.

**Frozen identity was enforced too late.** Benchmark reports used to be written
before same-version drift was rejected, and evaluation accepted an arbitrary
local normalization config with the same ID/version. Preparation now performs a
catalog preflight before touching canonical files. Evaluation requires the exact
registered config and manifest hashes, while the runner hashes every derived
audio input and refuses a manifest that changes during inference.

**Formatting and term multiplicity were miscounted.** NFKC mapped a decorative
full-width `～` to ASCII `~` (`Sm`), so all backends paid one non-speech deletion.
The entity metric also counted term presence per sample rather than actual term
occurrences. Benchmark v2 removes `~`, inherits the exact v1 sample set, and
counts multiplicity: 39 entity occurrences rather than 37 sample-term presences.

## Decision

**Select qwen3_asr (Qwen3-ASR-0.6B) as the verification backend.** It leads on
every accuracy measure, is one of two backends that pass both gates, has the best
license position (Apache-2.0), holds up across all four emotion labels rather
than collapsing on `sad`, and emits internally consistent punctuation. Its 0.115
realtime factor is the slowest of the three and still processes the full 277
assets in a few minutes.

Keep SenseVoice as the qualified alternative: about 15x faster, second on accuracy, but
punctuation-free output makes it unsuitable when a diff has to show anything
beyond characters, and its license is the most restrictive of the three.

Reject faster_whisper for this corpus. Beyond ranking last, it misrecognized the
speaker's self-reference `本座` as `本作` in 5 samples and 6 occurrences — a high-frequency,
identity-bearing term — and mixes punctuation width.

Gates and the explicit selection live in `configs/lab/asr/comparison_v2.yaml`:
`selection_max_aggregate_cer: 0.10` and `selection_max_realtime_factor: 0.5`.
faster_whisper fails the CER gate at 0.1063. Named-entity recall is reported but
deliberately not a gate: 39 occurrences over 28 terms is too small to threshold,
and it informs the decision rather than deciding it.

The final backend is no longer inferred from CER alone. `comparison_v2.yaml`
records `selected_backend_id` and a required human-readable rationale, and the
comparison refuses the selection if that backend does not pass the objective
gates. It separately reports the CER leader so the manual decision remains
auditable.

## The conclusion that matters more than the ranking

**No backend may overwrite `text_candidate`.** The best named-entity recall is
15/39 (0.3846), meaning the selected backend destroys roughly 61.5% of the domain terms it
encounters. Representative unanimous failures across all three backends:

- `符玄` → 浮玄 / 扶悬 / 浮悬
- `建木` → 剑木 / 剑目 / 剑幕
- `鳞渊境` → 凌渊境 / 灵渊禁 / 陵渊禁
- `持明龙尊` → 池明 / 驰名 / 驰明
- `太卜司` → 泰普斯 (all three)
- `穷观阵` → 穷关镇 (all three)
- `狩原毛峰`, `鳄蝽蜕`, `伏冬桑` — the herbal names, missed by all three

These are exactly the terms that make this corpus worth training on. ASR output
is stored as `text_asr` and surfaced as a diff; promotion to `text_train` stays a
reviewed human action.

## Ground truth is not flawless either

Cross-backend agreement also probes the reference. Where backends agree closely
with each other yet all differ from the reference, one of two things is true: the
reference is wrong, or all three models share the same blind spot. The
`consensus_conflicts_reference` flag cannot tell those apart, and it does not try
to — 11 of 40 samples carry it (#5, #12, #14, #15, #16, #17, #18, #23, #26, #29,
#39), and both kinds are present:

Reference defects, where the audio supports the backends:

- Missing leading interjections. All three backends hear one the filename omits
  in #29, #31, #36, and #40 (`唉`/`哎` twice, `哼` once).
- Reduplication. #5 reads `哼哼`; all three hear a single `哼`.
- Homophone spellings the audio cannot disambiguate: `不只`/`不止` (#12),
  `汇合`/`会合` (#15), `寸功不竟`/`不尽` (#26), `事务`/`事物` (#18, two
  backends), `当作`/`当做` (#31, two backends), `唔`/`嗯` (#39).

Shared model blind spots, where the reference is right and every backend is
wrong: `太卜司` → `泰普斯` (#14), `穷观阵` → `穷关阵`/`穷关镇` (#16, #39),
`云上五骁` → `云上五霄` (#17), `演算` → `也算`/`眼算` (#23).

Two consequences. First, this benchmark's CER has a floor no better model can
reach, because part of the residual distance is reference error rather than
transcription error. Second, the `user_confirmed_filename` provenance stands as
recorded, but Slice 7 must resolve these samples against the audio instead of
treating the filename as final — and the homophone cases must be resolved in
favor of the reference, since the audio carries no information about which
spelling was intended.

## Review routing

`asr-compare` reads the newest successful run per backend from the catalog —
never a superseded one — and emits a priority-ordered queue. Flags:

| Flag | Condition | Count |
|---|---|---:|
| `all_backends_wrong` | every backend has CER > 0 | 25 |
| `consensus_conflicts_reference` | backends agree within 0.10 pairwise yet all differ from the reference | 11 |
| `unanimous_named_entity_miss` | every backend loses the same lexicon term | 11 |
| `named_entity_miss` | some but not all backends lose a term | 11 |
| `high_cer_delta` | best-to-worst backend CER gap ≥ 0.20 | 4 |

`consensus_conflicts_reference` and `unanimous_named_entity_miss` raise a sample
to `high`; the other two to `medium`. The queue holds 31 of 40 samples: 18 high,
9 medium, 4 low. The 9 samples with no flag are those where at least one backend
transcribed the line exactly and no lexicon term was lost. The lexicon is
versioned at `configs/lab/asr/lexicon_fuxuan_v1.yaml` (28 named entities, 13
archaic terms), drawn only from these 40 references and hashed into every
comparison.

## Sources consulted

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 CUDA requirements](https://opennmt.net/CTranslate2/installation.html)
- [FunASR / SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-0.6B model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Unicode general categories](https://www.unicode.org/reports/tr44/#General_Category_Values)

## Consequences and exit conditions

- Backend choice rests on 40 locally scored samples with a frozen config and
  manifest hash, verified derived-audio hashes, a model revision, and an
  inference hash. Scores can be recomputed from the catalog alone.
- Every evaluated run is retained; comparison always reads the newest per
  backend, so a methodology fix never competes with the run it replaced.
- A backend that decorates its output is refused rather than silently scored.
- Re-run the comparison when a new backend is added, when the lexicon or
  benchmark version changes, or when Slice 7 corrects references — the corrected
  ground truth will lower every CER and may reorder the ranking.
- Reconsider qwen3_asr if a domain-adapted or hotword-capable backend raises
  named-entity recall materially above 0.3846, since that, not aggregate CER, is
  the binding limit on how useful verification can be here.
