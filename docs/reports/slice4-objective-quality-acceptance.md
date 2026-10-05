# Slice 4 objective audio-quality acceptance

- Date: 2026-08-27
- Result: Passed
- Input: 277 verified raw objects
- ADR: `docs/adr/0004-objective-audio-quality.md`
- Analysis: `configs/lab/quality/signal_analysis_v1.yaml`
- Policy: `configs/lab/quality/training_source_review_v1.yaml`

## Delivered

- BS.1770-4 integrated LUFS through pyloudnorm 0.2.0;
- 4× polyphase inter-sample peak estimate, explicitly labeled non-certified;
- sample peak, RMS, crest factor, and absolute DC offset;
- 20 ms / 10 ms leading/trailing silence and silence ratio;
- digital-silence ratio plus nullable noise-floor/speech-level/SNR proxies;
- separate near-peak evidence and quantization-aware flat-top clipping runs;
- strict versioned analysis configuration and review policy;
- catalog schema v4 with cached metrics and policy-specific assessments;
- zero-decode reassessment when only policy thresholds change;
- `dots.tts.lab quality` CLI with `--force` for deliberate remeasurement;
- traceable JSON, CSV, and standalone HTML reports with source paths, structured
  reasons, counts, and distribution percentiles.

## Research and prototype gate

The implementation compared pyloudnorm, FFmpeg ebur128, and custom loudness;
polyphase true-peak estimation; waveform clipping methods; WADA-SNR; and deferred
DNSMOS/NISQA model metrics. The accepted tradeoffs are documented in ADR 0004.

A 24-file prototype used six real files from each weak emotion directory. The
first noise proxy used the lowest energy percentile of all frames and produced
false 80–100 dB values because digital-zero silence was treated as background
noise. That prototype was rejected. The accepted algorithm:

- reports digital-zero frame ratio separately;
- estimates noise only from non-digital frames below the activity threshold;
- returns null instead of inventing a noise/SNR value when no such frames exist.

Accepted prototype run `646cf214-e75d-4284-b04f-1d2cb50b857d`:

- 24 analyzed, 0 errors;
- 23 pass, 1 review, 0 reject;
- the review was `【生气_angry】…又是我出主意？.wav` for silence ratio
  0.5449 above the initial 0.45 review threshold;
- no flat-top clipping runs.

Cached prototype run `8a97bc38-3b38-4776-bbb2-344013f67498`:

- 0 analyzed and 24 cached;
- identical measurements and decision counts.

The isolated prototype workspace was removed after results were recorded.

## Automated tests

Command:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Twenty-one tests passed. Slice 4 coverage includes:

- clean sine versus synthetic hard-clipped flat tops;
- leading/trailing silence and standard LUFS availability;
- digital-zero silence never becoming a fake noise floor;
- multiple threshold crossings produce structured review reasons;
- analysis errors produce reject decisions;
- matching metrics are cached and bypass the decoder;
- a policy version change reassesses cached metrics without decoding;
- analysis ID/version content drift is blocked;
- a corrupt raw file becomes a recorded analysis error/reject;
- report failure marks the run failed and commits no metrics.

All Slice 1–3 regressions remain green. `pip check` reports no broken
requirements after adding pyloudnorm 0.2.0 and the direct SciPy dependency.

## Full dataset acceptance

Before migration, the schema v3 catalog was copied to a verified temporary
backup. Run `c1ae7615-328f-48d5-a4f5-0afc69db23cf` migrated the working catalog
to schema v4 and analyzed all 277 raw assets:

- 277 analyzed, 0 cached, 0 errors;
- 261 pass, 16 review, 0 reject;
- 277 metric rows and 277 policy assessments;
- all 277 prior raw objects and provenance remained intact.

Review reasons are deliberately conservative and non-destructive:

| Reason | Assets | Interpretation |
|---|---:|---|
| 4× true-peak estimate above 0 dBTP | 7 | inspect before later normalization |
| silence ratio above 0.45 | 6 | often short/sparse delivery, inspect timing |
| trailing silence above 0.5 s | 2 | known tail-trim candidates |
| source sample rate below 40 kHz | 1 | known 36 kHz file |

The reason groups happen not to overlap in this batch, yielding 16 review assets.
No record is excluded automatically.

Key full-data distributions:

| Metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Integrated loudness | -19.84 LUFS | -16.98 LUFS | -14.14 LUFS |
| Sample peak | -5.23 dBFS | -3.01 dBFS | -0.0003 dBFS |
| 4× true-peak estimate | -5.18 dBTP | -2.97 dBTP | +0.123 dBTP |
| RMS | -21.88 dBFS | -17.57 dBFS | -15.10 dBFS |
| Absolute DC offset | 0.000004 | 0.000461 | 0.002131 |
| Leading silence | 0.000 s | 0.020 s | 0.340 s |
| Trailing silence | 0.000 s | 0.047 s | 0.620 s |
| Silence ratio | 0.103 | 0.241 | 0.583 |

The earlier six “near full scale” files were reproduced exactly using sample
peak above -0.01 dBFS. All 277 assets have zero qualifying flat-top runs. They
are therefore not labeled as hard-clipped; seven assets only receive a true-peak
estimate review because interpolation slightly exceeds 0 dBTP.

Final cached run `ce39a56f-009f-45b7-8a4c-8600355b71b7`:

- 0 analyzed and 277 cached;
- 261 pass, 16 review, 0 reject;
- runtime metric work completed without decoding any audio.

Database verification:

- schema version: 4;
- raw objects: 277;
- quality metrics: 277;
- assessments for policy v1: 277;
- metric errors: 0.

## Generated artifacts

- `data/catalog/catalog.sqlite` schema v4;
- `data/reports/quality/quality.json`;
- `data/reports/quality/quality.csv`;
- `data/reports/quality/quality.html`.

Local data and reports remain intentionally ignored by Git.

## Known limits deferred by design

- True peak is a 4× estimate, not a certified BS.1770 meter implementation.
- Noise/SNR values are relative frame-energy proxies, not calibrated acoustic
  SNR. Their high values in this clean, digitally edited source should not be
  compared directly with WADA-SNR or microphone specifications.
- Silence metrics detect energy boundaries, not phoneme truncation. VAD and
  listening checks remain necessary before automatic trimming.
- Reverb, music/background-source classification, DNSMOS, and NISQA require a
  human-labeled local benchmark and remain deferred.
- Current threshold crossings request review; they do not delete, transcode, or
  exclude source data.

## Next gate

Slice 5 must benchmark resampling and standardize derived training audio to
48 kHz mono without touching raw. It should use the two long-tail candidates and
the 36 kHz asset as required cases, preserve sample/text identity, cache by input
plus processing-config hash, and prove no new clipping is introduced.
