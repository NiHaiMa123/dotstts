# ADR 0004: Objective audio-quality measurement and review policy

- Status: Accepted
- Date: 2026-08-27
- Scope: Slice 4 signal metrics, cache identity, threshold policy, and review

## Context

The raw store now contains 277 verified original assets. Before standardization,
ASR, or training, the workbench needs reproducible signal measurements and a
conservative way to surface anomalies. Measurements must not destructively alter
audio, and changing a review threshold must not decode every asset again.

The current source is unusually consistent—mono PCM16 sentence clips—but it has
known edge cases: one 36 kHz file, several near-full-scale peaks, and long silence
tails. A useful first quality slice must distinguish “near peak” from convincing
flat-top clipping and must not label a fragile proxy as ground-truth SNR.

## Options considered

### Loudness

- **pyloudnorm 0.2.0** implements ITU-R BS.1770-4 integrated loudness, is MIT
  licensed, supports Python 3.9+, and received a 2026 release. It is small and
  uses the NumPy/SciPy stack already present in dots.tts.
- **FFmpeg ebur128/loudnorm** is mature and useful as a future independent
  cross-check, but would make this slice depend on an external executable and
  its build/version.
- **Custom K-weighting and gating** avoids a dependency but duplicates a
  standardized algorithm with more verification burden.

Use pyloudnorm for integrated LUFS. It targets BS.1770-4; this decision does not
claim BS.1770-5 conformance.

### True peak

Use SciPy `resample_poly` at 4× and report the result explicitly as
`true_peak_estimate_dbtp`. Polyphase interpolation can expose inter-sample peaks,
but it is not asserted to use the exact normative BS.1770 reference filter and
must not be presented as a certified meter reading.

### Clipping

A count of samples above -0.1 dBFS cannot distinguish a clean normalized peak
from clipping. Instead store both:

- near-peak sample count/ratio above a configured level;
- flat-top runs near peak that remain equal within PCM quantization tolerance for
  at least a configured number of samples.

Synthetic clipped sine tests must create flat-top runs while a clean sine must
not. Flat-top detection is still a review signal, not irreversible rejection.

### Noise and SNR

WADA-SNR is a known single-channel speech estimator, but common Python ports have
unclear licensing and its absolute values can be inaccurate. DNSMOS and NISQA
add model downloads, sample-rate/domain assumptions, and GPU/CPU cost; they also
need a locally labeled benchmark before their scores can drive policy.

Slice 4 therefore records an explicitly named proxy:

- speech level: configured percentile of active-frame RMS;
- noise floor: configured percentile of non-digital-silence frames below the
  activity threshold;
- SNR proxy: their difference when both populations exist.

Digital-zero frames are excluded from the noise estimate and recorded separately
as `digital_silence_frame_ratio`. If no usable noise frames exist, the noise/SNR
proxy is null rather than reporting an implausibly large value. This proxy is
relative triage only and never an automatic reject rule.

## Decision

Implement two immutable, independently versioned YAML documents:

1. `signal_analysis_v1.yaml` controls waveform measurement semantics;
2. `training_source_review_v1.yaml` applies thresholds to stored metrics.

Both are strict Pydantic models with canonical JSON SHA-256 identity. Reusing an
ID/version with different content is blocked. Metric identity includes asset
SHA-256, analysis ID/version/config hash, and an explicit implementation version.
Any algorithm code change must increment the implementation version.

The first analysis records:

- sample rate, channels, frames, duration;
- sample peak, 4× true-peak estimate, RMS, integrated LUFS, crest factor;
- absolute DC offset;
- leading/trailing silence and silence ratio from 20 ms / 10 ms RMS frames;
- digital-silence ratio, noise-floor proxy, speech-level proxy, SNR proxy;
- near-peak samples and quantization-aware flat-top runs.

The initial policy is conservative. Decode/analysis failure is rejected;
threshold crossings produce `review`, never automatic deletion or exclusion.
Reasons are structured records containing code, metric, actual value, threshold,
relation, and severity.

## Catalog and execution model

Schema version 4 adds:

- immutable analysis-config and policy registries;
- `quality_run` and per-run asset actions;
- cached `audio_quality_metric` rows;
- policy-specific `audio_quality_assessment` rows.

On a matching analysis identity, `quality` reuses cached metrics and only applies
the current policy. A policy version change therefore performs zero audio decodes.
`--force` exists for deliberate remeasurement. Reports retain raw SHA/path and a
preferred available source path, speaker, weak emotion, and candidate text.

JSON, CSV, and standalone HTML reports contain per-asset measurements, decisions,
structured reasons, reason counts, and min/p05/p50/p95/max distributions.

## Prototype finding and correction

The first 24-file prototype computed noise from the lowest percentile of all
frames. Digital-zero silence produced false SNR proxies around 80–100 dB. That
method was rejected before full-data use. The accepted implementation excludes
digital-zero frames from noise-floor estimation and reports their ratio
separately. This is why research prototypes precede the 277-file run.

## Sources consulted

- [ITU-R BS.1770 recommendation](https://www.itu.int/rec/R-REC-BS.1770/)
- [EBU loudness resources and R 128](https://tech.ebu.ch/loudness)
- [pyloudnorm repository](https://github.com/csteinmetz1/pyloudnorm)
- [SciPy `resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [WADA-SNR paper](https://www.isca-archive.org/interspeech_2008/kim08e_interspeech.html)
- [NISQA repository](https://github.com/gabrielmittag/NISQA)
- [Speech clipping detection study](https://pubmed.ncbi.nlm.nih.gov/35784517/)

## Consequences

- Threshold tuning and new policy versions are fast and do not re-decode audio.
- Standard LUFS is available without FFmpeg; true peak remains honestly labeled
  as an estimate.
- Near-full-scale and flat-top evidence are separate, reducing false clipping
  accusations.
- SNR/noise fields are useful relative diagnostics with explicit nullability and
  limitations, not claims of calibrated acoustic SNR.
- Reverb, music/background-source classification, DNSMOS, and NISQA remain
  deferred until a small human-labeled local benchmark can measure their value.
