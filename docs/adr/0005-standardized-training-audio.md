# ADR 0005: Standardized 48 kHz training audio

- Status: Accepted
- Date: 2026-08-27
- Scope: Slice 5 resampling, edge trimming, peak protection, artifact identity

## Context

The immutable raw store contains 277 mono PCM16 WAV assets: 276 at 44.1 kHz and
one at 36 kHz. Training and later segmentation need a uniform, rebuildable audio
representation without changing raw bytes. The choices here must preserve speech
and expressive dynamics, avoid new clipping, and remain practical on Windows.

Slice 4 measured a compact loudness distribution (p05 -18.34, median -16.98,
p95 -15.21 LUFS) and very small DC offsets (maximum absolute mean 0.00213). Seven
true-peak estimates exceed 0 dBTP, while only two files have trailing silence
above 0.5 seconds. These facts argue for targeted intervention rather than broad
normalization or filtering.

## Resampler benchmark

The reproducible benchmark is `scripts/benchmark_resamplers.py`; its complete
machine-readable result is `data/reports/research/resampler_benchmark_v1.json`.
It ran on Python 3.12.13 / Windows 11 with python-soxr 1.1.0, SciPy 1.18.1,
PyTorch/torchaudio 2.8.0, NumPy 2.2.6.

Three implementations were compared:

- python-soxr/libsoxr `HQ`;
- SciPy `resample_poly` with its default FIR window;
- torchaudio Kaiser-windowed sinc using the documented high-quality parameters.

The synthetic input is an analytically generated, common-bandwidth multitone at
36 and 44.1 kHz. Reconstruction at 48 kHz is compared with the same analytical
signal sampled directly at 48 kHz; 100 ms edges are excluded to avoid conflating
finite-filter boundary behavior with steady-state error. A separate tone at 46%
of source sample rate measures near-Nyquist gain and the first interpolation
image. Four real clips cover the 36 kHz exception and all weak emotion labels.

| Backend | Multitone SNR, 36/44.1 kHz | Near-Nyquist gain | Image rejection | Real-clip median time |
|---|---:|---:|---:|---:|
| SoXR HQ | 133.00 / 132.67 dB | -0.002 / -0.011 dB | -134.4 / -86.4 dB | 1.06–2.31 ms |
| SciPy default polyphase | 62.42 / 64.38 dB | about -1.19 dB | about -16.6 dB | 1.68–3.86 ms |
| torchaudio HQ sinc | 155.65 / 152.68 dB | about -0.55 dB | -156.0 / -86.4 dB | 0.91–1.62 ms |

SoXR produced exactly the rounded expected frame count on every synthetic and
real input. SciPy and torchaudio produced one extra frame on some 44.1 kHz real
clips. The absolute synthetic figures depend on this benchmark's signals and are
not general codec scores; their purpose is a reproducible local comparison.

## Decision

Use python-soxr 1.1.0 / libsoxr `HQ` as the default backend. It combines strong
measured reconstruction, predictable length, millisecond-scale CPU cost, a small
audio-focused dependency, Windows wheels, and LGPL-2.1-or-later licensing.
torchaudio remains a qualified alternative when the standardization worker
already loads PyTorch and its length convention is made explicit. Do not use the
SciPy default polyphase settings here; SciPy remains the independent 4x
true-peak estimator and a possible fallback with a deliberately designed window.

librosa 0.11 is not a separate candidate because its default `soxr_hq` path is a
wrapper around python-soxr. Adding it would not provide an independent engine.

The fixed processing order is:

```text
immutable raw WAV
-> float64 decode
-> arithmetic-mean mono mix
-> SoXR HQ resample to 48 kHz
-> conservative edge-silence trim
-> attenuation-only peak protection
-> mono PCM24 WAV + SHA-256 verification
```

Edge trimming uses 20 ms frames, 10 ms hops, and -50 dBFS activity. An edge is
eligible only when estimated silence exceeds 0.5 seconds; approximately 0.1
seconds of padding is retained. Internal pauses are never removed. All-silent
audio is left intact for later review rather than converted to an empty file.

Peak protection estimates inter-sample peak by 4x polyphase interpolation. Gain
is applied only if needed to target -1 dBTP; it can never amplify a clip. The
post-PCM24 file is decoded again and its sample/true-peak estimates are recorded.
No limiter is used because nonlinear peak shaping could alter training targets.

Do not apply dataset-wide loudness normalization: the current distribution is
already consistent, and relative intensity carries emotion/prosody information.
Do not remove DC by default because the observed offset is negligible. Do not
denoise or dereverberate in this slice.

PCM24 is used for derived WAVs so the float-domain resampling result is not
immediately reduced to the source's PCM16 precision. Raw remains the archival
truth; storage can be reconsidered if dataset scale makes PCM24 cost material.

## Identity, cache, and recovery

`derived_id` is SHA-256 over source asset SHA-256, canonical processing-config
SHA-256, and implementation version. Files live at:

`data/work/standardized/<derived-id-prefix>/<derived-id>.wav`

Catalog schema v5 registers immutable config versions, runs, derived metadata,
output hash/size, trim/gain facts, and per-run actions. A cache hit verifies file
size and full SHA-256 without decoding. Missing or corrupt derived files are
rebuilt atomically through a same-directory `.partial` file, `fsync`, decode and
metadata verification, hash calculation, and replace. A report/catalog failure
can leave a valid rebuildable file but never a falsely successful run.

## Sources consulted

- [python-soxr repository and usage](https://github.com/dofuuz/python-soxr)
- [libsoxr project](https://sourceforge.net/projects/soxr/)
- [SciPy `resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [torchaudio resampling tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html)
- [librosa `resample`](https://librosa.org/doc/latest/generated/librosa.resample.html)
- [WAVE subtype support in python-soundfile](https://python-soundfile.readthedocs.io/)

## Consequences and exit conditions

- Later stages receive consistent 48 kHz mono inputs while raw bytes stay intact.
- Reprocessing is deterministic by explicit config/code identity and recoverable
  from missing/corrupt work artifacts.
- Standardization preserves loudness and DC rather than silently changing data.
- Rebenchmark or replace SoXR if its wheel/platform support changes, a local
  listening test exposes speech artifacts, frame-count semantics become
  incompatible with downstream timestamps, or a future corpus contains source
  rates/bandwidths not covered by this benchmark.
