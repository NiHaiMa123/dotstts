# Slice 5 acceptance: Standardized training audio

- Date: 2026-08-27
- Result: Passed
- Config: `training_audio@1`
- Config SHA-256: `a2a2c7d00ec638dc3e420692785c604b86d0997202dd35abdcff17c8abdda4d6`
- Implementation version: 1

## Delivered behavior

The `dots.tts.lab standardize` command now derives reproducible training WAVs
from the immutable raw store. Catalog schema v5 records config identity, run
status, per-asset actions, output hashes and signal-transform metadata.

The accepted pipeline is float64 decode, mean mono mix, SoXR HQ to 48 kHz,
conservative edge trim, attenuation-only -1 dBTP protection, and atomic PCM24
WAV output. Loudness normalization, DC removal, denoising, limiting, and internal
silence removal are disabled.

Artifacts are content/config/code addressed below `data/work/standardized`.
Cache reuse verifies size and full SHA-256. Missing or corrupt outputs rebuild;
config drift under an existing version is blocked; report failure marks the run
failed without committing derived catalog rows.

## Research and prototype

ADR 0005 records the local SoXR/SciPy/torchaudio benchmark and processing
decision. The reproducible benchmark data is in
`data/reports/research/resampler_benchmark_v1.json`.

The 24-asset prototype is fixed in
`data/reports/research/slice5_prototype_assets.json`:

- 10 angry, 4 happy, 4 neutral, and all 6 sad assets;
- the only 36 kHz source;
- all seven source true-peak estimates above 0 dBTP;
- both source tails above 0.5 seconds.

First prototype run: 24 built, 0 errors. Exactly the two known long-tail assets
were trimmed; nine assets needed attenuation. The applied range was about
-0.91 to -1.13 dB, and the maximum output true-peak estimate was approximately
-1.000 dBTP. The immediate repeat was 24 cached, 0 decoded/rebuilt/errors.

## Full-data result

First full run after the prototype:

| Metric | Result |
|---|---:|
| Discovered | 277 |
| Newly built | 253 |
| Reused from prototype | 24 |
| Errors | 0 |
| Output sample rate / channels / subtype | 48 kHz / mono / PCM24 |
| Trimmed assets | 2 |
| Total removed edge audio | 46,119 samples / 0.9608 s |
| Attenuated assets | 24 |
| Gain range among attenuated assets | -1.1285 to -0.0271 dB |
| Output duration | 29.3334 min |
| Output size | 253,452,604 bytes |
| Maximum output true-peak estimate | -0.999999 dBTP |

The two trimmed assets removed 21,160 and 24,959 trailing samples respectively.
No leading edge or internal silence was removed. The duration difference from
raw is fully accounted for by these 46,119 output-rate samples.

The second full run reported 0 built, 277 cached, 0 rebuilt, and 0 errors.

## Independent audit

`scripts/audit_standardized.py` independently inspected the catalog and files:

- 277/277 raw objects retained the exact asset SHA-256;
- 277/277 derived objects matched catalog SHA-256 and size;
- 277/277 decoded as 48 kHz mono PCM24 with the recorded frame count;
- recomputed maximum 4x true-peak estimate was -0.999999 dBTP;
- zero audit errors.

The machine-readable audit is
`data/reports/standardization/integrity_audit.json`.

## Automated verification

The suite now contains 29 passing tests (21 previous plus 8 Slice 5 tests). New
coverage includes 36/44.1 kHz resampling, stereo mean mix, long versus short edge
silence, retained padding, attenuation-only protection, PCM24 metadata, cache
reuse without decoding, config drift, corrupt-artifact rebuild, report-failure
recovery, and raw immutability. Python compilation checks and `pip check` pass.

## Acceptance conclusion

Slice 5 meets its acceptance gates: every source decodes into a verified
48 kHz mono training artifact, correspondence remains one-to-one, only configured
edge trimming changes duration, no new clipping is present, raw is unchanged,
and a repeated run fully reuses verified cache. The project may proceed to
Slice 6, the manually anchored ASR/transcript-verification benchmark.
