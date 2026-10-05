# Slice 3 metadata profile acceptance

- Date: 2026-08-27
- Result: Passed
- Input: `data/inbox`
- ADR: `docs/adr/0003-versioned-metadata-profiles.md`
- Profile: `configs/lab/metadata_profiles/speaker_emotion_filename_v1.yaml`

## Delivered

- strict, frozen Pydantic metadata profile loaded with PyYAML `safe_load`;
- generic one-based directory selectors for speaker and weak emotion;
- validated filename regex with named emotion/text groups;
- canonical profile JSON and SHA-256 identity;
- refusal to reuse a profile ID/version after its content changes;
- separate directory and filename emotion observations;
- `review_required` routing for conflicts without making inventory fail;
- catalog schema v3 with profile registry, current parse state, and versioned
  parse history;
- `--profile` support on both `inventory` and `ingest`;
- explicit UTF-8 CLI output for reliable Chinese JSON on Windows.

## Automated tests

Command:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Fifteen tests passed. Slice 3 coverage includes:

- unknown YAML fields are rejected;
- invalid directory selectors and regular expressions are rejected;
- missing named filename groups are rejected before scanning;
- directory levels are controlled by the profile rather than hardcoded Python;
- directory/filename emotion conflicts retain both labels and enter
  `review_required` while inventory succeeds;
- changing content under the same profile ID/version is blocked;
- incrementing `profile_version` succeeds and retains both parse histories.

All Slice 1 and Slice 2 inventory, ingest, interruption, integrity, and
idempotency regressions remain green.

## 24-file prototype

The prototype copied six real files from each weak emotion directory into an
isolated temporary inbox.

First run `021fb9a5-850f-4550-86a4-309a84a14e9c`:

- 24 discovered/readable/added;
- 24 metadata `ok`, 0 review-required, 0 errors;
- profile `speaker_emotion_filename@1`;
- profile SHA-256
  `027cc46de0b478368bf10d8d543314ea53e7bf2635515b24584a849893053b60`.

Second run `6ebf18a1-0866-43b9-9a37-8f32a858ee33`:

- 24 unchanged;
- 0 added/changed/moved/missing;
- still one registered profile and 24 parse-history rows, not 48.

A third run confirmed the CLI UTF-8 correction; Chinese speaker, emotion, and
path values were already correct in JSON/SQLite and then also rendered correctly
on stdout. The temporary prototype workspace was removed after verification.

## Full dataset acceptance

Before migration, the schema v2 catalog was copied to a temporary verified
backup. Full inventory run `68adc7ea-4901-4231-95c1-15e8705f6bff` then migrated
the working catalog to schema v3:

- 277 discovered and 277 readable;
- 277 unchanged;
- 277 metadata `ok`;
- 0 review-required and 0 errors;
- speaker: `崩铁符玄` for all records;
- neutral 81, happy 68, angry 122, sad 6;
- total duration 29.349375 minutes;
- 276 at 44.1 kHz and one at 36 kHz.

Database verification:

- schema version: 3;
- registered metadata profiles: 1;
- parse-history rows: 277;
- current metadata statuses: `ok=277`;
- existing raw objects preserved: 277;
- all prior import provenance rows preserved.

Final immutable-ingest regression run `552f58a1-0b0f-40d6-a1be-5288c9edc19e`:

- 277 unique assets;
- 0 imported, 277 reused;
- 0 bytes written and 0 errors;
- inventory refresh and raw verification succeeded.

## Generated artifacts

- `configs/lab/metadata_profiles/speaker_emotion_filename_v1.yaml`;
- `data/catalog/catalog.sqlite` schema v3;
- refreshed `data/reports/inventory/inventory.{json,csv,html}`;
- refreshed `data/reports/ingest/ingest.{json,csv,html}`.

Local data and reports remain intentionally ignored by Git.

## Known limits deferred by design

- Profiles currently support directory-level selection and one filename regex;
  sidecar manifests, arbitrary path expressions, and per-source profile routing
  should be added only when a real source requires them.
- Regex profiles are trusted local configuration. A future web editor for
  untrusted regex would require execution limits or a safer pattern language.
- Weak emotion labels are not human-verified truth and are not automatically
  promoted to training controls.
- `transcript_candidate` still requires ASR comparison and/or human review before
  becoming `text_train`.

## Next gate

Slice 4 must research objective audio-quality metrics and threshold behavior on
the actual distribution. Metrics should be cached independently from policy so
changing a YAML threshold does not require re-decoding all 277 files. The known
36 kHz file, six near-full-scale files, and two long-tail-silence files must be
reproduced before adding model-based quality estimators.
