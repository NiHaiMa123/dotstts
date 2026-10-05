# Slice 2 immutable ingest acceptance

- Date: 2026-08-26
- Result: Passed
- Input: `data/inbox`
- ADR: `docs/adr/0002-immutable-raw-ingest.md`

## Delivered

- `dots.tts.lab ingest` CLI with an inventory refresh before every import;
- independent-copy storage at `data/raw/sha256/<prefix>/<sha256>.<ext>`;
- exact-byte deduplication across source paths;
- streamed copy hash, file `fsync`, persisted-temp rehash, atomic replacement,
  and final-path size/SHA-256 verification;
- single-writer raw-store lock and importer-owned crash-partial cleanup;
- safe recovery when a verified file exists but its catalog transaction did not
  complete;
- explicit refusal to overwrite an existing content-addressed path whose bytes
  do not match its name;
- SQLite schema v2 with `import_run`, `raw_object`, and `import_item` provenance;
- atomic JSON, CSV, and standalone HTML ingest reports.

## Research and prototype gate

Copy, same-volume hard link, and external-reference modes were compared before
implementation. Independent copy was selected because it alone both survives
source deletion and isolates raw bytes from later source writes.

A 24-file prototype used six real files from each weak emotion directory:

- first run `506e4adf-27a0-4a25-9101-593c703b8799`: 24 imported, 0 reused,
  0 errors, 11,268,748 bytes written;
- second run `249f3574-9e12-4370-8c58-7b79f6f91796`: 0 imported, 24 reused,
  0 errors, 0 bytes written.

The temporary prototype workspace was removed after its results were recorded.

## Automated tests

Command:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Twelve tests passed, including five Slice 2 cases:

- exact duplicate sources produce one raw object;
- repeated import is idempotent and source deletion leaves raw available;
- simulated stream interruption removes the partial and the next run recovers;
- `KeyboardInterrupt` leaves no final/partial file and marks the run failed;
- an orphan crash partial is removed under the single-writer lock;
- an existing corrupt raw path is reported and not overwritten;
- report failure leaves a verified object that the next run safely adopts.

The seven Slice 1 inventory and catalog regressions also remain green.

## Full dataset acceptance

Initial full import run: `1c9239cc-ad66-46b6-877d-e96ca7100441`

- status: succeeded;
- 277 source locations and 277 unique assets;
- 277 imported, 0 reused, 0 errors;
- 155,261,416 bytes written;
- no orphan partials.

Independent post-import audit used PowerShell `Get-FileHash`, not the importer:

- 277 raw objects;
- 155,261,416 total bytes, matching the inbox byte total;
- 0 SHA-256/filename mismatches;
- 0 `.partial` files.

Final run with the accepted code: `040e843d-a3a2-4043-aaa7-e7fff400e8ed`

- status: succeeded;
- 0 imported, 277 reused, 0 errors;
- 0 bytes written;
- inventory refresh succeeded with 0 inventory errors.

Database verification after migration:

- schema version: 2;
- 277 assets;
- 277 available source locations;
- 277 raw objects;
- 277 source items in the latest import batch.

## Generated local artifacts

- `data/raw/sha256/`;
- `data/catalog/catalog.sqlite`;
- `data/reports/ingest/ingest.json`;
- `data/reports/ingest/ingest.csv`;
- `data/reports/ingest/ingest.html`.

These large/local artifacts are intentionally ignored by Git.

## Known limits deferred by design

- Slice 2 preserves bytes; it does not standardize sample rate, trim silence, or
  reject audio based on subjective quality.
- Raw corruption is detected but not auto-repaired or quarantined. A future
  explicit repair command must preserve evidence and audit history.
- The catalog and filesystem cannot share one atomic transaction. Verified
  orphan adoption covers every process-crash boundary without claiming a
  distributed transaction.
- Broad codec fallback remains deferred; this inbox is fully readable through
  soundfile/libsndfile.

## Next gate

Slice 3 must move the current filename/directory interpretation into a versioned
configuration profile. It must preserve the raw weak label, record parser
version, and route directory/filename conflicts to review rather than embedding
the current `崩铁符玄` layout in code.
