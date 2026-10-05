# Slice 1 inventory acceptance

- Date: 2026-08-26
- Result: Passed
- Input: `data/inbox`
- ADR: `docs/adr/0001-inventory-foundations.md`

## Delivered

- `dots.tts.lab inventory` CLI;
- SQLite schema v1 with `ingest_run`, `asset`, and `source_location`;
- SHA-256 content identity and source-change detection;
- speaker, weak emotion, and filename transcript parsing;
- soundfile/libsndfile audio metadata probing;
- added/changed/moved/missing/unchanged classification;
- atomic JSON, CSV, and standalone HTML reports;
- explicit failed and completed-with-errors run states;
- Windows Unicode and near-MAX_PATH diagnostics.

## Automated tests

Command:

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Seven tests passed:

- empty and repeated inventory;
- Unicode paths and deterministic metadata;
- content changes, moves, and missing files;
- corrupt audio and invalid filename reporting;
- directory/filename emotion conflicts;
- near-legacy Windows path diagnostics;
- failed report generation never producing a successful ingest run.

The first run exposed leaked SQLite handles on Windows. The catalog now owns an
explicit connection-closing context manager, and the regression is covered by
the full integration suite.

## Real dataset acceptance

First run ID: `f12c3172-d47a-44c1-ab8a-92aee8c9243f`

- status: succeeded;
- 277 discovered and 277 readable;
- 277 added, 0 errors;
- 1,760.962502 seconds / 29.349375 minutes;
- neutral 81, happy 68, angry 122, sad 6;
- 44.1 kHz: 276, 36 kHz: 1.

Second run ID: `495d0abd-7905-4336-8cb1-c3b441c2b5ae`

- status: succeeded;
- 0 added/changed/moved/missing;
- 277 unchanged.

Database verification:

- schema version: 1;
- successful ingest runs: 2;
- unique assets: 277;
- available source locations: 277;
- JSON file records: 277;
- CSV file records: 277.

## Generated local artifacts

- `data/catalog/catalog.sqlite`;
- `data/reports/inventory/inventory.json`;
- `data/reports/inventory/inventory.csv`;
- `data/reports/inventory/inventory.html`.

These artifacts are intentionally ignored by Git and can be rebuilt from the
inbox.

## Known limits deferred by design

- No source audio has been copied into the immutable `raw` store yet.
- Soundfile is the only probe; ffprobe fallback is deferred until broader
  containers/codecs require it.
- Inventory records header metadata, not full-decode signal quality.
- Filename parsing follows the current three-level convention and becomes a
  configurable profile in a later slice.
- mtime/size are recorded, but SHA-256 remains canonical and is recomputed on
  every Slice 1 scan; safe hash caching can be evaluated after immutable ingest.

## Next gate

Slice 2 must research and benchmark copy versus same-volume hardlink versus
external-reference modes. The default must survive deletion or replacement of
files under `data/inbox`, use atomic temporary files plus rename, verify the
destination hash, and remain idempotent after interruption.
