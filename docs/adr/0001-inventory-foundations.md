# ADR 0001: Slice 1 inventory foundations

- Status: Accepted
- Date: 2026-08-26
- Scope: read-only inventory, catalog schema, file identity, Windows paths

## Context

Slice 1 must inventory local audio without modifying it. It needs reliable WAV
metadata, deterministic content identity, a small resumable catalog, and correct
handling of the existing Chinese filenames on Windows. The current inbox has
277 mono PCM16 WAV files (276 at 44.1 kHz and one at 36 kHz), so this decision
optimizes for correctness and a low dependency footprint before adding wider
container support.

## Decision

### Audio probing

Use `soundfile` 0.13.x / libsndfile as the primary Slice 1 probe.

- It is already a dots.tts dependency.
- Its platform wheels bundle libsndfile on Windows.
- It exposes format, subtype, channels, sample rate, frames, and duration.
- Probe errors retain their exception type and message in the catalog.

Do not require FFmpeg in Slice 1. `ffprobe` is the planned fallback when the
inbox accepts containers/codecs that libsndfile cannot identify. The fallback
must be explicit in each asset record so results do not depend silently on what
executables happen to be installed.

### SQLite and migrations

Use Python's standard-library `sqlite3` directly and maintain ordered SQL
migrations with `PRAGMA user_version`.

Slice 1 has three small application-owned tables and no ORM requirement.
Alembic would add SQLAlchemy, generated environments, and SQLite batch-migration
complexity before it provides value. Revisit Alembic if the workbench later
adopts SQLAlchemy or schema changes become frequent and multi-branch.

Each migration runs transactionally. Connections enable foreign keys, a busy
timeout, and WAL mode. An inventory run is inserted as `running` before work;
unexpected failures mark it `failed`, while bad source files produce
`completed_with_errors` and remain visible for review.

### File identity

Use SHA-256 of the original bytes as `asset_id`. Hash files in 8 MiB chunks.

`hashlib.file_digest()` is efficient, but a small explicit loop preserves a
future hook for progress, cancellation, and rate limiting. The implementation
checks size and `mtime_ns` before and after hashing and retries once if a source
changes during the scan. Partial digests are never committed.

### Windows and Unicode paths

Use `pathlib.Path` and preserve the exact Unicode filename. Do not apply Unicode
normalization: Windows treats filenames as opaque WCHAR sequences. Store a
portable relative path with `/` separators plus a platform comparison key. On
Windows the comparison key is case-folded because the normal filesystem is
case-insensitive but case-preserving.

Do not mutate the machine-wide `LongPathsEnabled` registry setting. Record path
length and surface an explicit diagnostic when a path approaches the legacy
260-character boundary. Tests cover Unicode and long components within the
filesystem's per-component limit.

## Alternatives considered

### ffprobe for every file

More container coverage and machine-readable JSON, but it adds a large external
runtime and subprocess overhead. Keep it as a later fallback, not the base WAV
dependency.

### SQLAlchemy plus Alembic

Appropriate for a larger service, but excessive for the first three tables.
Alembic's own documentation notes SQLite's move-and-copy workflow for many
ALTER operations.

### Fast metadata key instead of a content hash

Size and mtime are useful scan accelerators but cannot prove identity across
moves or timestamp changes. They remain catalog fields; SHA-256 is canonical.

### Automatic Unicode normalization or filename rewriting

Rejected because it can merge distinct source names and violates the rule that
inbox content is immutable.

## Sources consulted

- [python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/en/0.13.1/)
- [ffprobe documentation](https://ffmpeg.org/ffprobe.html)
- [SQLite PRAGMA documentation](https://www.sqlite.org/pragma.html#pragma_user_version)
- [Alembic SQLite batch migrations](https://alembic.sqlalchemy.org/en/latest/batch.html)
- [Python hashlib file hashing](https://docs.python.org/3/library/hashlib.html#file-hashing)
- [Microsoft MAX_PATH documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)

## Consequences

- Slice 1 remains small, offline, and Windows-friendly.
- Unsupported compressed containers are cataloged as probe errors until the
  ffprobe fallback is implemented.
- Schema migrations are deliberately explicit and must include upgrade tests.
- Catalog paths are traceable without renaming or mutating source files.
