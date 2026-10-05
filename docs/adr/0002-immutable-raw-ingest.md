# ADR 0002: Immutable raw ingest

- Status: Accepted
- Date: 2026-08-26
- Scope: Slice 2 raw storage, durability, deduplication, and provenance

## Context

The inventory catalog identifies source bytes by SHA-256, but the only copy still
lives under `data/inbox`. The workbench needs a local raw object that survives an
inbox rename, replacement, or deletion and can be verified independently before
later preprocessing starts.

Three storage modes were compared on Windows:

| Mode | Survives source deletion | Isolated from source writes | Cross-volume | Space cost |
|---|---|---|---|---|
| Independent copy | Yes | Yes | Yes | One full copy per unique hash |
| Same-volume hard link | Yes | No | No | Directory entry only |
| External reference | No | No | Yes | None |

Microsoft documents that hard links are multiple paths to the same file on one
volume and that changes through any link are immediately visible through every
other link. That violates raw immutability: editing an inbox path would silently
edit the alleged archive too. An external reference fails as soon as a user
removes or replaces the source. Therefore neither is an acceptable default.

A small local timing check created 30 objects from 17,127,780 bytes on the `D:`
workspace. PowerShell-level copy took 61.81 ms and hard-link creation took 66.43
ms. Command overhead and filesystem caching dominate such a small sample, so the
result is not used as a performance claim; it does show that storage semantics,
not speed, should decide this slice.

## Decision

Use independent byte-for-byte copies as the only Slice 2 storage mode.

Each unique asset is stored once at:

```text
data/raw/sha256/<first-two-hex>/<sha256>.<original-extension>
```

The first observed safe lowercase extension is retained for usability; SHA-256,
not the extension or filename, is the identity. Later source locations with the
same byte hash reuse the cataloged object even if their names differ.

The import protocol is:

1. Refresh the read-only inventory and obtain the expected hash and byte size.
2. Acquire a single-writer lock for the raw root and remove only importer-owned
   orphan `.<hash>.<random>.partial` files from a prior crashed process.
3. Stream the source into a uniquely named temporary file in the destination
   directory while calculating SHA-256.
4. Flush and `fsync` the temporary file, then close it.
5. Reopen the persisted temporary file and calculate SHA-256 again.
6. Atomically replace the final content-addressed name with `os.replace`.
7. Reopen the final path and verify size plus SHA-256 before catalog commit.

Writing the temporary file in the target directory keeps the replacement on one
filesystem. The final content-addressed path is therefore either absent or a
fully verified file; it is never a partially copied file. A crash after the file
replacement but before the SQLite transaction leaves a verified untracked
object. The next run verifies and adopts it instead of copying again.

An existing final path with the wrong size or hash is reported as a raw integrity
error and is not silently overwritten. Repair or quarantine behavior requires a
future explicit command so evidence of storage corruption is not destroyed.

## Catalog model

Schema version 2 adds:

- `import_run`: one immutable-ingest batch and its terminal status/counters;
- `raw_object`: the one verified managed object for each asset SHA-256;
- `import_item`: every source path, original name, batch, action, raw path, and
  error recorded for that run.

`source_location` remains the first/last-discovery record. Together these tables
retain original Unicode paths and names, first and latest observations, import
batches, and the stable raw object without encoding provenance in filenames.

## Alternatives considered

### Prefer hard links and fall back to copy

Rejected. A hard link can save space, but its contents are not isolated from
source writes and it only works within one volume. Having two different
immutability guarantees under the same raw namespace would also make later
audits depend on historical storage topology.

### Keep only external references

Rejected. It is suitable for a read-only data lake managed elsewhere, but this
inbox is explicitly user-managed and may be cleaned after import.

### Copy metadata with `shutil.copy2`

Rejected for the object payload. Python documents that even `copy2` cannot copy
all Windows metadata, including owners, ACLs, and alternate data streams. The
workbench needs the original bytes plus explicit catalog provenance, not a
partial and platform-dependent metadata clone.

### Commit the database before copying

Rejected because the catalog could claim a raw object exists while only a
partial file is present. Files are verified first; the database transaction is
the final registration step.

## Sources consulted

- [Python `os` filesystem operations](https://docs.python.org/3/library/os.html#os.replace)
- [Python high-level file operations](https://docs.python.org/3/library/shutil.html#shutil.copyfile)
- [Microsoft hard links and junctions](https://learn.microsoft.com/en-us/windows/win32/fileio/hard-links-and-junctions)
- [Microsoft alternatives to Transactional NTFS](https://learn.microsoft.com/en-us/windows/win32/fileio/deprecation-of-txf)

## Consequences

- Deleting the inbox after a successful import does not remove the raw object.
- Exact duplicate bytes consume one raw object regardless of source filenames.
- Import costs one additional full copy of each unique asset, which is the price
  of isolation from user edits.
- The raw root is single-writer during an import; concurrent attempts fail
  visibly instead of racing on partial or final paths.
- Database and filesystem are not one distributed transaction, but verified
  orphan adoption makes every crash boundary idempotently recoverable.
