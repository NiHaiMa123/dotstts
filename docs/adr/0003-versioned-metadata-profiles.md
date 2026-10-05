# ADR 0003: Versioned metadata parsing profiles

- Status: Accepted
- Date: 2026-08-27
- Scope: Slice 3 directory/filename metadata parsing and review routing

## Context

Slice 1 inferred the speaker from the first directory, the weak emotion from the
second directory, and candidate text from filenames shaped like
`【emotion】text.wav`. Those rules were Python constants. That was sufficient to
inventory the first `崩铁符玄` batch, but it made a repository code change
necessary for every future source layout and did not identify which parser
semantics produced a catalog value.

The parser now needs to be configurable while preserving these guarantees:

- malformed or misspelled configuration fails before a catalog run begins;
- the exact parser semantics are reproducible later;
- changing a published profile cannot silently reinterpret old data;
- directory and filename emotion values remain separate weak observations;
- a disagreement becomes review work, not an implicit winner.

## Options considered

### Standard-library dataclass plus manual validation

This adds no dependency, but every unknown-field, type, selector, regex, and
cross-field error requires bespoke validation and error formatting. The project
already depends on Pydantic, so duplicating a weaker schema layer has no benefit.

### PyYAML plus Pydantic v2

PyYAML provides a readable version-controlled format and `safe_load` avoids
constructing arbitrary Python objects. Pydantic can reject extra fields, enforce
strict types and constrained values, and run cross-field validation for regex
named groups. Both are existing project dependencies.

### Hydra or OmegaConf composition

These are useful for large experiment configuration graphs, interpolation, and
command-line overrides. Slice 3 has one small immutable profile and does not need
another dependency or dynamic composition semantics. Revisit this only if the
whole training configuration system later standardizes on one of them.

## Decision

Use UTF-8 YAML loaded with `yaml.safe_load`, then validate it as a frozen,
strict Pydantic model with `extra="forbid"`.

The initial generic profile is:

```yaml
schema_version: 1
profile_id: speaker_emotion_filename
profile_version: 1
speaker_from: level_1_directory
emotion_from: level_2_directory
filename_pattern: '^【(?P<emotion>[^】]+)】(?P<text>.+)$'
filename_emotion_group: emotion
transcript_group: text
trim_values: true
conflict_policy: review_required
```

Directory levels are one-based relative to the supplied inventory root. The
filename regex is compiled before scanning and must contain the configured named
groups. Matching uses `fullmatch` against the filename stem so partial matches do
not silently accept suffix garbage. Profiles are trusted, version-controlled
local configuration; accepting user-supplied arbitrary regex through a web UI is
outside this decision and would require additional complexity/time limits.

The validated model, including explicit defaults, is serialized as canonical
sorted compact JSON and hashed with SHA-256. The catalog registers
`(profile_id, profile_version, config_sha256, config_json, source_path)`. If the
same ID/version is later supplied with different content, inventory fails with a
required version-bump error. A new version creates a new parse history rather
than overwriting the prior interpretation.

## Metadata and review semantics

Each current source location and each versioned `metadata_parse` history row
stores:

- `speaker_id`;
- `directory_emotion_weak_label`;
- `filename_emotion_weak_label`;
- effective `emotion_weak_label`;
- `transcript_candidate`;
- profile ID, version, and config SHA-256;
- status, parse error, and review reason.

If directory and filename emotions disagree, both values are retained,
`metadata_status` becomes `review_required`, and the directory value remains the
effective weak label for current compatibility. The inventory itself succeeds
and increments `review_required_count`; this is not a decode/hash failure. A
missing required directory, filename mismatch, or empty candidate text is a true
metadata error and contributes to `error_count`.

`text_candidate` remains a filename-derived hypothesis. This slice does not
promote it to reviewed training text and does not let ASR or a classifier replace
human-provided weak observations.

## Catalog migration

Schema version 3 adds:

- `metadata_profile`, one immutable registered definition per ID/version;
- version/status fields on `source_location` for the latest interpretation;
- `metadata_parse`, keyed by source location plus profile identity/hash, for
  reproducible history.

SQLite foreign keys connect parse history to both the source location and the
registered profile. Repeated scans with the same profile update the timestamp and
current derived fields without duplicating history rows.

## Sources consulted

- [Pydantic model configuration](https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict)
- [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [PyYAML documentation and `safe_load`](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Python regular-expression named groups](https://docs.python.org/3/library/re.html)
- [SQLite foreign keys](https://sqlite.org/foreignkeys.html)

## Consequences

- New directory conventions normally require a new YAML profile, not Python
  edits.
- Changing semantics requires an explicit `profile_version` bump and preserves
  old parse history.
- Unknown keys and invalid regex/groups fail early with structured validation
  errors.
- Review-required records are queryable separately from processing failures.
- The current profile supports directory-level selection and named filename
  groups; arbitrary path expressions and multi-file sidecars remain future
  extensions driven by real input formats.
