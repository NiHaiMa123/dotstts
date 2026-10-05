# Local data workspace

Put new source audio only in:

`data/inbox/`（仓库根目录下）

Subdirectories are allowed and can be used to represent speakers, batches, or
source collections. Supported formats will initially include WAV, FLAC, MP3,
M4A, and OGG.

If an audio file already has an exact verbatim transcript, place a UTF-8 text
file with the same stem next to it, for example `clip_001.wav` and
`clip_001.txt`. Do not use category labels, filenames, or approximate summaries
as transcripts. Audio without a transcript should simply have no matching text
file; the future preprocessing pipeline will transcribe and review it.

The pipeline treats `inbox` as append-only input. `dots.tts.lab ingest` copies
each unique source byte stream into the verified immutable content-addressed
store at `data/raw/sha256`. It does not use hard links, so later edits or deletion
under `inbox` cannot modify or remove the raw object. Source files are never
modified in place.

Everything under `data` except this file and `inbox/.gitkeep` is intentionally
ignored by Git.

Directory and filename interpretation is controlled by the versioned profile at
`configs/lab/metadata_profiles/speaker_emotion_filename_v1.yaml`. Use
`dots.tts.lab inventory --profile <profile.yaml>` for a different source layout;
increment `profile_version` whenever an existing profile's semantics change.

Run cached objective signal analysis and the conservative training-source review
policy with:

```powershell
.venv\Scripts\dots.tts.lab.exe quality
```

Measurement semantics and review thresholds are versioned separately under
`configs/lab/quality/`. Editing a registered config requires a version bump;
changing only the policy reuses stored metrics and does not decode audio again.
