# Slice 14 Acceptance Report

Date: 2026-09-11

## Scope

Slice 14 implements a local, single-user Fuxuan TTS WebUI around the accepted
Step-400 runtime and the existing TXT batch service. It is intentionally not a
general model laboratory or a replacement for the earlier review pages.

## Delivered behavior

- Root-level `启动扶玄TTS.vbs` launches the application with a hidden Python
  process and opens the browser only after the local server is bound.
- The primary UI has one model selector and one TTS action. The accepted
  `fuxuan_step400_v1` model is selected by default.
- The TTS action force-regenerates every UTF-8 TXT under
  `inputs/fuxuan_text/` to a corresponding final WAV under
  `outputs/fuxuan_audio/`.
- The existing batch service remains authoritative for splitting, inference,
  safe edge trimming, concatenation, the E brightness profile, true-peak
  protection, and atomic final writes.
- Progress is persisted across model loading, files, sentence synthesis,
  finalization, completion, failure, interruption, and shutdown.
- Successful completion enables an action that opens the output directory in
  Windows Explorer.
- Browser close notification is debounced for refreshes. A heartbeat watchdog
  handles abrupt tab/browser termination and shuts down the local server.
- Startup recovers an unfinished prior state as `interrupted`, removes only the
  dedicated work directory and `*.partial.wav`, and preserves completed WAVs.
- A second launch reuses the healthy existing server instead of loading a
  second model instance.

## Persistent evidence

- Current state: `outputs/fuxuan_webui/state.json`
- Append-only transition history: `outputs/fuxuan_webui/history.jsonl`
- Ephemeral server registration: `outputs/fuxuan_webui/server.json`
- Hidden-launch failure report: `outputs/fuxuan_webui/startup-error.log`

## Verification

- Unit coverage includes text batching, cancellation before final write,
  state-transition validation, interrupted-run recovery, stale-artifact
  cleanup, completed-output preservation, requested primary controls, and a
  complete fake-runtime worker path through `completed`.
- A live HTTP lifecycle returned the page and state API, accepted heartbeat and
  close events, exited with code 0, removed the server registration, and left
  zero work or partial files.
- The actual VBS launcher was executed. It produced a healthy registration at
  `http://127.0.0.1:7868`, with no visible terminal window.
- Browser inspection confirmed the default model, TTS action, idle state, and
  progress surface render in the first viewport.
- The complete repository suite passed 197/197 tests after the near-field
  postprocess integration.
