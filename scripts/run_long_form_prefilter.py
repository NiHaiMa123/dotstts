from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.long_form_prefilter import (
    DEFAULT_PREFILTER_CONFIG_PATH,
    load_prefilter_config,
    run_prefilter,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Low-cost multi-source pre-screen for long-form audio (LF-12C)."
    )
    parser.add_argument("--config", default=str(DEFAULT_PREFILTER_CONFIG_PATH))
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Source audio paths. Default: every file in data/inbox sorted by name.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON output path (default: <report_root>/prefilter_run.json)",
    )
    args = parser.parse_args()

    config = load_prefilter_config(args.config)
    root = Path(__file__).resolve().parents[1]

    if args.sources:
        sources = [Path(item).resolve() for item in args.sources]
    else:
        inbox = (root / "data" / "inbox").resolve()
        sources = sorted(
            path
            for path in inbox.rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        )
    if not sources:
        print("no sources found", file=sys.stderr)
        return 2

    started = time.time()
    result = run_prefilter(sources, config=config)
    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["source_count"] = len(sources)

    report_root = validate_output_path(root / config.paths.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    output = (
        Path(args.report).resolve()
        if args.report
        else report_root / "prefilter_run.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
