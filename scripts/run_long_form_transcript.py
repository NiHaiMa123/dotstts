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
)
from dots_tts_lab.long_form_transcript import (
    DEFAULT_TRANSCRIPT_CONFIG_PATH,
    load_transcript_config,
    run_transcript,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dual-ASR transcript + sentence slicing (LF-12F)."
    )
    parser.add_argument("--config", default=str(DEFAULT_TRANSCRIPT_CONFIG_PATH))
    parser.add_argument(
        "--prefilter-config", default=str(DEFAULT_PREFILTER_CONFIG_PATH)
    )
    parser.add_argument("--sources", nargs="*", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_transcript_config(args.config)
    prefilter = load_prefilter_config(args.prefilter_config)
    root = Path(__file__).resolve().parents[1]

    from dots_tts_lab.long_form_audio import file_sha256

    jobs: list[tuple[Path, list[dict]]] = []
    for source_arg in args.sources:
        source = Path(source_arg).resolve()
        digest = file_sha256(source)
        manifest_path = (
            root / prefilter.paths.work_root / digest[:2] / digest / "regions.json"
        )
        if not manifest_path.is_file():
            print(
                f"missing prefilter regions for {source}",
                file=sys.stderr,
            )
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != digest:
            print(f"prefilter manifest drift for {source}", file=sys.stderr)
            return 2
        jobs.append((source, manifest.get("regions", [])))

    started = time.time()
    result = run_transcript(jobs, config=config)
    result["elapsed_seconds"] = round(time.time() - started, 3)

    report_root = validate_output_path(root / config.paths.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    output = (
        Path(args.report).resolve()
        if args.report
        else report_root / "transcript_run.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
