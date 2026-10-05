from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_identity import (
    DEFAULT_IDENTITY_CONFIG_PATH,
    load_identity_config,
    run_identity,
)
from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.long_form_prefilter import (
    DEFAULT_PREFILTER_CONFIG_PATH,
    load_prefilter_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Speaker timeline + target identity evidence (LF-12D). "
        "Consumes prefilter regions.json manifests."
    )
    parser.add_argument("--config", default=str(DEFAULT_IDENTITY_CONFIG_PATH))
    parser.add_argument(
        "--prefilter-config", default=str(DEFAULT_PREFILTER_CONFIG_PATH)
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        required=True,
        help="Source audio paths; each must have a completed prefilter stage.",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_identity_config(args.config)
    prefilter = load_prefilter_config(args.prefilter_config)
    root = Path(__file__).resolve().parents[1]

    from dots_tts_lab.long_form_audio import file_sha256

    jobs: list[tuple[Path, list[dict]]] = []
    for source_arg in args.sources:
        source = Path(source_arg).resolve()
        digest = file_sha256(source)
        manifest_path = (
            root
            / prefilter.paths.work_root
            / digest[:2]
            / digest
            / "regions.json"
        )
        if not manifest_path.is_file():
            print(
                f"missing prefilter regions for {source} — run "
                "scripts/run_long_form_prefilter.py first",
                file=sys.stderr,
            )
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != digest:
            print(f"prefilter manifest drift for {source}", file=sys.stderr)
            return 2
        jobs.append((source, manifest.get("regions", [])))

    started = time.time()
    result = run_identity(jobs, config=config)
    result["elapsed_seconds"] = round(time.time() - started, 3)

    report_root = validate_output_path(root / config.paths.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    output = (
        Path(args.report).resolve()
        if args.report
        else report_root / "identity_run.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
