from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.release import export_preset, load_release_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Export accepted Slice 13 audio")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lab/slice13/release_presets_v1.yaml"),
    )
    parser.add_argument("--preset", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--job-id", action="append")
    parser.add_argument(
        "--run-label",
        help="Required for filtered smoke/debug exports; isolates their manifests",
    )
    args = parser.parse_args()
    config = load_release_config(args.config)
    presets = args.preset or list(config.presets)
    reports = [
        export_preset(
            args.config,
            preset,
            limit=args.limit,
            job_ids=set(args.job_id) if args.job_id else None,
            run_label=args.run_label,
        )
        for preset in presets
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
