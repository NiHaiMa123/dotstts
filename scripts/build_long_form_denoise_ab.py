from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.long_form_enhancement import build_denoise_ab, rebuild_denoise_ab_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative DeepFilterNet blind AB package.")
    parser.add_argument("--source-root", type=Path, default=Path("data/work/long_form/sources"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lab/long_form/denoise_ab_v1.yaml"),
    )
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    if args.review_only:
        manifest = Path("data/reports/long_form/denoise_ab_v1/manifest.json")
        print(json.dumps(rebuild_denoise_ab_review(manifest), ensure_ascii=False, indent=2))
        return
    print(
        json.dumps(
            build_denoise_ab(source_root=args.source_root, config_path=args.config),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
