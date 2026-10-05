from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dots_tts_lab.long_form_export import (
    DEFAULT_LONG_FORM_EXPORT_CONFIG_PATH,
    export_long_form_dataset,
)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Export an approved long-form dataset fragment.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("review_snapshot", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_LONG_FORM_EXPORT_CONFIG_PATH)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_long_form_dataset(
                args.manifest,
                args.review_snapshot,
                config_path=args.config,
                target=args.target,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

