from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.long_form_refinement import refine_long_form_manifest
from dots_tts_lab.long_form_review import build_long_form_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim timestamp-ASR sentences and build a reduced long-form review page."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--refinement-config", type=Path)
    parser.add_argument("--long-form-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    keyword = {}
    if args.refinement_config is not None:
        keyword["refinement_config_path"] = args.refinement_config
    result = refine_long_form_manifest(
        args.manifest,
        args.source,
        long_form_config_path=args.long_form_config,
        output_root=args.output_root,
        **keyword,
    )
    result["review"] = build_long_form_review(result["manifest_path"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

