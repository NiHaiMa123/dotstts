from __future__ import annotations

import argparse
import json
from pathlib import Path

from dots_tts_lab.long_form_pipeline import run_long_form_pipeline
from dots_tts_lab.long_form_refinement import refine_long_form_manifest
from dots_tts_lab.long_form_review import build_long_form_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build reviewable training candidates from long-form audio."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Keep the broad ASR candidates without building the reduced review set.",
    )
    parser.add_argument("--refinement-config", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-sources", type=int)
    args = parser.parse_args()
    result = run_long_form_pipeline(
        args.input_dir,
        config_path=args.config,
        work_root=args.work_root,
        report_root=args.report_root,
        skip_asr=args.skip_asr,
        force=args.force,
        limit_sources=args.limit_sources,
    )
    refinements = []
    pending_asr = [] if args.skip_asr else [
        source for source in result["sources"] if source["asr_status"] in {"partial", "failed"}
    ]
    if not args.skip_asr and not args.skip_refinement:
        for source in result["sources"]:
            if source["asr_status"] != "succeeded":
                continue
            keyword = {}
            if args.refinement_config is not None:
                keyword["refinement_config_path"] = args.refinement_config
            refined = refine_long_form_manifest(
                source["manifest_path"],
                args.input_dir.resolve() / source["source_relative_path"],
                long_form_config_path=args.config,
                **keyword,
            )
            refined["source_sha256"] = source["source_sha256"]
            refined["source_relative_path"] = source["source_relative_path"]
            refined["review"] = build_long_form_review(refined["manifest_path"])
            refinements.append(refined)
    result["refinements"] = refinements
    result["pending_asr_sources"] = pending_asr
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if pending_asr else 0


if __name__ == "__main__":
    raise SystemExit(main())
