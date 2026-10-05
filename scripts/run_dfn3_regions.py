from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_dfn3 import (
    DEFAULT_DFN3_CONFIG_PATH,
    Dfn3VerificationError,
    load_dfn3_config,
    verify_backend,
)
from dots_tts_lab.long_form_dfn3_regions import (
    enhance_source_regions,
    make_df_enhancer,
    verify_delay_alignment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded context-window DeepFilterNet3 region enhancement "
        "(LF-13B). Runs inside the pinned DFN3 image."
    )
    parser.add_argument("--config", default=str(DEFAULT_DFN3_CONFIG_PATH))
    parser.add_argument("--source", required=True, help="Source audio path")
    parser.add_argument(
        "--regions",
        required=True,
        help="Prefilter regions.json (or any manifest with region rows)",
    )
    parser.add_argument(
        "--spatial-gate",
        default=None,
        help="JSON mapping region_index → downmix_allowed. Missing entries "
        "fail closed (no downmix).",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_dfn3_config(args.config)

    try:
        verify_backend(config, package_version=_df_version())
    except Dfn3VerificationError as error:
        print(f"backend verification failed: {error}", file=sys.stderr)
        return 1

    with Path(args.regions).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    regions = [
        row
        for row in manifest.get("regions", [])
        if row.get("status", "usable") == "usable"
    ]
    spatial_gate: dict[int, bool] | None = None
    if args.spatial_gate:
        with Path(args.spatial_gate).open("r", encoding="utf-8") as handle:
            raw_gate = json.load(handle)
        spatial_gate = {int(key): bool(value) for key, value in raw_gate.items()}

    started = time.time()
    enhancer = make_df_enhancer(config)
    residual = verify_delay_alignment(enhancer)
    if abs(residual) > 2:
        print(
            f"delay compensation residual {residual} samples exceeds ±2",
            file=sys.stderr,
        )
        return 1
    result = enhance_source_regions(
        args.source,
        regions,
        config=config,
        enhance_fn=enhancer,
        spatial_gate=spatial_gate,
    )
    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["delay_residual_samples"] = residual

    output = (
        Path(args.report).resolve()
        if args.report
        else Path(result["run_dir"]) / "run_report.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _df_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("deepfilternet")
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
