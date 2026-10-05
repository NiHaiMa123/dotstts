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
from dots_tts_lab.long_form_route import (
    DEFAULT_ROUTE_CONFIG_PATH,
    load_route_config,
    run_routes,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-region raw/DFN3 routing and materialisation (LF-13C). "
        "The DFN3 subset only runs when the pinned backend is importable "
        "(inside the DFN3 image); without it, dfn3-routed regions abort the "
        "run rather than falling back to raw."
    )
    parser.add_argument("--config", default=str(DEFAULT_ROUTE_CONFIG_PATH))
    parser.add_argument("--dfn3-config", default=str(DEFAULT_DFN3_CONFIG_PATH))
    parser.add_argument("--source", required=True, help="Source audio path")
    parser.add_argument("--regions", required=True, help="Prefilter regions.json")
    parser.add_argument(
        "--style-event",
        default=None,
        help="style_event.json for stationary-noise evidence (optional; "
        "missing evidence keeps regions on the raw route)",
    )
    parser.add_argument(
        "--spatial-gate",
        default=None,
        help="JSON mapping region_index → downmix_allowed (fail closed)",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_route_config(args.config)
    dfn3_config = load_dfn3_config(args.dfn3_config)

    with Path(args.regions).open("r", encoding="utf-8") as handle:
        regions = json.load(handle).get("regions", [])

    evidence: dict[int, dict] = {}
    if args.style_event:
        with Path(args.style_event).open("r", encoding="utf-8") as handle:
            for row in json.load(handle).get("regions", []):
                evidence[int(row["region_index"])] = row.get("event_evidence")

    spatial_gate: dict[int, bool] | None = None
    if args.spatial_gate:
        with Path(args.spatial_gate).open("r", encoding="utf-8") as handle:
            spatial_gate = {
                int(key): bool(value) for key, value in json.load(handle).items()
            }

    # Decide whether any region will need the DFN3 backend before loading it.
    from dots_tts_lab.long_form_route import decide_route

    needs_dfn3 = any(
        decide_route(
            region,
            event_evidence=evidence.get(int(region["region_index"])),
            downmix_allowed=bool(
                spatial_gate and spatial_gate.get(int(region["region_index"]), False)
            ),
            source_channels=_probe_channels(args.source),
            config=config.routing,
        )[0]
        == "dfn3_denoised"
        for region in regions
    )

    enhancer = None
    if needs_dfn3:
        try:
            verify_backend(dfn3_config, package_version=_df_version())
        except Dfn3VerificationError as error:
            print(f"backend verification failed: {error}", file=sys.stderr)
            return 1
        from dots_tts_lab.long_form_dfn3_regions import (
            make_df_enhancer,
            verify_delay_alignment,
        )

        enhancer = make_df_enhancer(dfn3_config)
        residual = verify_delay_alignment(enhancer)
        if abs(residual) > 2:
            print(
                f"delay compensation residual {residual} samples exceeds ±2",
                file=sys.stderr,
            )
            return 1

    started = time.time()
    result = run_routes(
        args.source,
        regions,
        config=config,
        dfn3_config=dfn3_config,
        enhance_fn=enhancer,
        event_evidence_by_region=evidence,
        spatial_gate=spatial_gate,
    )
    result["elapsed_seconds"] = round(time.time() - started, 3)

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


def _probe_channels(source: str) -> int:
    import soundfile as sf

    return int(sf.info(str(source)).channels)


def _df_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("deepfilternet")
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
