from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_quality import (
    DEFAULT_QUALITY_CONFIG_PATH,
    DnsmosBackend,
    load_quality_config,
    run_quality_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DNSMOS P.835 in-domain quality acceptance for routed "
        "regions (LF-12G)."
    )
    parser.add_argument("--config", default=str(DEFAULT_QUALITY_CONFIG_PATH))
    parser.add_argument("--source", required=True, help="Source audio path")
    parser.add_argument(
        "--routes",
        required=True,
        help="routes.json from the LF-13C routing run",
    )
    parser.add_argument(
        "--style-event",
        default=None,
        help="style_event.json to join event-risk evidence (optional)",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_quality_config(args.config)
    with Path(args.routes).open("r", encoding="utf-8") as handle:
        routes_manifest = json.load(handle)
    routes_manifest.setdefault("run_dir", str(Path(args.routes).resolve().parent))

    risk: dict[int, dict] = {}
    if args.style_event:
        with Path(args.style_event).open("r", encoding="utf-8") as handle:
            for row in json.load(handle).get("regions", []):
                risk[int(row["region_index"])] = row.get("event_evidence") or {}

    started = time.time()
    backend = DnsmosBackend(config.models)
    result = run_quality_acceptance(
        args.source,
        routes_manifest,
        config=config,
        backend=backend,
        event_risk_by_region=risk,
    )
    result["elapsed_seconds"] = round(time.time() - started, 3)

    if args.report:
        Path(args.report).resolve().write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
