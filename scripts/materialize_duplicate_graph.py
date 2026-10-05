from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    load_dataset_freeze_config,
)
from dots_tts_lab.duplicate_graph import build_duplicate_edges, load_verified_json
from dots_tts_lab.threshold_calibration import (
    DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
    assert_dataset_ready_with_calibration,
    load_threshold_calibration_overlay,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persist verified duplicate evidence and stable connected components."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=DEFAULT_THRESHOLD_CALIBRATION_CONFIG_PATH,
    )
    for name in (
        "candidate-snapshot",
        "exact-audio",
        "exact-text",
        "near-text",
        "acoustic-fingerprint",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()

    dataset_config = load_dataset_freeze_config(args.dataset_config)
    overlay = load_threshold_calibration_overlay(args.calibration_config)
    thresholds = assert_dataset_ready_with_calibration(dataset_config, overlay)
    source_sha256s = {
        "candidate_snapshot": args.candidate_snapshot_sha256,
        "exact_audio": args.exact_audio_sha256,
        "exact_text": args.exact_text_sha256,
        "near_text": args.near_text_sha256,
        "acoustic_fingerprint": args.acoustic_fingerprint_sha256,
    }
    if source_sha256s["candidate_snapshot"] != overlay.candidate_snapshot_sha256:
        raise RuntimeError("Candidate snapshot SHA-256 differs from calibration overlay")
    graph = build_duplicate_edges(
        candidate_snapshot=load_verified_json(
            args.candidate_snapshot, source_sha256s["candidate_snapshot"]
        ),
        exact_audio_report=load_verified_json(
            args.exact_audio, source_sha256s["exact_audio"]
        ),
        exact_text_report=load_verified_json(
            args.exact_text, source_sha256s["exact_text"]
        ),
        near_text_report=load_verified_json(
            args.near_text, source_sha256s["near_text"]
        ),
        acoustic_fingerprint_report=load_verified_json(
            args.acoustic_fingerprint,
            source_sha256s["acoustic_fingerprint"],
        ),
        text_threshold=thresholds.text_near_similarity,
        acoustic_threshold=thresholds.acoustic_near_cosine,
        acoustic_duration_ratio_min=dataset_config.acoustic_fingerprint.duration_ratio_min,
        source_sha256s=source_sha256s,
    )
    result = Catalog(args.catalog).store_dataset_similarity_graph(
        run_id=args.run_id,
        candidate_asset_sha256s=graph["candidate_asset_sha256s"],
        edges=graph["edges"],
        created_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )
    print(
        json.dumps(
            {
                **result,
                "edge_counts_by_evidence_type": graph[
                    "edge_counts_by_evidence_type"
                ],
                "acoustic_scored_pair_count": graph["acoustic_scored_pair_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
