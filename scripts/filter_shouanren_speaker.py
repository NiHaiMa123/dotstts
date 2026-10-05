from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_identity import load_identity_config
from dots_tts_lab.speaker_embedding import (
    load_or_compute_speaker_embedding,
    load_speaker_encoder,
)

ROOT = Path(__file__).resolve().parents[1]
STANDARDIZATION_REPORT = ROOT / "data/reports/standardization/standardization.json"
STANDARDIZED_ROOT = ROOT / "data/work/standardized"
IDENTITY_CONFIG = ROOT / "configs/lab/long_form/identity_v1.yaml"
REPORT_DIR = ROOT / "data/reports/speaker_filter"


def _load_candidates(speaker_id: str) -> list[dict]:
    report = json.loads(
        STANDARDIZATION_REPORT.read_text(encoding="utf-8")
    )
    candidates = []
    for asset in report["assets"]:
        if asset.get("speaker_id") != speaker_id:
            continue
        audio_path = (STANDARDIZED_ROOT / asset["relative_path"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"standardized audio missing: {audio_path}"
            )
        candidates.append(
            {
                "asset_sha256": asset["asset_sha256"],
                "audio_sha256": asset["output_sha256"],
                "audio_absolute_path": str(audio_path),
                "source_relative_path": asset["source_relative_path"],
                "transcript_candidate": asset.get("transcript_candidate"),
                "duration_seconds": float(asset["duration_seconds"]),
            }
        )
    if not candidates:
        raise RuntimeError(f"no standardized assets for speaker {speaker_id}")
    return candidates


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("zero embedding vector")
    return matrix / norms


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster CAM++ speaker embeddings of 守岸人 standardized assets "
            "and keep only the dominant speaker (largest cluster by duration)."
        )
    )
    parser.add_argument("--speaker-id", default="守岸人")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.5,
        help="Agglomerative average-linkage merge limit on 1-cosine distance",
    )
    parser.add_argument(
        "--rescue-cosine",
        type=float,
        default=0.6,
        help=(
            "Assets outside the dominant cluster with centroid cosine at or "
            "above this are marked boundary_rescue instead of exclude"
        ),
    )
    parser.add_argument(
        "--report-dir", default=str(REPORT_DIR), help="output report directory"
    )
    args = parser.parse_args()

    encoder_config = load_identity_config(IDENTITY_CONFIG).encoder
    candidates = _load_candidates(args.speaker_id)
    encoder = load_speaker_encoder(encoder_config)

    features = []
    for candidate in candidates:
        feature = load_or_compute_speaker_embedding(
            candidate, config=encoder_config, encoder=encoder
        )
        features.append(
            {
                "asset_sha256": candidate["asset_sha256"],
                "vector": feature["vector"],
            }
        )
    matrix = _l2(
        np.stack([f["vector"].astype(np.float64) for f in features])
    )
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=args.distance_threshold,
    ).fit_predict(distance)

    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["asset_sha256"])
    clusters: dict[int, dict] = {}
    for i in order:
        label = int(labels[i])
        bucket = clusters.setdefault(label, {"indices": [], "duration": 0.0})
        bucket["indices"].append(i)
        bucket["duration"] += candidates[i]["duration_seconds"]
    ranked = sorted(
        clusters.items(),
        key=lambda kv: (-kv[1]["duration"], -len(kv[1]["indices"]), kv[0]),
    )
    dominant_label = ranked[0][0]
    dominant_idx = clusters[dominant_label]["indices"]
    centroid = np.median(matrix[dominant_idx], axis=0)
    centroid /= np.linalg.norm(centroid)
    centroid_cosines = np.clip(matrix @ centroid, -1.0, 1.0)

    records = []
    for i in order:
        label = int(labels[i])
        candidate = candidates[i]
        if label == dominant_label:
            decision = "keep"
        elif float(centroid_cosines[i]) >= args.rescue_cosine:
            decision = "boundary_rescue"
        else:
            decision = "exclude"
        records.append(
            {
                "asset_sha256": candidate["asset_sha256"],
                "audio_sha256": candidate["audio_sha256"],
                "source_relative_path": candidate["source_relative_path"],
                "transcript_candidate": candidate["transcript_candidate"],
                "duration_seconds": round(candidate["duration_seconds"], 4),
                "cluster_id": label,
                "cluster_rank": next(
                    r for r, (label_r, _) in enumerate(ranked, 1) if label_r == label
                ),
                "centroid_cosine": round(float(centroid_cosines[i]), 4),
                "decision": decision,
            }
        )

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    json_path = report_dir / f"{args.speaker_id}_speaker_filter.json"
    csv_path = report_dir / f"{args.speaker_id}_speaker_filter.csv"

    summary = {
        "run_id": run_id,
        "speaker_id": args.speaker_id,
        "asset_count": len(records),
        "cluster_count": len(clusters),
        "distance_threshold": args.distance_threshold,
        "rescue_cosine": args.rescue_cosine,
        "dominant_cluster_id": dominant_label,
        "dominant_asset_count": len(dominant_idx),
        "dominant_duration_seconds": round(clusters[dominant_label]["duration"], 3),
        "clusters": [
            {
                "cluster_id": label,
                "rank": rank,
                "asset_count": len(bucket["indices"]),
                "duration_seconds": round(bucket["duration"], 3),
                "example_sources": [
                    candidates[i]["source_relative_path"]
                    for i in bucket["indices"][:3]
                ],
            }
            for rank, (label, bucket) in enumerate(ranked, 1)
        ],
        "decisions": {
            decision: sum(r["decision"] == decision for r in records)
            for decision in ("keep", "boundary_rescue", "exclude")
        },
        "encoder_config_sha256": encoder_config.weights_sha256,
        "records": records,
    }

    for path, writer in (
        (json_path, lambda p: p.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )),
        (csv_path, lambda p: _write_csv(p, records)),
    ):
        partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
        writer(partial)
        partial.replace(path)

    print(json.dumps({k: v for k, v in summary.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")
    print(f"json_sha256: {hashlib.sha256(json_path.read_bytes()).hexdigest()}")
    return 0


def _write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    raise SystemExit(main())
