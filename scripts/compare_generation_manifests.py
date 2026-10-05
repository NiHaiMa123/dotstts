from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _audio_path(row: dict[str, Any]) -> Path:
    path = Path(row["output_path"])
    return path if path.is_absolute() else ROOT / path


def _validated_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    path = _audio_path(row).resolve()
    expected = row.get("output_sha256")
    if not path.is_file():
        raise RuntimeError(f"Audio file does not exist: {path}")
    if expected and sha256_file(path) != expected:
        raise RuntimeError(f"Audio hash mismatch: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim != 1:
        raise RuntimeError(f"Only mono audio is supported: {path}")
    return audio, int(sample_rate)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _manifest_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    rtfs = [float(row["rtf"]) for row in successful if row.get("rtf") is not None]
    peak_bytes = max(
        (int(row.get("peak_torch_cuda_bytes") or 0) for row in successful),
        default=0,
    )
    return {
        "item_count": len(rows),
        "successful_count": len(successful),
        "mean_rtf": _mean(rtfs),
        "steady_state_mean_rtf": _mean(rtfs[1:]),
        "peak_torch_cuda_bytes": peak_bytes,
        "peak_torch_cuda_gib": peak_bytes / (1024**3),
    }


def compare(left_manifest: Path, right_manifest: Path) -> dict[str, Any]:
    left_manifest = left_manifest.resolve()
    right_manifest = right_manifest.resolve()
    left_rows = load_jsonl(left_manifest)
    right_rows = load_jsonl(right_manifest)
    left_by_id = {row["job_id"]: row for row in left_rows if row.get("status") == "ok"}
    right_by_id = {row["job_id"]: row for row in right_rows if row.get("status") == "ok"}
    common_ids = sorted(left_by_id.keys() & right_by_id.keys())
    if not common_ids:
        raise RuntimeError("The manifests have no successful job IDs in common")

    comparisons: list[dict[str, Any]] = []
    for job_id in common_ids:
        left_row = left_by_id[job_id]
        right_row = right_by_id[job_id]
        left_audio, left_rate = _validated_audio(left_row)
        right_audio, right_rate = _validated_audio(right_row)
        if left_rate != right_rate:
            raise RuntimeError(
                f"Sample-rate mismatch for {job_id}: {left_rate} != {right_rate}"
            )
        overlap = min(len(left_audio), len(right_audio))
        left_overlap = left_audio[:overlap]
        right_overlap = right_audio[:overlap]
        denominator = float(np.linalg.norm(left_overlap) * np.linalg.norm(right_overlap))
        cosine = (
            float(np.dot(left_overlap, right_overlap) / denominator)
            if denominator > 0.0
            else float(np.array_equal(left_overlap, right_overlap))
        )
        rmse = float(np.sqrt(np.mean(np.square(left_overlap - right_overlap))))
        comparisons.append(
            {
                "job_id": job_id,
                "sample_rate": left_rate,
                "left_sample_count": len(left_audio),
                "right_sample_count": len(right_audio),
                "sample_count_equal": len(left_audio) == len(right_audio),
                "output_hash_equal": left_row.get("output_sha256")
                == right_row.get("output_sha256"),
                "overlap_waveform_cosine": cosine,
                "overlap_rmse": rmse,
            }
        )

    left_stats = _manifest_stats(left_rows)
    right_stats = _manifest_stats(right_rows)
    left_rtf = left_stats["mean_rtf"]
    right_rtf = right_stats["mean_rtf"]
    return {
        "schema_version": 1,
        "left_manifest": left_manifest.relative_to(ROOT).as_posix(),
        "left_manifest_sha256": sha256_file(left_manifest),
        "right_manifest": right_manifest.relative_to(ROOT).as_posix(),
        "right_manifest_sha256": sha256_file(right_manifest),
        "left": left_stats,
        "right": right_stats,
        "right_to_left_mean_rtf_ratio": (
            float(right_rtf / left_rtf) if left_rtf and right_rtf is not None else None
        ),
        "common_successful_count": len(comparisons),
        "sample_count_equal_count": sum(
            int(row["sample_count_equal"]) for row in comparisons
        ),
        "output_hash_equal_count": sum(
            int(row["output_hash_equal"]) for row in comparisons
        ),
        "minimum_overlap_waveform_cosine": min(
            row["overlap_waveform_cosine"] for row in comparisons
        ),
        "maximum_overlap_rmse": max(row["overlap_rmse"] for row in comparisons),
        "items": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_manifest", type=Path)
    parser.add_argument("right_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.left_manifest, args.right_manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
