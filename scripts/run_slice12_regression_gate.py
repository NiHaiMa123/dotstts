#!/usr/bin/env python3
"""Evaluate Slice12 trained metrics against explicit absolute and control gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def summarize(metrics: dict[str, Any], generation: list[dict[str, Any]]) -> dict[str, float | int | None]:
    expected = int(metrics.get("item_count", len(metrics.get("items", []))))
    items = list(metrics.get("items", []))
    successful = [item for item in items if item.get("status") == "ok"]
    failed = max(int(metrics.get("item_error_count", expected - len(successful))), expected - len(successful))
    peak_values = [int(row["peak_torch_cuda_bytes"]) for row in generation if row.get("peak_torch_cuda_bytes") is not None]
    return {
        "expected_count": expected,
        "successful_count": len(successful),
        "failure_rate": failed / max(1, expected),
        "mean_cer": mean([float(item["cer"]) for item in successful if item.get("cer") is not None]),
        "mean_speaker_cosine": mean([float(item["speaker_cosine"]) for item in successful if item.get("speaker_cosine") is not None]),
        "quality_pass_rate": sum(item.get("quality_decision") == "pass" for item in successful) / max(1, len(successful)),
        "mean_rtf": mean([float(item["rtf"]) for item in successful if item.get("rtf") is not None]),
        "inference_peak_cuda_gib": max(peak_values, default=0) / (1024 ** 3),
    }


def check_max(name: str, actual: float | None, limit: float) -> dict[str, Any]:
    return {"gate": name, "actual": actual, "limit": limit, "passed": actual is not None and math.isfinite(actual) and actual <= limit}


def check_min(name: str, actual: float | None, limit: float) -> dict[str, Any]:
    return {"gate": name, "actual": actual, "limit": limit, "passed": actual is not None and math.isfinite(actual) and actual >= limit}


def difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1e-12)


def evaluate(control: dict[str, Any], trained: dict[str, Any], gates: dict[str, Any]) -> list[dict[str, Any]]:
    absolute = gates["candidate_absolute_gates"]
    regression = gates["regression_gates_against_training_control"]
    checks = [
        check_max("max_failure_rate", trained["failure_rate"], float(absolute["max_failure_rate"])),
        check_max("max_mean_cer", trained["mean_cer"], float(absolute["max_mean_cer"])),
        check_min("min_mean_speaker_cosine", trained["mean_speaker_cosine"], float(absolute["min_mean_speaker_cosine"])),
        check_min("min_quality_pass_rate", trained["quality_pass_rate"], float(absolute["min_quality_pass_rate"])),
        check_max("max_mean_rtf", trained["mean_rtf"], float(absolute["max_mean_rtf"])),
        check_max("max_inference_peak_cuda_gib", trained["inference_peak_cuda_gib"], float(absolute["max_inference_peak_cuda_gib"])),
    ]
    checks.extend(
        [
            check_max("max_mean_cer_increase", difference(trained["mean_cer"], control["mean_cer"]), float(regression["max_mean_cer_increase"])),
            check_max("max_mean_speaker_cosine_drop", difference(control["mean_speaker_cosine"], trained["mean_speaker_cosine"]), float(regression["max_mean_speaker_cosine_drop"])),
            check_max("max_quality_pass_rate_drop", difference(control["quality_pass_rate"], trained["quality_pass_rate"]), float(regression["max_quality_pass_rate_drop"])),
            check_max("max_mean_rtf_ratio", ratio(trained["mean_rtf"], control["mean_rtf"]), float(regression["max_mean_rtf_ratio"])),
        ]
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice12/experiment_contract_v2.yaml"))
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    paths = {
        key: (ROOT / outputs[key]).resolve()
        for key in ("training_control_metrics", "training_control_generation", "trained_metrics", "trained_generation")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    output_path = (ROOT / outputs["regression_report"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        payload = {
            "schema_version": 1,
            "status": "blocked_missing_inputs",
            "experiment_id": config["experiment_id"],
            "missing_inputs": missing,
        }
        if paths["training_control_metrics"].is_file() and paths["training_control_generation"].is_file():
            payload["control"] = summarize(
                json.loads(paths["training_control_metrics"].read_text(encoding="utf-8")),
                load_jsonl(paths["training_control_generation"]),
            )
        if paths["trained_metrics"].is_file() and paths["trained_generation"].is_file():
            payload["trained"] = summarize(
                json.loads(paths["trained_metrics"].read_text(encoding="utf-8")),
                load_jsonl(paths["trained_generation"]),
            )
    else:
        control = summarize(json.loads(paths["training_control_metrics"].read_text(encoding="utf-8")), load_jsonl(paths["training_control_generation"]))
        trained = summarize(json.loads(paths["trained_metrics"].read_text(encoding="utf-8")), load_jsonl(paths["trained_generation"]))
        checks = evaluate(control, trained, config["metrics"])
        payload = {
            "schema_version": 1,
            "status": "passed" if all(check["passed"] for check in checks) else "failed",
            "experiment_id": config["experiment_id"],
            "control": control,
            "trained": trained,
            "checks": checks,
            "manual_review_required": any(not check["passed"] for check in checks),
        }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
