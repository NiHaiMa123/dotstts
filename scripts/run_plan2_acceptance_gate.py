from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a mean from an empty list")
    return sum(values) / len(values)


def generation_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Generation manifest is empty")
    successful = [row for row in rows if row.get("status") == "ok"]
    return {
        "item_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "failure_rate": (len(rows) - len(successful)) / len(rows),
        "mean_rtf": mean([float(row["rtf"]) for row in successful]),
        "peak_cuda_gib": max(
            int(row.get("peak_torch_cuda_bytes") or 0) for row in successful
        )
        / (1024**3),
        "maximum_output_patches": max(
            int(row.get("sample_count") or 0) / 7680 for row in successful
        ),
    }


def metric_stats(report: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in report.get("items", []) if row.get("status") == "ok"]
    if not rows:
        raise ValueError("Metrics report has no successful rows")
    quality_passes = sum(row.get("quality_decision") == "pass" for row in rows)
    return {
        "item_count": len(report.get("items", [])),
        "success_count": len(rows),
        "cer": mean([float(row["cer"]) for row in rows]),
        "speaker_cosine": mean([float(row["speaker_cosine"]) for row in rows]),
        "quality_pass_rate": quality_passes / len(rows),
        "maximum_true_peak_dbtp": max(
            float(row["quality"]["true_peak_estimate_dbtp"]) for row in rows
        ),
        "flat_top_run_count": sum(
            int(row["quality"]["flat_top_run_count"]) for row in rows
        ),
        "maximum_leading_silence_seconds": max(
            float(row["quality"]["leading_silence_seconds"]) for row in rows
        ),
    }


def threshold_report(
    control_generation: dict[str, Any],
    candidate_generation: dict[str, Any],
    control_raw: dict[str, Any],
    candidate_raw: dict[str, Any],
    control_trimmed: dict[str, Any],
    candidate_trimmed: dict[str, Any],
    manual: dict[str, Any],
    *,
    same_postprocess_config: bool,
    provenance_matches: bool,
) -> dict[str, Any]:
    preference = manual["preference_counts"]
    rtf_ratio = candidate_generation["mean_rtf"] / control_generation["mean_rtf"]
    checks = {
        "model_failure_rate": {
            "actual": candidate_generation["failure_rate"],
            "operator": "<=",
            "threshold": 0.05,
            "passed": candidate_generation["failure_rate"] <= 0.05,
        },
        "model_cer_increase": {
            "actual": candidate_raw["cer"] - control_raw["cer"],
            "operator": "<=",
            "threshold": 0.02,
            "passed": candidate_raw["cer"] - control_raw["cer"] <= 0.02,
        },
        "model_speaker_drop": {
            "actual": control_raw["speaker_cosine"]
            - candidate_raw["speaker_cosine"],
            "operator": "<=",
            "threshold": 0.02,
            "passed": control_raw["speaker_cosine"]
            - candidate_raw["speaker_cosine"]
            <= 0.02,
        },
        "manual_decisive_wins": {
            "actual": {
                "trained": int(preference["trained"]),
                "control": int(preference["control"]),
            },
            "operator": "trained > control",
            "passed": int(preference["trained"]) > int(preference["control"]),
        },
        "same_postprocess_config": {
            "actual": same_postprocess_config,
            "operator": "is",
            "threshold": True,
            "passed": same_postprocess_config,
        },
        "postprocess_provenance": {
            "actual": provenance_matches,
            "operator": "is",
            "threshold": True,
            "passed": provenance_matches,
        },
        "postprocess_quality_pass_rate": {
            "actual": candidate_trimmed["quality_pass_rate"],
            "operator": ">=",
            "threshold": 0.50,
            "passed": candidate_trimmed["quality_pass_rate"] >= 0.50,
        },
        "postprocess_quality_drop": {
            "actual": control_trimmed["quality_pass_rate"]
            - candidate_trimmed["quality_pass_rate"],
            "operator": "<=",
            "threshold": 0.05,
            "passed": control_trimmed["quality_pass_rate"]
            - candidate_trimmed["quality_pass_rate"]
            <= 0.05,
        },
        "postprocess_cer_increase": {
            "actual": candidate_trimmed["cer"] - candidate_raw["cer"],
            "operator": "<=",
            "threshold": 0.01,
            "passed": candidate_trimmed["cer"] - candidate_raw["cer"] <= 0.01,
        },
        "postprocess_no_clipping": {
            "actual": {
                "flat_top_run_count": candidate_trimmed["flat_top_run_count"],
                "maximum_true_peak_dbtp": candidate_trimmed[
                    "maximum_true_peak_dbtp"
                ],
            },
            "operator": "flat_top_run_count == 0 and maximum_true_peak_dbtp <= 0",
            "passed": candidate_trimmed["flat_top_run_count"] == 0
            and candidate_trimmed["maximum_true_peak_dbtp"] <= 0.0,
        },
        "performance_mean_rtf": {
            "actual": candidate_generation["mean_rtf"],
            "operator": "<=",
            "threshold": 1.0,
            "passed": candidate_generation["mean_rtf"] <= 1.0,
        },
        "performance_rtf_ratio": {
            "actual": rtf_ratio,
            "operator": "<=",
            "threshold": 1.15,
            "passed": rtf_ratio <= 1.15,
        },
        "performance_peak_cuda_gib": {
            "actual": candidate_generation["peak_cuda_gib"],
            "operator": "<=",
            "threshold": 12.0,
            "passed": candidate_generation["peak_cuda_gib"] <= 12.0,
        },
    }
    groups = {
        "model_gate": [
            "model_failure_rate",
            "model_cer_increase",
            "model_speaker_drop",
            "manual_decisive_wins",
        ],
        "postprocess_gate": [
            "same_postprocess_config",
            "postprocess_provenance",
            "postprocess_quality_pass_rate",
            "postprocess_quality_drop",
            "postprocess_cer_increase",
            "postprocess_no_clipping",
        ],
        "performance_gate": [
            "performance_mean_rtf",
            "performance_rtf_ratio",
            "performance_peak_cuda_gib",
        ],
    }
    gates = {
        name: {
            "passed": all(checks[check]["passed"] for check in names),
            "checks": names,
        }
        for name, names in groups.items()
    }
    return {
        "checks": checks,
        "gates": gates,
        "accepted": all(gate["passed"] for gate in gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "control_generation",
            "candidate_generation",
            "control_raw_metrics",
            "candidate_raw_metrics",
            "control_trimmed_generation",
            "candidate_trimmed_generation",
            "control_trimmed_metrics",
            "candidate_trimmed_metrics",
            "manual_review",
        )
    }
    loaded_generation = {
        name: load_jsonl(path)
        for name, path in paths.items()
        if name.endswith("generation")
    }
    loaded_metrics = {
        name: load_json(path)
        for name, path in paths.items()
        if name.endswith("metrics")
    }
    manual = load_json(paths["manual_review"])

    control_raw_rows = loaded_generation["control_generation"]
    candidate_raw_rows = loaded_generation["candidate_generation"]
    control_trimmed_rows = loaded_generation["control_trimmed_generation"]
    candidate_trimmed_rows = loaded_generation["candidate_trimmed_generation"]
    raw_by_side = {
        "control": {row["job_id"]: row for row in control_raw_rows},
        "candidate": {row["job_id"]: row for row in candidate_raw_rows},
    }
    config_hashes = {
        row.get("postprocess", {}).get("config_sha256")
        for row in control_trimmed_rows + candidate_trimmed_rows
    }
    same_postprocess_config = len(config_hashes) == 1 and None not in config_hashes
    provenance_matches = all(
        row.get("source_output_sha256")
        == raw_by_side[side].get(row["job_id"], {}).get("output_sha256")
        for side, rows in (
            ("control", control_trimmed_rows),
            ("candidate", candidate_trimmed_rows),
        )
        for row in rows
    )

    control_generation = generation_stats(control_raw_rows)
    candidate_generation = generation_stats(candidate_raw_rows)
    control_raw = metric_stats(loaded_metrics["control_raw_metrics"])
    candidate_raw = metric_stats(loaded_metrics["candidate_raw_metrics"])
    control_trimmed = metric_stats(loaded_metrics["control_trimmed_metrics"])
    candidate_trimmed = metric_stats(loaded_metrics["candidate_trimmed_metrics"])
    result = threshold_report(
        control_generation,
        candidate_generation,
        control_raw,
        candidate_raw,
        control_trimmed,
        candidate_trimmed,
        manual,
        same_postprocess_config=same_postprocess_config,
        provenance_matches=provenance_matches,
    )
    return {
        "schema_version": 1,
        "status": "accepted" if result["accepted"] else "rejected",
        "task": "P2-7",
        "control_generation": control_generation,
        "candidate_generation": candidate_generation,
        "control_raw_metrics": control_raw,
        "candidate_raw_metrics": candidate_raw,
        "control_trimmed_metrics": control_trimmed,
        "candidate_trimmed_metrics": candidate_trimmed,
        "manual_review": {
            "preference_counts": manual["preference_counts"],
            "decisive_pair_count": manual["decisive_pair_count"],
            "trained_win_rate_among_decisive": manual[
                "trained_win_rate_among_decisive"
            ],
        },
        **result,
        "inputs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    base = Path("data/reports/datasets/fuxuan_v1/slice12")
    parser.add_argument(
        "--control-generation",
        type=Path,
        default=base / "soar_control_optimized_formal_generation.jsonl",
    )
    parser.add_argument(
        "--candidate-generation",
        type=Path,
        default=base / "soar_lora_step400_optimized_formal_generation.jsonl",
    )
    parser.add_argument(
        "--control-raw-metrics",
        type=Path,
        default=base / "soar_control_optimized_formal_metrics.json",
    )
    parser.add_argument(
        "--candidate-raw-metrics",
        type=Path,
        default=base / "soar_lora_step400_optimized_formal_metrics.json",
    )
    parser.add_argument(
        "--control-trimmed-generation",
        type=Path,
        default=base / "soar_control_optimized_formal_trimmed_generation.jsonl",
    )
    parser.add_argument(
        "--candidate-trimmed-generation",
        type=Path,
        default=base / "soar_lora_step400_optimized_formal_trimmed_generation.jsonl",
    )
    parser.add_argument(
        "--control-trimmed-metrics",
        type=Path,
        default=base / "soar_control_optimized_formal_trimmed_metrics.json",
    )
    parser.add_argument(
        "--candidate-trimmed-metrics",
        type=Path,
        default=base / "soar_lora_step400_optimized_formal_trimmed_metrics.json",
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=base / "slice12_step400_manual_review.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/plan2-acceptance-v1.json"),
    )
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
