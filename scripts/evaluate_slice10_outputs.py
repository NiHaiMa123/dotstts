from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from dots_tts_lab.quality import (
    assess_metrics,
    analyze_audio,
    load_quality_analysis_config,
    load_quality_policy,
)
from dots_tts_lab.speaker_embedding import compute_speaker_embedding, load_speaker_encoder
from dots_tts_lab.dataset_freeze import load_dataset_freeze_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("~", "")
    return "".join(char for char in value if unicodedata.category(char)[0] not in {"P", "Z", "C"})


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def load_center(report_path: Path) -> tuple[np.ndarray, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    vectors = []
    for feature in report.get("features", []):
        path = Path(str(feature["cache_path"]))
        vector = np.fromfile(path, dtype="<f4")
        if vector.shape != (512,) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"Invalid cached speaker vector: {path}")
        vectors.append(vector.astype(np.float64))
    if len(vectors) < 2:
        raise RuntimeError("Speaker center requires at least two cached vectors")
    matrix = np.stack(vectors)
    center = np.median(matrix, axis=0)
    center /= np.linalg.norm(center)
    center_f32 = np.asarray(center, dtype="<f4")
    return center, hashlib.sha256(center_f32.tobytes()).hexdigest()


def finite_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def numeric_or_default(value: Any, default: float) -> float:
    """Return a numeric metric without treating the valid value 0 as missing."""
    return default if value is None else float(value)


def minmax(values: dict[str, float], *, higher_is_better: bool = True) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    output = {key: (value - low) / (high - low) for key, value in values.items()}
    if not higher_is_better:
        output = {key: 1.0 - value for key, value in output.items()}
    return output


def run(config_path: Path) -> dict[str, Any]:
    repo_root = config_path.resolve().parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    generation_path = (repo_root / outputs["generation_manifest_path"]).resolve()
    asr_path = (repo_root / outputs["asr_output_path"]).resolve()
    generation = load_jsonl(generation_path)
    asr_rows = {row["job_id"]: row for row in load_jsonl(asr_path)}
    if not generation or len(asr_rows) != len(generation):
        raise RuntimeError("Generation and ASR result sets do not match")

    dataset_cfg = load_dataset_freeze_config(repo_root / config["metrics"]["speaker_config_path"])
    encoder = load_speaker_encoder(dataset_cfg.speaker_embedding)
    center, center_sha = load_center(repo_root / config["metrics"]["speaker_report_path"])
    quality_cfg = load_quality_analysis_config(repo_root / config["metrics"]["quality_analysis_config_path"])
    quality_policy = load_quality_policy(repo_root / config["metrics"]["quality_policy_path"])

    item_metrics: list[dict[str, Any]] = []
    for generation_row in generation:
        job_id = generation_row["job_id"]
        asr = asr_rows[job_id]
        item: dict[str, Any] = {
            "job_id": job_id,
            "ordinal": int(generation_row["ordinal"]),
            "asset_sha256": generation_row["asset_sha256"],
            "pool_ids": generation_row["pool_ids"],
            "sentence_id": generation_row["sentence_id"],
            "text": generation_row["text"],
            "seed": int(generation_row["seed"]),
            "output_path": generation_row["output_path"],
            "output_sha256": generation_row["output_sha256"],
            "generation_seconds": generation_row.get("generation_seconds"),
            "rtf": generation_row.get("rtf"),
            "status": "ok",
            "errors": [],
        }
        output_path = (repo_root / generation_row["output_path"]).resolve()
        try:
            if not output_path.is_file() or sha256_file(output_path) != generation_row["output_sha256"]:
                raise RuntimeError("generated audio SHA-256 mismatch")
            reference = normalize_text(str(generation_row["text"]))
            hypothesis = normalize_text(str(asr.get("hypothesis", "")))
            distance = edit_distance(reference, hypothesis)
            item.update({
                "reference_normalized": reference,
                "hypothesis_raw": asr.get("hypothesis", ""),
                "hypothesis_normalized": hypothesis,
                "edit_distance": distance,
                "reference_char_count": len(reference),
                "cer": distance / max(1, len(reference)),
                "asr_seconds": asr.get("asr_seconds"),
            })
            if asr.get("status") != "ok":
                raise RuntimeError(f"ASR status is {asr.get('status')}: {asr.get('error_message', '')}")
            audio, sample_rate = sf.read(str(output_path), dtype="float32", always_2d=False)
            vector = compute_speaker_embedding(audio, sample_rate=int(sample_rate), config=dataset_cfg.speaker_embedding, encoder=encoder)
            item["speaker_cosine"] = float(np.clip(np.dot(vector.astype(np.float64), center), -1.0, 1.0))
            info = sf.info(str(output_path))
            quality = analyze_audio(output_path, subtype=info.subtype, config=quality_cfg)
            decision = assess_metrics(quality, policy=quality_policy)
            item["quality"] = quality
            item["quality_decision"] = decision["decision"]
            item["quality_reasons"] = decision["reasons"]
            item["duration_seconds"] = quality["duration_seconds"]
        except Exception as error:
            item["status"] = "metric_error"
            item["errors"].append({"type": type(error).__name__, "message": str(error)})
        item_metrics.append(item)

    by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_metrics:
        by_asset[row["asset_sha256"]].append(row)
    manifest_rows = {row["asset_sha256"]: row for row in load_jsonl((repo_root / outputs["benchmark_manifest_path"]).resolve())}
    candidate_rows: list[dict[str, Any]] = []
    stability_thresholds = config["metrics"]["stability"]
    for asset, rows in sorted(by_asset.items()):
        ok_rows = [row for row in rows if row["status"] == "ok"]
        cer_values = [float(row["cer"]) for row in ok_rows if row.get("cer") is not None]
        speaker_values = [float(row["speaker_cosine"]) for row in ok_rows if row.get("speaker_cosine") is not None]
        durations = [float(row["duration_seconds"]) for row in ok_rows if row.get("duration_seconds") is not None]
        rtfs = [float(row["rtf"]) for row in ok_rows if row.get("rtf") is not None]
        quality_pass = sum(row.get("quality_decision") == "pass" for row in ok_rows)
        per_sentence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in ok_rows:
            per_sentence[row["sentence_id"]].append(row)
        cer_ranges = [max((float(row["cer"]) for row in values), default=0.0) - min((float(row["cer"]) for row in values), default=0.0) for values in per_sentence.values()]
        embedding_ranges = [max((float(row["speaker_cosine"]) for row in values), default=0.0) - min((float(row["speaker_cosine"]) for row in values), default=0.0) for values in per_sentence.values()]
        duration_cvs = []
        for values in per_sentence.values():
            ds = [float(row["duration_seconds"]) for row in values]
            if len(ds) > 1 and statistics.fmean(ds) > 0:
                duration_cvs.append(statistics.pstdev(ds) / statistics.fmean(ds))
        max_cer_range = max(cer_ranges, default=0.0)
        max_embedding_range = max(embedding_ranges, default=0.0)
        max_duration_cv = max(duration_cvs, default=0.0)
        stability_penalty = max(
            max_cer_range / float(stability_thresholds["max_cer_range"]),
            max_embedding_range / float(stability_thresholds["max_embedding_range"]),
            max_duration_cv / float(stability_thresholds["max_duration_cv"]),
        )
        candidate_rows.append({
            "asset_sha256": asset,
            "ordinal": manifest_rows[asset]["ordinal"],
            "pool_ids": manifest_rows[asset]["pool_ids"],
            "static_composite_score": manifest_rows[asset]["static_composite_score"],
            "success_count": len(ok_rows),
            "expected_count": len(rows),
            "generation_success_rate": len(ok_rows) / max(1, len(rows)),
            "mean_cer": finite_mean(cer_values),
            "mean_speaker_cosine": finite_mean(speaker_values),
            "quality_pass_rate": quality_pass / max(1, len(ok_rows)),
            "mean_rtf": finite_mean(rtfs),
            "mean_duration_seconds": finite_mean(durations),
            "max_seed_cer_range": max_cer_range,
            "max_seed_embedding_range": max_embedding_range,
            "max_seed_duration_cv": max_duration_cv,
            "stability_score": max(0.0, 1.0 - min(1.0, stability_penalty)),
            "metric_error_count": sum(row["status"] != "ok" for row in rows),
        })

    complete = all(row["metric_error_count"] == 0 for row in candidate_rows)
    cer_component = minmax({row["asset_sha256"]: -numeric_or_default(row["mean_cer"], 1.0) for row in candidate_rows})
    speaker_component = minmax({row["asset_sha256"]: numeric_or_default(row["mean_speaker_cosine"], -1.0) for row in candidate_rows})
    quality_component = {row["asset_sha256"]: float(row["quality_pass_rate"]) for row in candidate_rows}
    success_component = {row["asset_sha256"]: float(row["generation_success_rate"]) for row in candidate_rows}
    stability_component = {row["asset_sha256"]: float(row["stability_score"]) for row in candidate_rows}
    rtf_values = {row["asset_sha256"]: numeric_or_default(row["mean_rtf"], 999.0) for row in candidate_rows}
    rtf_component = minmax(rtf_values, higher_is_better=False)
    weights = {"cer": 0.35, "speaker": 0.25, "quality": 0.15, "success": 0.10, "stability": 0.10, "rtf": 0.05}
    for row in candidate_rows:
        asset = row["asset_sha256"]
        components = {
            "cer": cer_component[asset], "speaker": speaker_component[asset],
            "quality": quality_component[asset], "success": success_component[asset],
            "stability": stability_component[asset], "rtf": rtf_component[asset],
        }
        row["closed_loop_components"] = components
        row["closed_loop_score"] = sum(weights[key] * components[key] for key in weights)

    static_order = sorted(candidate_rows, key=lambda row: (-float(row["static_composite_score"]), row["asset_sha256"]))
    closed_order = sorted(candidate_rows, key=lambda row: (-float(row["closed_loop_score"]), numeric_or_default(row["mean_cer"], 1.0), -numeric_or_default(row["mean_speaker_cosine"], -1.0), row["asset_sha256"]))
    static_rank = {row["asset_sha256"]: index for index, row in enumerate(static_order, start=1)}
    for index, row in enumerate(closed_order, start=1):
        row["closed_loop_rank"] = index
        row["static_rank"] = static_rank[row["asset_sha256"]]
        row["rank_delta"] = row["static_rank"] - row["closed_loop_rank"]
    candidate_rows.sort(key=lambda row: row["closed_loop_rank"])

    run_root = (repo_root / outputs["run_root"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "schema_version": 1,
        "status": "succeeded" if complete else "incomplete",
        "benchmark_id": config["benchmark_id"],
        "benchmark_version": config["benchmark_version"],
        "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
        "generation_manifest_path": str(generation_path),
        "asr_output_path": str(asr_path),
        "asr_backend": {key: config["metrics"][key] for key in ("asr_backend_id", "asr_model_id", "asr_model_revision")},
        "speaker_center_sha256": center_sha,
        "item_count": len(item_metrics),
        "item_success_count": sum(item["status"] == "ok" for item in item_metrics),
        "item_error_count": sum(item["status"] != "ok" for item in item_metrics),
        "items": item_metrics,
    }
    (repo_root / outputs["metrics_path"]).write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    ranking_payload = {
        "schema_version": 1,
        "status": "succeeded" if complete else "incomplete",
        "benchmark_id": config["benchmark_id"],
        "benchmark_version": config["benchmark_version"],
        "benchmark_manifest_sha256": outputs["benchmark_manifest_sha256"],
        "static_score_source": "slice9 frozen ranking manifest",
        "closed_loop_model": config["model"],
        "closed_loop_weights": weights,
        "stability_thresholds": stability_thresholds,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "pool_rankings": {pool: [row["asset_sha256"] for row in sorted((r for r in candidate_rows if pool in r["pool_ids"]), key=lambda r: r["closed_loop_rank"])] for pool in sorted({pool for row in candidate_rows for pool in row["pool_ids"]})},
    }
    ranking_path = (repo_root / outputs["ranking_path"]).resolve()
    ranking_path.write_text(json.dumps(ranking_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    csv_path = (repo_root / outputs["ranking_csv_path"]).resolve()
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["closed_loop_rank", "static_rank", "rank_delta", "asset_sha256", "pool_ids", "static_composite_score", "closed_loop_score", "mean_cer", "mean_speaker_cosine", "quality_pass_rate", "generation_success_rate", "stability_score", "mean_rtf", "metric_error_count"])
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({
                "closed_loop_rank": row["closed_loop_rank"],
                "static_rank": row["static_rank"],
                "rank_delta": row["rank_delta"],
                "asset_sha256": row["asset_sha256"],
                "pool_ids": " ".join(row["pool_ids"]),
                "static_composite_score": row["static_composite_score"],
                "closed_loop_score": f"{row['closed_loop_score']:.8f}",
                "mean_cer": row["mean_cer"],
                "mean_speaker_cosine": row["mean_speaker_cosine"],
                "quality_pass_rate": row["quality_pass_rate"],
                "generation_success_rate": row["generation_success_rate"],
                "stability_score": row["stability_score"],
                "mean_rtf": row["mean_rtf"],
                "metric_error_count": row["metric_error_count"],
            })
    html_rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td>{:.4f}</td><td>{:.2%}</td><td>{:.2%}</td><td>{:.4f}</td></tr>".format(
            row["closed_loop_rank"], row["static_rank"], html.escape(row["asset_sha256"][:12]), float(row["closed_loop_score"]), float(row["static_composite_score"]), numeric_or_default(row["mean_cer"], 1.0), float(row["quality_pass_rate"]), float(row["generation_success_rate"]), float(row["stability_score"])
        ) for row in candidate_rows
    )
    html_path = (repo_root / outputs["ranking_html_path"]).resolve()
    html_path.write_text("<!doctype html><meta charset='utf-8'><title>Slice10 reference ranking</title><style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.4rem}th{background:#f3f4f6}</style><h1>Slice10 model-closed-loop reference ranking</h1><p>静态分数与闭环分数分开；音频文件位于本地 generation_root。</p><table><tr><th>闭环#</th><th>静态#</th><th>资产</th><th>闭环分</th><th>静态分</th><th>均值 CER</th><th>音质通过率</th><th>生成成功率</th><th>稳定性</th></tr>" + html_rows + "</table>\n", encoding="utf-8", newline="\n")
    return {"status": ranking_payload["status"], "candidate_count": len(candidate_rows), "item_count": len(item_metrics), "item_error_count": ranking_payload["candidate_count"] and sum(item["status"] != "ok" for item in item_metrics), "metrics_path": str(repo_root / outputs["metrics_path"]), "ranking_path": str(ranking_path), "ranking_html_path": str(html_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Slice10 generated audio and rank references.")
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice10/benchmark_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
