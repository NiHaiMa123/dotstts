from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dots_tts_lab.asr_benchmark import (
    DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    character_edit_distance,
    load_asr_benchmark_config,
    normalize_asr_text,
)
from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import utc_now
from dots_tts_lab.reports import _atomic_write_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def residual_symbols(text: str) -> Counter[str]:
    """Characters that are neither letters nor numbers after normalization.

    Benchmark normalization removes punctuation and separators only, so a
    backend that decorates its output with rich-transcription markers (emoji
    emotion or event tags) would have every marker charged as an insertion
    error. Such a run measures formatting, not transcription accuracy, so
    evaluation refuses it instead of publishing a polluted CER.
    """
    return Counter(
        character
        for character in text
        if unicodedata.category(character)[:1] not in ("L", "N")
    )


def _write_reports(
    output_dir: Path, summary: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, str]:
    json_path = output_dir / "evaluation.json"
    csv_path = output_dir / "evaluation.csv"
    html_path = output_dir / "evaluation.html"
    _atomic_write_text(
        json_path,
        json.dumps(
            {"schema_version": 1, "summary": summary, "assets": items},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    fields = (
        "ordinal",
        "asset_sha256",
        "emotion_weak_label",
        "reference_exact",
        "hypothesis_raw",
        "reference_normalized",
        "hypothesis_normalized",
        "reference_char_count",
        "edit_distance",
        "cer",
        "runtime_seconds",
        "detected_language",
        "status",
        "error_message",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(items)
    _atomic_write_text(csv_path, output.getvalue())
    summary_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if not isinstance(value, (dict, list))
    )
    rows = "\n".join(
        "<tr class=\"{}\"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2%}</td><td>{:.3f}</td></tr>".format(
            "error" if item["status"] == "error" else "",
            item["ordinal"],
            html.escape(str(item["emotion_weak_label"])),
            html.escape(str(item["reference_exact"])),
            html.escape(str(item.get("hypothesis_raw") or "")),
            float(item.get("cer") or 0.0),
            float(item.get("runtime_seconds") or 0.0),
        )
        for item in sorted(
            items,
            key=lambda item: (
                item["status"] != "error",
                -(float(item.get("cer") or 0.0)),
            ),
        )
    )
    _atomic_write_text(
        html_path,
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>dots.tts ASR evaluation</title><style>
body {{ font:14px/1.5 system-ui,sans-serif; margin:2rem; color:#1f2937 }}
table {{ border-collapse:collapse; width:100%; margin-bottom:2rem }}
th,td {{ border:1px solid #d1d5db; padding:.4rem .55rem; text-align:left }}
th {{ background:#f3f4f6 }} tr.error td {{ background:#fef2f2 }}
</style></head><body><h1>ASR evaluation</h1><h2>Summary</h2>
<table><tbody>{summary_rows}</tbody></table><h2>Worst first</h2><table><thead>
<tr><th>#</th><th>Emotion</th><th>Reference</th><th>Hypothesis</th><th>CER</th><th>Seconds</th></tr>
</thead><tbody>{rows}</tbody></table></body></html>""",
    )
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def evaluate_asr_output(
    *,
    result_path: str | Path,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    benchmark_config_path: str | Path = DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    report_root: str | Path = "data/reports/asr",
) -> dict[str, Any]:
    raw_path = Path(result_path).resolve()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported ASR backend output schema_version")
    config = load_asr_benchmark_config(benchmark_config_path)
    catalog = Catalog(catalog_path)
    catalog.initialize()
    registered_benchmark = catalog.load_asr_benchmark(
        benchmark_id=config.benchmark_id,
        benchmark_version=config.benchmark_version,
    )
    if registered_benchmark is None:
        raise RuntimeError(
            "ASR benchmark is not registered: "
            f"{config.benchmark_id}@{config.benchmark_version}"
        )
    if registered_benchmark["config_sha256"] != config.config_sha256():
        raise RuntimeError(
            "ASR benchmark config does not match the registered frozen config: "
            f"{config.benchmark_id}@{config.benchmark_version}"
        )
    manifest_value = payload.get("manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("ASR backend output must record manifest_path")
    source_manifest_path = Path(manifest_value).resolve()
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"ASR backend source manifest is missing: {source_manifest_path}"
        )
    source_manifest_sha256 = _sha256(source_manifest_path)
    if payload.get("manifest_sha256") != source_manifest_sha256:
        raise ValueError("ASR backend manifest SHA-256 does not match manifest_path")
    if source_manifest_sha256 != registered_benchmark["manifest_sha256"]:
        raise RuntimeError(
            "ASR backend output was not produced from the registered frozen manifest"
        )
    benchmark_items = catalog.load_asr_benchmark_items(
        benchmark_id=config.benchmark_id,
        benchmark_version=config.benchmark_version,
    )
    expected = {str(item["asset_sha256"]): item for item in benchmark_items}
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("ASR backend output must contain a results list")
    supplied: dict[str, dict[str, Any]] = {}
    for result in raw_results:
        asset_sha256 = str(result["asset_sha256"])
        if asset_sha256 in supplied:
            raise ValueError(f"Duplicate ASR result for {asset_sha256}")
        supplied[asset_sha256] = result
    missing = expected.keys() - supplied.keys()
    unexpected = supplied.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(
            f"ASR result set mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    inference_config = payload["inference_config"]
    inference_json = json.dumps(
        inference_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    inference_sha256 = hashlib.sha256(inference_json.encode("utf-8")).hexdigest()
    if inference_sha256 != payload.get("inference_config_sha256"):
        raise ValueError("Inference config SHA-256 does not match backend output")
    started_at = utc_now()
    run_id = catalog.begin_asr_run(
        benchmark_id=config.benchmark_id,
        benchmark_version=config.benchmark_version,
        backend_id=str(payload["backend_id"]),
        backend_version=str(payload["backend_version"]),
        model_id=str(payload["model_id"]),
        model_revision=str(payload["model_revision"]),
        inference_config_sha256=inference_sha256,
        inference_config_json=inference_json,
        started_at=started_at,
    )
    try:
        items = []
        catalog_results = []
        polluted: dict[int, Counter[str]] = {}
        for asset_sha256, benchmark in expected.items():
            raw = supplied[asset_sha256]
            status = str(raw.get("status", "ok"))
            if status not in {"ok", "error"}:
                raise ValueError(
                    f"Invalid ASR result status for {asset_sha256}: {status!r}"
                )
            runtime_seconds = raw.get("runtime_seconds")
            if runtime_seconds is not None and (
                not isinstance(runtime_seconds, (int, float))
                or isinstance(runtime_seconds, bool)
                or not math.isfinite(float(runtime_seconds))
                or float(runtime_seconds) < 0.0
            ):
                raise ValueError(
                    f"Invalid runtime_seconds for {asset_sha256}: {runtime_seconds!r}"
                )
            reference = normalize_asr_text(str(benchmark["text_exact"]), config)
            result: dict[str, Any] = {
                "ordinal": benchmark["ordinal"],
                "asset_sha256": asset_sha256,
                "emotion_weak_label": benchmark["emotion_weak_label"],
                "reference_exact": benchmark["text_exact"],
                "reference_normalized": reference,
                "reference_char_count": len(reference),
                "runtime_seconds": runtime_seconds,
                "detected_language": raw.get("detected_language"),
                "backend_metadata_json": json.dumps(
                    raw.get("metadata", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "status": status,
                "error_type": raw.get("error_type"),
                "error_message": raw.get("error_message"),
            }
            if status == "ok":
                hypothesis_raw = str(raw.get("hypothesis", ""))
                hypothesis = normalize_asr_text(hypothesis_raw, config)
                leftover = residual_symbols(hypothesis)
                if leftover:
                    polluted[int(benchmark["ordinal"])] = leftover
                distance = character_edit_distance(reference, hypothesis)
                result.update(
                    {
                        "hypothesis_raw": hypothesis_raw,
                        "hypothesis_normalized": hypothesis,
                        "edit_distance": distance,
                        "cer": distance / max(1, len(reference)),
                    }
                )
            items.append(result)
            catalog_results.append(result)
        if polluted:
            offenders = Counter[str]()
            for leftover in polluted.values():
                offenders.update(leftover)
            listed = ", ".join(
                f"{character!r} U+{ord(character):04X}×{count}"
                for character, count in sorted(offenders.most_common())
            )
            raise ValueError(
                f"Backend {payload['backend_id']} emitted non-transcript symbols in "
                f"{len(polluted)} of {len(items)} hypotheses ({listed}); strip rich "
                "transcription markers in the backend before evaluating, otherwise "
                "CER measures formatting instead of accuracy"
            )
        successful = [item for item in items if item["status"] == "ok"]
        error_count = len(items) - len(successful)
        total_reference_chars = sum(item["reference_char_count"] for item in successful)
        aggregate_cer = (
            sum(int(item["edit_distance"]) for item in successful)
            / total_reference_chars
            if total_reference_chars
            else None
        )
        by_emotion: dict[str, dict[str, float | int]] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in successful:
            grouped[str(item["emotion_weak_label"])].append(item)
        for emotion, group in grouped.items():
            characters = sum(item["reference_char_count"] for item in group)
            by_emotion[emotion] = {
                "count": len(group),
                "reference_char_count": characters,
                "edit_distance": sum(int(item["edit_distance"]) for item in group),
                "cer": sum(int(item["edit_distance"]) for item in group)
                / max(1, characters),
            }
        finished_at = utc_now()
        runtime_metadata = dict(payload.get("runtime_metadata") or {})
        runtime_metadata.update(
            {
                "environment": payload.get("environment") or {},
                "capabilities": payload.get("capabilities") or {},
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": source_manifest_sha256,
            }
        )
        summary = {
            "run_id": run_id,
            "status": "completed_with_errors" if error_count else "succeeded",
            "benchmark_id": config.benchmark_id,
            "benchmark_version": config.benchmark_version,
            "backend_id": payload["backend_id"],
            "backend_version": payload["backend_version"],
            "model_id": payload["model_id"],
            "model_revision": payload["model_revision"],
            "model_license": payload.get("model_license"),
            "inference_config_sha256": inference_sha256,
            "raw_result_path": str(raw_path),
            "raw_result_sha256": _sha256(raw_path),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha256,
            "started_at": started_at,
            "finished_at": finished_at,
            "item_count": len(items),
            "success_count": len(successful),
            "error_count": error_count,
            "reference_char_count": total_reference_chars,
            "edit_distance": sum(int(item["edit_distance"]) for item in successful),
            "aggregate_cer": aggregate_cer,
            "exact_match_count": sum(item.get("edit_distance") == 0 for item in successful),
            "total_audio_seconds": sum(float(item["duration_seconds"]) for item in benchmark_items),
            "total_runtime_seconds": sum(
                float(item.get("runtime_seconds") or 0.0) for item in items
            ),
            "wall_runtime_seconds": payload.get("wall_runtime_seconds"),
            "model_load_seconds": payload.get("runtime_metadata", {}).get(
                "model_load_seconds"
            ),
            "realtime_factor": sum(
                float(item.get("runtime_seconds") or 0.0) for item in items
            )
            / max(1e-12, sum(float(item["duration_seconds"]) for item in benchmark_items)),
            "by_emotion": by_emotion,
            "detected_language_counts": dict(
                Counter(item.get("detected_language") for item in successful)
            ),
            "runtime_metadata": runtime_metadata,
        }
        output_dir = Path(report_root).resolve() / str(payload["backend_id"])
        reports = _write_reports(output_dir, summary, items)
        catalog.complete_asr_run(
            run_id=run_id,
            finished_at=finished_at,
            results=catalog_results,
            summary=summary,
        )
        summary["reports"] = reports
        summary["catalog_path"] = str(Path(catalog_path).resolve())
        return summary
    except BaseException as error:
        catalog.fail_asr_run(
            run_id=run_id,
            finished_at=utc_now(),
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
