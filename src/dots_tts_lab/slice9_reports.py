"""Render Slice 9 selection and boundary candidates for manual listening."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_constraints import DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH
from dots_tts_lab.slice9_sensitivity import DEFAULT_SLICE9_SENSITIVITY_REPORT_PATH


DEFAULT_SLICE9_REVIEW_JSON_PATH = Path("data/reports/datasets/fuxuan_v1/analysis/slice9_review_report_v1.json")
DEFAULT_SLICE9_CSV_PATH = Path("data/reports/datasets/fuxuan_v1/analysis/slice9_candidates_v1.csv")
DEFAULT_SLICE9_HTML_PATH = Path("data/reports/datasets/fuxuan_v1/analysis/slice9_candidates_v1.html")
_STANDARDIZED_ROOT = Path("data/work/standardized")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _waveform_svg(audio_path: Path, *, width: int = 360, height: int = 80, points: int = 180) -> tuple[str, str]:
    if not audio_path.exists():
        return "", "missing"
    try:
        import soundfile as sf

        samples, _ = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if len(samples) == 0:
            return "", "empty"
        envelope = []
        for index in range(points):
            start = int(index * len(samples) / points)
            end = max(start + 1, int((index + 1) * len(samples) / points))
            envelope.append(float(max(abs(samples[start:end]))))
        peak = max(envelope) or 1.0
        center = height / 2.0
        top = [(index * (width - 2) / max(1, points - 1) + 1, center - min(1.0, value / peak) * (height / 2.0 - 2)) for index, value in enumerate(envelope)]
        bottom = [(x, 2 * center - y) for x, y in reversed(top)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in top + bottom)
        return f'<svg class="waveform" viewBox="0 0 {width} {height}" role="img" aria-label="waveform"><polygon points="{polygon}" /></svg>', "ok"
    except Exception as error:  # pragma: no cover - codec/platform failures are reported in the artifact.
        return html.escape(f"waveform unavailable: {type(error).__name__}"), "error"


def _audio_uri(relative_path: str) -> str:
    return (_STANDARDIZED_ROOT / relative_path).resolve().as_uri()


def _row(pool_id: str, kind: str, rank: int | None, item: dict[str, Any]) -> dict[str, Any]:
    relative = str(item["audio_relative_path"])
    audio_path = (_STANDARDIZED_ROOT / relative).resolve()
    return {
        "pool_id": pool_id,
        "selection_kind": kind,
        "rank": rank,
        "asset_sha256": item["asset_sha256"],
        "audio_relative_path": relative,
        "audio_sha256": item.get("audio_sha256"),
        "audio_uri": _audio_uri(relative),
        "text_exact": item["text_exact"],
        "text_source": item.get("text_source"),
        "confidence_tier": item.get("confidence_tier"),
        "emotion_primary": item.get("emotion_primary"),
        "composite_score": item.get("composite_score"),
        "available_weight": item.get("available_weight"),
        "diversity_score": item.get("diversity_score"),
        "review_flags": item.get("review_flags", []),
        "score_reasons": item.get("score_reasons", []),
        "duplicate_group_id": item.get("duplicate_group_id"),
        "waveform_status": "ok" if audio_path.exists() else "missing",
    }


def build_slice9_reports(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    constrained_report_path: str | Path = DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH,
    sensitivity_report_path: str | Path = DEFAULT_SLICE9_SENSITIVITY_REPORT_PATH,
    output_json_path: str | Path = DEFAULT_SLICE9_REVIEW_JSON_PATH,
    output_csv_path: str | Path = DEFAULT_SLICE9_CSV_PATH,
    output_html_path: str | Path = DEFAULT_SLICE9_HTML_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    constrained_path = Path(constrained_report_path).resolve()
    sensitivity_path = Path(sensitivity_report_path).resolve()
    constrained = _load_json(constrained_path, label="constrained Slice 9 selection")
    sensitivity = _load_json(sensitivity_path, label="Slice 9 sensitivity report")
    rows: list[dict[str, Any]] = []
    pool_summaries: dict[str, Any] = {}
    for pool_id, pool in constrained["pools"].items():
        selected_rows = [_row(pool_id, "selected", item.get("constrained_rank"), item) for item in pool["selected"]]
        boundary_rows = [_row(pool_id, "boundary", None, item) for item in pool["boundary"]]
        rows.extend(selected_rows + boundary_rows)
        pool_summaries[pool_id] = {
            "candidate_count": pool["candidate_count"],
            "selected_count": len(selected_rows),
            "boundary_count": len(boundary_rows),
            "missing_audio_count": sum(row["waveform_status"] == "missing" for row in selected_rows + boundary_rows),
        }
    rows.sort(key=lambda row: (list(config["pools"]).index(row["pool_id"]), row["selection_kind"] != "selected", row["rank"] or 999, row["asset_sha256"]))
    json_payload = {
        "schema_version": 1,
        "review_report_schema_version": "slice9_manual_review_report@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": {
            "config_sha256": _sha256(Path(config_path).resolve()),
            "constrained_report_sha256": _sha256(constrained_path),
            "sensitivity_report_sha256": _sha256(sensitivity_path),
        },
        "manual_review_required": True,
        "manual_review_scope": "每个池 selected Top-K + boundary 样本；自动分数只建议，不静默定稿",
        "sensitivity_summary": sensitivity["summary"],
        "pool_summaries": pool_summaries,
        "row_count": len(rows),
        "rows": rows,
    }
    json_path = Path(output_json_path).resolve()
    csv_path = Path(output_csv_path).resolve()
    html_path = Path(output_html_path).resolve()
    _atomic_write_text(json_path, json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    csv_fields = ["pool_id", "selection_kind", "rank", "asset_sha256", "audio_relative_path", "audio_sha256", "audio_uri", "text_exact", "text_source", "confidence_tier", "emotion_primary", "composite_score", "available_weight", "diversity_score", "review_flags", "score_reasons", "duplicate_group_id", "waveform_status"]
    csv_lines: list[str] = []
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field], ensure_ascii=False) if isinstance(row[field], (list, dict)) else row[field] for field in csv_fields})
    sections: list[str] = []
    for pool_id in config["pools"]:
        pool_rows = [row for row in rows if row["pool_id"] == pool_id]
        body: list[str] = []
        for row in pool_rows:
            audio_path = (_STANDARDIZED_ROOT / row["audio_relative_path"]).resolve()
            waveform, _ = _waveform_svg(audio_path)
            reasons = html.escape(json.dumps(row["score_reasons"] or row["review_flags"], ensure_ascii=False))
            body.append(
                "<tr>"
                f"<td>{html.escape(str(row['selection_kind']))}</td><td>{html.escape(str(row['rank'] or ''))}</td>"
                f"<td><code>{html.escape(row['asset_sha256'][:16])}</code></td>"
                f"<td>{html.escape(str(row['confidence_tier']))}</td><td>{html.escape(str(row['composite_score']))}</td>"
                f"<td class=\"text\">{html.escape(row['text_exact'])}</td>"
                f"<td>{waveform}<audio controls preload=\"none\" src=\"{html.escape(row['audio_uri'], quote=True)}\"></audio><small>{html.escape(row['waveform_status'])}</small></td>"
                f"<td>{reasons}</td></tr>"
            )
        sections.append(f"<h2>{html.escape(pool_id)}</h2><table><thead><tr><th>kind</th><th>rank</th><th>asset</th><th>tier</th><th>score</th><th>text</th><th>audio/waveform</th><th>reasons</th></tr></thead><tbody>{''.join(body)}</tbody></table>")
    html_content = "<!doctype html><meta charset=\"utf-8\"><title>Slice 9 manual review</title><style>body{font:14px system-ui,sans-serif;margin:24px;background:#fafafa}table{border-collapse:collapse;width:100%;margin-bottom:28px;background:white}th,td{border:1px solid #ddd;padding:6px;vertical-align:top}th{background:#eee;position:sticky;top:0}.text{max-width:360px}.waveform{display:block;width:360px;height:80px;background:#f4f7fb;border:1px solid #ccd6e0}.waveform polygon{fill:#4f7cac}audio{width:360px}small{color:#888}code{font-size:11px}</style><p><strong>人工听审必需：</strong>自动分数仅建议，不静默定稿。每个池包含 selected Top-K 和 boundary 样本。</p>" + "".join(sections)
    _atomic_write_text(html_path, html_content)
    return {**json_payload, "report_paths": {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}}

