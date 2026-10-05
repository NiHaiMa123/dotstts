from __future__ import annotations

import csv
import html
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

_CSV_FIELDS = (
    "relative_path",
    "change_kind",
    "scan_status",
    "scan_error",
    "sha256",
    "size_bytes",
    "mtime_ns",
    "path_length",
    "path_warning",
    "speaker_id",
    "emotion_weak_label",
    "directory_emotion_weak_label",
    "filename_emotion_weak_label",
    "transcript_candidate",
    "parse_status",
    "parse_error",
    "metadata_status",
    "metadata_review_reason",
    "parser_profile_id",
    "parser_profile_version",
    "parser_config_sha256",
    "format",
    "subtype",
    "sample_rate",
    "channels",
    "frames",
    "duration_seconds",
    "probe_status",
    "probe_error_type",
    "probe_error_message",
    "moved_from",
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as fout:
            fout.write(text)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _csv_text(rows: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in _CSV_FIELDS})
    return output.getvalue()


def _summary_table(summary: dict[str, Any]) -> str:
    keys = (
        "status",
        "run_id",
        "root_path",
        "discovered_count",
        "readable_count",
        "error_count",
        "review_required_count",
        "added_count",
        "changed_count",
        "moved_count",
        "missing_count",
        "unchanged_count",
        "total_duration_minutes",
        "path_warning_count",
        "parser_profile_id",
        "parser_profile_version",
        "parser_config_sha256",
    )
    return "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(key), html.escape(str(summary.get(key, "")))
        )
        for key in keys
    )


def _mapping_table(mapping: dict[str, Any]) -> str:
    if not mapping:
        return '<tr><td colspan="2">None</td></tr>'
    return "\n".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(key)), html.escape(str(value))
        )
        for key, value in mapping.items()
    )


def _file_rows(records: Iterable[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        status = str(record.get("scan_status") or record.get("change_kind") or "")
        row_class = status if status in {"error", "review_required"} else ""
        rows.append(
            "<tr class=\"{}\"><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                row_class,
                html.escape(str(record.get("relative_path", ""))),
                html.escape(str(record.get("change_kind", ""))),
                html.escape(str(record.get("speaker_id", ""))),
                html.escape(str(record.get("emotion_weak_label", ""))),
                html.escape(str(record.get("sample_rate", ""))),
                html.escape(str(record.get("duration_seconds", ""))),
                html.escape(str(record.get("scan_error", ""))),
                html.escape(str(record.get("scan_review_reason", ""))),
            )
        )
    return "\n".join(rows)


def _html_text(
    *,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    missing_records: list[dict[str, Any]],
) -> str:
    all_rows = [*records, *missing_records]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>dots.tts inventory</title>
  <style>
    body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: .4rem .55rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr.error td {{ background: #fef2f2; }}
    tr.review_required td {{ background: #fffbeb; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>dots.tts inventory</h1>
  <h2>Summary</h2>
  <table><tbody>{_summary_table(summary)}</tbody></table>
  <h2>Emotion weak labels</h2>
  <table><thead><tr><th>Label</th><th>Count</th></tr></thead>
  <tbody>{_mapping_table(summary.get('emotion_counts', {}))}</tbody></table>
  <h2>Sample rates</h2>
  <table><thead><tr><th>Hz</th><th>Count</th></tr></thead>
  <tbody>{_mapping_table(summary.get('sample_rate_counts', {}))}</tbody></table>
  <h2>Files</h2>
  <table><thead><tr><th>Relative path</th><th>Change</th><th>Speaker</th>
  <th>Emotion</th><th>Sample rate</th><th>Duration</th><th>Error</th>
  <th>Review reason</th></tr></thead>
  <tbody>{_file_rows(all_rows)}</tbody></table>
</body>
</html>
"""


def write_inventory_reports(
    report_dir: str | Path,
    *,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    missing_records: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir = Path(report_dir).resolve()
    json_path = output_dir / "inventory.json"
    csv_path = output_dir / "inventory.csv"
    html_path = output_dir / "inventory.html"
    payload = {
        "schema_version": 1,
        "summary": summary,
        "files": records,
        "missing_locations": missing_records,
    }

    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(csv_path, _csv_text([*records, *missing_records]))
    _atomic_write_text(
        html_path,
        _html_text(
            summary=summary,
            records=records,
            missing_records=missing_records,
        ),
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }


_INGEST_CSV_FIELDS = (
    "source_relative_path",
    "original_name",
    "action",
    "asset_sha256",
    "raw_relative_path",
    "error_type",
    "error_message",
)


def _ingest_csv_text(items: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=_INGEST_CSV_FIELDS, extrasaction="ignore"
    )
    writer.writeheader()
    for item in items:
        writer.writerow({field: item.get(field) for field in _INGEST_CSV_FIELDS})
    return output.getvalue()


def _ingest_html_text(
    *, summary: dict[str, Any], items: list[dict[str, Any]]
) -> str:
    summary_keys = (
        "status",
        "run_id",
        "root_path",
        "raw_root_path",
        "storage_mode",
        "discovered_count",
        "unique_asset_count",
        "imported_count",
        "reused_count",
        "error_count",
        "bytes_written",
    )
    summary_rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(key), html.escape(str(summary.get(key, "")))
        )
        for key in summary_keys
    )
    item_rows = "\n".join(
        '<tr class="{}"><td>{}</td><td>{}</td><td><code>{}</code></td>'
        "<td><code>{}</code></td><td>{}</td></tr>".format(
            "error" if item.get("action") == "error" else "",
            html.escape(str(item.get("source_relative_path", ""))),
            html.escape(str(item.get("action", ""))),
            html.escape(str(item.get("asset_sha256", ""))),
            html.escape(str(item.get("raw_relative_path", ""))),
            html.escape(str(item.get("error_message", ""))),
        )
        for item in items
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>dots.tts immutable ingest</title>
  <style>
    body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: .4rem .55rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr.error td {{ background: #fef2f2; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>dots.tts immutable ingest</h1>
  <h2>Summary</h2>
  <table><tbody>{summary_rows}</tbody></table>
  <h2>Source locations</h2>
  <table><thead><tr><th>Source</th><th>Action</th><th>SHA-256</th>
  <th>Raw object</th><th>Error</th></tr></thead><tbody>{item_rows}</tbody></table>
</body>
</html>
"""


def write_ingest_reports(
    report_dir: str | Path,
    *,
    summary: dict[str, Any],
    items: list[dict[str, Any]],
    raw_objects: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir = Path(report_dir).resolve()
    json_path = output_dir / "ingest.json"
    csv_path = output_dir / "ingest.csv"
    html_path = output_dir / "ingest.html"
    payload = {
        "schema_version": 1,
        "summary": summary,
        "raw_objects": raw_objects,
        "source_locations": items,
    }
    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(csv_path, _ingest_csv_text(items))
    _atomic_write_text(html_path, _ingest_html_text(summary=summary, items=items))
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }


_QUALITY_CSV_FIELDS = (
    "asset_sha256",
    "raw_relative_path",
    "source_relative_path",
    "speaker_id",
    "emotion_weak_label",
    "transcript_candidate",
    "metric_action",
    "decision",
    "sample_rate",
    "channels",
    "duration_seconds",
    "sample_peak_dbfs",
    "true_peak_estimate_dbtp",
    "rms_dbfs",
    "integrated_loudness_lufs",
    "crest_factor_db",
    "abs_dc_offset",
    "leading_silence_seconds",
    "trailing_silence_seconds",
    "silence_ratio",
    "digital_silence_frame_ratio",
    "noise_floor_proxy_dbfs",
    "speech_level_proxy_dbfs",
    "snr_proxy_db",
    "near_peak_sample_count",
    "near_peak_sample_ratio",
    "flat_top_run_count",
    "max_flat_top_run_samples",
    "reasons_json",
    "error_type",
    "error_message",
)


def _quality_csv_text(items: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=_QUALITY_CSV_FIELDS,
        extrasaction="ignore",
    )
    writer.writeheader()
    for item in items:
        writer.writerow(
            {field: item.get(field) for field in _QUALITY_CSV_FIELDS}
        )
    return output.getvalue()


def _quality_html_text(
    *, summary: dict[str, Any], items: list[dict[str, Any]]
) -> str:
    summary_keys = (
        "status",
        "run_id",
        "analysis_id",
        "analysis_version",
        "policy_id",
        "policy_version",
        "discovered_count",
        "analyzed_count",
        "cached_count",
        "pass_count",
        "review_count",
        "reject_count",
        "error_count",
    )
    summary_rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(key), html.escape(str(summary.get(key, "")))
        )
        for key in summary_keys
    )
    reason_rows = _mapping_table(summary.get("reason_counts", {}))
    file_rows = "\n".join(
        '<tr class="{}"><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td>'
        "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(item.get("decision", ""))),
            html.escape(str(item.get("source_relative_path", ""))),
            html.escape(str(item.get("raw_relative_path", ""))),
            html.escape(str(item.get("metric_action", ""))),
            html.escape(str(item.get("decision", ""))),
            html.escape(str(item.get("integrated_loudness_lufs", ""))),
            html.escape(str(item.get("true_peak_estimate_dbtp", ""))),
            html.escape(str(item.get("snr_proxy_db", ""))),
            html.escape(str(item.get("trailing_silence_seconds", ""))),
            html.escape(str(item.get("reasons_json", ""))),
        )
        for item in items
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>dots.tts objective audio quality</title>
  <style>
    body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: .4rem .55rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr.review td {{ background: #fffbeb; }}
    tr.reject td {{ background: #fef2f2; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>dots.tts objective audio quality</h1>
  <h2>Summary</h2>
  <table><tbody>{summary_rows}</tbody></table>
  <h2>Review reasons</h2>
  <table><thead><tr><th>Reason</th><th>Count</th></tr></thead>
  <tbody>{reason_rows}</tbody></table>
  <h2>Assets</h2>
  <table><thead><tr><th>Source</th><th>Raw object</th><th>Metric</th><th>Decision</th>
  <th>LUFS</th><th>True peak est.</th><th>SNR proxy</th>
  <th>Trailing silence</th><th>Reasons</th></tr></thead>
  <tbody>{file_rows}</tbody></table>
</body>
</html>
"""


def write_quality_reports(
    report_dir: str | Path,
    *,
    summary: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir = Path(report_dir).resolve()
    json_path = output_dir / "quality.json"
    csv_path = output_dir / "quality.csv"
    html_path = output_dir / "quality.html"
    payload = {
        "schema_version": 1,
        "summary": summary,
        "assets": items,
    }
    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(csv_path, _quality_csv_text(items))
    _atomic_write_text(
        html_path,
        _quality_html_text(summary=summary, items=items),
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }


_STANDARDIZATION_CSV_FIELDS = (
    "asset_sha256",
    "derived_id",
    "source_relative_path",
    "raw_relative_path",
    "derived_relative_path",
    "speaker_id",
    "emotion_weak_label",
    "action",
    "source_sample_rate",
    "source_channels",
    "source_frames",
    "output_sample_rate",
    "output_channels",
    "output_frames",
    "duration_seconds",
    "leading_samples_removed",
    "trailing_samples_removed",
    "gain_applied_db",
    "output_sample_peak_dbfs",
    "output_true_peak_estimate_dbtp",
    "output_sha256",
    "size_bytes",
    "error_type",
    "error_message",
)


def _standardization_csv_text(items: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=_STANDARDIZATION_CSV_FIELDS, extrasaction="ignore"
    )
    writer.writeheader()
    for item in items:
        writer.writerow(
            {field: item.get(field) for field in _STANDARDIZATION_CSV_FIELDS}
        )
    return output.getvalue()


def _standardization_html_text(
    *, summary: dict[str, Any], items: Iterable[dict[str, Any]]
) -> str:
    summary_keys = (
        "status",
        "run_id",
        "config_id",
        "config_version",
        "config_sha256",
        "implementation_version",
        "discovered_count",
        "built_count",
        "cached_count",
        "rebuilt_count",
        "trimmed_count",
        "attenuated_count",
        "error_count",
        "output_duration_minutes",
        "output_size_bytes",
    )
    summary_rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(key), html.escape(str(summary.get(key, "")))
        )
        for key in summary_keys
    )
    item_rows = "\n".join(
        """<tr class=\"{}\"><td><code>{}</code></td><td>{}</td><td>{}</td>"
        "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>""".format(
            "error" if item.get("action") == "error" else "",
            html.escape(str(item.get("asset_sha256", ""))),
            html.escape(str(item.get("source_relative_path", ""))),
            html.escape(str(item.get("derived_relative_path", ""))),
            html.escape(str(item.get("action", ""))),
            html.escape(str(item.get("output_frames", ""))),
            html.escape(str(item.get("trailing_samples_removed", ""))),
            html.escape(str(item.get("gain_applied_db", ""))),
            html.escape(str(item.get("error_message", ""))),
        )
        for item in items
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>dots.tts standardized training audio</title>
  <style>
    body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: .4rem .55rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tr.error td {{ background: #fef2f2; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>dots.tts standardized training audio</h1>
  <h2>Summary</h2>
  <table><tbody>{summary_rows}</tbody></table>
  <h2>Assets</h2>
  <table><thead><tr><th>Asset</th><th>Source</th><th>Derived</th><th>Action</th>
  <th>Frames</th><th>Tail removed</th><th>Gain dB</th><th>Error</th></tr></thead>
  <tbody>{item_rows}</tbody></table>
</body>
</html>
"""


def write_standardization_reports(
    report_dir: str | Path,
    *,
    summary: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir = Path(report_dir).resolve()
    json_path = output_dir / "standardization.json"
    csv_path = output_dir / "standardization.csv"
    html_path = output_dir / "standardization.html"
    payload = {"schema_version": 1, "summary": summary, "assets": items}
    _atomic_write_text(
        json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    _atomic_write_text(csv_path, _standardization_csv_text(items))
    _atomic_write_text(
        html_path, _standardization_html_text(summary=summary, items=items)
    )
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}
