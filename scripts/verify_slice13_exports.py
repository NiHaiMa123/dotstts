from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dots_tts_lab.release import ROOT, load_jsonl, load_release_config, sha256_file


def parse_expected_hashes(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, digest = value.partition("=")
        if not separator or len(digest) != 64:
            raise ValueError(f"Expected PRESET=SHA256, got {value!r}")
        result[name] = digest.lower()
    return result


def verify(
    config_path: Path,
    presets: list[str] | None,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_release_config(config_path)
    config_hash = sha256_file(config_path)
    source_manifest = (ROOT / config.source.generation_manifest).resolve()
    source_manifest_hash = sha256_file(source_manifest)
    selected = presets or list(config.presets)
    reports: dict[str, Any] = {}
    errors: list[str] = []
    for preset_name in selected:
        if preset_name not in config.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        preset = config.presets[preset_name]
        preset_root = (ROOT / config.output_root / preset_name).resolve()
        manifest = preset_root / "manifest.jsonl"
        manifest_hash = sha256_file(manifest)
        if preset_name in expected_hashes and expected_hashes[preset_name] != manifest_hash:
            errors.append(f"{preset_name}: manifest hash changed across rerun")
        rows = load_jsonl(manifest)
        by_job: dict[str, set[str]] = {}
        peaks: list[float] = []
        loudness: list[float] = []
        for row in rows:
            job_id = str(row["job_id"])
            by_job.setdefault(job_id, set()).add(str(row["format"]))
            output = (ROOT / row["output_path"]).resolve()
            sidecar_path = (ROOT / row["sidecar_path"]).resolve()
            if sha256_file(output) != row["output_sha256"]:
                errors.append(f"{preset_name}/{job_id}: output hash mismatch")
            if sha256_file(sidecar_path) != row["sidecar_sha256"]:
                errors.append(f"{preset_name}/{job_id}: sidecar hash mismatch")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("release_config_sha256") != config_hash:
                errors.append(f"{preset_name}/{job_id}: release config drift")
            if sidecar.get("source_manifest_sha256") != source_manifest_hash:
                errors.append(f"{preset_name}/{job_id}: source manifest drift")
            if sidecar.get("output", {}).get("sha256") != row["output_sha256"]:
                errors.append(f"{preset_name}/{job_id}: sidecar output hash mismatch")
            job = sidecar.get("job", {})
            reference = sidecar.get("reference", {})
            model = sidecar.get("model", {})
            if not (
                job.get("text")
                and job.get("seed") is not None
                and reference.get("audio_sha256")
                and model.get("model_id")
                and model.get("model_adapter")
                and model.get("sampling_options")
            ):
                errors.append(f"{preset_name}/{job_id}: incomplete provenance")
            processing = sidecar.get("processing", {})
            peak = processing.get("output_true_peak_dbtp")
            measured_loudness = processing.get("output_loudness_lufs")
            if peak is not None:
                peaks.append(float(peak))
            if measured_loudness is not None:
                loudness.append(float(measured_loudness))
            if preset_name in {"raw", "training"}:
                if row["output_sha256"] != sidecar.get("source_audio_sha256"):
                    errors.append(f"{preset_name}/{job_id}: passthrough is not byte-identical")
                if processing.get("total_gain_db") != 0.0 or processing.get("edge_trim"):
                    errors.append(f"{preset_name}/{job_id}: passthrough was processed")
            else:
                ceiling = float(preset.true_peak_ceiling_dbtp)
                target = float(preset.target_loudness_lufs)
                if peak is None or float(peak) > ceiling + 0.01:
                    errors.append(f"{preset_name}/{job_id}: true-peak ceiling failed")
                if measured_loudness is None:
                    errors.append(f"{preset_name}/{job_id}: output loudness is missing")
                elif not processing.get("limited_by_true_peak") and float(
                    processing.get("encoding_safety_attenuation_db", 0.0)
                ) == 0.0 and abs(float(measured_loudness) - target) > 0.25:
                    errors.append(f"{preset_name}/{job_id}: loudness target failed")
                elif float(measured_loudness) > target + 0.25:
                    errors.append(f"{preset_name}/{job_id}: loudness exceeds target")
        expected_formats = set(preset.formats)
        for job_id, formats in by_job.items():
            if formats != expected_formats:
                errors.append(
                    f"{preset_name}/{job_id}: formats {sorted(formats)} != "
                    f"{sorted(expected_formats)}"
                )
        reports[preset_name] = {
            "job_count": len(by_job),
            "output_count": len(rows),
            "formats": sorted(expected_formats),
            "manifest_sha256": manifest_hash,
            "idempotence_hash_matched": (
                None
                if preset_name not in expected_hashes
                else expected_hashes[preset_name] == manifest_hash
            ),
            "minimum_loudness_lufs": min(loudness) if loudness else None,
            "maximum_loudness_lufs": max(loudness) if loudness else None,
            "maximum_true_peak_dbtp": max(peaks) if peaks else None,
        }
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "source_manifest_sha256": source_manifest_hash,
        "presets": reports,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Slice 13 exported audio")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lab/slice13/release_presets_v1.yaml"),
    )
    parser.add_argument("--preset", action="append")
    parser.add_argument("--expected-manifest-sha", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/slice13-export-smoke-v1.json"),
    )
    args = parser.parse_args()
    report = verify(
        args.config,
        args.preset,
        parse_expected_hashes(args.expected_manifest_sha),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
