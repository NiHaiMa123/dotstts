from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dots_tts_lab.dataset_freeze import DatasetFreezeConfig, canonical_dataset_config
from dots_tts_lab.duplicate_graph import canonical_json


_SPLITS = ("train", "validation", "test")
_PARQUET_SCHEMA = pa.schema(
    [
        ("ordinal", pa.int32()),
        ("fid", pa.string()),
        ("asset_sha256", pa.string()),
        ("split", pa.string()),
        ("analysis_run_id", pa.string()),
        ("group_id", pa.string()),
        ("source_root_key", pa.string()),
        ("source_path_key", pa.string()),
        ("source_relative_path", pa.string()),
        ("raw_relative_path", pa.string()),
        ("derived_id", pa.string()),
        ("audio_relative_path", pa.string()),
        ("audio_absolute_path", pa.string()),
        ("audio_sha256", pa.string()),
        ("duration_seconds", pa.float64()),
        ("quality_run_id", pa.string()),
        ("quality_decision", pa.string()),
        ("quality_reasons_json", pa.string()),
        ("text_exact", pa.string()),
        ("text_sha256", pa.string()),
        ("text_source", pa.string()),
        ("speaker_id", pa.string()),
        ("emotion_weak_label", pa.string()),
        ("emotion_primary", pa.string()),
        ("emotion_secondary", pa.string()),
        ("intensity", pa.string()),
        ("label_source", pa.string()),
        ("review_decision_id", pa.string()),
        ("review_round", pa.int32()),
        ("review_batch_id", pa.string()),
        ("lineage_json", pa.string()),
    ],
    metadata={
        b"artifact": b"dots_tts_lab.canonical_dataset",
        b"schema_version": b"1",
    },
)


def build_dataset_artifact_set(
    root: str | Path,
    *,
    config: DatasetFreezeConfig,
    selection: dict[str, Any],
    split_plan: dict[str, Any],
    split_audit: dict[str, Any],
    input_file_sha256s: dict[str, str],
    standardized_root: str | Path,
) -> dict[str, Any]:
    output_root = Path(root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _dataset_rows(
        config=config,
        selection=selection,
        split_plan=split_plan,
        split_audit=split_audit,
        standardized_root=standardized_root,
    )
    paths = _output_paths(output_root, config)
    _assert_distinct_paths(paths)

    table = pa.Table.from_pylist(rows, schema=_PARQUET_SCHEMA)
    pq.write_table(
        table,
        paths["parquet"],
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
        store_schema=True,
    )
    _fsync_file(paths["parquet"])

    for split in _SPLITS:
        content = "".join(
            canonical_json(
                {
                    "fid": row["fid"],
                    "audio": row["audio_absolute_path"],
                    "text": row["text_exact"],
                }
            )
            + "\n"
            for row in rows
            if row["split"] == split
        )
        _write_bytes(paths[split], content.encode("utf-8"))

    stats = _build_stats(rows, selection, split_plan)
    _write_json(paths["stats"], stats)
    frozen_config = {
        "schema_version": 1,
        "dataset_config": config.model_dump(mode="json"),
        "dataset_config_sha256": canonical_dataset_config(config)["sha256"],
        "freeze_inputs": {
            "analysis_run_id": selection["analysis_run_id"],
            "candidate_snapshot_sha256": selection["candidate_snapshot_sha256"],
            "edge_review_snapshot_sha256": selection[
                "edge_review_snapshot_sha256"
            ],
            "selection_sha256": selection["selection_sha256"],
            "assignment_sha256": split_plan["assignment_sha256"],
            "split_audit_sha256": split_audit["audit_sha256"],
            "input_file_sha256s": dict(sorted(input_file_sha256s.items())),
        },
        "writer": {
            "implementation": "dots_tts_lab.dataset_artifacts@1",
            "pyarrow_version": pa.__version__,
            "row_order": "split(train,validation,test),asset_sha256",
            "parquet": {
                "compression": "zstd",
                "compression_level": 9,
                "data_page_version": "1.0",
                "format_version": "2.6",
                "use_dictionary": False,
                "write_statistics": True,
            },
        },
    }
    config_yaml = yaml.safe_dump(
        frozen_config,
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
    )
    _write_bytes(paths["config"], config_yaml.encode("utf-8"))

    core_names = ("parquet", "train", "validation", "test", "stats", "config")
    core_artifacts = {
        paths[name].relative_to(output_root).as_posix(): _file_identity(paths[name])
        for name in core_names
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "analysis_run_id": selection["analysis_run_id"],
        "candidate_snapshot_sha256": selection["candidate_snapshot_sha256"],
        "edge_review_snapshot_sha256": selection["edge_review_snapshot_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "assignment_sha256": split_plan["assignment_sha256"],
        "split_audit_sha256": split_audit["audit_sha256"],
        "item_count": len(rows),
        "total_duration_seconds": stats["total_duration_seconds"],
        "split_counts": stats["split_counts"],
        "artifacts": core_artifacts,
    }
    manifest["artifact_set_sha256"] = hashlib.sha256(
        canonical_json(core_artifacts).encode("utf-8")
    ).hexdigest()
    _write_json(paths["manifest"], manifest)

    checksum_names = (
        "config",
        "parquet",
        "manifest",
        "stats",
        "test",
        "train",
        "validation",
    )
    checksum_rows = sorted(
        (
            _sha256_file(paths[name]),
            paths[name].relative_to(output_root).as_posix(),
        )
        for name in checksum_names
    )
    checksum_content = "".join(
        f"{digest}  {relative}\n" for digest, relative in checksum_rows
    )
    _write_bytes(paths["checksums"], checksum_content.encode("utf-8"))
    return {
        "rows": rows,
        "stats": stats,
        "manifest": manifest,
        "checksums_sha256": _sha256_file(paths["checksums"]),
    }


def validate_dataset_artifact_set(
    root: str | Path, *, config: DatasetFreezeConfig
) -> dict[str, Any]:
    output_root = Path(root).resolve()
    paths = _output_paths(output_root, config)
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"Missing frozen dataset artifact: {path}")
    expected_checksums = {}
    for line in paths["checksums"].read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in expected_checksums:
            raise RuntimeError("Malformed or duplicate checksums.txt row")
        expected_checksums[relative] = digest
    for relative, expected in expected_checksums.items():
        artifact = (output_root / relative).resolve()
        if output_root not in artifact.parents or not artifact.is_file():
            raise RuntimeError(f"Unsafe or missing checksummed artifact: {relative}")
        if _sha256_file(artifact) != expected:
            raise RuntimeError(f"Frozen artifact checksum mismatch: {relative}")

    table = pq.read_table(paths["parquet"])
    if not table.schema.equals(_PARQUET_SCHEMA, check_metadata=True):
        raise RuntimeError("Canonical Parquet schema drift")
    parquet_rows = table.to_pylist()
    jsonl_rows = []
    for split in _SPLITS:
        for line in paths[split].read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if list(row) != ["audio", "fid", "text"] or not row["text"].strip():
                raise RuntimeError(f"Invalid trainer JSONL row in {split}")
            jsonl_rows.append((split, row))
    expected = [
        (
            row["split"],
            {
                "audio": row["audio_absolute_path"],
                "fid": row["fid"],
                "text": row["text_exact"],
            },
        )
        for row in parquet_rows
    ]
    if jsonl_rows != expected:
        raise RuntimeError("Parquet and trainer JSONL rows differ")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("item_count") != len(parquet_rows):
        raise RuntimeError("Manifest item count differs from Parquet")
    return {
        "item_count": len(parquet_rows),
        "split_counts": dict(Counter(row["split"] for row in parquet_rows)),
        "checksummed_artifact_count": len(expected_checksums),
    }


def _dataset_rows(
    *,
    config: DatasetFreezeConfig,
    selection: dict[str, Any],
    split_plan: dict[str, Any],
    split_audit: dict[str, Any],
    standardized_root: str | Path,
) -> list[dict[str, Any]]:
    items = selection.get("selected_items")
    groups = selection.get("selected_groups")
    assignments = split_plan.get("asset_assignments")
    if not isinstance(items, list) or not isinstance(groups, list) or not isinstance(
        assignments, dict
    ):
        raise RuntimeError("Selection or split input is malformed")
    if split_audit.get("status") != "passed" or split_audit.get("violation_count") != 0:
        raise RuntimeError("Split audit must pass before artifact generation")
    identities = (
        (split_plan, "selection_sha256", selection.get("selection_sha256")),
        (split_audit, "selection_sha256", selection.get("selection_sha256")),
        (split_audit, "assignment_sha256", split_plan.get("assignment_sha256")),
    )
    if any(payload.get(field) != expected for payload, field, expected in identities):
        raise RuntimeError("Selection, split, and split-audit identities differ")

    group_by_asset = {}
    for group in groups:
        for asset in group.get("member_asset_sha256s", []):
            if asset in group_by_asset:
                raise RuntimeError("Selected asset belongs to multiple groups")
            group_by_asset[asset] = group["group_id"]
    assets = {item.get("asset_sha256") for item in items}
    if None in assets or set(assignments) != assets or set(group_by_asset) != assets:
        raise RuntimeError("Selection, groups, and split assignments do not agree")
    if any(split not in _SPLITS for split in assignments.values()):
        raise RuntimeError("Unsupported split assignment")

    audio_root = Path(standardized_root).resolve()
    rows = []
    ordered = sorted(items, key=lambda row: (_SPLITS.index(assignments[row["asset_sha256"]]), row["asset_sha256"]))
    for ordinal, item in enumerate(ordered, start=1):
        asset = item["asset_sha256"]
        audio = (audio_root / item["audio_relative_path"]).resolve()
        if audio_root not in audio.parents:
            raise RuntimeError(f"Audio path escapes standardized root: {asset}")
        rows.append(
            {
                "ordinal": ordinal,
                "fid": item["fid"],
                "asset_sha256": asset,
                "split": assignments[asset],
                "analysis_run_id": selection["analysis_run_id"],
                "group_id": group_by_asset[asset],
                "source_root_key": item["source_root_key"],
                "source_path_key": item["source_path_key"],
                "source_relative_path": item["source_relative_path"],
                "raw_relative_path": item["raw_relative_path"],
                "derived_id": item["derived_id"],
                "audio_relative_path": item["audio_relative_path"],
                "audio_absolute_path": str(audio),
                "audio_sha256": item["audio_sha256"],
                "duration_seconds": float(item["duration_seconds"]),
                "quality_run_id": item["quality_run_id"],
                "quality_decision": item["quality_decision"],
                "quality_reasons_json": canonical_json(item["quality_reasons"]),
                "text_exact": item["text_exact"],
                "text_sha256": item["text_sha256"],
                "text_source": item["text_source"],
                "speaker_id": item["speaker_id"],
                "emotion_weak_label": item["emotion_weak_label"],
                "emotion_primary": item["emotion_primary"],
                "emotion_secondary": item["emotion_secondary"],
                "intensity": item["intensity"],
                "label_source": item["label_source"],
                "review_decision_id": item["review_decision_id"],
                "review_round": item["review_round"],
                "review_batch_id": item["review_batch_id"],
                "lineage_json": canonical_json(item["lineage"]),
            }
        )
    return rows


def _build_stats(
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
    split_plan: dict[str, Any],
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    split_durations = defaultdict(float)
    emotion_counts = {split: Counter() for split in _SPLITS}
    text_sources = Counter()
    quality_decisions = Counter()
    for row in rows:
        split_durations[row["split"]] += row["duration_seconds"]
        emotion_counts[row["split"]][row["emotion_primary"]] += 1
        text_sources[row["text_source"]] += 1
        quality_decisions[row["quality_decision"]] += 1
    return {
        "schema_version": 1,
        "item_count": len(rows),
        "candidate_count": selection["candidate_count"],
        "excluded_count": selection["excluded_count"],
        "oversampled_count": selection["oversampled_count"],
        "total_duration_seconds": sum(row["duration_seconds"] for row in rows),
        "split_counts": {split: split_counts[split] for split in _SPLITS},
        "split_duration_seconds": {
            split: split_durations[split] for split in _SPLITS
        },
        "emotion_primary_counts": {
            split: dict(sorted(emotion_counts[split].items())) for split in _SPLITS
        },
        "text_source_counts": dict(sorted(text_sources.items())),
        "quality_decision_counts": dict(sorted(quality_decisions.items())),
        "split_algorithm": split_plan["algorithm"],
    }


def _output_paths(root: Path, config: DatasetFreezeConfig) -> dict[str, Path]:
    output = config.output
    return {
        "parquet": root / output.canonical_parquet,
        "manifest": root / output.manifest_json,
        "stats": root / output.stats_json,
        "train": root / output.train_jsonl,
        "validation": root / output.validation_jsonl,
        "test": root / output.test_jsonl,
        "config": root / output.frozen_config,
        "checksums": root / output.checksums,
    }


def _assert_distinct_paths(paths: dict[str, Path]) -> None:
    if len(set(paths.values())) != len(paths):
        raise RuntimeError("Frozen dataset artifact paths must be distinct")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as source:
        os.fsync(source.fileno())


def _file_identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
