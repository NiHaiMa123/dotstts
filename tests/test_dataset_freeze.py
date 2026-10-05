from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import yaml

from dots_tts.data.source_adapters.base_adapter import SourceContext
from dots_tts.data.source_adapters.jsonl_manifest_adapter import (
    JsonlManifestSourceAdapter,
)
from dots_tts_lab.dataset_freeze import (
    DEFAULT_DATASET_FREEZE_CONFIG_PATH,
    DatasetFreezeConfig,
    build_candidate_snapshot,
    canonical_dataset_config,
    load_dataset_freeze_config,
    resolve_dataset_freeze_candidates,
    write_candidate_snapshot,
)
from dots_tts_lab.catalog import Catalog, SCHEMA_VERSION, _MIGRATIONS
from dots_tts_lab.cli import build_parser
from dots_tts_lab.dataset_audit import _load_quality_report
from dots_tts_lab.speaker_embedding import (
    SPEAKER_EMBEDDING_DIMENSION,
    assess_speaker_embeddings,
    build_speaker_embedding_cache_and_assessment,
    compute_speaker_embedding,
    load_or_compute_speaker_embedding,
    load_speaker_encoder,
    write_speaker_embedding_report,
)
from dots_tts_lab.threshold_calibration import (
    ThresholdCalibrationOverlay,
    assert_dataset_ready_with_calibration,
    load_threshold_calibration_overlay,
    load_threshold_calibration_config,
    robust_lower_threshold,
    select_stratified_duration_samples,
    separated_midpoint_threshold,
)
from dots_tts_lab.acoustic_fingerprint import (
    FINGERPRINT_DIMENSION,
    build_acoustic_fingerprint_cache,
    compute_acoustic_fingerprint,
    cosine_similarity,
    load_or_compute_acoustic_fingerprint,
    write_acoustic_fingerprint_report,
)
from dots_tts_lab.dataset_analysis import (
    analyze_exact_audio,
    analyze_exact_text,
    analyze_near_text,
    normalize_text_for_similarity,
    symmetric_levenshtein_similarity,
    write_exact_audio_report,
    write_exact_text_report,
    write_near_text_report,
)
from dots_tts_lab.duplicate_graph import (
    build_duplicate_edges,
    canonical_json,
    canonical_sha256,
    stable_similarity_groups,
)
from dots_tts_lab.dataset_analysis_report import (
    build_dataset_analysis_report,
    write_dataset_analysis_reports,
    write_portable_dataset_analysis_bundle,
)
from dots_tts_lab.dataset_split import (
    assert_no_group_crosses_splits,
    balanced_strata_targets,
    largest_remainder_targets,
    plan_emotion_stratified_split,
    plan_group_aware_split,
)
from dots_tts_lab.dataset_selection import apply_reviewed_exclusions
from dots_tts_lab.dataset_split_audit import (
    assert_split_audit_passes,
    audit_split_leakage,
)
from dots_tts_lab.dataset_publish import (
    DatasetVersionConflict,
    atomic_publish_dataset,
    dataset_tree_inventory,
)
from dots_tts_lab.dataset_artifacts import (
    build_dataset_artifact_set,
    validate_dataset_artifact_set,
)
from dots_tts_lab.dataset_registry import validate_frozen_dataset


class TrainingManifestContractTests(unittest.TestCase):
    def context(self) -> SourceContext:
        return SourceContext(
            epoch=0,
            rank=0,
            world_size=1,
            worker_id=0,
            num_workers=1,
            seed=42,
        )

    def test_official_three_field_jsonl_contract_and_lineage_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = (root / "sample.wav").resolve()
            manifest_path = root / "train.jsonl"
            record = {
                "fid": "asset-0001",
                "audio": str(audio_path),
                "text": "冻结清单契约",
                "asset_sha256": "a" * 64,
                "split": "train",
            }
            manifest_path.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            adapter = JsonlManifestSourceAdapter(
                manifest_path=str(manifest_path),
                shuffle=False,
            )
            sample = next(iter(adapter.iter_samples(self.context())))

            self.assertEqual(sample["fid"], record["fid"])
            self.assertEqual(sample["audio"], record["audio"])
            self.assertEqual(sample["text"], record["text"])
            self.assertTrue(Path(sample["audio"]).is_absolute())
            self.assertEqual(sample["asset_sha256"], record["asset_sha256"])
            self.assertEqual(sample["split"], "train")

    def test_official_adapter_rejects_each_missing_required_field(self) -> None:
        complete = {"fid": "asset-0001", "audio": "sample.wav", "text": "文本"}
        for missing in ("fid", "audio", "text"):
            with self.subTest(missing=missing):
                record = dict(complete)
                del record[missing]
                with tempfile.TemporaryDirectory() as temp_dir:
                    manifest_path = Path(temp_dir) / "train.jsonl"
                    manifest_path.write_text(
                        json.dumps(record, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    adapter = JsonlManifestSourceAdapter(
                        manifest_path=str(manifest_path),
                        shuffle=False,
                    )
                    with self.assertRaisesRegex(KeyError, missing):
                        next(iter(adapter.iter_samples(self.context())))


class DatasetFreezeConfigTests(unittest.TestCase):
    def payload(self) -> dict:
        return yaml.safe_load(
            DEFAULT_DATASET_FREEZE_CONFIG_PATH.read_text(encoding="utf-8")
        )

    def test_loads_versioned_audit_config_and_blocks_premature_freeze(self) -> None:
        config = load_dataset_freeze_config()

        self.assertEqual(config.dataset_id, "fuxuan")
        self.assertEqual(config.dataset_version, 1)
        self.assertEqual(config.output.trainer_jsonl_fields, ["fid", "audio", "text"])
        self.assertEqual(config.speaker_embedding.embedding_size, 512)
        with self.assertRaisesRegex(RuntimeError, "audit-only"):
            config.assert_freeze_ready()

    def test_rejects_unknown_fields_and_noncanonical_training_contract(self) -> None:
        payload = self.payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            DatasetFreezeConfig.model_validate(payload, strict=True)

        payload = self.payload()
        payload["output"]["trainer_jsonl_fields"] = ["audio", "text", "fid"]
        with self.assertRaisesRegex(ValueError, "exactly fid, audio, text"):
            DatasetFreezeConfig.model_validate(payload, strict=True)

    def test_rejects_unsafe_paths_ratios_and_integrity_opt_out(self) -> None:
        payload = self.payload()
        payload["output"]["dataset_root"] = "../outside"
        with self.assertRaisesRegex(ValueError, "relative POSIX"):
            DatasetFreezeConfig.model_validate(payload, strict=True)

        payload = self.payload()
        payload["split"]["test_ratio"] = 0.20
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            DatasetFreezeConfig.model_validate(payload, strict=True)

        payload = self.payload()
        payload["eligibility"]["require_verified_standardized_hash"] = False
        with self.assertRaisesRegex(ValueError, "cannot be disabled"):
            DatasetFreezeConfig.model_validate(payload, strict=True)

    def test_thresholds_are_null_only_while_calibration_is_required(self) -> None:
        payload = self.payload()
        payload["text_similarity"]["near_similarity_threshold"] = 0.95
        with self.assertRaisesRegex(ValueError, "text_similarity"):
            DatasetFreezeConfig.model_validate(payload, strict=True)


class CatalogV10MigrationTests(unittest.TestCase):
    expected_tables = {
        "dataset_build_config",
        "dataset_analysis_run",
        "dataset_asset_feature",
        "dataset_similarity_edge",
        "dataset_similarity_edge_review",
        "dataset_similarity_group",
        "dataset_similarity_group_member",
        "dataset_speaker_assessment",
        "dataset_speaker_review",
        "dataset_version",
        "dataset_item",
        "dataset_threshold_calibration",
        "dataset_analysis_calibration",
    }

    def test_empty_catalog_initializes_to_v10_with_all_dataset_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            with catalog.session() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

            self.assertEqual(version, SCHEMA_VERSION)
            self.assertEqual(SCHEMA_VERSION, 10)
            self.assertTrue(self.expected_tables.issubset(tables))

    def test_v8_catalog_upgrades_without_losing_existing_data_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.sqlite"
            connection = sqlite3.connect(path)
            try:
                for version in range(1, 9):
                    connection.executescript(_MIGRATIONS[version])
                connection.execute(
                    """
                    INSERT INTO asset (
                        sha256, size_bytes, probe_status, first_seen_at, last_seen_at
                    ) VALUES (?, ?, 'ok', ?, ?)
                    """,
                    ("a" * 64, 123, "2026-09-08T00:00:00+00:00", "2026-09-08T00:00:00+00:00"),
                )
                connection.commit()
            finally:
                connection.close()

            catalog = Catalog(path)
            catalog.initialize()
            catalog.initialize()
            with catalog.session() as upgraded:
                version = upgraded.execute("PRAGMA user_version").fetchone()[0]
                asset = upgraded.execute(
                    "SELECT size_bytes FROM asset WHERE sha256 = ?", ("a" * 64,)
                ).fetchone()
                dataset_table_count = upgraded.execute(
                    """
                    SELECT count(*) FROM sqlite_master
                    WHERE type = 'table' AND name LIKE 'dataset_%'
                    """
                ).fetchone()[0]

            self.assertEqual(version, 10)
            self.assertEqual(asset[0], 123)
            self.assertEqual(dataset_table_count, len(self.expected_tables))

    def test_v10_enforces_canonical_edge_order_and_feature_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            with catalog.session() as connection:
                sql = connection.execute(
                    """
                    SELECT sql FROM sqlite_master
                    WHERE type = 'table' AND name = 'dataset_similarity_edge'
                    """
                ).fetchone()[0]
                feature_sql = connection.execute(
                    """
                    SELECT sql FROM sqlite_master
                    WHERE type = 'table' AND name = 'dataset_asset_feature'
                    """
                ).fetchone()[0]

            self.assertIn("left_asset_sha256 < right_asset_sha256", sql)
            self.assertIn("fingerprint_dimension = 4096", feature_sql)
            self.assertIn("speaker_embedding_dimension = 512", feature_sql)


class DatasetCandidateQueryTests(unittest.TestCase):
    def seed_candidate(self, catalog: Catalog) -> None:
        asset = "a" * 64
        derived = "d" * 64
        connection = sqlite3.connect(catalog.path)
        try:
            connection.execute(
                """
                INSERT INTO asset (
                    sha256, size_bytes, sample_rate, channels, frames,
                    duration_seconds, probe_status, first_seen_at, last_seen_at
                ) VALUES (?, 123, 48000, 1, 48000, 1.0, 'ok', 't0', 't0')
                """,
                (asset,),
            )
            connection.execute(
                """
                INSERT INTO raw_object (
                    asset_sha256, relative_path, extension, storage_mode,
                    size_bytes, first_imported_at, last_verified_at,
                    first_import_run_id, last_import_run_id
                ) VALUES (?, 'aa/raw.wav', '.wav', 'copy', 123, 't0', 't0', 'i0', 'i0')
                """,
                (asset,),
            )
            connection.execute(
                """
                INSERT INTO source_location (
                    root_key, root_path, path_key, relative_path, asset_sha256,
                    size_bytes, mtime_ns, path_length, speaker_id,
                    emotion_weak_label, transcript_candidate, parse_status,
                    availability_status, first_seen_at, last_seen_at,
                    last_seen_run_id, metadata_status, parser_profile_id,
                    parser_profile_version, parser_config_sha256
                ) VALUES (
                    'root', '/source', 'path', 'speaker/file.wav', ?,
                    123, 1, 16, 'speaker', '中立_neutral', 'filename text', 'ok',
                    'available', 't0', 't0', 'r0', 'ok', 'profile', 1, ?
                )
                """,
                (asset, "1" * 64),
            )
            connection.execute(
                """
                INSERT INTO derived_audio (
                    derived_id, asset_sha256, config_id, config_version,
                    config_sha256, implementation_version, relative_path,
                    output_sha256, size_bytes, created_at, last_verified_at,
                    source_sample_rate, source_channels, source_frames,
                    output_sample_rate, output_channels, output_frames,
                    duration_seconds, leading_samples_removed,
                    trailing_samples_removed, gain_applied_db,
                    output_subtype, created_run_id
                ) VALUES (
                    ?, ?, 'training_audio', 1, ?, 1, 'dd/derived.wav', ?,
                    456, 't0', 't0', 48000, 1, 48000, 48000, 1, 48000,
                    1.0, 0, 0, 0.0, 'PCM_24', 's0'
                )
                """,
                (derived, asset, "2" * 64, "e" * 64),
            )
            for run_id, started_at, decision in (
                ("quality-old", "2026-01-01", "pass"),
                ("quality-new", "2026-02-01", "reject"),
            ):
                connection.execute(
                    """
                    INSERT INTO quality_run (
                        run_id, raw_root_path, analysis_id, analysis_version,
                        analysis_config_sha256, implementation_version,
                        policy_id, policy_version, policy_config_sha256,
                        started_at, finished_at, status
                    ) VALUES (?, '/raw', 'analysis', 1, ?, 1, 'policy', 1, ?, ?, ?, 'succeeded')
                    """,
                    (run_id, "3" * 64, "4" * 64, started_at, started_at),
                )
                connection.execute(
                    """
                    INSERT INTO quality_run_item (
                        run_id, asset_sha256, raw_relative_path,
                        metric_action, decision, reasons_json
                    ) VALUES (?, ?, 'aa/raw.wav', 'cached', ?, '[]')
                    """,
                    (run_id, asset, decision),
                )
            for decision_id, benchmark_version, review_round, text in (
                ("decision-old", 2, 1, "old review text"),
                ("decision-latest", 2, 2, "latest review text"),
                ("decision-other-benchmark", 3, 7, "foreign benchmark text"),
            ):
                connection.execute(
                    """
                    INSERT INTO review_decision (
                        decision_id, asset_sha256, benchmark_id,
                        benchmark_version, review_round, text_decision,
                        text_final, emotion_primary, label_source,
                        review_status, auto_rules_applied_json, created_at,
                        export_batch_id, source_row_index
                    ) VALUES (?, ?, 'benchmark', ?, ?, 'accept_edited', ?,
                              '中立_neutral', 'human_corrected', 'approved',
                              '[]', ?, ?, 0)
                    """,
                    (
                        decision_id,
                        asset,
                        benchmark_version,
                        review_round,
                        text,
                        f"t{review_round}",
                        f"batch-{benchmark_version}-{review_round}",
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def test_query_pins_quality_run_and_review_benchmark_latest_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            self.seed_candidate(catalog)

            rows = catalog.load_dataset_freeze_candidates(
                standardization_config_id="training_audio",
                standardization_config_version=1,
                quality_run_id="quality-old",
                review_benchmark_id="benchmark",
                review_benchmark_version=2,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["quality_run_id"], "quality-old")
            self.assertEqual(rows[0]["quality_decision"], "pass")
            self.assertEqual(rows[0]["review_decision_id"], "decision-latest")
            self.assertEqual(rows[0]["review_round"], 2)
            self.assertEqual(rows[0]["review_text_final"], "latest review text")
            self.assertEqual(rows[0]["derived_output_sha256"], "e" * 64)
            self.assertEqual(rows[0]["parser_config_sha256"], "1" * 64)

            newer_quality = catalog.load_dataset_freeze_candidates(
                standardization_config_id="training_audio",
                standardization_config_version=1,
                quality_run_id="quality-new",
                review_benchmark_id="benchmark",
                review_benchmark_version=2,
            )
            self.assertEqual(newer_quality[0]["quality_decision"], "reject")

    def test_query_does_not_leak_review_from_another_benchmark_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            self.seed_candidate(catalog)

            rows = catalog.load_dataset_freeze_candidates(
                standardization_config_id="training_audio",
                standardization_config_version=1,
                quality_run_id="quality-old",
                review_benchmark_id="benchmark",
                review_benchmark_version=3,
            )

            self.assertEqual(rows[0]["review_decision_id"], "decision-other-benchmark")
            self.assertEqual(rows[0]["review_round"], 7)


class DatasetCandidateResolutionTests(unittest.TestCase):
    def row(self, root: Path, *, reviewed: bool = False) -> dict:
        audio = root / "aa" / "audio.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"deterministic standardized audio")
        asset = "a" * 64
        row = {
            "asset_sha256": asset,
            "raw_relative_path": "aa/raw.wav",
            "source_root_key": "root",
            "source_path_key": "path",
            "source_relative_path": "speaker/file.wav",
            "source_availability_status": "available",
            "speaker_id": "speaker",
            "emotion_weak_label": "中立_neutral",
            "transcript_candidate": "filename text",
            "parser_profile_id": "profile",
            "parser_profile_version": 1,
            "parser_config_sha256": "1" * 64,
            "derived_id": "d" * 64,
            "derived_relative_path": "aa/audio.wav",
            "derived_output_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "derived_size_bytes": audio.stat().st_size,
            "derived_duration_seconds": 1.25,
            "output_sample_rate": 48000,
            "output_channels": 1,
            "output_subtype": "PCM_24",
            "standardization_config_sha256": "2" * 64,
            "standardization_implementation_version": 1,
            "quality_run_id": "quality-run",
            "quality_analysis_id": "objective_signal_metrics",
            "quality_analysis_version": 1,
            "quality_analysis_config_sha256": "3" * 64,
            "quality_implementation_version": 1,
            "quality_policy_id": "training_source_review",
            "quality_policy_version": 1,
            "quality_policy_config_sha256": "4" * 64,
            "quality_decision": "pass",
            "quality_reasons_json": "[]",
            "review_decision_id": None,
            "review_round": None,
            "review_text_decision": None,
            "review_text_final": None,
            "review_emotion_primary": None,
            "review_emotion_secondary": None,
            "review_intensity": None,
            "review_label_source": None,
            "review_status": None,
            "review_export_batch_id": None,
        }
        if reviewed:
            row.update(
                {
                    "review_decision_id": "decision",
                    "review_round": 2,
                    "review_text_decision": "accept_edited",
                    "review_text_final": "human text",
                    "review_emotion_primary": "开心_happy",
                    "review_emotion_secondary": "中立_neutral",
                    "review_intensity": "medium",
                    "review_label_source": "human_corrected",
                    "review_status": "approved",
                    "review_export_batch_id": "batch",
                }
            )
        return row

    def test_resolves_reviewed_and_unreviewed_provenance(self) -> None:
        config = load_dataset_freeze_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reviewed = self.row(root, reviewed=True)
            reviewed["asset_sha256"] = "b" * 64
            unreviewed = self.row(root)
            result = resolve_dataset_freeze_candidates(
                [reviewed, unreviewed], config=config, standardized_root=root
            )

        self.assertEqual(len(result["eligible"]), 2)
        by_asset = {item["asset_sha256"]: item for item in result["eligible"]}
        self.assertEqual(by_asset["b" * 64]["text_exact"], "human text")
        self.assertEqual(by_asset["b" * 64]["text_source"], "human_review")
        self.assertEqual(by_asset["b" * 64]["label_source"], "human_corrected")
        self.assertEqual(by_asset["a" * 64]["text_exact"], "filename text")
        self.assertEqual(
            by_asset["a" * 64]["text_source"], "filename_candidate_unreviewed"
        )
        self.assertEqual(by_asset["a" * 64]["label_source"], "weak_label_unreviewed")

    def test_policy_exclusions_are_explicit(self) -> None:
        config = load_dataset_freeze_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pending = self.row(root, reviewed=True)
            pending["review_status"] = "pending"
            result = resolve_dataset_freeze_candidates(
                [pending], config=config, standardized_root=root
            )
            self.assertEqual(result["excluded"][0]["reason"], "review_pending")

            unapproved_quality = self.row(root)
            unapproved_quality["quality_decision"] = "review"
            result = resolve_dataset_freeze_candidates(
                [unapproved_quality], config=config, standardized_root=root
            )
            self.assertEqual(
                result["excluded"][0]["reason"],
                "quality_review_without_latest_approval",
            )

    def test_missing_file_hash_drift_and_missing_source_fail_closed(self) -> None:
        config = load_dataset_freeze_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = self.row(root)
            (root / "aa" / "audio.wav").unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_dataset_freeze_candidates(
                    [missing], config=config, standardized_root=root
                )

            drift = self.row(root)
            drift["derived_output_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256 drift"):
                resolve_dataset_freeze_candidates(
                    [drift], config=config, standardized_root=root
                )

            missing_source = self.row(root)
            missing_source["source_path_key"] = None
            with self.assertRaisesRegex(RuntimeError, "source_path_key"):
                resolve_dataset_freeze_candidates(
                    [missing_source], config=config, standardized_root=root
                )


class DatasetCandidateSnapshotTests(unittest.TestCase):
    def candidate(self, asset: str, text: str, absolute_audio: str) -> dict:
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {
            "fid": asset,
            "asset_sha256": asset,
            "raw_relative_path": f"{asset[:2]}/raw.wav",
            "source_root_key": "root",
            "source_path_key": asset,
            "source_relative_path": f"speaker/{asset}.wav",
            "speaker_id": "speaker",
            "derived_id": "d" * 64,
            "audio_relative_path": f"{asset[:2]}/audio.wav",
            "audio_absolute_path": absolute_audio,
            "audio_sha256": "e" * 64,
            "duration_seconds": 1.25,
            "quality_run_id": "quality-run",
            "quality_decision": "pass",
            "quality_reasons": [],
            "text_exact": text,
            "text_sha256": text_sha256,
            "text_source": "filename_candidate_unreviewed",
            "emotion_weak_label": "中立_neutral",
            "emotion_primary": "中立_neutral",
            "emotion_secondary": None,
            "intensity": None,
            "label_source": "weak_label_unreviewed",
            "review_decision_id": None,
            "review_round": None,
            "review_batch_id": None,
            "lineage": {"config": "frozen"},
        }

    def test_snapshot_is_order_and_machine_path_independent(self) -> None:
        first = self.candidate("a" * 64, "甲", "C:/machine-one/audio.wav")
        second = self.candidate("b" * 64, "乙", "D:/machine-one/audio.wav")
        snapshot = build_candidate_snapshot([second, first])

        first["audio_absolute_path"] = "Z:/other-machine/audio.wav"
        second["audio_absolute_path"] = "/mnt/other/audio.wav"
        rebuilt = build_candidate_snapshot([first, second])

        self.assertEqual(snapshot["sha256"], rebuilt["sha256"])
        self.assertEqual(snapshot["canonical_json"], rebuilt["canonical_json"])
        self.assertEqual(snapshot["payload"]["item_count"], 2)
        self.assertNotIn("audio_absolute_path", snapshot["canonical_json"])

    def test_snapshot_changes_with_semantic_text_and_rejects_duplicates(self) -> None:
        candidate = self.candidate("a" * 64, "甲", "C:/audio.wav")
        original = build_candidate_snapshot([candidate])
        candidate["text_exact"] = "乙"
        candidate["text_sha256"] = hashlib.sha256("乙".encode("utf-8")).hexdigest()
        changed = build_candidate_snapshot([candidate])

        self.assertNotEqual(original["sha256"], changed["sha256"])
        with self.assertRaisesRegex(RuntimeError, "Duplicate asset"):
            build_candidate_snapshot([candidate, candidate])

    def test_catalog_rejects_same_version_config_and_snapshot_drift(self) -> None:
        config = load_dataset_freeze_config()
        identity = canonical_dataset_config(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            arguments = {
                "dataset_id": config.dataset_id,
                "dataset_version": config.dataset_version,
                "config_sha256": identity["sha256"],
                "config_json": identity["canonical_json"],
                "implementation_version": config.implementation_version,
                "source_path": str(DEFAULT_DATASET_FREEZE_CONFIG_PATH),
                "registered_at": "2026-09-08T00:00:00+00:00",
            }
            catalog.register_dataset_build_config(**arguments)
            catalog.register_dataset_build_config(**arguments)
            with self.assertRaisesRegex(RuntimeError, "config changed"):
                catalog.register_dataset_build_config(
                    **{**arguments, "config_sha256": "f" * 64}
                )

            run_arguments = {
                "dataset_id": config.dataset_id,
                "dataset_version": config.dataset_version,
                "candidate_snapshot_sha256": "a" * 64,
                "calibration_id": "thresholds",
                "calibration_version": 1,
                "calibration_config_sha256": "c" * 64,
                "calibration_report_sha256": "d" * 64,
                "started_at": "2026-09-08T00:00:00+00:00",
            }
            with self.assertRaisesRegex(RuntimeError, "is not registered"):
                catalog.begin_dataset_analysis_run(**run_arguments)
            with catalog.session() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM dataset_analysis_run"
                    ).fetchone()[0],
                    0,
                )

            calibration_arguments = {
                "dataset_id": config.dataset_id,
                "dataset_version": config.dataset_version,
                "calibration_id": "thresholds",
                "calibration_version": 1,
                "calibration_config_sha256": "c" * 64,
                "calibration_report_sha256": "d" * 64,
                "report_path": "calibration.json",
                "thresholds": {"text": 0.8},
                "registered_at": "2026-09-08T00:00:00+00:00",
            }
            catalog.register_dataset_threshold_calibration(**calibration_arguments)
            catalog.register_dataset_threshold_calibration(**calibration_arguments)
            with self.assertRaisesRegex(RuntimeError, "calibration changed"):
                catalog.register_dataset_threshold_calibration(
                    **{
                        **calibration_arguments,
                        "calibration_report_sha256": "e" * 64,
                    }
                )
            registered = catalog.load_dataset_threshold_calibration(
                dataset_id=config.dataset_id,
                dataset_version=config.dataset_version,
                calibration_id="thresholds",
                calibration_version=1,
            )
            self.assertEqual(registered["calibration_report_sha256"], "d" * 64)

            catalog.begin_dataset_analysis_run(
                **run_arguments,
            )
            catalog.begin_dataset_analysis_run(
                **{
                    **run_arguments,
                    "started_at": "2026-09-08T00:01:00+00:00",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "snapshot changed"):
                catalog.begin_dataset_analysis_run(
                    **{
                        **run_arguments,
                        "candidate_snapshot_sha256": "b" * 64,
                        "started_at": "2026-09-08T00:02:00+00:00",
                    },
                )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                catalog.begin_dataset_analysis_run(
                    **{
                        **run_arguments,
                        "calibration_report_sha256": "e" * 64,
                        "started_at": "2026-09-08T00:03:00+00:00",
                    },
                )
            with catalog.session() as connection:
                bindings = connection.execute(
                    """
                    SELECT COUNT(*) FROM dataset_analysis_calibration
                    WHERE calibration_report_sha256 = ?
                    """,
                    ("d" * 64,),
                ).fetchone()[0]
            self.assertEqual(bindings, 2)

    def test_snapshot_write_is_atomic_idempotent_and_refuses_overwrite(self) -> None:
        candidate = self.candidate("a" * 64, "甲", "C:/audio.wav")
        snapshot = build_candidate_snapshot([candidate])
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "candidate_snapshot.json"
            first = write_candidate_snapshot(snapshot, target)
            second = write_candidate_snapshot(snapshot, target)

            self.assertEqual(first["action"], "written")
            self.assertEqual(second["action"], "cached")
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), snapshot["sha256"])

            changed = build_candidate_snapshot(
                [self.candidate("a" * 64, "乙", "C:/audio.wav")]
            )
            with self.assertRaisesRegex(RuntimeError, "without a version bump"):
                write_candidate_snapshot(changed, target)


class DatasetAuditContractTests(unittest.TestCase):
    def test_cli_exposes_read_only_dataset_audit_inputs(self) -> None:
        args = build_parser().parse_args(
            [
                "dataset-audit",
                "--catalog",
                "catalog.sqlite",
                "--standardized-root",
                "standardized",
                "--report",
                "audit.json",
            ]
        )

        self.assertEqual(args.command, "dataset-audit")
        self.assertEqual(args.catalog, "catalog.sqlite")
        self.assertEqual(args.standardized_root, "standardized")
        self.assertEqual(args.report, "audit.json")

    def test_read_only_catalog_open_does_not_create_a_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing" / "catalog.sqlite"
            catalog = Catalog(path)

            with self.assertRaises(FileNotFoundError):
                catalog.load_dataset_build_config(dataset_id="fuxuan", dataset_version=1)

            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())

    def test_quality_report_identity_is_pinned_to_dataset_config(self) -> None:
        config = load_dataset_freeze_config()
        payload = {
            "schema_version": 1,
            "summary": {
                "status": "succeeded",
                "run_id": "quality-run",
                "analysis_id": config.inputs.quality_analysis_id,
                "analysis_version": config.inputs.quality_analysis_version,
                "policy_id": config.inputs.quality_policy_id,
                "policy_version": config.inputs.quality_policy_version,
            },
            "assets": [{"asset_sha256": "a" * 64, "decision": "pass", "reasons": []}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "quality.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = _load_quality_report(path, config=config)
            self.assertEqual(set(loaded["assets_by_sha256"]), {"a" * 64})

            payload["summary"]["policy_version"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "quality policy version mismatch"):
                _load_quality_report(path, config=config)


class ExactAudioAnalysisTests(unittest.TestCase):
    def candidate(self, root: Path, asset: str, name: str, content: bytes) -> dict:
        path = root / name
        path.write_bytes(content)
        return {
            "asset_sha256": asset,
            "audio_absolute_path": str(path),
            "audio_relative_path": name,
            "audio_sha256": hashlib.sha256(content).hexdigest(),
        }

    def test_rehashes_bytes_and_builds_stable_exact_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self.candidate(root, "a" * 64, "first.wav", b"same audio")
            second = self.candidate(root, "b" * 64, "second.wav", b"same audio")
            unique = self.candidate(root, "c" * 64, "unique.wav", b"unique audio")

            report = analyze_exact_audio([unique, second, first])
            reordered = analyze_exact_audio([first, unique, second])

            self.assertEqual(report, reordered)
            self.assertEqual(report["checked_asset_count"], 3)
            self.assertEqual(report["unique_audio_sha256_count"], 2)
            self.assertEqual(report["duplicate_group_count"], 1)
            self.assertEqual(report["duplicate_item_count"], 2)
            self.assertEqual(
                report["groups"][0]["group_id"],
                "exact-audio:" + hashlib.sha256(b"same audio").hexdigest(),
            )
            self.assertEqual(
                [item["asset_sha256"] for item in report["groups"][0]["members"]],
                ["a" * 64, "b" * 64],
            )

    def test_rejects_hash_drift_missing_files_and_duplicate_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = self.candidate(root, "a" * 64, "audio.wav", b"original")
            Path(candidate["audio_absolute_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 drift"):
                analyze_exact_audio([candidate])

            missing = self.candidate(root, "b" * 64, "missing.wav", b"audio")
            Path(missing["audio_absolute_path"]).unlink()
            with self.assertRaises(FileNotFoundError):
                analyze_exact_audio([missing])

            duplicate = self.candidate(root, "c" * 64, "duplicate.wav", b"audio")
            with self.assertRaisesRegex(RuntimeError, "Duplicate exact-audio"):
                analyze_exact_audio([duplicate, duplicate])

    def test_report_write_is_atomic_and_deterministic(self) -> None:
        report = {
            "schema_version": 1,
            "status": "succeeded",
            "checked_asset_count": 0,
            "unique_audio_sha256_count": 0,
            "duplicate_group_count": 0,
            "duplicate_item_count": 0,
            "groups": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "exact_audio.json"
            first = write_exact_audio_report(report, target)
            first_content = target.read_bytes()
            second = write_exact_audio_report(report, target)

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first_content, target.read_bytes())
            self.assertFalse(list(target.parent.glob("*.partial")))


class ExactTextAnalysisTests(unittest.TestCase):
    def test_normalization_is_configured_and_explains_changes(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        result = normalize_text_for_similarity("Ａ， B\tß！", config=config)

        self.assertEqual(result["text_normalized"], "abss")
        self.assertTrue(result["unicode_changed"])
        self.assertTrue(result["casefold_changed"])
        self.assertEqual(
            {item["category"][:1] for item in result["removed_characters"]},
            {"P", "Z", "C"},
        )
        self.assertTrue(result["diff"])

    def test_groups_exact_normalized_text_stably_and_skips_empty(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        candidates = [
            {
                "asset_sha256": "b" * 64,
                "text_exact": "Hello， world!",
                "text_source": "human_review",
            },
            {
                "asset_sha256": "a" * 64,
                "text_exact": "hello world",
                "text_source": "filename_candidate_unreviewed",
            },
            {
                "asset_sha256": "c" * 64,
                "text_exact": "！？ \t",
                "text_source": "human_review",
            },
        ]

        report = analyze_exact_text(candidates, config=config)
        reordered = analyze_exact_text(reversed(candidates), config=config)

        self.assertEqual(report, reordered)
        self.assertEqual(report["duplicate_group_count"], 1)
        self.assertEqual(report["duplicate_item_count"], 2)
        self.assertEqual(report["empty_after_normalization_count"], 1)
        self.assertEqual(
            report["groups"][0]["member_asset_sha256s"],
            ["a" * 64, "b" * 64],
        )
        self.assertEqual(
            report["empty_items"][0]["reason"], "empty_after_normalization"
        )

    def test_rejects_missing_text_and_duplicate_assets(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        missing = {"asset_sha256": "a" * 64}
        with self.assertRaisesRegex(RuntimeError, "Missing exact text"):
            analyze_exact_text([missing], config=config)

        candidate = {"asset_sha256": "b" * 64, "text_exact": "text"}
        with self.assertRaisesRegex(RuntimeError, "Duplicate exact-text"):
            analyze_exact_text([candidate, candidate], config=config)

    def test_exact_text_report_write_is_deterministic(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        report = analyze_exact_text([], config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "exact_text.json"
            first = write_exact_text_report(report, target)
            first_content = target.read_bytes()
            second = write_exact_text_report(report, target)

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first_content, target.read_bytes())
            self.assertFalse(list(target.parent.glob("*.partial")))


class NearTextAnalysisTests(unittest.TestCase):
    def test_levenshtein_score_is_symmetric(self) -> None:
        forward = symmetric_levenshtein_similarity("abc", "ax")
        reverse = symmetric_levenshtein_similarity("ax", "abc")

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["edit_distance"], 2)
        self.assertAlmostEqual(forward["similarity"], 1 / 3)
        self.assertEqual(
            symmetric_levenshtein_similarity("", "")["similarity"], 1.0
        )

    def test_recall_is_stable_and_flags_false_merge_risks(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        candidates = [
            {
                "asset_sha256": "a" * 64,
                "text_exact": "Hello，Bob!",
            },
            {
                "asset_sha256": "b" * 64,
                "text_exact": "hello Alice",
            },
            {
                "asset_sha256": "c" * 64,
                "text_exact": "hello bob",
            },
            {
                "asset_sha256": "d" * 64,
                "text_exact": "！？ \t",
            },
        ]
        report = analyze_near_text(
            candidates,
            config=config,
            named_entities=["Bob", "Alice"],
            short_text_max_chars=8,
        )
        reordered = analyze_near_text(
            reversed(candidates),
            config=config,
            named_entities=["Alice", "Bob"],
            short_text_max_chars=8,
        )

        self.assertEqual(report, reordered)
        self.assertEqual(report["skipped_empty_asset_count"], 1)
        self.assertEqual(report["exact_pair_count"], 1)
        pair = report["pairs"][0]
        self.assertIn("short_text", pair["risk_flags"])
        self.assertIn("punctuation_removed", pair["risk_flags"])
        self.assertIn("named_entity_difference", pair["risk_flags"])
        self.assertIsNone(report["threshold"])
        self.assertTrue(report["calibration_required"])

    def test_length_recall_and_bad_inputs_fail_closed(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        report = analyze_near_text(
            [
                {"asset_sha256": "a" * 64, "text_exact": "ab"},
                {"asset_sha256": "b" * 64, "text_exact": "abcdefghij"},
            ],
            config=config,
        )
        self.assertEqual(report["near_candidate_pair_count"], 0)
        self.assertEqual(report["length_filtered_pair_count"], 1)

        with self.assertRaisesRegex(RuntimeError, "Missing near-text input"):
            analyze_near_text([{"asset_sha256": "c" * 64}], config=config)
        duplicate = {"asset_sha256": "d" * 64, "text_exact": "text"}
        with self.assertRaisesRegex(RuntimeError, "Duplicate near-text"):
            analyze_near_text([duplicate, duplicate], config=config)

    def test_near_text_report_write_is_deterministic(self) -> None:
        config = load_dataset_freeze_config().text_similarity
        report = analyze_near_text([], config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "near_text.json"
            first = write_near_text_report(report, target)
            content = target.read_bytes()
            second = write_near_text_report(report, target)

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(content, target.read_bytes())
            self.assertFalse(list(target.parent.glob("*.partial")))


class DuplicateGraphTests(unittest.TestCase):
    def test_connected_components_and_group_ids_ignore_edge_order(self) -> None:
        assets = ["d" * 64, "b" * 64, "a" * 64, "c" * 64]
        pairs = [("b" * 64, "c" * 64), ("a" * 64, "b" * 64)]

        groups = stable_similarity_groups(assets, pairs)
        reordered = stable_similarity_groups(reversed(assets), reversed(pairs))

        self.assertEqual(groups, reordered)
        self.assertEqual(groups[0]["member_asset_sha256s"], ["a" * 64, "b" * 64, "c" * 64])
        self.assertEqual(
            groups[0]["group_id"],
            canonical_sha256(groups[0]["member_asset_sha256s"]),
        )
        self.assertEqual(groups[1]["member_asset_sha256s"], ["d" * 64])

    def test_builds_typed_scored_edges_from_verified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = ["a" * 64, "b" * 64, "c" * 64]
            features = []
            for index, asset in enumerate(assets):
                vector = np.zeros(FINGERPRINT_DIMENSION, dtype="<f4")
                vector[index] = 1.0
                blob = vector.tobytes()
                path = root / f"{asset}.f32"
                path.write_bytes(blob)
                features.append(
                    {
                        "asset_sha256": asset,
                        "cache_path": str(path),
                        "fingerprint_sha256": hashlib.sha256(blob).hexdigest(),
                    }
                )
            snapshot = {
                "item_count": 3,
                "items": [
                    {"asset_sha256": asset, "duration_seconds": 1.0}
                    for asset in reversed(assets)
                ],
            }
            exact_audio = {
                "status": "succeeded",
                "checked_asset_count": 3,
                "groups": [
                    {
                        "group_id": "audio-group",
                        "members": [
                            {"asset_sha256": "b" * 64},
                            {"asset_sha256": "a" * 64},
                        ],
                    }
                ],
            }
            exact_text = {
                "status": "succeeded",
                "checked_asset_count": 3,
                "groups": [
                    {
                        "group_id": "text-group",
                        "member_asset_sha256s": ["c" * 64, "b" * 64],
                    }
                ],
            }
            near_text = {
                "status": "succeeded",
                "input_asset_count": 3,
                "metric": "metric",
                "pairs": [
                    {
                        "left_asset_sha256": "c" * 64,
                        "right_asset_sha256": "a" * 64,
                        "similarity": 0.9,
                    },
                    {
                        "left_asset_sha256": "a" * 64,
                        "right_asset_sha256": "b" * 64,
                        "similarity": 0.6,
                    },
                ],
            }
            fingerprints = {
                "status": "succeeded",
                "feature_count": 3,
                "dimension": FINGERPRINT_DIMENSION,
                "dtype": "float32_le",
                "fingerprint_config_sha256": "f" * 64,
                "features": list(reversed(features)),
            }
            graph = build_duplicate_edges(
                candidate_snapshot=snapshot,
                exact_audio_report=exact_audio,
                exact_text_report=exact_text,
                near_text_report=near_text,
                acoustic_fingerprint_report=fingerprints,
                text_threshold=0.8,
                acoustic_threshold=0.95,
                acoustic_duration_ratio_min=0.9,
                source_sha256s={
                    "candidate_snapshot": "1" * 64,
                    "exact_audio": "2" * 64,
                    "exact_text": "3" * 64,
                    "near_text": "4" * 64,
                    "acoustic_fingerprint": "5" * 64,
                },
            )

            self.assertEqual(graph["edge_count"], 3)
            self.assertEqual(
                graph["edge_counts_by_evidence_type"],
                {"exact_audio": 1, "exact_text": 1, "near_audio": 0, "near_text": 1},
            )
            near_edge = next(
                edge for edge in graph["edges"] if edge["evidence_type"] == "near_text"
            )
            self.assertEqual(near_edge["analysis_status"], "pending_review")
            self.assertEqual(near_edge["threshold"], 0.8)
            self.assertEqual(near_edge["left_asset_sha256"], "a" * 64)

    def test_catalog_persists_edges_and_rematerializes_latest_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            assets = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
            run_id = "duplicate-run"
            with catalog.session() as connection:
                connection.executemany(
                    """
                    INSERT INTO asset (
                        sha256, size_bytes, probe_status, first_seen_at, last_seen_at
                    ) VALUES (?, 1, 'ok', 't0', 't0')
                    """,
                    [(asset,) for asset in assets],
                )
                connection.execute(
                    """
                    INSERT INTO dataset_build_config (
                        dataset_id, dataset_version, config_sha256, config_json,
                        implementation_version, source_path, registered_at
                    ) VALUES ('dataset', 1, ?, '{}', 1, 'config.yaml', 't0')
                    """,
                    ("1" * 64,),
                )
                connection.execute(
                    """
                    INSERT INTO dataset_analysis_run (
                        run_id, dataset_id, dataset_version,
                        candidate_snapshot_sha256, started_at, status
                    ) VALUES (?, 'dataset', 1, ?, 't0', 'running')
                    """,
                    (run_id, "2" * 64),
                )
            edges = [
                {
                    "left_asset_sha256": "a" * 64,
                    "right_asset_sha256": "b" * 64,
                    "evidence_type": "exact_audio",
                    "score": 1.0,
                    "threshold": None,
                    "analysis_status": "accepted_exact",
                    "evidence_json": canonical_json({"source": "exact"}),
                },
                {
                    "left_asset_sha256": "b" * 64,
                    "right_asset_sha256": "c" * 64,
                    "evidence_type": "near_text",
                    "score": 0.9,
                    "threshold": 0.8,
                    "analysis_status": "pending_review",
                    "evidence_json": canonical_json({"source": "near"}),
                },
            ]
            first = catalog.store_dataset_similarity_graph(
                run_id=run_id,
                candidate_asset_sha256s=reversed(assets),
                edges=reversed(edges),
                created_at="t1",
            )
            cached = catalog.store_dataset_similarity_graph(
                run_id=run_id,
                candidate_asset_sha256s=assets,
                edges=edges,
                created_at="t2",
            )
            self.assertEqual(first["group_count"], 3)
            self.assertEqual(first["singleton_group_count"], 2)
            self.assertEqual(cached["action"], "cached")

            review_rows = [
                {
                    "review_id": "review-1",
                    "run_id": run_id,
                    "left_asset_sha256": "b" * 64,
                    "right_asset_sha256": "c" * 64,
                    "evidence_type": "near_text",
                    "review_status": "accepted",
                    "review_note": "same utterance",
                    "created_at": "t3",
                    "review_batch_id": "batch-1",
                    "source_row_index": 0,
                }
            ]
            self.assertEqual(
                catalog.record_dataset_similarity_edge_reviews(review_rows),
                (1, 0),
            )
            self.assertEqual(
                catalog.record_dataset_similarity_edge_reviews(review_rows),
                (0, 1),
            )
            changed_review = [dict(review_rows[0], review_status="rejected")]
            with self.assertRaisesRegex(RuntimeError, "batch replay changed"):
                catalog.record_dataset_similarity_edge_reviews(changed_review)
            reviewed = catalog.store_dataset_similarity_graph(
                run_id=run_id,
                candidate_asset_sha256s=assets,
                edges=edges,
                created_at="t4",
            )
            self.assertEqual(reviewed["action"], "rematerialized")
            self.assertEqual(reviewed["group_count"], 2)
            with catalog.session() as connection:
                members = connection.execute(
                    """
                    SELECT group_id, asset_sha256
                    FROM dataset_similarity_group_member
                    WHERE run_id = ?
                    ORDER BY asset_sha256
                    """,
                    (run_id,),
                ).fetchall()
            self.assertEqual(
                [row["group_id"] for row in members[:3]],
                [canonical_sha256(assets[:3])] * 3,
            )

    def test_catalog_rejects_edge_drift_without_partial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            assets = ["a" * 64, "b" * 64]
            with catalog.session() as connection:
                connection.executemany(
                    """
                    INSERT INTO asset (
                        sha256, size_bytes, probe_status, first_seen_at, last_seen_at
                    ) VALUES (?, 1, 'ok', 't0', 't0')
                    """,
                    [(asset,) for asset in assets],
                )
                connection.execute(
                    """
                    INSERT INTO dataset_build_config (
                        dataset_id, dataset_version, config_sha256, config_json,
                        implementation_version, source_path, registered_at
                    ) VALUES ('dataset', 1, ?, '{}', 1, 'config.yaml', 't0')
                    """,
                    ("1" * 64,),
                )
                connection.execute(
                    """
                    INSERT INTO dataset_analysis_run (
                        run_id, dataset_id, dataset_version,
                        candidate_snapshot_sha256, started_at, status
                    ) VALUES ('run', 'dataset', 1, ?, 't0', 'running')
                    """,
                    ("2" * 64,),
                )
            edge = {
                "left_asset_sha256": assets[0],
                "right_asset_sha256": assets[1],
                "evidence_type": "near_audio",
                "score": 0.95,
                "threshold": 0.9,
                "analysis_status": "pending_review",
                "evidence_json": canonical_json({"metric": "cosine"}),
            }
            catalog.store_dataset_similarity_graph(
                run_id="run",
                candidate_asset_sha256s=assets,
                edges=[edge],
                created_at="t1",
            )
            changed = {**edge, "score": 0.96}
            with self.assertRaisesRegex(RuntimeError, "edge set changed"):
                catalog.store_dataset_similarity_graph(
                    run_id="run",
                    candidate_asset_sha256s=assets,
                    edges=[changed],
                    created_at="t2",
                )
            with catalog.session() as connection:
                score = connection.execute(
                    "SELECT score FROM dataset_similarity_edge WHERE run_id = 'run'"
                ).fetchone()[0]
                group_count = connection.execute(
                    "SELECT count(*) FROM dataset_similarity_group WHERE run_id = 'run'"
                ).fetchone()[0]
            self.assertEqual(score, 0.95)
            self.assertEqual(group_count, 2)

    def test_catalog_persists_features_and_speaker_reviews_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Catalog(Path(temp_dir) / "catalog.sqlite")
            catalog.initialize()
            DatasetCandidateQueryTests().seed_candidate(catalog)
            asset = "a" * 64
            run_id = "speaker-run"
            with catalog.session() as connection:
                connection.execute(
                    """
                    INSERT INTO dataset_build_config (
                        dataset_id, dataset_version, config_sha256, config_json,
                        implementation_version, source_path, registered_at
                    ) VALUES ('dataset', 1, ?, '{}', 1, 'config.yaml', 't0')
                    """,
                    ("6" * 64,),
                )
                connection.execute(
                    """
                    INSERT INTO dataset_analysis_run (
                        run_id, dataset_id, dataset_version,
                        candidate_snapshot_sha256, started_at, status
                    ) VALUES (?, 'dataset', 1, ?, 't0', 'running')
                    """,
                    (run_id, "7" * 64),
                )
            catalog.store_dataset_similarity_graph(
                run_id=run_id,
                candidate_asset_sha256s=[asset],
                edges=[],
                created_at="t1",
            )
            fingerprint_blob = bytes(FINGERPRINT_DIMENSION * 4)
            embedding_blob = bytes(SPEAKER_EMBEDDING_DIMENSION * 4)
            features = [
                {
                    "asset_sha256": asset,
                    "derived_id": "d" * 64,
                    "standardized_sha256": "e" * 64,
                    "duration_seconds": 1.0,
                    "fingerprint_blob": fingerprint_blob,
                    "fingerprint_dimension": FINGERPRINT_DIMENSION,
                    "fingerprint_sha256": hashlib.sha256(
                        fingerprint_blob
                    ).hexdigest(),
                    "speaker_embedding_blob": embedding_blob,
                    "speaker_embedding_dimension": SPEAKER_EMBEDDING_DIMENSION,
                    "speaker_embedding_sha256": hashlib.sha256(
                        embedding_blob
                    ).hexdigest(),
                }
            ]
            assessments = [
                {
                    "asset_sha256": asset,
                    "center_cosine": 0.5,
                    "knn_cosine": 0.6,
                    "knn_k": 1,
                    "center_robust_z": -2.0,
                    "knn_robust_z": -1.0,
                    "candidate_outlier": 1,
                    "evidence_json": canonical_json({"rank": 1}),
                }
            ]
            stored = catalog.store_dataset_features_and_speaker_assessments(
                run_id=run_id,
                features=features,
                assessments=assessments,
            )
            cached = catalog.store_dataset_features_and_speaker_assessments(
                run_id=run_id,
                features=features,
                assessments=assessments,
            )
            self.assertEqual(stored["action"], "stored")
            self.assertEqual(cached["action"], "cached")

            reviews = [
                {
                    "review_id": "speaker-review-1",
                    "run_id": run_id,
                    "asset_sha256": asset,
                    "review_status": "exclude_uncertain",
                    "review_note": "poor audio quality",
                    "created_at": "t2",
                    "review_batch_id": "speaker-batch-1",
                    "source_row_index": 0,
                }
            ]
            self.assertEqual(catalog.record_dataset_speaker_reviews(reviews), (1, 0))
            self.assertEqual(catalog.record_dataset_speaker_reviews(reviews), (0, 1))
            changed = [dict(reviews[0], review_status="confirmed_same_speaker")]
            with self.assertRaisesRegex(RuntimeError, "batch replay changed"):
                catalog.record_dataset_speaker_reviews(changed)


class GroupAwareSplitTests(unittest.TestCase):
    def test_largest_remainder_matches_frozen_277_targets(self) -> None:
        self.assertEqual(
            largest_remainder_targets(
                277,
                {"train": 0.8, "validation": 0.1, "test": 0.1},
            ),
            {"train": 221, "validation": 28, "test": 28},
        )

    def test_split_is_order_independent_and_keeps_groups_atomic(self) -> None:
        assets = [f"{index:064x}" for index in range(1, 11)]
        items = [
            {"asset_sha256": asset, "duration_seconds": float(index + 1)}
            for index, asset in enumerate(assets)
        ]
        member_sets = [assets[:2], *[[asset] for asset in assets[2:]]]
        groups = [
            {
                "group_id": canonical_sha256(members),
                "member_asset_sha256s": members,
            }
            for members in member_sets
        ]
        split_config = load_dataset_freeze_config().split
        first = plan_group_aware_split(
            items=items,
            groups=groups,
            split_config=split_config,
            dataset_config_sha256="1" * 64,
        )
        reordered = plan_group_aware_split(
            items=reversed(items),
            groups=(
                {
                    **group,
                    "member_asset_sha256s": list(
                        reversed(group["member_asset_sha256s"])
                    ),
                }
                for group in reversed(groups)
            ),
            split_config=split_config,
            dataset_config_sha256="1" * 64,
        )

        self.assertEqual(first, reordered)
        self.assertEqual(first["summary"]["train"]["actual_item_count"], 8)
        self.assertEqual(first["summary"]["validation"]["actual_item_count"], 1)
        self.assertEqual(first["summary"]["test"]["actual_item_count"], 1)
        self.assertEqual(
            first["asset_assignments"][assets[0]],
            first["asset_assignments"][assets[1]],
        )
        assert_no_group_crosses_splits(first)

    def test_split_rejects_incomplete_groups_and_detects_cross_split_drift(self) -> None:
        assets = ["a" * 64, "b" * 64]
        items = [
            {"asset_sha256": asset, "duration_seconds": 1.0} for asset in assets
        ]
        split_config = load_dataset_freeze_config().split
        with self.assertRaisesRegex(RuntimeError, "do not cover"):
            plan_group_aware_split(
                items=items,
                groups=[
                    {
                        "group_id": canonical_sha256([assets[0]]),
                        "member_asset_sha256s": [assets[0]],
                    }
                ],
                split_config=split_config,
                dataset_config_sha256="1" * 64,
            )

        group_id = canonical_sha256(assets)
        plan = plan_group_aware_split(
            items=items,
            groups=[
                {
                    "group_id": group_id,
                    "member_asset_sha256s": assets,
                }
            ],
            split_config=split_config,
            dataset_config_sha256="1" * 64,
        )
        plan["asset_assignments"][assets[1]] = "test"
        with self.assertRaisesRegex(RuntimeError, "crosses splits"):
            assert_no_group_crosses_splits(plan)

    def test_balanced_strata_targets_preserve_row_and_column_totals(self) -> None:
        totals = {
            "中立_neutral": 86,
            "开心_happy": 64,
            "生气_angry": 120,
            "难过_sad": 7,
        }
        split_targets = {"train": 221, "validation": 28, "test": 28}
        targets = balanced_strata_targets(
            totals,
            split_targets,
            {"train": 0.8, "validation": 0.1, "test": 0.1},
        )
        for label, total in totals.items():
            self.assertEqual(sum(targets[label].values()), total)
        for split, target in split_targets.items():
            self.assertEqual(
                sum(targets[label][split] for label in totals),
                target,
            )
        self.assertEqual(targets["难过_sad"], {"train": 5, "validation": 1, "test": 1})

        reduced_totals = {
            "中立_neutral": 84,
            "开心_happy": 63,
            "生气_angry": 119,
            "难过_sad": 7,
        }
        reduced_split_targets = {"train": 219, "validation": 27, "test": 27}
        reduced = balanced_strata_targets(
            reduced_totals,
            reduced_split_targets,
            {"train": 0.8, "validation": 0.1, "test": 0.1},
            minimums={"难过_sad": {"validation": 1, "test": 1}},
        )
        self.assertEqual(
            reduced["难过_sad"],
            {"train": 5, "validation": 1, "test": 1},
        )
        for split, target in reduced_split_targets.items():
            self.assertEqual(
                sum(reduced[label][split] for label in reduced_totals),
                target,
            )

    def test_emotion_strata_is_deterministic_and_covers_low_resource_eval(self) -> None:
        assets = [f"{index:064x}" for index in range(1, 11)]
        items = [
            {
                "asset_sha256": asset,
                "duration_seconds": float(index + 1),
                "emotion_primary": "难过_sad" if index < 3 else "生气_angry",
                "emotion_weak_label": "难过_sad" if index < 2 else "生气_angry",
            }
            for index, asset in enumerate(assets)
        ]
        groups = [
            {
                "group_id": canonical_sha256([asset]),
                "member_asset_sha256s": [asset],
            }
            for asset in assets
        ]
        split_config = load_dataset_freeze_config().split
        first = plan_emotion_stratified_split(
            items=items,
            groups=groups,
            split_config=split_config,
            dataset_config_sha256="1" * 64,
        )
        repeated = plan_emotion_stratified_split(
            items=reversed(items),
            groups=reversed(groups),
            split_config=split_config,
            dataset_config_sha256="1" * 64,
        )

        self.assertEqual(first, repeated)
        low = first["summary"]["low_resource"]
        self.assertGreaterEqual(
            low["final_primary_by_split"]["validation"]["group_count"],
            1,
        )
        self.assertGreaterEqual(
            low["final_primary_by_split"]["test"]["group_count"],
            1,
        )
        self.assertEqual(low["final_primary_total"], 3)
        self.assertEqual(low["weak_label_total"], 2)
        self.assertEqual(first["summary"]["train"]["actual_item_count"], 8)
        self.assertEqual(first["summary"]["validation"]["actual_item_count"], 1)
        self.assertEqual(first["summary"]["test"]["actual_item_count"], 1)


class ReviewedSelectionTests(unittest.TestCase):
    def fixture(self) -> tuple[list[dict], list[dict], dict]:
        assets = [f"{index:064x}" for index in range(1, 6)]
        items = [
            {
                "asset_sha256": asset,
                "fid": asset,
                "duration_seconds": 1.0,
                "emotion_primary": "难过_sad" if index == 4 else "生气_angry",
                "emotion_weak_label": "难过_sad" if index == 4 else "生气_angry",
            }
            for index, asset in enumerate(assets)
        ]
        groups = [
            {
                "group_id": canonical_sha256(assets[:2]),
                "member_asset_sha256s": assets[:2],
            },
            *[
                {
                    "group_id": canonical_sha256([asset]),
                    "member_asset_sha256s": [asset],
                }
                for asset in assets[2:]
            ],
        ]
        review = {
            "edge_decisions": [
                {
                    "left_asset_sha256": assets[0],
                    "right_asset_sha256": assets[1],
                    "review_status": "accepted",
                    "representative_asset_sha256": assets[1],
                    "exclude_asset_sha256s": [assets[0]],
                    "review_note": "left is cropped",
                }
            ],
            "speaker_decisions": [
                {
                    "asset_sha256": assets[4],
                    "review_status": "exclude_uncertain",
                    "review_note": "poor quality",
                }
            ],
        }
        return items, groups, review

    def test_applies_one_representative_and_speaker_exclusion_without_sampling(self) -> None:
        items, groups, review = self.fixture()
        selection = apply_reviewed_exclusions(
            items=reversed(items),
            groups=reversed(groups),
            review=review,
        )
        selected_assets = {
            item["asset_sha256"] for item in selection["selected_items"]
        }
        self.assertEqual(selection["candidate_count"], 5)
        self.assertEqual(selection["selected_count"], 3)
        self.assertEqual(selection["excluded_count"], 2)
        self.assertEqual(selection["oversampled_count"], 0)
        self.assertNotIn(items[0]["asset_sha256"], selected_assets)
        self.assertIn(items[1]["asset_sha256"], selected_assets)
        self.assertNotIn(items[4]["asset_sha256"], selected_assets)
        duplicate_group = next(
            group
            for group in selection["selected_groups"]
            if "canonical_member_asset_sha256s" in group
        )
        self.assertEqual(
            duplicate_group["member_asset_sha256s"],
            [items[1]["asset_sha256"]],
        )
        split = plan_group_aware_split(
            items=selection["selected_items"],
            groups=selection["selected_groups"],
            split_config=load_dataset_freeze_config().split,
            dataset_config_sha256="1" * 64,
        )
        self.assertEqual(split["candidate_count"], 3)

    def test_rejects_multi_asset_group_without_explicit_representative(self) -> None:
        items, groups, review = self.fixture()
        review["edge_decisions"] = []
        with self.assertRaisesRegex(RuntimeError, "explicit representative"):
            apply_reviewed_exclusions(items=items, groups=groups, review=review)


class SplitAuditTests(unittest.TestCase):
    def item(
        self,
        asset: str,
        *,
        audio_sha256: str,
        text: str,
        source_path: str,
    ) -> dict:
        return {
            "asset_sha256": asset,
            "audio_sha256": audio_sha256,
            "text_exact": text,
            "source_root_key": "root",
            "source_path_key": source_path,
            "raw_relative_path": source_path,
        }

    def plan(self, assets: list[str], assignments: dict[str, str]) -> dict:
        return {
            "asset_assignments": assignments,
            "groups": [
                {
                    "group_id": canonical_sha256([asset]),
                    "split": assignments[asset],
                    "member_asset_sha256s": [asset],
                }
                for asset in assets
            ],
        }

    def test_passes_unique_assets_and_same_split_similarity_edge(self) -> None:
        assets = ["a" * 64, "b" * 64, "c" * 64]
        assignments = {assets[0]: "train", assets[1]: "train", assets[2]: "test"}
        items = [
            self.item(
                asset,
                audio_sha256=str(index + 1) * 64,
                text=f"text {index}",
                source_path=f"source-{index}.wav",
            )
            for index, asset in enumerate(assets)
        ]
        report = audit_split_leakage(
            items=items,
            split_plan=self.plan(assets, assignments),
            similarity_edges=[
                {
                    "left_asset_sha256": assets[0],
                    "right_asset_sha256": assets[1],
                    "evidence_type": "near_text",
                }
            ],
            text_config=load_dataset_freeze_config().text_similarity,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["violation_count"], 0)
        assert_split_audit_passes(report)

    def test_reports_audio_text_edge_and_source_leakage(self) -> None:
        assets = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        assignments = {
            assets[0]: "train",
            assets[1]: "test",
            assets[2]: "train",
            assets[3]: "validation",
        }
        items = [
            self.item(
                assets[0],
                audio_sha256="1" * 64,
                text="unique a",
                source_path="shared.wav",
            ),
            self.item(
                assets[1],
                audio_sha256="1" * 64,
                text="unique b",
                source_path="b.wav",
            ),
            self.item(
                assets[2],
                audio_sha256="2" * 64,
                text="相同，文本。",
                source_path="c.wav",
            ),
            self.item(
                assets[3],
                audio_sha256="3" * 64,
                text="相同文本",
                source_path="shared.wav",
            ),
        ]
        report = audit_split_leakage(
            items=items,
            split_plan=self.plan(assets, assignments),
            similarity_edges=[
                {
                    "left_asset_sha256": assets[1],
                    "right_asset_sha256": assets[3],
                    "evidence_type": "near_audio",
                }
            ],
            text_config=load_dataset_freeze_config().text_similarity,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            {violation["check"] for violation in report["violations"]},
            {
                "exact_audio_sha256",
                "exact_normalized_text",
                "similarity_edge",
                "source_provenance_identity",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "4 violations"):
            assert_split_audit_passes(report)


class AtomicDatasetPublishTests(unittest.TestCase):
    def test_publish_is_atomic_and_identical_rebuild_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "datasets" / "fuxuan" / "v1"

            def build(staging: Path) -> None:
                (staging / "manifest.json").write_text(
                    '{"version":1}\n', encoding="utf-8", newline=""
                )

            def validate(staging: Path) -> None:
                self.assertEqual(
                    (staging / "manifest.json").read_text(encoding="utf-8"),
                    '{"version":1}\n',
                )

            published = atomic_publish_dataset(
                target,
                build=build,
                validate=validate,
            )
            cached = atomic_publish_dataset(target, build=build, validate=validate)
            self.assertEqual(published["action"], "published")
            self.assertEqual(cached["action"], "cached")
            self.assertEqual(published["tree_sha256"], cached["tree_sha256"])
            self.assertEqual(dataset_tree_inventory(target)["files"], [
                {
                    "path": "manifest.json",
                    "size_bytes": 14,
                    "sha256": hashlib.sha256(b'{"version":1}\n').hexdigest(),
                }
            ])

    def test_conflict_and_validation_failure_leave_published_data_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "fuxuan" / "v1"

            def good(staging: Path) -> None:
                (staging / "manifest.json").write_text(
                    "stable\n", encoding="utf-8", newline=""
                )

            atomic_publish_dataset(target, build=good, validate=lambda _: None)
            before = dataset_tree_inventory(target)

            def changed(staging: Path) -> None:
                (staging / "manifest.json").write_text(
                    "changed\n", encoding="utf-8", newline=""
                )

            with self.assertRaisesRegex(DatasetVersionConflict, "different content"):
                atomic_publish_dataset(target, build=changed, validate=lambda _: None)
            self.assertEqual(dataset_tree_inventory(target), before)

            missing_target = root / "fuxuan" / "v2"
            with self.assertRaisesRegex(RuntimeError, "invalid staged data"):
                atomic_publish_dataset(
                    missing_target,
                    build=good,
                    validate=lambda _: (_ for _ in ()).throw(
                        RuntimeError("invalid staged data")
                    ),
                )
            self.assertFalse(missing_target.exists())
            self.assertEqual(list((root / "fuxuan").glob(".*.staging")), [])


class DatasetArtifactTests(unittest.TestCase):
    def fixture(self, audio_root: Path) -> tuple[dict, dict, dict]:
        asset = "a" * 64
        audio_path = audio_root / "sample.wav"
        audio_path.parent.mkdir(parents=True)
        audio_path.write_bytes(b"canonical-audio")
        item = {
            "fid": asset,
            "asset_sha256": asset,
            "raw_relative_path": "raw/sample.wav",
            "source_root_key": "source",
            "source_path_key": "sample.wav",
            "source_relative_path": "sample.wav",
            "speaker_id": "fuxuan",
            "derived_id": "derived-1",
            "audio_relative_path": "sample.wav",
            "audio_sha256": hashlib.sha256(b"canonical-audio").hexdigest(),
            "duration_seconds": 1.25,
            "quality_run_id": "quality-1",
            "quality_decision": "pass",
            "quality_reasons": [],
            "text_exact": "冻结样本",
            "text_sha256": hashlib.sha256("冻结样本".encode()).hexdigest(),
            "text_source": "filename_candidate_unreviewed",
            "emotion_weak_label": "中立_neutral",
            "emotion_primary": "中立_neutral",
            "emotion_secondary": None,
            "intensity": None,
            "label_source": "weak_label_unreviewed",
            "review_decision_id": None,
            "review_round": None,
            "review_batch_id": None,
            "lineage": {"source": "synthetic"},
        }
        selection = {
            "analysis_run_id": "analysis-1",
            "candidate_snapshot_sha256": "1" * 64,
            "edge_review_snapshot_sha256": "2" * 64,
            "selection_sha256": "3" * 64,
            "candidate_count": 1,
            "selected_count": 1,
            "excluded_count": 0,
            "oversampled_count": 0,
            "selected_items": [item],
            "selected_groups": [
                {"group_id": "4" * 64, "member_asset_sha256s": [asset]}
            ],
        }
        split = {
            "algorithm": "deterministic_group_emotion_swap_v1",
            "selection_sha256": "3" * 64,
            "assignment_sha256": "5" * 64,
            "asset_assignments": {asset: "train"},
        }
        audit = {
            "status": "passed",
            "violation_count": 0,
            "selection_sha256": "3" * 64,
            "assignment_sha256": "5" * 64,
            "audit_sha256": "6" * 64,
        }
        return selection, split, audit

    def test_canonical_artifact_set_is_complete_and_stable(self) -> None:
        config = load_dataset_freeze_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_root = root / "audio"
            selection, split, audit = self.fixture(audio_root)
            identities = {"selection.json": "7" * 64}
            inventories = []
            for name in ("first", "second"):
                output = root / name
                build_dataset_artifact_set(
                    output,
                    config=config,
                    selection=selection,
                    split_plan=split,
                    split_audit=audit,
                    input_file_sha256s=identities,
                    standardized_root=audio_root,
                )
                validation = validate_dataset_artifact_set(output, config=config)
                self.assertEqual(validation["item_count"], 1)
                self.assertEqual(validation["checksummed_artifact_count"], 7)
                inventories.append(dataset_tree_inventory(output))
            self.assertEqual(inventories[0], inventories[1])
            trainer_row = json.loads(
                (root / "first" / config.output.train_jsonl).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(trainer_row), {"fid", "audio", "text"})
            self.assertTrue(Path(trainer_row["audio"]).is_absolute())
            deep = validate_frozen_dataset(root / "first", config=config)
            self.assertEqual(deep["audio_hashes_verified"], 1)
            stats_path = root / "first" / config.output.stats_json
            stats_content = stats_path.read_bytes()
            stats_path.write_bytes(stats_content + b"drift")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                validate_dataset_artifact_set(root / "first", config=config)
            stats_path.write_bytes(stats_content)
            (audio_root / "sample.wav").write_bytes(b"drifted-audio")
            with self.assertRaisesRegex(RuntimeError, "audio SHA-256 drift"):
                validate_frozen_dataset(root / "first", config=config)


class DatasetAnalysisReportTests(unittest.TestCase):
    def seed_report_fixture(
        self, root: Path
    ) -> tuple[Catalog, str, dict, dict, dict, ThresholdCalibrationOverlay]:
        audio_root = root / "standardized"
        audio_root.mkdir()
        assets = ["a" * 64, "b" * 64]
        items = []
        for index, asset in enumerate(assets):
            content = f"audio-{index}".encode()
            path = audio_root / f"{index}.wav"
            path.write_bytes(content)
            items.append(
                {
                    "asset_sha256": asset,
                    "audio_relative_path": path.name,
                    "audio_sha256": hashlib.sha256(content).hexdigest(),
                    "duration_seconds": 1.0,
                    "emotion_primary": "neutral",
                    "fid": f"item-{index}",
                    "text_exact": "<script>alert(1)</script>" if index == 0 else "safe",
                    "text_source": "filename_candidate_unreviewed",
                }
            )
        snapshot = {"item_count": 2, "items": items}
        overlay = ThresholdCalibrationOverlay.model_validate(
            {
                "calibration_id": "calibration",
                "calibration_version": 1,
                "calibration_config_sha256": "3" * 64,
                "calibration_report_sha256": "4" * 64,
                "calibration_report_path": str(root / "calibration.json"),
                "dataset_id": "dataset",
                "dataset_version": 1,
                "dataset_config_sha256": "1" * 64,
                "candidate_snapshot_sha256": "2" * 64,
                "thresholds": {
                    "text_near_similarity": 0.8,
                    "acoustic_near_cosine": 0.9,
                    "speaker_center_cosine": 0.6,
                    "speaker_knn_cosine": 0.7,
                },
            },
            strict=True,
        )
        catalog = Catalog(root / "catalog.sqlite")
        catalog.initialize()
        with catalog.session() as connection:
            connection.executemany(
                """
                INSERT INTO asset (
                    sha256, size_bytes, probe_status, first_seen_at, last_seen_at
                ) VALUES (?, 1, 'ok', 't0', 't0')
                """,
                [(asset,) for asset in assets],
            )
        catalog.register_dataset_build_config(
            dataset_id="dataset",
            dataset_version=1,
            config_sha256="1" * 64,
            config_json="{}",
            implementation_version=1,
            source_path="config.yaml",
            registered_at="t0",
        )
        catalog.register_dataset_threshold_calibration(
            dataset_id="dataset",
            dataset_version=1,
            calibration_id="calibration",
            calibration_version=1,
            calibration_config_sha256="3" * 64,
            calibration_report_sha256="4" * 64,
            thresholds=overlay.thresholds.model_dump(mode="json"),
            report_path="calibration.json",
            registered_at="t0",
        )
        run_id = catalog.begin_dataset_analysis_run(
            dataset_id="dataset",
            dataset_version=1,
            candidate_snapshot_sha256="2" * 64,
            calibration_id="calibration",
            calibration_version=1,
            calibration_config_sha256="3" * 64,
            calibration_report_sha256="4" * 64,
            started_at="t0",
        )
        edge = {
            "left_asset_sha256": assets[0],
            "right_asset_sha256": assets[1],
            "evidence_type": "near_text",
            "score": 0.9,
            "threshold": 0.8,
            "analysis_status": "pending_review",
            "evidence_json": canonical_json(
                {"pair": {"risk_flags": ["named_entity_difference"]}}
            ),
        }
        catalog.store_dataset_similarity_graph(
            run_id=run_id,
            candidate_asset_sha256s=assets,
            edges=[edge],
            created_at="t1",
        )
        speaker_sha256 = "5" * 64
        speaker_report = {
            "status": "succeeded",
            "assessments": [
                {
                    "asset_sha256": assets[0],
                    "center_cosine": 0.5,
                    "knn_cosine": 0.65,
                    "neighbor_asset_sha256s": [assets[1]],
                    "neighbor_cosines": [0.65],
                    "outlier_rank": 1,
                    "outlier_score": 0.425,
                },
                {
                    "asset_sha256": assets[1],
                    "center_cosine": 0.8,
                    "knn_cosine": 0.85,
                    "neighbor_asset_sha256s": [assets[0]],
                    "neighbor_cosines": [0.65],
                    "outlier_rank": 2,
                    "outlier_score": 0.175,
                },
            ],
        }
        calibration_report = {
            "status": "succeeded",
            "inputs": {"speaker_report_sha256": speaker_sha256},
            "speaker": {
                "candidate_count": 1,
                "candidates": [{"asset_sha256": assets[0]}],
            },
        }
        return (
            catalog,
            run_id,
            snapshot,
            speaker_report,
            calibration_report,
            overlay,
        )

    def test_builds_and_writes_deterministic_listenable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, run_id, snapshot, speaker, calibration, overlay = (
                self.seed_report_fixture(root)
            )
            catalog_hash_before = hashlib.sha256(catalog.path.read_bytes()).hexdigest()
            report = build_dataset_analysis_report(
                catalog=catalog,
                run_id=run_id,
                candidate_snapshot=snapshot,
                candidate_snapshot_sha256="2" * 64,
                speaker_report=speaker,
                speaker_report_sha256="5" * 64,
                calibration_report=calibration,
                overlay=overlay,
                standardized_root=root / "standardized",
            )
            outputs = write_dataset_analysis_reports(report, root / "reports")
            first_html = Path(outputs["html_path"]).read_text(encoding="utf-8")
            repeated = write_dataset_analysis_reports(report, root / "reports")

            self.assertEqual(report["status"], "review_required")
            self.assertEqual(report["summary"]["duplicate_edge_count"], 1)
            self.assertEqual(report["summary"]["speaker_outlier_count"], 1)
            self.assertEqual(len(report["speaker_outliers"][0]["neighbors"]), 1)
            self.assertIn("<audio controls", first_html)
            self.assertNotIn("<script>alert(1)</script>", first_html)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", first_html)
            self.assertEqual(outputs, repeated)
            self.assertEqual(
                hashlib.sha256(catalog.path.read_bytes()).hexdigest(),
                catalog_hash_before,
            )
            csv_lines = Path(outputs["csv_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), 3)

            bundle = write_portable_dataset_analysis_bundle(
                report,
                root / "bundle",
                online_url="https://example.test/report",
            )
            repeated_bundle = write_portable_dataset_analysis_bundle(
                report,
                root / "bundle",
                online_url="https://example.test/report",
            )
            portable_json = (root / "bundle" / "dataset_analysis.json").read_text(
                encoding="utf-8"
            )
            portable_html = (root / "bundle" / "index.html").read_text(
                encoding="utf-8"
            )
            self.assertEqual(bundle, repeated_bundle)
            self.assertEqual(bundle["audio_file_count"], 2)
            self.assertNotIn(str(root.resolve()), portable_json)
            self.assertIn('src="audio/', portable_html)
            self.assertIn(
                "https://example.test/report",
                (root / "bundle" / "README.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                len((root / "bundle" / "checksums.txt").read_text().splitlines()),
                7,
            )

    def test_reviewed_html_makes_keep_and_exclude_decisions_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, run_id, snapshot, speaker, calibration, overlay = (
                self.seed_report_fixture(root)
            )
            report = build_dataset_analysis_report(
                catalog=catalog,
                run_id=run_id,
                candidate_snapshot=snapshot,
                candidate_snapshot_sha256="2" * 64,
                speaker_report=speaker,
                speaker_report_sha256="5" * 64,
                calibration_report=calibration,
                overlay=overlay,
                standardized_root=root / "standardized",
            )
            edge = report["duplicate_edges"][0]
            edge["review_status"] = "accepted"
            edge["review"] = {
                "review_note": canonical_json(
                    {
                        "representative_asset_sha256": "b" * 64,
                        "exclude_asset_sha256s": ["a" * 64],
                        "note": "keep the complete sample",
                    }
                )
            }
            outlier = report["speaker_outliers"][0]
            outlier["review_status"] = "exclude_uncertain"
            outlier["review"] = {"review_note": "poor audio quality"}
            report["status"] = "reviewed"
            report["summary"]["pending_duplicate_review_count"] = 0
            report["summary"]["pending_speaker_review_count"] = 0
            report["summary"]["speaker_exclusion_count"] = 1

            outputs = write_dataset_analysis_reports(report, root / "reviewed")
            rendered = Path(outputs["html_path"]).read_text(encoding="utf-8")
            self.assertIn("人工复核已完成", rendered)
            self.assertIn("保留此条", rendered)
            self.assertIn("排除此条（重复/截短）", rendered)
            self.assertIn("从冻结数据集排除", rendered)
            self.assertIn("poor audio quality", rendered)

    def test_missing_or_drifted_review_audio_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, run_id, snapshot, speaker, calibration, overlay = (
                self.seed_report_fixture(root)
            )
            (root / "standardized" / "0.wav").write_bytes(b"drift")
            with self.assertRaisesRegex(RuntimeError, "audio SHA-256 drift"):
                build_dataset_analysis_report(
                    catalog=catalog,
                    run_id=run_id,
                    candidate_snapshot=snapshot,
                    candidate_snapshot_sha256="2" * 64,
                    speaker_report=speaker,
                    speaker_report_sha256="5" * 64,
                    calibration_report=calibration,
                    overlay=overlay,
                    standardized_root=root / "standardized",
                )


class AcousticFingerprintTests(unittest.TestCase):
    def signal(self, sample_rate: int = 48000) -> np.ndarray:
        time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.3 * time)
        signal = envelope * (
            0.45 * np.sin(2 * np.pi * 220 * time)
            + 0.25 * np.sin(2 * np.pi * 503 * time)
            + 0.15 * np.sin(2 * np.pi * 997 * time)
        )
        return signal.astype(np.float32)

    def candidate(self, path: Path, asset: str) -> dict:
        return {
            "asset_sha256": asset,
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "audio_absolute_path": str(path),
        }

    def test_gain_silence_and_resampling_variants_remain_close(self) -> None:
        config = load_dataset_freeze_config().acoustic_fingerprint
        base = self.signal()
        gain = base * 0.5
        silence = np.pad(base, (4800, 7200))
        resampled = soxr.resample(base, 48000, 44100, quality="HQ")

        base_feature = compute_acoustic_fingerprint(
            base, sample_rate=48000, config=config
        )
        gain_feature = compute_acoustic_fingerprint(
            gain, sample_rate=48000, config=config
        )
        silence_feature = compute_acoustic_fingerprint(
            silence, sample_rate=48000, config=config
        )
        resampled_feature = compute_acoustic_fingerprint(
            resampled, sample_rate=44100, config=config
        )

        self.assertEqual(base_feature.shape, (FINGERPRINT_DIMENSION,))
        self.assertEqual(base_feature.dtype, np.dtype("<f4"))
        self.assertAlmostEqual(float(np.linalg.norm(base_feature)), 1.0, places=5)
        self.assertGreater(cosine_similarity(base_feature, gain_feature), 0.999)
        self.assertGreater(cosine_similarity(base_feature, silence_feature), 0.99)
        self.assertGreater(cosine_similarity(base_feature, resampled_feature), 0.99)
        self.assertEqual(
            cosine_similarity(base_feature, resampled_feature),
            cosine_similarity(resampled_feature, base_feature),
        )

    def test_cache_is_content_addressed_and_validated_before_reuse(self) -> None:
        config = load_dataset_freeze_config().acoustic_fingerprint
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio.wav"
            sf.write(audio_path, self.signal(), 48000, subtype="PCM_24")
            candidate = self.candidate(audio_path, "a" * 64)
            cache_root = root / "cache"

            first = load_or_compute_acoustic_fingerprint(
                candidate, config=config, cache_root=cache_root
            )
            second = load_or_compute_acoustic_fingerprint(
                candidate, config=config, cache_root=cache_root
            )
            self.assertEqual(first["action"], "computed")
            self.assertEqual(second["action"], "cached")
            self.assertEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])
            self.assertEqual(first["cache_path"], second["cache_path"])

            cache_path = Path(first["cache_path"])
            corrupted = bytearray(cache_path.read_bytes())
            corrupted[-1] ^= 1
            cache_path.write_bytes(corrupted)
            with self.assertRaisesRegex(RuntimeError, "content hash or metadata drift"):
                load_or_compute_acoustic_fingerprint(
                    candidate, config=config, cache_root=cache_root
                )
            cache_path.write_bytes(first["vector"].astype("<f4").tobytes())

            sf.write(audio_path, self.signal() * 0.25, 48000, subtype="PCM_24")
            with self.assertRaisesRegex(RuntimeError, "input SHA-256 drift"):
                load_or_compute_acoustic_fingerprint(
                    candidate, config=config, cache_root=cache_root
                )

    def test_silence_and_bad_cosine_inputs_fail_closed(self) -> None:
        config = load_dataset_freeze_config().acoustic_fingerprint
        with self.assertRaisesRegex(RuntimeError, "silent"):
            compute_acoustic_fingerprint(
                np.zeros(48000, dtype=np.float32),
                sample_rate=48000,
                config=config,
            )
        with self.assertRaises(ValueError):
            cosine_similarity(np.zeros(2), np.ones(2))
        with self.assertRaises(ValueError):
            cosine_similarity(np.ones(2), np.ones(3))

    def test_batch_cache_and_report_are_stable(self) -> None:
        config = load_dataset_freeze_config().acoustic_fingerprint
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.wav"
            second_path = root / "second.wav"
            sf.write(first_path, self.signal(), 48000, subtype="PCM_24")
            sf.write(second_path, self.signal() * 0.75, 48000, subtype="PCM_24")
            candidates = [
                self.candidate(second_path, "b" * 64),
                self.candidate(first_path, "a" * 64),
            ]
            first = build_acoustic_fingerprint_cache(
                candidates, config=config, cache_root=root / "cache"
            )
            second = build_acoustic_fingerprint_cache(
                reversed(candidates), config=config, cache_root=root / "cache"
            )

            self.assertEqual(first["report"], second["report"])
            self.assertEqual(first["actions"], {"computed": 2})
            self.assertEqual(second["actions"], {"cached": 2})
            report_path = root / "fingerprints.json"
            written = write_acoustic_fingerprint_report(first["report"], report_path)
            rewritten = write_acoustic_fingerprint_report(second["report"], report_path)
            self.assertEqual(written["sha256"], rewritten["sha256"])


class SpeakerEmbeddingTests(unittest.TestCase):
    class FakeEncoder:
        def __call__(self, audio, audio_lengths=None):
            base = torch.arange(1, SPEAKER_EMBEDDING_DIMENSION + 1).float()
            return (base + audio.mean()).reshape(1, -1)

    def signal(self, sample_rate: int = 48000) -> np.ndarray:
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        return (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

    def candidate(self, path: Path, asset: str) -> dict:
        return {
            "asset_sha256": asset,
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "audio_absolute_path": str(path),
        }

    def test_pinned_model_loads_strictly_and_rejects_weight_drift(self) -> None:
        config = load_dataset_freeze_config().speaker_embedding
        encoder = load_speaker_encoder(config)
        self.assertEqual(encoder.model.xvector.dense.linear.out_channels, 512)

        drifted = config.model_copy(update={"weights_sha256": "0" * 64})
        with self.assertRaisesRegex(RuntimeError, "speaker weights SHA-256 drift"):
            load_speaker_encoder(drifted)

    def test_embedding_is_finite_l2_normalized_and_strict_about_audio(self) -> None:
        config = load_dataset_freeze_config().speaker_embedding
        embedding = compute_speaker_embedding(
            self.signal(),
            sample_rate=48000,
            config=config,
            encoder=self.FakeEncoder(),
        )
        self.assertEqual(embedding.shape, (SPEAKER_EMBEDDING_DIMENSION,))
        self.assertEqual(embedding.dtype, np.dtype("<f4"))
        self.assertTrue(np.all(np.isfinite(embedding)))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
        with self.assertRaisesRegex(RuntimeError, "sample rate mismatch"):
            compute_speaker_embedding(
                self.signal(),
                sample_rate=16000,
                config=config,
                encoder=self.FakeEncoder(),
            )
        with self.assertRaisesRegex(RuntimeError, "silent"):
            compute_speaker_embedding(
                np.zeros(48000, dtype=np.float32),
                sample_rate=48000,
                config=config,
                encoder=self.FakeEncoder(),
            )

    def test_cache_reuses_valid_content_and_rejects_corruption(self) -> None:
        config = load_dataset_freeze_config().speaker_embedding
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio.wav"
            sf.write(audio_path, self.signal(), 48000, subtype="PCM_24")
            candidate = self.candidate(audio_path, "a" * 64)
            cache_root = root / "cache"
            first = load_or_compute_speaker_embedding(
                candidate,
                config=config,
                encoder=self.FakeEncoder(),
                cache_root=cache_root,
            )
            second = load_or_compute_speaker_embedding(
                candidate,
                config=config,
                encoder=self.FakeEncoder(),
                cache_root=cache_root,
            )
            self.assertEqual(first["action"], "computed")
            self.assertEqual(second["action"], "cached")
            self.assertEqual(
                first["speaker_embedding_sha256"],
                second["speaker_embedding_sha256"],
            )

            cache_path = Path(first["cache_path"])
            damaged = bytearray(cache_path.read_bytes())
            damaged[-1] ^= 1
            cache_path.write_bytes(damaged)
            with self.assertRaisesRegex(
                RuntimeError, "content hash or metadata drift"
            ):
                load_or_compute_speaker_embedding(
                    candidate,
                    config=config,
                    encoder=self.FakeEncoder(),
                    cache_root=cache_root,
                )

    def test_center_knn_and_outlier_score_are_stable(self) -> None:
        def unit(first: float, second: float) -> np.ndarray:
            vector = np.zeros(SPEAKER_EMBEDDING_DIMENSION, dtype=np.float32)
            vector[:2] = (first, second)
            return vector / np.linalg.norm(vector)

        features = [
            {"asset_sha256": "a" * 64, "vector": unit(1.0, 0.0)},
            {"asset_sha256": "b" * 64, "vector": unit(0.99, 0.1)},
            {"asset_sha256": "c" * 64, "vector": unit(0.0, 1.0)},
        ]
        assessment = assess_speaker_embeddings(features, knn_k=1)
        reordered = assess_speaker_embeddings(reversed(features), knn_k=1)

        self.assertEqual(assessment, reordered)
        by_asset = {
            item["asset_sha256"]: item for item in assessment["assessments"]
        }
        self.assertGreater(
            by_asset["c" * 64]["outlier_score"],
            by_asset["a" * 64]["outlier_score"],
        )
        self.assertEqual(by_asset["c" * 64]["outlier_rank"], 1)
        self.assertIsNone(by_asset["c" * 64]["candidate_outlier"])

    def test_batch_report_is_stable_across_cache_actions(self) -> None:
        config = load_dataset_freeze_config().speaker_embedding
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = []
            for index, asset in enumerate(("a" * 64, "b" * 64)):
                path = root / f"audio-{index}.wav"
                sf.write(
                    path,
                    self.signal() * (1 - index * 0.1),
                    48000,
                    subtype="PCM_24",
                )
                candidates.append(self.candidate(path, asset))
            first = build_speaker_embedding_cache_and_assessment(
                candidates,
                config=config,
                cache_root=root / "cache",
                encoder=self.FakeEncoder(),
            )
            second = build_speaker_embedding_cache_and_assessment(
                reversed(candidates),
                config=config,
                cache_root=root / "cache",
                encoder=self.FakeEncoder(),
            )
            self.assertEqual(first["report"], second["report"])
            self.assertEqual(first["actions"], {"computed": 2})
            self.assertEqual(second["actions"], {"cached": 2})
            report_path = root / "speaker.json"
            written = write_speaker_embedding_report(first["report"], report_path)
            rewritten = write_speaker_embedding_report(second["report"], report_path)
            self.assertEqual(written["sha256"], rewritten["sha256"])


class ThresholdCalibrationTests(unittest.TestCase):
    def test_versioned_config_is_strict_and_has_stable_identity(self) -> None:
        config = load_threshold_calibration_config()
        self.assertEqual(config.calibration_id, "fuxuan_dataset_thresholds")
        self.assertEqual(config.calibration_version, 1)
        self.assertEqual(config.identity(), config.identity())
        self.assertEqual(len(config.text.positive_pairs), 3)

    def test_overlay_promotes_thresholds_without_mutating_dataset_config(self) -> None:
        dataset_config = load_dataset_freeze_config()
        with self.assertRaisesRegex(RuntimeError, "audit-only"):
            dataset_config.assert_freeze_ready()

        overlay = load_threshold_calibration_overlay()
        thresholds = assert_dataset_ready_with_calibration(dataset_config, overlay)
        self.assertEqual(overlay.dataset_id, "fuxuan")
        self.assertEqual(overlay.dataset_version, 1)
        self.assertGreater(thresholds.text_near_similarity, 0.0)
        self.assertGreater(thresholds.acoustic_near_cosine, 0.0)

    def test_midpoint_requires_separated_distributions(self) -> None:
        result = separated_midpoint_threshold([0.9, 0.8], [0.2, 0.6])
        self.assertEqual(result["positive_min"], 0.8)
        self.assertEqual(result["background_max"], 0.6)
        self.assertAlmostEqual(result["threshold"], 0.7)
        with self.assertRaisesRegex(RuntimeError, "distributions overlap"):
            separated_midpoint_threshold([0.5, 0.7], [0.2, 0.6])

    def test_robust_lower_threshold_uses_median_and_mad(self) -> None:
        result = robust_lower_threshold(
            [0.8, 0.9, 0.9, 1.0],
            consistency_scale=1.4826,
            sigma_cutoff=4.5,
        )
        self.assertEqual(result["median"], 0.9)
        self.assertAlmostEqual(result["mad"], 0.05)
        self.assertAlmostEqual(
            result["threshold"], 0.9 - 4.5 * 1.4826 * 0.05
        )

    def test_duration_stratified_selection_is_order_stable(self) -> None:
        items = [
            {
                "asset_sha256": f"{index:064x}",
                "emotion_primary": emotion,
                "duration_seconds": duration,
            }
            for index, (emotion, duration) in enumerate(
                [
                    ("happy", 1.0),
                    ("happy", 2.0),
                    ("happy", 3.0),
                    ("happy", 4.0),
                    ("happy", 5.0),
                    ("sad", 1.5),
                    ("sad", 2.5),
                ]
            )
        ]
        selected = select_stratified_duration_samples(
            items, samples_per_emotion=3
        )
        reordered = select_stratified_duration_samples(
            list(reversed(items)), samples_per_emotion=3
        )
        self.assertEqual(selected, reordered)
        self.assertEqual(
            Counter(item["emotion_primary"] for item in selected),
            Counter({"happy": 3, "sad": 2}),
        )


if __name__ == "__main__":
    unittest.main()
