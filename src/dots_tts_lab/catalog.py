from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from dots_tts_lab.duplicate_graph import canonical_json, stable_similarity_groups

SCHEMA_VERSION = 10

_MIGRATIONS: dict[int, str] = {
    1: """
        BEGIN IMMEDIATE;

        CREATE TABLE ingest_run (
            run_id TEXT PRIMARY KEY,
            root_path TEXT NOT NULL,
            root_key TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'completed_with_errors', 'failed')
            ),
            discovered_count INTEGER NOT NULL DEFAULT 0,
            readable_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            added_count INTEGER NOT NULL DEFAULT 0,
            changed_count INTEGER NOT NULL DEFAULT 0,
            moved_count INTEGER NOT NULL DEFAULT 0,
            missing_count INTEGER NOT NULL DEFAULT 0,
            unchanged_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        );

        CREATE TABLE asset (
            sha256 TEXT PRIMARY KEY CHECK (length(sha256) = 64),
            size_bytes INTEGER NOT NULL,
            format TEXT,
            subtype TEXT,
            sample_rate INTEGER,
            channels INTEGER,
            frames INTEGER,
            duration_seconds REAL,
            probe_status TEXT NOT NULL CHECK (probe_status IN ('ok', 'error')),
            probe_error_type TEXT,
            probe_error_message TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE source_location (
            root_key TEXT NOT NULL,
            root_path TEXT NOT NULL,
            path_key TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            asset_sha256 TEXT,
            size_bytes INTEGER,
            mtime_ns INTEGER,
            path_length INTEGER NOT NULL,
            speaker_id TEXT,
            emotion_weak_label TEXT,
            transcript_candidate TEXT,
            parse_status TEXT NOT NULL CHECK (parse_status IN ('ok', 'error')),
            parse_error TEXT,
            availability_status TEXT NOT NULL CHECK (
                availability_status IN ('available', 'missing')
            ),
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_seen_run_id TEXT NOT NULL,
            missing_since_run_id TEXT,
            PRIMARY KEY (root_key, path_key),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (last_seen_run_id) REFERENCES ingest_run(run_id),
            FOREIGN KEY (missing_since_run_id) REFERENCES ingest_run(run_id)
        );

        CREATE INDEX idx_source_location_asset
            ON source_location(asset_sha256);
        CREATE INDEX idx_source_location_availability
            ON source_location(root_key, availability_status);
        CREATE INDEX idx_ingest_run_root_started
            ON ingest_run(root_key, started_at);

        PRAGMA user_version = 1;
        COMMIT;
    """,
    2: """
        BEGIN IMMEDIATE;

        CREATE TABLE import_run (
            run_id TEXT PRIMARY KEY,
            root_path TEXT NOT NULL,
            root_key TEXT NOT NULL,
            raw_root_path TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'completed_with_errors', 'failed')
            ),
            discovered_count INTEGER NOT NULL DEFAULT 0,
            unique_asset_count INTEGER NOT NULL DEFAULT 0,
            imported_count INTEGER NOT NULL DEFAULT 0,
            reused_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            bytes_written INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        );

        CREATE TABLE raw_object (
            asset_sha256 TEXT PRIMARY KEY,
            relative_path TEXT NOT NULL UNIQUE,
            extension TEXT NOT NULL,
            storage_mode TEXT NOT NULL CHECK (storage_mode IN ('copy')),
            size_bytes INTEGER NOT NULL,
            first_imported_at TEXT NOT NULL,
            last_verified_at TEXT NOT NULL,
            first_import_run_id TEXT NOT NULL,
            last_import_run_id TEXT NOT NULL,
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (first_import_run_id) REFERENCES import_run(run_id),
            FOREIGN KEY (last_import_run_id) REFERENCES import_run(run_id)
        );

        CREATE TABLE import_item (
            run_id TEXT NOT NULL,
            root_key TEXT NOT NULL,
            path_key TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            source_relative_path TEXT NOT NULL,
            original_name TEXT NOT NULL,
            raw_relative_path TEXT,
            action TEXT NOT NULL CHECK (action IN ('imported', 'reused', 'error')),
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, root_key, path_key),
            FOREIGN KEY (run_id) REFERENCES import_run(run_id),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (root_key, path_key)
                REFERENCES source_location(root_key, path_key)
        );

        CREATE INDEX idx_import_run_root_started
            ON import_run(root_key, started_at);
        CREATE INDEX idx_import_item_asset
            ON import_item(asset_sha256);

        PRAGMA user_version = 2;
        COMMIT;
    """,
    3: """
        BEGIN IMMEDIATE;

        CREATE TABLE metadata_profile (
            profile_id TEXT NOT NULL,
            profile_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            config_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (profile_id, profile_version),
            UNIQUE (config_sha256)
        );

        ALTER TABLE source_location
            ADD COLUMN directory_emotion_weak_label TEXT;
        ALTER TABLE source_location
            ADD COLUMN filename_emotion_weak_label TEXT;
        ALTER TABLE source_location
            ADD COLUMN metadata_status TEXT NOT NULL DEFAULT 'error' CHECK (
                metadata_status IN ('ok', 'review_required', 'error')
            );
        ALTER TABLE source_location
            ADD COLUMN metadata_review_reason TEXT;
        ALTER TABLE source_location
            ADD COLUMN parser_profile_id TEXT;
        ALTER TABLE source_location
            ADD COLUMN parser_profile_version INTEGER;
        ALTER TABLE source_location
            ADD COLUMN parser_config_sha256 TEXT;

        UPDATE source_location
        SET directory_emotion_weak_label = emotion_weak_label,
            metadata_status = parse_status;

        CREATE TABLE metadata_parse (
            root_key TEXT NOT NULL,
            path_key TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            profile_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL,
            parsed_at TEXT NOT NULL,
            speaker_id TEXT,
            directory_emotion_weak_label TEXT,
            filename_emotion_weak_label TEXT,
            emotion_weak_label TEXT,
            transcript_candidate TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('ok', 'review_required', 'error')
            ),
            error_message TEXT,
            review_reason TEXT,
            PRIMARY KEY (
                root_key, path_key, profile_id, profile_version, config_sha256
            ),
            FOREIGN KEY (root_key, path_key)
                REFERENCES source_location(root_key, path_key),
            FOREIGN KEY (profile_id, profile_version)
                REFERENCES metadata_profile(profile_id, profile_version)
        );

        CREATE INDEX idx_metadata_parse_status
            ON metadata_parse(status, profile_id, profile_version);

        PRAGMA user_version = 3;
        COMMIT;
    """,
    4: """
        BEGIN IMMEDIATE;

        CREATE TABLE quality_analysis_config (
            analysis_id TEXT NOT NULL,
            analysis_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            config_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (analysis_id, analysis_version),
            UNIQUE (config_sha256)
        );

        CREATE TABLE quality_policy (
            policy_id TEXT NOT NULL,
            policy_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            config_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (policy_id, policy_version),
            UNIQUE (config_sha256)
        );

        CREATE TABLE quality_run (
            run_id TEXT PRIMARY KEY,
            raw_root_path TEXT NOT NULL,
            analysis_id TEXT NOT NULL,
            analysis_version INTEGER NOT NULL,
            analysis_config_sha256 TEXT NOT NULL,
            implementation_version INTEGER NOT NULL,
            policy_id TEXT NOT NULL,
            policy_version INTEGER NOT NULL,
            policy_config_sha256 TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'completed_with_errors', 'failed')
            ),
            discovered_count INTEGER NOT NULL DEFAULT 0,
            analyzed_count INTEGER NOT NULL DEFAULT 0,
            cached_count INTEGER NOT NULL DEFAULT 0,
            pass_count INTEGER NOT NULL DEFAULT 0,
            review_count INTEGER NOT NULL DEFAULT 0,
            reject_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY (analysis_id, analysis_version)
                REFERENCES quality_analysis_config(analysis_id, analysis_version),
            FOREIGN KEY (policy_id, policy_version)
                REFERENCES quality_policy(policy_id, policy_version)
        );

        CREATE TABLE audio_quality_metric (
            asset_sha256 TEXT NOT NULL,
            analysis_id TEXT NOT NULL,
            analysis_version INTEGER NOT NULL,
            analysis_config_sha256 TEXT NOT NULL,
            implementation_version INTEGER NOT NULL,
            analyzed_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
            error_type TEXT,
            error_message TEXT,
            sample_rate INTEGER,
            channels INTEGER,
            frames INTEGER,
            duration_seconds REAL,
            sample_peak_dbfs REAL,
            true_peak_estimate_dbtp REAL,
            rms_dbfs REAL,
            integrated_loudness_lufs REAL,
            crest_factor_db REAL,
            abs_dc_offset REAL,
            leading_silence_seconds REAL,
            trailing_silence_seconds REAL,
            silence_ratio REAL,
            digital_silence_frame_ratio REAL,
            noise_floor_proxy_dbfs REAL,
            speech_level_proxy_dbfs REAL,
            snr_proxy_db REAL,
            near_peak_sample_count INTEGER,
            near_peak_sample_ratio REAL,
            flat_top_run_count INTEGER,
            max_flat_top_run_samples INTEGER,
            PRIMARY KEY (
                asset_sha256, analysis_id, analysis_version,
                analysis_config_sha256, implementation_version
            ),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (analysis_id, analysis_version)
                REFERENCES quality_analysis_config(analysis_id, analysis_version)
        );

        CREATE TABLE audio_quality_assessment (
            asset_sha256 TEXT NOT NULL,
            analysis_id TEXT NOT NULL,
            analysis_version INTEGER NOT NULL,
            analysis_config_sha256 TEXT NOT NULL,
            implementation_version INTEGER NOT NULL,
            policy_id TEXT NOT NULL,
            policy_version INTEGER NOT NULL,
            policy_config_sha256 TEXT NOT NULL,
            assessed_at TEXT NOT NULL,
            decision TEXT NOT NULL CHECK (
                decision IN ('pass', 'review', 'reject')
            ),
            reasons_json TEXT NOT NULL,
            PRIMARY KEY (
                asset_sha256, analysis_id, analysis_version,
                analysis_config_sha256, implementation_version,
                policy_id, policy_version, policy_config_sha256
            ),
            FOREIGN KEY (
                asset_sha256, analysis_id, analysis_version,
                analysis_config_sha256, implementation_version
            ) REFERENCES audio_quality_metric (
                asset_sha256, analysis_id, analysis_version,
                analysis_config_sha256, implementation_version
            ),
            FOREIGN KEY (policy_id, policy_version)
                REFERENCES quality_policy(policy_id, policy_version)
        );

        CREATE TABLE quality_run_item (
            run_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            raw_relative_path TEXT NOT NULL,
            metric_action TEXT NOT NULL CHECK (
                metric_action IN ('analyzed', 'cached', 'error')
            ),
            decision TEXT NOT NULL CHECK (
                decision IN ('pass', 'review', 'reject')
            ),
            reasons_json TEXT NOT NULL,
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, asset_sha256),
            FOREIGN KEY (run_id) REFERENCES quality_run(run_id),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256)
        );

        CREATE INDEX idx_quality_metric_status
            ON audio_quality_metric(status, analysis_id, analysis_version);
        CREATE INDEX idx_quality_assessment_decision
            ON audio_quality_assessment(decision, policy_id, policy_version);
        CREATE INDEX idx_quality_run_started
            ON quality_run(started_at);

        PRAGMA user_version = 4;
        COMMIT;
    """,
    5: """
        BEGIN IMMEDIATE;

        CREATE TABLE standardization_config (
            config_id TEXT NOT NULL,
            config_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            config_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (config_id, config_version),
            UNIQUE (config_sha256)
        );

        CREATE TABLE standardization_run (
            run_id TEXT PRIMARY KEY,
            raw_root_path TEXT NOT NULL,
            output_root_path TEXT NOT NULL,
            config_id TEXT NOT NULL,
            config_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL,
            implementation_version INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'completed_with_errors', 'failed')
            ),
            discovered_count INTEGER NOT NULL DEFAULT 0,
            built_count INTEGER NOT NULL DEFAULT 0,
            cached_count INTEGER NOT NULL DEFAULT 0,
            rebuilt_count INTEGER NOT NULL DEFAULT 0,
            trimmed_count INTEGER NOT NULL DEFAULT 0,
            attenuated_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY (config_id, config_version)
                REFERENCES standardization_config(config_id, config_version)
        );

        CREATE TABLE derived_audio (
            derived_id TEXT PRIMARY KEY CHECK (length(derived_id) = 64),
            asset_sha256 TEXT NOT NULL,
            config_id TEXT NOT NULL,
            config_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL,
            implementation_version INTEGER NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            output_sha256 TEXT NOT NULL CHECK (length(output_sha256) = 64),
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_verified_at TEXT NOT NULL,
            source_sample_rate INTEGER NOT NULL,
            source_channels INTEGER NOT NULL,
            source_frames INTEGER NOT NULL,
            output_sample_rate INTEGER NOT NULL,
            output_channels INTEGER NOT NULL,
            output_frames INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            leading_samples_removed INTEGER NOT NULL,
            trailing_samples_removed INTEGER NOT NULL,
            gain_applied_db REAL NOT NULL,
            output_sample_peak_dbfs REAL,
            output_true_peak_estimate_dbtp REAL,
            output_subtype TEXT NOT NULL,
            created_run_id TEXT NOT NULL,
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (config_id, config_version)
                REFERENCES standardization_config(config_id, config_version),
            FOREIGN KEY (created_run_id) REFERENCES standardization_run(run_id)
        );

        CREATE TABLE standardization_run_item (
            run_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            derived_id TEXT NOT NULL,
            raw_relative_path TEXT NOT NULL,
            derived_relative_path TEXT,
            action TEXT NOT NULL CHECK (
                action IN ('built', 'cached', 'rebuilt', 'error')
            ),
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, asset_sha256),
            FOREIGN KEY (run_id) REFERENCES standardization_run(run_id),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256)
        );

        CREATE INDEX idx_derived_audio_source
            ON derived_audio(asset_sha256, config_id, config_version);
        CREATE INDEX idx_standardization_run_started
            ON standardization_run(started_at);

        PRAGMA user_version = 5;
        COMMIT;
    """,
    6: """
        BEGIN IMMEDIATE;

        CREATE TABLE transcript_ground_truth (
            asset_sha256 TEXT PRIMARY KEY,
            text_exact TEXT NOT NULL,
            text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64),
            provenance_kind TEXT NOT NULL CHECK (
                provenance_kind IN ('user_confirmed_filename')
            ),
            provenance_note TEXT NOT NULL,
            confirmed_at TEXT NOT NULL,
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256)
        );

        CREATE TABLE asr_benchmark (
            benchmark_id TEXT NOT NULL,
            benchmark_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            config_json TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64),
            created_at TEXT NOT NULL,
            PRIMARY KEY (benchmark_id, benchmark_version),
            UNIQUE (config_sha256),
            UNIQUE (manifest_sha256)
        );

        CREATE TABLE asr_benchmark_item (
            benchmark_id TEXT NOT NULL,
            benchmark_version INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            asset_sha256 TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            derived_id TEXT NOT NULL,
            derived_relative_path TEXT NOT NULL,
            emotion_weak_label TEXT NOT NULL,
            selection_reasons_json TEXT NOT NULL,
            PRIMARY KEY (benchmark_id, benchmark_version, asset_sha256),
            UNIQUE (benchmark_id, benchmark_version, ordinal),
            FOREIGN KEY (benchmark_id, benchmark_version)
                REFERENCES asr_benchmark(benchmark_id, benchmark_version),
            FOREIGN KEY (asset_sha256)
                REFERENCES transcript_ground_truth(asset_sha256),
            FOREIGN KEY (derived_id) REFERENCES derived_audio(derived_id)
        );

        CREATE TABLE asr_run (
            run_id TEXT PRIMARY KEY,
            benchmark_id TEXT NOT NULL,
            benchmark_version INTEGER NOT NULL,
            backend_id TEXT NOT NULL,
            backend_version TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            inference_config_sha256 TEXT NOT NULL CHECK (
                length(inference_config_sha256) = 64
            ),
            inference_config_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'completed_with_errors', 'failed')
            ),
            item_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            total_audio_seconds REAL NOT NULL DEFAULT 0,
            total_runtime_seconds REAL NOT NULL DEFAULT 0,
            aggregate_cer REAL,
            error_message TEXT,
            FOREIGN KEY (benchmark_id, benchmark_version)
                REFERENCES asr_benchmark(benchmark_id, benchmark_version)
        );

        CREATE TABLE asr_result (
            run_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            hypothesis_raw TEXT,
            reference_normalized TEXT NOT NULL,
            hypothesis_normalized TEXT,
            reference_char_count INTEGER NOT NULL,
            edit_distance INTEGER,
            cer REAL,
            runtime_seconds REAL,
            detected_language TEXT,
            backend_metadata_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, asset_sha256),
            FOREIGN KEY (run_id) REFERENCES asr_run(run_id),
            FOREIGN KEY (asset_sha256)
                REFERENCES transcript_ground_truth(asset_sha256)
        );

        CREATE INDEX idx_asr_result_cer ON asr_result(run_id, cer);
        CREATE INDEX idx_asr_run_started ON asr_run(started_at);

        PRAGMA user_version = 6;
        COMMIT;
    """,
    7: """
        BEGIN IMMEDIATE;

        ALTER TABLE asr_run ADD COLUMN runtime_metadata_json TEXT;
        ALTER TABLE asr_run ADD COLUMN model_license TEXT;
        ALTER TABLE asr_run ADD COLUMN exact_match_count INTEGER;

        PRAGMA user_version = 7;
        COMMIT;
    """,
    8: """
        BEGIN IMMEDIATE;

        CREATE TABLE review_decision (
            decision_id TEXT PRIMARY KEY,
            asset_sha256 TEXT NOT NULL,
            benchmark_id TEXT NOT NULL,
            benchmark_version INTEGER NOT NULL,
            review_round INTEGER NOT NULL,
            text_decision TEXT NOT NULL CHECK (text_decision IN (
                'accept_reference', 'accept_edited', 'flag_defect'
            )),
            text_final TEXT,
            text_note TEXT,
            emotion_primary TEXT,
            emotion_secondary TEXT,
            intensity TEXT CHECK (intensity IN ('low', 'medium', 'high')),
            label_source TEXT NOT NULL CHECK (label_source IN (
                'weak_label_confirmed', 'human_corrected'
            )),
            review_status TEXT NOT NULL CHECK (review_status IN (
                'approved', 'rejected', 'pending'
            )),
            auto_rules_applied_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            export_batch_id TEXT NOT NULL,
            source_row_index INTEGER NOT NULL,
            UNIQUE (asset_sha256, benchmark_id, benchmark_version, review_round),
            UNIQUE (export_batch_id, source_row_index),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (benchmark_id, benchmark_version)
                REFERENCES asr_benchmark(benchmark_id, benchmark_version)
        );

        CREATE INDEX idx_review_decision_asset
            ON review_decision(asset_sha256);

        PRAGMA user_version = 8;
        COMMIT;
    """,
    9: """
        BEGIN IMMEDIATE;

        CREATE TABLE dataset_build_config (
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            config_json TEXT NOT NULL,
            implementation_version INTEGER NOT NULL CHECK (implementation_version >= 1),
            source_path TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, dataset_version),
            UNIQUE (config_sha256)
        );

        CREATE TABLE dataset_analysis_run (
            run_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            candidate_snapshot_sha256 TEXT NOT NULL CHECK (
                length(candidate_snapshot_sha256) = 64
            ),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('running', 'succeeded', 'completed_with_errors', 'failed')
            ),
            candidate_count INTEGER NOT NULL DEFAULT 0 CHECK (candidate_count >= 0),
            feature_count INTEGER NOT NULL DEFAULT 0 CHECK (feature_count >= 0),
            error_count INTEGER NOT NULL DEFAULT 0 CHECK (error_count >= 0),
            error_message TEXT,
            FOREIGN KEY (dataset_id, dataset_version)
                REFERENCES dataset_build_config(dataset_id, dataset_version)
        );

        CREATE TABLE dataset_asset_feature (
            run_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            derived_id TEXT NOT NULL,
            standardized_sha256 TEXT NOT NULL CHECK (
                length(standardized_sha256) = 64
            ),
            duration_seconds REAL NOT NULL CHECK (duration_seconds > 0),
            fingerprint_blob BLOB NOT NULL,
            fingerprint_dimension INTEGER NOT NULL CHECK (
                fingerprint_dimension = 4096
            ),
            fingerprint_sha256 TEXT NOT NULL CHECK (
                length(fingerprint_sha256) = 64
            ),
            speaker_embedding_blob BLOB NOT NULL,
            speaker_embedding_dimension INTEGER NOT NULL CHECK (
                speaker_embedding_dimension = 512
            ),
            speaker_embedding_sha256 TEXT NOT NULL CHECK (
                length(speaker_embedding_sha256) = 64
            ),
            PRIMARY KEY (run_id, asset_sha256),
            FOREIGN KEY (run_id) REFERENCES dataset_analysis_run(run_id),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (derived_id) REFERENCES derived_audio(derived_id)
        );

        CREATE TABLE dataset_similarity_edge (
            run_id TEXT NOT NULL,
            left_asset_sha256 TEXT NOT NULL,
            right_asset_sha256 TEXT NOT NULL,
            evidence_type TEXT NOT NULL CHECK (evidence_type IN (
                'exact_audio', 'exact_text', 'near_audio', 'near_text'
            )),
            score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
            threshold REAL CHECK (threshold >= 0 AND threshold <= 1),
            analysis_status TEXT NOT NULL CHECK (
                analysis_status IN ('accepted_exact', 'pending_review')
            ),
            evidence_json TEXT NOT NULL,
            PRIMARY KEY (
                run_id, left_asset_sha256, right_asset_sha256, evidence_type
            ),
            CHECK (left_asset_sha256 < right_asset_sha256),
            CHECK (
                (evidence_type IN ('exact_audio', 'exact_text')
                 AND analysis_status = 'accepted_exact' AND threshold IS NULL)
                OR
                (evidence_type IN ('near_audio', 'near_text')
                 AND analysis_status = 'pending_review' AND threshold IS NOT NULL)
            ),
            FOREIGN KEY (run_id) REFERENCES dataset_analysis_run(run_id),
            FOREIGN KEY (left_asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (right_asset_sha256) REFERENCES asset(sha256)
        );

        CREATE TABLE dataset_similarity_edge_review (
            review_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            left_asset_sha256 TEXT NOT NULL,
            right_asset_sha256 TEXT NOT NULL,
            evidence_type TEXT NOT NULL CHECK (
                evidence_type IN ('near_audio', 'near_text')
            ),
            review_round INTEGER NOT NULL CHECK (review_round >= 1),
            review_status TEXT NOT NULL CHECK (review_status IN ('accepted', 'rejected')),
            review_note TEXT,
            created_at TEXT NOT NULL,
            review_batch_id TEXT NOT NULL,
            source_row_index INTEGER NOT NULL CHECK (source_row_index >= 0),
            UNIQUE (
                run_id, left_asset_sha256, right_asset_sha256,
                evidence_type, review_round
            ),
            UNIQUE (review_batch_id, source_row_index),
            FOREIGN KEY (
                run_id, left_asset_sha256, right_asset_sha256, evidence_type
            ) REFERENCES dataset_similarity_edge(
                run_id, left_asset_sha256, right_asset_sha256, evidence_type
            )
        );

        CREATE TABLE dataset_similarity_group (
            run_id TEXT NOT NULL,
            group_id TEXT NOT NULL CHECK (length(group_id) = 64),
            member_count INTEGER NOT NULL CHECK (member_count >= 1),
            edge_review_snapshot_sha256 TEXT NOT NULL CHECK (
                length(edge_review_snapshot_sha256) = 64
            ),
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, group_id),
            FOREIGN KEY (run_id) REFERENCES dataset_analysis_run(run_id)
        );

        CREATE TABLE dataset_similarity_group_member (
            run_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            PRIMARY KEY (run_id, group_id, asset_sha256),
            UNIQUE (run_id, asset_sha256),
            FOREIGN KEY (run_id, group_id)
                REFERENCES dataset_similarity_group(run_id, group_id),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256)
        );

        CREATE TABLE dataset_speaker_assessment (
            run_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            center_cosine REAL NOT NULL CHECK (center_cosine >= -1 AND center_cosine <= 1),
            knn_cosine REAL NOT NULL CHECK (knn_cosine >= -1 AND knn_cosine <= 1),
            knn_k INTEGER NOT NULL CHECK (knn_k >= 1),
            center_robust_z REAL NOT NULL,
            knn_robust_z REAL NOT NULL,
            candidate_outlier INTEGER NOT NULL CHECK (candidate_outlier IN (0, 1)),
            evidence_json TEXT NOT NULL,
            PRIMARY KEY (run_id, asset_sha256),
            FOREIGN KEY (run_id, asset_sha256)
                REFERENCES dataset_asset_feature(run_id, asset_sha256)
        );

        CREATE TABLE dataset_speaker_review (
            review_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            review_round INTEGER NOT NULL CHECK (review_round >= 1),
            review_status TEXT NOT NULL CHECK (review_status IN (
                'confirmed_same_speaker',
                'exclude_wrong_speaker',
                'exclude_uncertain'
            )),
            review_note TEXT,
            created_at TEXT NOT NULL,
            review_batch_id TEXT NOT NULL,
            source_row_index INTEGER NOT NULL CHECK (source_row_index >= 0),
            UNIQUE (run_id, asset_sha256, review_round),
            UNIQUE (review_batch_id, source_row_index),
            FOREIGN KEY (run_id, asset_sha256)
                REFERENCES dataset_speaker_assessment(run_id, asset_sha256)
        );

        CREATE TABLE dataset_version (
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            analysis_run_id TEXT NOT NULL,
            candidate_snapshot_sha256 TEXT NOT NULL CHECK (
                length(candidate_snapshot_sha256) = 64
            ),
            edge_review_snapshot_sha256 TEXT NOT NULL CHECK (
                length(edge_review_snapshot_sha256) = 64
            ),
            speaker_review_snapshot_sha256 TEXT NOT NULL CHECK (
                length(speaker_review_snapshot_sha256) = 64
            ),
            item_count INTEGER NOT NULL CHECK (item_count >= 1),
            total_duration_seconds REAL NOT NULL CHECK (total_duration_seconds > 0),
            artifact_set_json TEXT NOT NULL,
            artifact_set_sha256 TEXT NOT NULL CHECK (length(artifact_set_sha256) = 64),
            parquet_path TEXT NOT NULL,
            parquet_sha256 TEXT NOT NULL CHECK (length(parquet_sha256) = 64),
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64),
            checksums_path TEXT NOT NULL,
            checksums_sha256 TEXT NOT NULL CHECK (length(checksums_sha256) = 64),
            published_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, dataset_version),
            FOREIGN KEY (dataset_id, dataset_version)
                REFERENCES dataset_build_config(dataset_id, dataset_version),
            FOREIGN KEY (analysis_run_id) REFERENCES dataset_analysis_run(run_id)
        );

        CREATE TABLE dataset_item (
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
            fid TEXT NOT NULL,
            asset_sha256 TEXT NOT NULL,
            split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
            analysis_run_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            source_root_key TEXT NOT NULL,
            source_path_key TEXT NOT NULL,
            derived_id TEXT NOT NULL,
            audio_relative_path TEXT NOT NULL,
            audio_sha256 TEXT NOT NULL CHECK (length(audio_sha256) = 64),
            quality_run_id TEXT NOT NULL,
            quality_decision TEXT NOT NULL CHECK (quality_decision IN ('pass', 'review')),
            quality_reasons_json TEXT NOT NULL,
            text_exact TEXT NOT NULL CHECK (length(trim(text_exact)) > 0),
            text_sha256 TEXT NOT NULL CHECK (length(text_sha256) = 64),
            text_source TEXT NOT NULL CHECK (text_source IN (
                'human_review', 'filename_candidate_unreviewed'
            )),
            emotion_weak_label TEXT NOT NULL,
            emotion_primary TEXT NOT NULL,
            emotion_secondary TEXT,
            intensity TEXT CHECK (intensity IN ('low', 'medium', 'high')),
            label_source TEXT NOT NULL CHECK (label_source IN (
                'weak_label_confirmed', 'human_corrected', 'weak_label_unreviewed'
            )),
            review_decision_id TEXT,
            review_round INTEGER,
            review_batch_id TEXT,
            lineage_json TEXT NOT NULL,
            PRIMARY KEY (dataset_id, dataset_version, asset_sha256),
            UNIQUE (dataset_id, dataset_version, ordinal),
            UNIQUE (dataset_id, dataset_version, fid),
            CHECK (
                (text_source = 'human_review'
                 AND review_decision_id IS NOT NULL
                 AND review_round IS NOT NULL
                 AND review_batch_id IS NOT NULL
                 AND label_source IN ('weak_label_confirmed', 'human_corrected'))
                OR
                (text_source = 'filename_candidate_unreviewed'
                 AND review_decision_id IS NULL
                 AND review_round IS NULL
                 AND review_batch_id IS NULL
                 AND label_source = 'weak_label_unreviewed')
            ),
            FOREIGN KEY (dataset_id, dataset_version)
                REFERENCES dataset_version(dataset_id, dataset_version),
            FOREIGN KEY (asset_sha256) REFERENCES asset(sha256),
            FOREIGN KEY (source_root_key, source_path_key)
                REFERENCES source_location(root_key, path_key),
            FOREIGN KEY (derived_id) REFERENCES derived_audio(derived_id),
            FOREIGN KEY (quality_run_id) REFERENCES quality_run(run_id),
            FOREIGN KEY (review_decision_id) REFERENCES review_decision(decision_id),
            FOREIGN KEY (analysis_run_id, group_id)
                REFERENCES dataset_similarity_group(run_id, group_id)
        );

        CREATE INDEX idx_dataset_analysis_config_started
            ON dataset_analysis_run(dataset_id, dataset_version, started_at);
        CREATE INDEX idx_dataset_analysis_status
            ON dataset_analysis_run(status, started_at);
        CREATE INDEX idx_dataset_asset_feature_asset
            ON dataset_asset_feature(asset_sha256);
        CREATE INDEX idx_dataset_asset_feature_derived
            ON dataset_asset_feature(derived_id);
        CREATE INDEX idx_dataset_edge_status
            ON dataset_similarity_edge(run_id, evidence_type, analysis_status);
        CREATE INDEX idx_dataset_edge_left
            ON dataset_similarity_edge(run_id, left_asset_sha256);
        CREATE INDEX idx_dataset_edge_right
            ON dataset_similarity_edge(run_id, right_asset_sha256);
        CREATE INDEX idx_dataset_edge_review_latest
            ON dataset_similarity_edge_review(
                run_id, left_asset_sha256, right_asset_sha256,
                evidence_type, review_round DESC
            );
        CREATE INDEX idx_dataset_group_member_asset
            ON dataset_similarity_group_member(run_id, asset_sha256);
        CREATE INDEX idx_dataset_speaker_outlier
            ON dataset_speaker_assessment(run_id, candidate_outlier, center_cosine);
        CREATE INDEX idx_dataset_speaker_review_latest
            ON dataset_speaker_review(run_id, asset_sha256, review_round DESC);
        CREATE INDEX idx_dataset_item_split
            ON dataset_item(dataset_id, dataset_version, split, ordinal);
        CREATE INDEX idx_dataset_item_group
            ON dataset_item(analysis_run_id, group_id);
        CREATE INDEX idx_dataset_item_sources
            ON dataset_item(dataset_id, dataset_version, text_source, label_source);
        CREATE INDEX idx_dataset_item_quality
            ON dataset_item(dataset_id, dataset_version, quality_decision);

        PRAGMA user_version = 9;
        COMMIT;
    """,
    10: """
        BEGIN IMMEDIATE;

        CREATE TABLE dataset_threshold_calibration (
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            calibration_id TEXT NOT NULL,
            calibration_version INTEGER NOT NULL CHECK (calibration_version >= 1),
            calibration_config_sha256 TEXT NOT NULL CHECK (
                length(calibration_config_sha256) = 64
            ),
            calibration_report_sha256 TEXT NOT NULL CHECK (
                length(calibration_report_sha256) = 64
            ),
            report_path TEXT NOT NULL,
            thresholds_json TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (
                dataset_id, dataset_version, calibration_id, calibration_version
            ),
            UNIQUE (
                dataset_id, dataset_version, calibration_id, calibration_version,
                calibration_config_sha256, calibration_report_sha256
            ),
            FOREIGN KEY (dataset_id, dataset_version)
                REFERENCES dataset_build_config(dataset_id, dataset_version)
        );

        CREATE TABLE dataset_analysis_calibration (
            run_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            calibration_id TEXT NOT NULL,
            calibration_version INTEGER NOT NULL,
            calibration_config_sha256 TEXT NOT NULL CHECK (
                length(calibration_config_sha256) = 64
            ),
            calibration_report_sha256 TEXT NOT NULL CHECK (
                length(calibration_report_sha256) = 64
            ),
            FOREIGN KEY (run_id) REFERENCES dataset_analysis_run(run_id),
            FOREIGN KEY (
                dataset_id, dataset_version, calibration_id, calibration_version,
                calibration_config_sha256, calibration_report_sha256
            ) REFERENCES dataset_threshold_calibration(
                dataset_id, dataset_version, calibration_id, calibration_version,
                calibration_config_sha256, calibration_report_sha256
            )
        );

        CREATE INDEX idx_dataset_calibration_identity
            ON dataset_threshold_calibration(
                calibration_id, calibration_version, calibration_report_sha256
            );
        CREATE INDEX idx_dataset_analysis_calibration_identity
            ON dataset_analysis_calibration(
                dataset_id, dataset_version, calibration_id, calibration_version
            );

        PRAGMA user_version = 10;
        COMMIT;
    """,
}


class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def connect_read_only(self) -> sqlite3.Connection:
        resolved = self.path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Catalog does not exist: {resolved}")
        connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro",
            uri=True,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def read_only_session(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect_read_only()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.session() as connection:
            current_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    "Catalog schema is newer than this application: "
                    f"database={current_version}, supported={SCHEMA_VERSION}."
                )
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                migration = _MIGRATIONS.get(version)
                if migration is None:
                    raise RuntimeError(f"Missing catalog migration {version}.")
                connection.executescript(migration)
                migrated_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                if migrated_version != version:
                    raise RuntimeError(
                        f"Migration {version} did not update PRAGMA user_version."
                    )

    def register_metadata_profile(
        self,
        *,
        profile_id: str,
        profile_version: int,
        config_sha256: str,
        config_json: str,
        source_path: str,
        registered_at: str,
    ) -> None:
        with self.session() as connection:
            existing = connection.execute(
                """
                SELECT config_sha256 FROM metadata_profile
                WHERE profile_id = ? AND profile_version = ?
                """,
                (profile_id, profile_version),
            ).fetchone()
            if existing is not None and existing["config_sha256"] != config_sha256:
                raise RuntimeError(
                    "Metadata profile content changed without a version bump: "
                    f"{profile_id}@{profile_version}; registered "
                    f"{existing['config_sha256']}, supplied {config_sha256}."
                )
            connection.execute(
                """
                INSERT INTO metadata_profile (
                    profile_id, profile_version, config_sha256, config_json,
                    source_path, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, profile_version) DO UPDATE SET
                    source_path = excluded.source_path
                """,
                (
                    profile_id,
                    profile_version,
                    config_sha256,
                    config_json,
                    source_path,
                    registered_at,
                ),
            )

    def begin_run(self, *, root_path: str, root_key: str, started_at: str) -> str:
        run_id = str(uuid.uuid4())
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO ingest_run (
                    run_id, root_path, root_key, started_at, status
                ) VALUES (?, ?, ?, ?, 'running')
                """,
                (run_id, root_path, root_key, started_at),
            )
        return run_id

    def load_locations(self, *, root_key: str) -> dict[str, dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_location
                WHERE root_key = ?
                """,
                (root_key,),
            ).fetchall()
        return {str(row["path_key"]): dict(row) for row in rows}

    def complete_run(
        self,
        *,
        run_id: str,
        root_path: str,
        root_key: str,
        finished_at: str,
        records: Iterable[dict[str, Any]],
        missing_path_keys: Iterable[str],
        summary: dict[str, Any],
    ) -> None:
        record_list = list(records)
        missing_keys = list(missing_path_keys)
        with self.session() as connection:
            for record in record_list:
                sha256 = record.get("sha256")
                if sha256:
                    connection.execute(
                        """
                        INSERT INTO asset (
                            sha256, size_bytes, format, subtype, sample_rate,
                            channels, frames, duration_seconds, probe_status,
                            probe_error_type, probe_error_message,
                            first_seen_at, last_seen_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(sha256) DO UPDATE SET
                            size_bytes = excluded.size_bytes,
                            format = excluded.format,
                            subtype = excluded.subtype,
                            sample_rate = excluded.sample_rate,
                            channels = excluded.channels,
                            frames = excluded.frames,
                            duration_seconds = excluded.duration_seconds,
                            probe_status = excluded.probe_status,
                            probe_error_type = excluded.probe_error_type,
                            probe_error_message = excluded.probe_error_message,
                            last_seen_at = excluded.last_seen_at
                        """,
                        (
                            sha256,
                            record.get("size_bytes"),
                            record.get("format"),
                            record.get("subtype"),
                            record.get("sample_rate"),
                            record.get("channels"),
                            record.get("frames"),
                            record.get("duration_seconds"),
                            record.get("probe_status"),
                            record.get("probe_error_type"),
                            record.get("probe_error_message"),
                            finished_at,
                            finished_at,
                        ),
                    )

                connection.execute(
                    """
                    INSERT INTO source_location (
                        root_key, root_path, path_key, relative_path,
                        asset_sha256, size_bytes, mtime_ns, path_length,
                        speaker_id, emotion_weak_label, transcript_candidate,
                        directory_emotion_weak_label,
                        filename_emotion_weak_label, metadata_status,
                        metadata_review_reason, parser_profile_id,
                        parser_profile_version, parser_config_sha256,
                        parse_status, parse_error, availability_status,
                        first_seen_at, last_seen_at, last_seen_run_id,
                        missing_since_run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, 'available', ?, ?, ?, NULL)
                    ON CONFLICT(root_key, path_key) DO UPDATE SET
                        root_path = excluded.root_path,
                        relative_path = excluded.relative_path,
                        asset_sha256 = excluded.asset_sha256,
                        size_bytes = excluded.size_bytes,
                        mtime_ns = excluded.mtime_ns,
                        path_length = excluded.path_length,
                        speaker_id = excluded.speaker_id,
                        emotion_weak_label = excluded.emotion_weak_label,
                        transcript_candidate = excluded.transcript_candidate,
                        directory_emotion_weak_label =
                            excluded.directory_emotion_weak_label,
                        filename_emotion_weak_label =
                            excluded.filename_emotion_weak_label,
                        metadata_status = excluded.metadata_status,
                        metadata_review_reason = excluded.metadata_review_reason,
                        parser_profile_id = excluded.parser_profile_id,
                        parser_profile_version = excluded.parser_profile_version,
                        parser_config_sha256 = excluded.parser_config_sha256,
                        parse_status = excluded.parse_status,
                        parse_error = excluded.parse_error,
                        availability_status = 'available',
                        last_seen_at = excluded.last_seen_at,
                        last_seen_run_id = excluded.last_seen_run_id,
                        missing_since_run_id = NULL
                    """,
                    (
                        root_key,
                        root_path,
                        record["path_key"],
                        record["relative_path"],
                        sha256,
                        record.get("size_bytes"),
                        record.get("mtime_ns"),
                        record["path_length"],
                        record.get("speaker_id"),
                        record.get("emotion_weak_label"),
                        record.get("transcript_candidate"),
                        record.get("directory_emotion_weak_label"),
                        record.get("filename_emotion_weak_label"),
                        record["metadata_status"],
                        record.get("metadata_review_reason"),
                        record["parser_profile_id"],
                        record["parser_profile_version"],
                        record["parser_config_sha256"],
                        record["parse_status"],
                        record.get("parse_error"),
                        finished_at,
                        finished_at,
                        run_id,
                    ),
                )

                connection.execute(
                    """
                    INSERT INTO metadata_parse (
                        root_key, path_key, profile_id, profile_version,
                        config_sha256, parsed_at, speaker_id,
                        directory_emotion_weak_label,
                        filename_emotion_weak_label, emotion_weak_label,
                        transcript_candidate, status, error_message,
                        review_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        root_key, path_key, profile_id, profile_version,
                        config_sha256
                    ) DO UPDATE SET
                        parsed_at = excluded.parsed_at,
                        speaker_id = excluded.speaker_id,
                        directory_emotion_weak_label =
                            excluded.directory_emotion_weak_label,
                        filename_emotion_weak_label =
                            excluded.filename_emotion_weak_label,
                        emotion_weak_label = excluded.emotion_weak_label,
                        transcript_candidate = excluded.transcript_candidate,
                        status = excluded.status,
                        error_message = excluded.error_message,
                        review_reason = excluded.review_reason
                    """,
                    (
                        root_key,
                        record["path_key"],
                        record["parser_profile_id"],
                        record["parser_profile_version"],
                        record["parser_config_sha256"],
                        finished_at,
                        record.get("speaker_id"),
                        record.get("directory_emotion_weak_label"),
                        record.get("filename_emotion_weak_label"),
                        record.get("emotion_weak_label"),
                        record.get("transcript_candidate"),
                        record["metadata_status"],
                        record.get("parse_error"),
                        record.get("metadata_review_reason"),
                    ),
                )

            for path_key in missing_keys:
                connection.execute(
                    """
                    UPDATE source_location
                    SET availability_status = 'missing',
                        missing_since_run_id = COALESCE(missing_since_run_id, ?)
                    WHERE root_key = ? AND path_key = ?
                    """,
                    (run_id, root_key, path_key),
                )

            connection.execute(
                """
                UPDATE ingest_run
                SET finished_at = ?, status = ?,
                    discovered_count = ?, readable_count = ?, error_count = ?,
                    added_count = ?, changed_count = ?, moved_count = ?,
                    missing_count = ?, unchanged_count = ?, error_message = NULL
                WHERE run_id = ?
                """,
                (
                    finished_at,
                    summary["status"],
                    summary["discovered_count"],
                    summary["readable_count"],
                    summary["error_count"],
                    summary["added_count"],
                    summary["changed_count"],
                    summary["moved_count"],
                    summary["missing_count"],
                    summary["unchanged_count"],
                    run_id,
                ),
            )

    def fail_run(self, *, run_id: str, finished_at: str, error_message: str) -> None:
        with self.session() as connection:
            connection.execute(
                """
                UPDATE ingest_run
                SET finished_at = ?, status = 'failed', error_message = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_at, error_message, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_run WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def load_import_candidates(self, *, root_key: str) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT source_location.*, asset.size_bytes AS asset_size_bytes
                FROM source_location
                JOIN asset ON asset.sha256 = source_location.asset_sha256
                WHERE source_location.root_key = ?
                  AND source_location.availability_status = 'available'
                  AND source_location.asset_sha256 IS NOT NULL
                ORDER BY source_location.path_key
                """,
                (root_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_raw_objects(self) -> dict[str, dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute("SELECT * FROM raw_object").fetchall()
        return {str(row["asset_sha256"]): dict(row) for row in rows}

    def begin_import_run(
        self,
        *,
        root_path: str,
        root_key: str,
        raw_root_path: str,
        started_at: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO import_run (
                    run_id, root_path, root_key, raw_root_path, started_at, status
                ) VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (run_id, root_path, root_key, raw_root_path, started_at),
            )
        return run_id

    def complete_import_run(
        self,
        *,
        run_id: str,
        finished_at: str,
        raw_objects: Iterable[dict[str, Any]],
        items: Iterable[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        with self.session() as connection:
            for raw_object in raw_objects:
                connection.execute(
                    """
                    INSERT INTO raw_object (
                        asset_sha256, relative_path, extension, storage_mode,
                        size_bytes, first_imported_at, last_verified_at,
                        first_import_run_id, last_import_run_id
                    ) VALUES (?, ?, ?, 'copy', ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_sha256) DO UPDATE SET
                        relative_path = excluded.relative_path,
                        extension = excluded.extension,
                        size_bytes = excluded.size_bytes,
                        last_verified_at = excluded.last_verified_at,
                        last_import_run_id = excluded.last_import_run_id
                    """,
                    (
                        raw_object["asset_sha256"],
                        raw_object["relative_path"],
                        raw_object["extension"],
                        raw_object["size_bytes"],
                        finished_at,
                        finished_at,
                        run_id,
                        run_id,
                    ),
                )

            for item in items:
                connection.execute(
                    """
                    INSERT INTO import_item (
                        run_id, root_key, path_key, asset_sha256,
                        source_relative_path, original_name, raw_relative_path,
                        action, error_type, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        item["root_key"],
                        item["path_key"],
                        item["asset_sha256"],
                        item["source_relative_path"],
                        item["original_name"],
                        item.get("raw_relative_path"),
                        item["action"],
                        item.get("error_type"),
                        item.get("error_message"),
                    ),
                )

            connection.execute(
                """
                UPDATE import_run
                SET finished_at = ?, status = ?, discovered_count = ?,
                    unique_asset_count = ?, imported_count = ?, reused_count = ?,
                    error_count = ?, bytes_written = ?, error_message = NULL
                WHERE run_id = ?
                """,
                (
                    finished_at,
                    summary["status"],
                    summary["discovered_count"],
                    summary["unique_asset_count"],
                    summary["imported_count"],
                    summary["reused_count"],
                    summary["error_count"],
                    summary["bytes_written"],
                    run_id,
                ),
            )

    def fail_import_run(
        self, *, run_id: str, finished_at: str, error_message: str
    ) -> None:
        with self.session() as connection:
            connection.execute(
                """
                UPDATE import_run
                SET finished_at = ?, status = 'failed', error_message = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_at, error_message, run_id),
            )

    def get_import_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute(
                "SELECT * FROM import_run WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def _register_quality_config(
        self,
        *,
        table: str,
        id_column: str,
        version_column: str,
        config_id: str,
        config_version: int,
        config_sha256: str,
        config_json: str,
        source_path: str,
        registered_at: str,
    ) -> None:
        if table not in {"quality_analysis_config", "quality_policy"}:
            raise ValueError(f"Unsupported quality config table: {table}")
        with self.session() as connection:
            existing = connection.execute(
                f"""
                SELECT config_sha256 FROM {table}
                WHERE {id_column} = ? AND {version_column} = ?
                """,
                (config_id, config_version),
            ).fetchone()
            if existing is not None and existing["config_sha256"] != config_sha256:
                raise RuntimeError(
                    "Quality configuration changed without a version bump: "
                    f"{config_id}@{config_version}; registered "
                    f"{existing['config_sha256']}, supplied {config_sha256}."
                )
            connection.execute(
                f"""
                INSERT INTO {table} (
                    {id_column}, {version_column}, config_sha256, config_json,
                    source_path, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT({id_column}, {version_column}) DO UPDATE SET
                    source_path = excluded.source_path
                """,
                (
                    config_id,
                    config_version,
                    config_sha256,
                    config_json,
                    source_path,
                    registered_at,
                ),
            )

    def register_quality_analysis_config(
        self,
        *,
        analysis_id: str,
        analysis_version: int,
        config_sha256: str,
        config_json: str,
        source_path: str,
        registered_at: str,
    ) -> None:
        self._register_quality_config(
            table="quality_analysis_config",
            id_column="analysis_id",
            version_column="analysis_version",
            config_id=analysis_id,
            config_version=analysis_version,
            config_sha256=config_sha256,
            config_json=config_json,
            source_path=source_path,
            registered_at=registered_at,
        )

    def register_quality_policy(
        self,
        *,
        policy_id: str,
        policy_version: int,
        config_sha256: str,
        config_json: str,
        source_path: str,
        registered_at: str,
    ) -> None:
        self._register_quality_config(
            table="quality_policy",
            id_column="policy_id",
            version_column="policy_version",
            config_id=policy_id,
            config_version=policy_version,
            config_sha256=config_sha256,
            config_json=config_json,
            source_path=source_path,
            registered_at=registered_at,
        )

    def load_quality_candidates(self) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT raw_object.asset_sha256, raw_object.relative_path,
                       raw_object.size_bytes, asset.subtype,
                       source_location.relative_path AS source_relative_path,
                       source_location.speaker_id,
                       source_location.emotion_weak_label,
                       source_location.transcript_candidate
                FROM raw_object
                JOIN asset ON asset.sha256 = raw_object.asset_sha256
                LEFT JOIN source_location ON source_location.rowid = (
                    SELECT candidate.rowid FROM source_location AS candidate
                    WHERE candidate.asset_sha256 = raw_object.asset_sha256
                      AND candidate.availability_status = 'available'
                    ORDER BY candidate.root_key, candidate.path_key
                    LIMIT 1
                )
                ORDER BY raw_object.asset_sha256
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def load_quality_metrics(
        self,
        *,
        analysis_id: str,
        analysis_version: int,
        analysis_config_sha256: str,
        implementation_version: int,
    ) -> dict[str, dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM audio_quality_metric
                WHERE analysis_id = ? AND analysis_version = ?
                  AND analysis_config_sha256 = ?
                  AND implementation_version = ?
                """,
                (
                    analysis_id,
                    analysis_version,
                    analysis_config_sha256,
                    implementation_version,
                ),
            ).fetchall()
        return {str(row["asset_sha256"]): dict(row) for row in rows}

    def begin_quality_run(
        self,
        *,
        raw_root_path: str,
        analysis_id: str,
        analysis_version: int,
        analysis_config_sha256: str,
        implementation_version: int,
        policy_id: str,
        policy_version: int,
        policy_config_sha256: str,
        started_at: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO quality_run (
                    run_id, raw_root_path, analysis_id, analysis_version,
                    analysis_config_sha256, implementation_version,
                    policy_id, policy_version, policy_config_sha256,
                    started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    raw_root_path,
                    analysis_id,
                    analysis_version,
                    analysis_config_sha256,
                    implementation_version,
                    policy_id,
                    policy_version,
                    policy_config_sha256,
                    started_at,
                ),
            )
        return run_id

    def complete_quality_run(
        self,
        *,
        run_id: str,
        finished_at: str,
        metrics_to_store: Iterable[dict[str, Any]],
        assessments: Iterable[dict[str, Any]],
        items: Iterable[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        metric_columns = (
            "asset_sha256",
            "analysis_id",
            "analysis_version",
            "analysis_config_sha256",
            "implementation_version",
            "analyzed_at",
            "status",
            "error_type",
            "error_message",
            "sample_rate",
            "channels",
            "frames",
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
        )
        primary_columns = metric_columns[:5]
        update_columns = metric_columns[5:]
        placeholders = ", ".join("?" for _ in metric_columns)
        update_assignments = ", ".join(
            f"{column} = excluded.{column}" for column in update_columns
        )
        with self.session() as connection:
            for metric in metrics_to_store:
                connection.execute(
                    f"""
                    INSERT INTO audio_quality_metric ({', '.join(metric_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT({', '.join(primary_columns)}) DO UPDATE SET
                        {update_assignments}
                    """,
                    tuple(metric.get(column) for column in metric_columns),
                )

            for assessment in assessments:
                connection.execute(
                    """
                    INSERT INTO audio_quality_assessment (
                        asset_sha256, analysis_id, analysis_version,
                        analysis_config_sha256, implementation_version,
                        policy_id, policy_version, policy_config_sha256,
                        assessed_at, decision, reasons_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        asset_sha256, analysis_id, analysis_version,
                        analysis_config_sha256, implementation_version,
                        policy_id, policy_version, policy_config_sha256
                    ) DO UPDATE SET
                        assessed_at = excluded.assessed_at,
                        decision = excluded.decision,
                        reasons_json = excluded.reasons_json
                    """,
                    (
                        assessment["asset_sha256"],
                        assessment["analysis_id"],
                        assessment["analysis_version"],
                        assessment["analysis_config_sha256"],
                        assessment["implementation_version"],
                        assessment["policy_id"],
                        assessment["policy_version"],
                        assessment["policy_config_sha256"],
                        assessment["assessed_at"],
                        assessment["decision"],
                        assessment["reasons_json"],
                    ),
                )

            for item in items:
                connection.execute(
                    """
                    INSERT INTO quality_run_item (
                        run_id, asset_sha256, raw_relative_path,
                        metric_action, decision, reasons_json,
                        error_type, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        item["asset_sha256"],
                        item["raw_relative_path"],
                        item["metric_action"],
                        item["decision"],
                        item["reasons_json"],
                        item.get("error_type"),
                        item.get("error_message"),
                    ),
                )

            connection.execute(
                """
                UPDATE quality_run
                SET finished_at = ?, status = ?, discovered_count = ?,
                    analyzed_count = ?, cached_count = ?, pass_count = ?,
                    review_count = ?, reject_count = ?, error_count = ?,
                    error_message = NULL
                WHERE run_id = ?
                """,
                (
                    finished_at,
                    summary["status"],
                    summary["discovered_count"],
                    summary["analyzed_count"],
                    summary["cached_count"],
                    summary["pass_count"],
                    summary["review_count"],
                    summary["reject_count"],
                    summary["error_count"],
                    run_id,
                ),
            )

    def fail_quality_run(
        self, *, run_id: str, finished_at: str, error_message: str
    ) -> None:
        with self.session() as connection:
            connection.execute(
                """
                UPDATE quality_run
                SET finished_at = ?, status = 'failed', error_message = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_at, error_message, run_id),
            )

    def get_quality_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute(
                "SELECT * FROM quality_run WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def register_standardization_config(
        self,
        *,
        config_id: str,
        config_version: int,
        config_sha256: str,
        config_json: str,
        source_path: str,
        registered_at: str,
    ) -> None:
        with self.session() as connection:
            existing = connection.execute(
                """
                SELECT config_sha256 FROM standardization_config
                WHERE config_id = ? AND config_version = ?
                """,
                (config_id, config_version),
            ).fetchone()
            if existing is not None and existing["config_sha256"] != config_sha256:
                raise RuntimeError(
                    "Standardization config content changed without a version bump: "
                    f"{config_id}@{config_version}; registered "
                    f"{existing['config_sha256']}, supplied {config_sha256}."
                )
            connection.execute(
                """
                INSERT INTO standardization_config (
                    config_id, config_version, config_sha256, config_json,
                    source_path, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_id, config_version) DO UPDATE SET
                    source_path = excluded.source_path
                """,
                (
                    config_id,
                    config_version,
                    config_sha256,
                    config_json,
                    source_path,
                    registered_at,
                ),
            )

    def load_standardization_candidates(self) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT raw_object.asset_sha256, raw_object.relative_path,
                       raw_object.size_bytes, asset.sample_rate,
                       asset.channels, asset.frames, asset.subtype,
                       source_location.relative_path AS source_relative_path,
                       source_location.speaker_id,
                       source_location.emotion_weak_label,
                       source_location.transcript_candidate
                FROM raw_object
                JOIN asset ON asset.sha256 = raw_object.asset_sha256
                LEFT JOIN source_location ON source_location.rowid = (
                    SELECT candidate.rowid FROM source_location AS candidate
                    WHERE candidate.asset_sha256 = raw_object.asset_sha256
                      AND candidate.availability_status = 'available'
                    ORDER BY candidate.root_key, candidate.path_key
                    LIMIT 1
                )
                ORDER BY raw_object.asset_sha256
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def load_derived_audio(
        self,
        *,
        config_id: str,
        config_version: int,
        config_sha256: str,
        implementation_version: int,
    ) -> dict[str, dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM derived_audio
                WHERE config_id = ? AND config_version = ?
                  AND config_sha256 = ? AND implementation_version = ?
                """,
                (
                    config_id,
                    config_version,
                    config_sha256,
                    implementation_version,
                ),
            ).fetchall()
        return {str(row["asset_sha256"]): dict(row) for row in rows}

    def begin_standardization_run(
        self,
        *,
        raw_root_path: str,
        output_root_path: str,
        config_id: str,
        config_version: int,
        config_sha256: str,
        implementation_version: int,
        started_at: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO standardization_run (
                    run_id, raw_root_path, output_root_path,
                    config_id, config_version, config_sha256,
                    implementation_version, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    raw_root_path,
                    output_root_path,
                    config_id,
                    config_version,
                    config_sha256,
                    implementation_version,
                    started_at,
                ),
            )
        return run_id

    def complete_standardization_run(
        self,
        *,
        run_id: str,
        finished_at: str,
        derived_to_store: Iterable[dict[str, Any]],
        items: Iterable[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        derived_columns = (
            "derived_id",
            "asset_sha256",
            "config_id",
            "config_version",
            "config_sha256",
            "implementation_version",
            "relative_path",
            "output_sha256",
            "size_bytes",
            "created_at",
            "last_verified_at",
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
            "output_subtype",
            "created_run_id",
        )
        placeholders = ", ".join("?" for _ in derived_columns)
        update_columns = derived_columns[6:]
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in update_columns
        )
        with self.session() as connection:
            for derived in derived_to_store:
                connection.execute(
                    f"""
                    INSERT INTO derived_audio ({', '.join(derived_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(derived_id) DO UPDATE SET {updates}
                    """,
                    tuple(derived.get(column) for column in derived_columns),
                )
            for item in items:
                connection.execute(
                    """
                    INSERT INTO standardization_run_item (
                        run_id, asset_sha256, derived_id, raw_relative_path,
                        derived_relative_path, action, error_type, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        item["asset_sha256"],
                        item["derived_id"],
                        item["raw_relative_path"],
                        item.get("derived_relative_path"),
                        item["action"],
                        item.get("error_type"),
                        item.get("error_message"),
                    ),
                )
            connection.execute(
                """
                UPDATE standardization_run
                SET finished_at = ?, status = ?, discovered_count = ?,
                    built_count = ?, cached_count = ?, rebuilt_count = ?,
                    trimmed_count = ?, attenuated_count = ?, error_count = ?,
                    error_message = NULL
                WHERE run_id = ?
                """,
                (
                    finished_at,
                    summary["status"],
                    summary["discovered_count"],
                    summary["built_count"],
                    summary["cached_count"],
                    summary["rebuilt_count"],
                    summary["trimmed_count"],
                    summary["attenuated_count"],
                    summary["error_count"],
                    run_id,
                ),
            )

    def fail_standardization_run(
        self, *, run_id: str, finished_at: str, error_message: str
    ) -> None:
        with self.session() as connection:
            connection.execute(
                """
                UPDATE standardization_run
                SET finished_at = ?, status = 'failed', error_message = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_at, error_message, run_id),
            )

    def register_filename_ground_truths(
        self, records: Iterable[dict[str, Any]]
    ) -> tuple[int, int]:
        registered = 0
        cached = 0
        with self.session() as connection:
            for record in records:
                existing = connection.execute(
                    """
                    SELECT text_exact, text_sha256 FROM transcript_ground_truth
                    WHERE asset_sha256 = ?
                    """,
                    (record["asset_sha256"],),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["text_exact"] != record["text_exact"]
                        or existing["text_sha256"] != record["text_sha256"]
                    ):
                        raise RuntimeError(
                            "Confirmed transcript changed for existing asset "
                            f"{record['asset_sha256']}; explicit correction history "
                            "is required instead of silent replacement."
                        )
                    cached += 1
                    continue
                connection.execute(
                    """
                    INSERT INTO transcript_ground_truth (
                        asset_sha256, text_exact, text_sha256,
                        provenance_kind, provenance_note, confirmed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["asset_sha256"],
                        record["text_exact"],
                        record["text_sha256"],
                        record["provenance_kind"],
                        record["provenance_note"],
                        record["confirmed_at"],
                    ),
                )
                registered += 1
        return registered, cached

    def load_asr_preparation_candidates(
        self,
        *,
        standardization_config_id: str,
        standardization_config_version: int,
    ) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT raw_object.asset_sha256,
                       source_location.relative_path AS source_relative_path,
                       source_location.speaker_id,
                       source_location.emotion_weak_label,
                       source_location.transcript_candidate,
                       asset.duration_seconds AS source_duration_seconds,
                       asset.sample_rate AS source_sample_rate,
                       derived_audio.derived_id,
                       derived_audio.relative_path AS derived_relative_path,
                       derived_audio.duration_seconds AS derived_duration_seconds,
                       derived_audio.output_sha256,
                       transcript_ground_truth.text_exact,
                       transcript_ground_truth.text_sha256
                FROM raw_object
                JOIN asset ON asset.sha256 = raw_object.asset_sha256
                JOIN derived_audio
                  ON derived_audio.asset_sha256 = raw_object.asset_sha256
                 AND derived_audio.config_id = ?
                 AND derived_audio.config_version = ?
                LEFT JOIN source_location ON source_location.rowid = (
                    SELECT candidate.rowid FROM source_location AS candidate
                    WHERE candidate.asset_sha256 = raw_object.asset_sha256
                      AND candidate.availability_status = 'available'
                    ORDER BY candidate.root_key, candidate.path_key
                    LIMIT 1
                )
                LEFT JOIN transcript_ground_truth
                  ON transcript_ground_truth.asset_sha256 = raw_object.asset_sha256
                ORDER BY raw_object.asset_sha256
                """,
                (standardization_config_id, standardization_config_version),
            ).fetchall()
        return [dict(row) for row in rows]

    def register_asr_benchmark(
        self,
        *,
        benchmark: dict[str, Any],
        items: Iterable[dict[str, Any]],
    ) -> None:
        with self.session() as connection:
            existing = connection.execute(
                """
                SELECT config_sha256, manifest_sha256 FROM asr_benchmark
                WHERE benchmark_id = ? AND benchmark_version = ?
                """,
                (benchmark["benchmark_id"], benchmark["benchmark_version"]),
            ).fetchone()
            if existing is not None and (
                existing["config_sha256"] != benchmark["config_sha256"]
                or existing["manifest_sha256"] != benchmark["manifest_sha256"]
            ):
                raise RuntimeError(
                    "ASR benchmark changed without a version bump: "
                    f"{benchmark['benchmark_id']}@{benchmark['benchmark_version']}."
                )
            connection.execute(
                """
                INSERT INTO asr_benchmark (
                    benchmark_id, benchmark_version, config_sha256, config_json,
                    manifest_path, manifest_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(benchmark_id, benchmark_version) DO UPDATE SET
                    manifest_path = excluded.manifest_path
                """,
                (
                    benchmark["benchmark_id"],
                    benchmark["benchmark_version"],
                    benchmark["config_sha256"],
                    benchmark["config_json"],
                    benchmark["manifest_path"],
                    benchmark["manifest_sha256"],
                    benchmark["created_at"],
                ),
            )
            if existing is None:
                for item in items:
                    connection.execute(
                        """
                        INSERT INTO asr_benchmark_item (
                            benchmark_id, benchmark_version, ordinal,
                            asset_sha256, text_sha256, derived_id,
                            derived_relative_path, emotion_weak_label,
                            selection_reasons_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            benchmark["benchmark_id"],
                            benchmark["benchmark_version"],
                            item["ordinal"],
                            item["asset_sha256"],
                            item["text_sha256"],
                            item["derived_id"],
                            item["derived_relative_path"],
                            item["emotion_weak_label"],
                            item["selection_reasons_json"],
                        ),
                    )

    def load_asr_benchmark(
        self, *, benchmark_id: str, benchmark_version: int
    ) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute(
                """
                SELECT * FROM asr_benchmark
                WHERE benchmark_id = ? AND benchmark_version = ?
                """,
                (benchmark_id, benchmark_version),
            ).fetchone()
        return dict(row) if row is not None else None

    def assert_asr_benchmark_compatible(
        self,
        *,
        benchmark_id: str,
        benchmark_version: int,
        config_sha256: str,
        manifest_sha256: str,
    ) -> None:
        """Reject same-version drift before canonical report files are touched."""
        existing = self.load_asr_benchmark(
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
        )
        if existing is not None and (
            existing["config_sha256"] != config_sha256
            or existing["manifest_sha256"] != manifest_sha256
        ):
            raise RuntimeError(
                "ASR benchmark changed without a version bump: "
                f"{benchmark_id}@{benchmark_version}."
            )

    def load_asr_benchmark_items(
        self, *, benchmark_id: str, benchmark_version: int
    ) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT item.*, truth.text_exact,
                       derived.duration_seconds, derived.output_sha256
                FROM asr_benchmark_item AS item
                JOIN transcript_ground_truth AS truth
                  ON truth.asset_sha256 = item.asset_sha256
                JOIN derived_audio AS derived
                  ON derived.derived_id = item.derived_id
                WHERE item.benchmark_id = ? AND item.benchmark_version = ?
                ORDER BY item.ordinal
                """,
                (benchmark_id, benchmark_version),
            ).fetchall()
        return [dict(row) for row in rows]

    def begin_asr_run(
        self,
        *,
        benchmark_id: str,
        benchmark_version: int,
        backend_id: str,
        backend_version: str,
        model_id: str,
        model_revision: str,
        inference_config_sha256: str,
        inference_config_json: str,
        started_at: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO asr_run (
                    run_id, benchmark_id, benchmark_version,
                    backend_id, backend_version, model_id, model_revision,
                    inference_config_sha256, inference_config_json,
                    started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    benchmark_id,
                    benchmark_version,
                    backend_id,
                    backend_version,
                    model_id,
                    model_revision,
                    inference_config_sha256,
                    inference_config_json,
                    started_at,
                ),
            )
        return run_id

    def complete_asr_run(
        self,
        *,
        run_id: str,
        finished_at: str,
        results: Iterable[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        with self.session() as connection:
            for result in results:
                connection.execute(
                    """
                    INSERT INTO asr_result (
                        run_id, asset_sha256, hypothesis_raw,
                        reference_normalized, hypothesis_normalized,
                        reference_char_count, edit_distance, cer,
                        runtime_seconds, detected_language,
                        backend_metadata_json, status, error_type, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result["asset_sha256"],
                        result.get("hypothesis_raw"),
                        result["reference_normalized"],
                        result.get("hypothesis_normalized"),
                        result["reference_char_count"],
                        result.get("edit_distance"),
                        result.get("cer"),
                        result.get("runtime_seconds"),
                        result.get("detected_language"),
                        result["backend_metadata_json"],
                        result["status"],
                        result.get("error_type"),
                        result.get("error_message"),
                    ),
                )
            connection.execute(
                """
                UPDATE asr_run
                SET finished_at = ?, status = ?, item_count = ?,
                    success_count = ?, error_count = ?,
                    total_audio_seconds = ?, total_runtime_seconds = ?,
                    aggregate_cer = ?, exact_match_count = ?,
                    model_license = ?, runtime_metadata_json = ?,
                    error_message = NULL
                WHERE run_id = ?
                """,
                (
                    finished_at,
                    summary["status"],
                    summary["item_count"],
                    summary["success_count"],
                    summary["error_count"],
                    summary["total_audio_seconds"],
                    summary["total_runtime_seconds"],
                    summary.get("aggregate_cer"),
                    summary.get("exact_match_count"),
                    summary.get("model_license"),
                    json.dumps(
                        summary.get("runtime_metadata", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    run_id,
                ),
            )

    def fail_asr_run(
        self, *, run_id: str, finished_at: str, error_message: str
    ) -> None:
        with self.session() as connection:
            connection.execute(
                """
                UPDATE asr_run
                SET finished_at = ?, status = 'failed', error_message = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_at, error_message, run_id),
            )

    def load_latest_asr_runs(
        self, *, benchmark_id: str, benchmark_version: int
    ) -> list[dict[str, Any]]:
        """Newest successful run per backend, so a superseded run never competes.

        A backend can be re-run after a methodology fix, and the earlier run is
        kept for audit. Comparison must read only the newest one per backend.
        """
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT run.* FROM asr_run AS run
                JOIN (
                    SELECT backend_id, MAX(started_at) AS started_at
                    FROM asr_run
                    WHERE benchmark_id = ? AND benchmark_version = ?
                      AND status = 'succeeded'
                    GROUP BY backend_id
                ) AS newest
                  ON newest.backend_id = run.backend_id
                 AND newest.started_at = run.started_at
                WHERE run.benchmark_id = ? AND run.benchmark_version = ?
                  AND run.status = 'succeeded'
                ORDER BY run.backend_id
                """,
                (
                    benchmark_id,
                    benchmark_version,
                    benchmark_id,
                    benchmark_version,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_asr_run(self, *, run_id: str) -> dict[str, Any] | None:
        with self.session() as connection:
            row = connection.execute(
                "SELECT * FROM asr_run WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def load_asr_results(self, *, run_id: str) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT result.*, item.ordinal, item.emotion_weak_label,
                       truth.text_exact
                FROM asr_result AS result
                JOIN asr_run AS run ON run.run_id = result.run_id
                JOIN asr_benchmark_item AS item
                  ON item.asset_sha256 = result.asset_sha256
                 AND item.benchmark_id = run.benchmark_id
                 AND item.benchmark_version = run.benchmark_version
                JOIN transcript_ground_truth AS truth
                  ON truth.asset_sha256 = result.asset_sha256
                WHERE result.run_id = ?
                ORDER BY item.ordinal
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_review_decisions(
        self, rows: Iterable[dict[str, Any]]
    ) -> tuple[int, int]:
        """Append review decisions; an identical batch replay is a no-op.

        Every column of the row set comes from a human or an exported rule
        decision, so rows are never updated in place. Changing one's mind
        inserts a new review_round instead, and the previous round stays for
        audit. Reusing a batch id with changed content is rejected rather than
        silently treating the change as an idempotent replay.
        """
        replay_columns = (
            "asset_sha256",
            "benchmark_id",
            "benchmark_version",
            "text_decision",
            "text_final",
            "text_note",
            "emotion_primary",
            "emotion_secondary",
            "intensity",
            "label_source",
            "review_status",
            "auto_rules_applied_json",
            "created_at",
        )
        row_list = list(rows)
        inserted = 0
        ignored = 0
        with self.session() as connection:
            # Serialize round allocation with the inserts it governs. Without an
            # immediate write transaction, two importers can both observe the
            # same MAX(review_round) and one decision is then lost to a conflict.
            connection.execute("BEGIN IMMEDIATE")
            for row in row_list:
                existing = connection.execute(
                    """
                    SELECT * FROM review_decision
                    WHERE export_batch_id = ? AND source_row_index = ?
                    """,
                    (row["export_batch_id"], row["source_row_index"]),
                ).fetchone()
                if existing is not None:
                    changed = [
                        column
                        for column in replay_columns
                        if existing[column] != row.get(column)
                    ]
                    if changed:
                        raise ValueError(
                            "Review batch replay changed previously imported "
                            f"row {row['source_row_index']}: {', '.join(changed)}"
                        )
                    row["review_round"] = int(existing["review_round"])
                    ignored += 1
                    continue
                current_round = connection.execute(
                    """
                    SELECT COALESCE(MAX(review_round), 0)
                    FROM review_decision
                    WHERE asset_sha256 = ? AND benchmark_id = ?
                      AND benchmark_version = ?
                    """,
                    (
                        row["asset_sha256"],
                        row["benchmark_id"],
                        row["benchmark_version"],
                    ),
                ).fetchone()[0]
                row["review_round"] = int(current_round) + 1
                connection.execute(
                    """
                    INSERT INTO review_decision (
                        decision_id, asset_sha256, benchmark_id,
                        benchmark_version, review_round, text_decision,
                        text_final, text_note, emotion_primary,
                        emotion_secondary, intensity, label_source,
                        review_status, auto_rules_applied_json, created_at,
                        export_batch_id, source_row_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["decision_id"],
                        row["asset_sha256"],
                        row["benchmark_id"],
                        row["benchmark_version"],
                        row["review_round"],
                        row["text_decision"],
                        row.get("text_final"),
                        row.get("text_note"),
                        row.get("emotion_primary"),
                        row.get("emotion_secondary"),
                        row.get("intensity"),
                        row["label_source"],
                        row["review_status"],
                        row["auto_rules_applied_json"],
                        row["created_at"],
                        row["export_batch_id"],
                        row["source_row_index"],
                    ),
                )
                inserted += 1
        return inserted, ignored

    def load_latest_review_decisions(
        self, *, benchmark_id: str, benchmark_version: int
    ) -> list[dict[str, Any]]:
        """Newest review round per asset; the source of truth for downstream slices."""
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT decision.* FROM review_decision AS decision
                JOIN (
                    SELECT asset_sha256, MAX(review_round) AS review_round
                    FROM review_decision
                    WHERE benchmark_id = ? AND benchmark_version = ?
                    GROUP BY asset_sha256
                ) AS latest
                  ON latest.asset_sha256 = decision.asset_sha256
                 AND latest.review_round = decision.review_round
                WHERE decision.benchmark_id = ? AND decision.benchmark_version = ?
                ORDER BY decision.asset_sha256
                """,
                (
                    benchmark_id,
                    benchmark_version,
                    benchmark_id,
                    benchmark_version,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_dataset_freeze_candidates(
        self,
        *,
        standardization_config_id: str,
        standardization_config_version: int,
        quality_run_id: str,
        review_benchmark_id: str,
        review_benchmark_version: int,
    ) -> list[dict[str, Any]]:
        """Load one provenance-pinned row per raw asset for dataset auditing.

        The quality run is explicit rather than selected by recency. Review rows
        are latest only inside the caller-specified benchmark version.
        """
        with self.read_only_session() as connection:
            rows = connection.execute(
                """
                SELECT raw_object.asset_sha256,
                       raw_object.relative_path AS raw_relative_path,
                       raw_object.size_bytes AS raw_size_bytes,
                       asset.duration_seconds AS raw_duration_seconds,
                       source.root_key AS source_root_key,
                       source.path_key AS source_path_key,
                       source.relative_path AS source_relative_path,
                       source.availability_status AS source_availability_status,
                       source.speaker_id,
                       source.emotion_weak_label,
                       source.transcript_candidate,
                       source.parser_profile_id,
                       source.parser_profile_version,
                       source.parser_config_sha256,
                       derived.derived_id,
                       derived.relative_path AS derived_relative_path,
                       derived.output_sha256 AS derived_output_sha256,
                       derived.size_bytes AS derived_size_bytes,
                       derived.duration_seconds AS derived_duration_seconds,
                       derived.output_sample_rate,
                       derived.output_channels,
                       derived.output_subtype,
                       derived.config_sha256 AS standardization_config_sha256,
                       derived.implementation_version AS standardization_implementation_version,
                       quality_run.run_id AS quality_run_id,
                       quality_run.analysis_id AS quality_analysis_id,
                       quality_run.analysis_version AS quality_analysis_version,
                       quality_run.analysis_config_sha256 AS quality_analysis_config_sha256,
                       quality_run.implementation_version AS quality_implementation_version,
                       quality_run.policy_id AS quality_policy_id,
                       quality_run.policy_version AS quality_policy_version,
                       quality_run.policy_config_sha256 AS quality_policy_config_sha256,
                       quality_item.decision AS quality_decision,
                       quality_item.reasons_json AS quality_reasons_json,
                       review.decision_id AS review_decision_id,
                       review.review_round,
                       review.text_decision AS review_text_decision,
                       review.text_final AS review_text_final,
                       review.text_note AS review_text_note,
                       review.emotion_primary AS review_emotion_primary,
                       review.emotion_secondary AS review_emotion_secondary,
                       review.intensity AS review_intensity,
                       review.label_source AS review_label_source,
                       review.review_status,
                       review.created_at AS review_created_at,
                       review.export_batch_id AS review_export_batch_id
                FROM raw_object
                JOIN asset ON asset.sha256 = raw_object.asset_sha256
                LEFT JOIN source_location AS source ON source.rowid = (
                    SELECT candidate.rowid
                    FROM source_location AS candidate
                    WHERE candidate.asset_sha256 = raw_object.asset_sha256
                      AND candidate.availability_status = 'available'
                    ORDER BY candidate.root_key, candidate.path_key
                    LIMIT 1
                )
                JOIN derived_audio AS derived
                  ON derived.asset_sha256 = raw_object.asset_sha256
                 AND derived.config_id = ?
                 AND derived.config_version = ?
                JOIN quality_run_item AS quality_item
                  ON quality_item.asset_sha256 = raw_object.asset_sha256
                 AND quality_item.run_id = ?
                JOIN quality_run
                  ON quality_run.run_id = quality_item.run_id
                 AND quality_run.status IN ('succeeded', 'completed_with_errors')
                LEFT JOIN review_decision AS review ON review.decision_id = (
                    SELECT candidate.decision_id
                    FROM review_decision AS candidate
                    WHERE candidate.asset_sha256 = raw_object.asset_sha256
                      AND candidate.benchmark_id = ?
                      AND candidate.benchmark_version = ?
                    ORDER BY candidate.review_round DESC,
                             candidate.created_at DESC,
                             candidate.decision_id DESC
                    LIMIT 1
                )
                ORDER BY raw_object.asset_sha256
                """,
                (
                    standardization_config_id,
                    standardization_config_version,
                    quality_run_id,
                    review_benchmark_id,
                    review_benchmark_version,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_dataset_build_config(
        self, *, dataset_id: str, dataset_version: int
    ) -> dict[str, Any] | None:
        with self.read_only_session() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM dataset_build_config
                WHERE dataset_id = ? AND dataset_version = ?
                """,
                (dataset_id, dataset_version),
            ).fetchone()
        return dict(row) if row is not None else None

    def load_dataset_freeze_coverage(
        self,
        *,
        standardization_config_id: str,
        standardization_config_version: int,
        quality_run_id: str,
    ) -> dict[str, int]:
        with self.read_only_session() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM raw_object) AS raw_object_count,
                    (SELECT COUNT(DISTINCT asset_sha256)
                     FROM derived_audio
                     WHERE config_id = ? AND config_version = ?) AS standardized_count,
                    (SELECT COUNT(*)
                     FROM quality_run_item
                     WHERE run_id = ?) AS quality_item_count,
                    (SELECT COUNT(DISTINCT asset_sha256)
                     FROM source_location
                     WHERE availability_status = 'available') AS available_source_count
                """,
                (
                    standardization_config_id,
                    standardization_config_version,
                    quality_run_id,
                ),
            ).fetchone()
        assert row is not None
        return {key: int(row[key]) for key in row.keys()}

    def register_dataset_build_config(
        self,
        *,
        dataset_id: str,
        dataset_version: int,
        config_sha256: str,
        config_json: str,
        implementation_version: int,
        source_path: str,
        registered_at: str,
    ) -> None:
        with self.session() as connection:
            existing = connection.execute(
                """
                SELECT config_sha256, config_json, implementation_version
                FROM dataset_build_config
                WHERE dataset_id = ? AND dataset_version = ?
                """,
                (dataset_id, dataset_version),
            ).fetchone()
            if existing is not None and (
                existing["config_sha256"] != config_sha256
                or existing["config_json"] != config_json
                or int(existing["implementation_version"]) != implementation_version
            ):
                raise RuntimeError(
                    "Dataset build config changed without a version bump: "
                    f"{dataset_id}@{dataset_version}"
                )
            connection.execute(
                """
                INSERT INTO dataset_build_config (
                    dataset_id, dataset_version, config_sha256, config_json,
                    implementation_version, source_path, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, dataset_version) DO UPDATE SET
                    source_path = excluded.source_path
                """,
                (
                    dataset_id,
                    dataset_version,
                    config_sha256,
                    config_json,
                    implementation_version,
                    source_path,
                    registered_at,
                ),
            )

    def register_dataset_threshold_calibration(
        self,
        *,
        dataset_id: str,
        dataset_version: int,
        calibration_id: str,
        calibration_version: int,
        calibration_config_sha256: str,
        calibration_report_sha256: str,
        report_path: str,
        thresholds: dict[str, float],
        registered_at: str,
    ) -> None:
        thresholds_json = json.dumps(
            thresholds,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.session() as connection:
            existing = connection.execute(
                """
                SELECT calibration_config_sha256, calibration_report_sha256,
                       thresholds_json
                FROM dataset_threshold_calibration
                WHERE dataset_id = ? AND dataset_version = ?
                  AND calibration_id = ? AND calibration_version = ?
                """,
                (
                    dataset_id,
                    dataset_version,
                    calibration_id,
                    calibration_version,
                ),
            ).fetchone()
            if existing is not None and (
                existing["calibration_config_sha256"] != calibration_config_sha256
                or existing["calibration_report_sha256"] != calibration_report_sha256
                or existing["thresholds_json"] != thresholds_json
            ):
                raise RuntimeError(
                    "Dataset threshold calibration changed without a version bump: "
                    f"{calibration_id}@{calibration_version} for "
                    f"{dataset_id}@{dataset_version}"
                )
            connection.execute(
                """
                INSERT INTO dataset_threshold_calibration (
                    dataset_id, dataset_version, calibration_id,
                    calibration_version, calibration_config_sha256,
                    calibration_report_sha256, report_path, thresholds_json,
                    registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    dataset_id, dataset_version, calibration_id, calibration_version
                ) DO UPDATE SET report_path = excluded.report_path
                """,
                (
                    dataset_id,
                    dataset_version,
                    calibration_id,
                    calibration_version,
                    calibration_config_sha256,
                    calibration_report_sha256,
                    report_path,
                    thresholds_json,
                    registered_at,
                ),
            )

    def load_dataset_threshold_calibration(
        self,
        *,
        dataset_id: str,
        dataset_version: int,
        calibration_id: str,
        calibration_version: int,
    ) -> dict[str, Any] | None:
        with self.read_only_session() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM dataset_threshold_calibration
                WHERE dataset_id = ? AND dataset_version = ?
                  AND calibration_id = ? AND calibration_version = ?
                """,
                (
                    dataset_id,
                    dataset_version,
                    calibration_id,
                    calibration_version,
                ),
            ).fetchone()
        return dict(row) if row is not None else None

    def begin_dataset_analysis_run(
        self,
        *,
        dataset_id: str,
        dataset_version: int,
        candidate_snapshot_sha256: str,
        calibration_id: str,
        calibration_version: int,
        calibration_config_sha256: str,
        calibration_report_sha256: str,
        started_at: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.session() as connection:
            calibration = connection.execute(
                """
                SELECT calibration_config_sha256, calibration_report_sha256
                FROM dataset_threshold_calibration
                WHERE dataset_id = ? AND dataset_version = ?
                  AND calibration_id = ? AND calibration_version = ?
                """,
                (
                    dataset_id,
                    dataset_version,
                    calibration_id,
                    calibration_version,
                ),
            ).fetchone()
            if calibration is None:
                raise RuntimeError(
                    "Dataset threshold calibration is not registered: "
                    f"{calibration_id}@{calibration_version} for "
                    f"{dataset_id}@{dataset_version}"
                )
            if (
                calibration["calibration_config_sha256"]
                != calibration_config_sha256
                or calibration["calibration_report_sha256"]
                != calibration_report_sha256
            ):
                raise RuntimeError(
                    "Dataset threshold calibration identity mismatch before run: "
                    f"{calibration_id}@{calibration_version}"
                )
            known_snapshots = {
                str(row["candidate_snapshot_sha256"])
                for row in connection.execute(
                    """
                    SELECT candidate_snapshot_sha256
                    FROM dataset_analysis_run
                    WHERE dataset_id = ? AND dataset_version = ?
                    UNION
                    SELECT candidate_snapshot_sha256
                    FROM dataset_version
                    WHERE dataset_id = ? AND dataset_version = ?
                    """,
                    (dataset_id, dataset_version, dataset_id, dataset_version),
                )
            }
            if known_snapshots and known_snapshots != {candidate_snapshot_sha256}:
                raise RuntimeError(
                    "Dataset candidate snapshot changed without a version bump: "
                    f"{dataset_id}@{dataset_version}; registered {sorted(known_snapshots)}, "
                    f"supplied {candidate_snapshot_sha256}"
                )
            known_calibrations = {
                (
                    str(row["calibration_id"]),
                    int(row["calibration_version"]),
                    str(row["calibration_config_sha256"]),
                    str(row["calibration_report_sha256"]),
                )
                for row in connection.execute(
                    """
                    SELECT calibration_id, calibration_version,
                           calibration_config_sha256, calibration_report_sha256
                    FROM dataset_analysis_calibration
                    WHERE dataset_id = ? AND dataset_version = ?
                    """,
                    (dataset_id, dataset_version),
                )
            }
            requested_calibration = (
                calibration_id,
                calibration_version,
                calibration_config_sha256,
                calibration_report_sha256,
            )
            if known_calibrations and known_calibrations != {requested_calibration}:
                raise RuntimeError(
                    "Dataset analysis calibration changed without a version bump: "
                    f"{dataset_id}@{dataset_version}"
                )
            connection.execute(
                """
                INSERT INTO dataset_analysis_run (
                    run_id, dataset_id, dataset_version,
                    candidate_snapshot_sha256, started_at, status
                ) VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    dataset_id,
                    dataset_version,
                    candidate_snapshot_sha256,
                    started_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO dataset_analysis_calibration (
                    run_id, dataset_id, dataset_version, calibration_id,
                    calibration_version, calibration_config_sha256,
                    calibration_report_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset_id,
                    dataset_version,
                    calibration_id,
                    calibration_version,
                    calibration_config_sha256,
                    calibration_report_sha256,
                ),
            )
        return run_id

    def store_dataset_similarity_graph(
        self,
        *,
        run_id: str,
        candidate_asset_sha256s: Iterable[str],
        edges: Iterable[dict[str, Any]],
        created_at: str,
    ) -> dict[str, Any]:
        """Persist one immutable edge set and materialize its reviewed components."""
        candidates = sorted(candidate_asset_sha256s)
        if not candidates or len(set(candidates)) != len(candidates):
            raise RuntimeError("Similarity graph candidates must be non-empty and unique")
        edge_rows = sorted(
            [dict(edge) for edge in edges],
            key=lambda edge: (
                edge["evidence_type"],
                edge["left_asset_sha256"],
                edge["right_asset_sha256"],
            ),
        )
        edge_keys = [
            (
                edge["left_asset_sha256"],
                edge["right_asset_sha256"],
                edge["evidence_type"],
            )
            for edge in edge_rows
        ]
        if len(set(edge_keys)) != len(edge_keys):
            raise RuntimeError("Similarity graph contains duplicate edge keys")

        with self.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                """
                SELECT status, candidate_count
                FROM dataset_analysis_run
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise RuntimeError(f"Dataset analysis run does not exist: {run_id}")
            if run["status"] != "running":
                raise RuntimeError(
                    f"Dataset analysis run is not writable: {run_id} ({run['status']})"
                )
            if int(run["candidate_count"]) not in (0, len(candidates)):
                raise RuntimeError("Similarity graph candidate count changed within the run")

            placeholders = ",".join("?" for _ in candidates)
            known_assets = {
                str(row["sha256"])
                for row in connection.execute(
                    f"SELECT sha256 FROM asset WHERE sha256 IN ({placeholders})",
                    candidates,
                )
            }
            if known_assets != set(candidates):
                missing = sorted(set(candidates) - known_assets)
                raise RuntimeError(f"Similarity graph has unknown catalog assets: {missing}")
            candidate_set = set(candidates)
            for edge in edge_rows:
                evidence_type = edge.get("evidence_type")
                if evidence_type not in (
                    "exact_audio",
                    "exact_text",
                    "near_audio",
                    "near_text",
                ):
                    raise RuntimeError("Similarity edge has an invalid evidence type")
                score = edge.get("score")
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise RuntimeError("Similarity edge has an invalid score")
                threshold = edge.get("threshold")
                if evidence_type.startswith("near_") and (
                    not isinstance(threshold, (int, float))
                    or not 0 <= threshold <= 1
                ):
                    raise RuntimeError("Near-similarity edge has an invalid threshold")
                if (
                    edge["left_asset_sha256"] not in candidate_set
                    or edge["right_asset_sha256"] not in candidate_set
                ):
                    raise RuntimeError("Similarity edge references a non-candidate asset")
                expected_status = (
                    "accepted_exact"
                    if evidence_type in ("exact_audio", "exact_text")
                    else "pending_review"
                )
                if (
                    edge["left_asset_sha256"] >= edge["right_asset_sha256"]
                    or edge["analysis_status"] != expected_status
                    or (
                        evidence_type in ("exact_audio", "exact_text")
                        and threshold is not None
                    )
                    or not isinstance(edge["evidence_json"], str)
                    or canonical_json(json.loads(edge["evidence_json"]))
                    != edge["evidence_json"]
                ):
                    raise RuntimeError("Similarity edge violates the canonical contract")

            existing_rows = connection.execute(
                """
                SELECT left_asset_sha256, right_asset_sha256, evidence_type,
                       score, threshold, analysis_status, evidence_json
                FROM dataset_similarity_edge
                WHERE run_id = ?
                ORDER BY evidence_type, left_asset_sha256, right_asset_sha256
                """,
                (run_id,),
            ).fetchall()
            existing_edges = [dict(row) for row in existing_rows]
            if existing_edges and existing_edges != edge_rows:
                raise RuntimeError("Similarity edge set changed within the analysis run")
            action = "cached" if existing_edges else "stored"
            if not existing_edges:
                connection.executemany(
                    """
                    INSERT INTO dataset_similarity_edge (
                        run_id, left_asset_sha256, right_asset_sha256,
                        evidence_type, score, threshold, analysis_status,
                        evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            edge["left_asset_sha256"],
                            edge["right_asset_sha256"],
                            edge["evidence_type"],
                            edge["score"],
                            edge["threshold"],
                            edge["analysis_status"],
                            edge["evidence_json"],
                        )
                        for edge in edge_rows
                    ],
                )
            connection.execute(
                """
                UPDATE dataset_analysis_run
                SET candidate_count = ?
                WHERE run_id = ?
                """,
                (len(candidates), run_id),
            )

            latest_reviews = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT review_id, left_asset_sha256, right_asset_sha256,
                           evidence_type, review_round, review_status,
                           review_note, created_at, review_batch_id,
                           source_row_index
                    FROM dataset_similarity_edge_review AS review
                    WHERE run_id = ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM dataset_similarity_edge_review AS newer
                          WHERE newer.run_id = review.run_id
                            AND newer.left_asset_sha256 = review.left_asset_sha256
                            AND newer.right_asset_sha256 = review.right_asset_sha256
                            AND newer.evidence_type = review.evidence_type
                            AND newer.review_round > review.review_round
                      )
                    ORDER BY evidence_type, left_asset_sha256, right_asset_sha256
                    """,
                    (run_id,),
                )
            ]
            review_snapshot_sha256 = hashlib.sha256(
                canonical_json(latest_reviews).encode("utf-8")
            ).hexdigest()
            accepted_pairs = {
                (edge["left_asset_sha256"], edge["right_asset_sha256"])
                for edge in edge_rows
                if edge["analysis_status"] == "accepted_exact"
            }
            accepted_pairs.update(
                (review["left_asset_sha256"], review["right_asset_sha256"])
                for review in latest_reviews
                if review["review_status"] == "accepted"
            )
            groups = stable_similarity_groups(candidates, accepted_pairs)
            existing_groups_by_id = {
                str(row["group_id"]): {
                    "group_id": str(row["group_id"]),
                    "member_count": int(row["member_count"]),
                    "edge_review_snapshot_sha256": str(
                        row["edge_review_snapshot_sha256"]
                    ),
                    "member_asset_sha256s": [],
                }
                for row in connection.execute(
                    """
                    SELECT group_id, member_count, edge_review_snapshot_sha256
                    FROM dataset_similarity_group
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )
            }
            for row in connection.execute(
                """
                SELECT group_id, asset_sha256
                FROM dataset_similarity_group_member
                WHERE run_id = ?
                ORDER BY asset_sha256
                """,
                (run_id,),
            ):
                existing_groups_by_id[str(row["group_id"])][
                    "member_asset_sha256s"
                ].append(str(row["asset_sha256"]))
            existing_groups = sorted(
                existing_groups_by_id.values(),
                key=lambda group: tuple(group["member_asset_sha256s"]),
            )
            desired_groups = [
                {**group, "edge_review_snapshot_sha256": review_snapshot_sha256}
                for group in groups
            ]
            if existing_groups != desired_groups:
                connection.execute(
                    "DELETE FROM dataset_similarity_group_member WHERE run_id = ?",
                    (run_id,),
                )
                connection.execute(
                    "DELETE FROM dataset_similarity_group WHERE run_id = ?",
                    (run_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO dataset_similarity_group (
                        run_id, group_id, member_count,
                        edge_review_snapshot_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            group["group_id"],
                            group["member_count"],
                            review_snapshot_sha256,
                            created_at,
                        )
                        for group in groups
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO dataset_similarity_group_member (
                        run_id, group_id, asset_sha256
                    ) VALUES (?, ?, ?)
                    """,
                    [
                        (run_id, group["group_id"], asset_sha256)
                        for group in groups
                        for asset_sha256 in group["member_asset_sha256s"]
                    ],
                )
                if existing_groups:
                    action = "rematerialized"
            return {
                "action": action,
                "run_id": run_id,
                "candidate_count": len(candidates),
                "edge_count": len(edge_rows),
                "accepted_pair_count": len(accepted_pairs),
                "group_count": len(groups),
                "singleton_group_count": sum(
                    group["member_count"] == 1 for group in groups
                ),
                "edge_review_snapshot_sha256": review_snapshot_sha256,
            }

    def store_dataset_features_and_speaker_assessments(
        self,
        *,
        run_id: str,
        features: Iterable[dict[str, Any]],
        assessments: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        feature_rows = sorted(
            [dict(row) for row in features], key=lambda row: row["asset_sha256"]
        )
        assessment_rows = sorted(
            [dict(row) for row in assessments], key=lambda row: row["asset_sha256"]
        )
        feature_assets = [row["asset_sha256"] for row in feature_rows]
        assessment_assets = [row["asset_sha256"] for row in assessment_rows]
        if (
            not feature_rows
            or len(set(feature_assets)) != len(feature_assets)
            or assessment_assets != feature_assets
        ):
            raise RuntimeError("Dataset feature and speaker assessment assets must match")
        for row in feature_rows:
            fingerprint_blob = row["fingerprint_blob"]
            embedding_blob = row["speaker_embedding_blob"]
            if (
                not isinstance(fingerprint_blob, bytes)
                or len(fingerprint_blob) != 4096 * 4
                or hashlib.sha256(fingerprint_blob).hexdigest()
                != row["fingerprint_sha256"]
                or not isinstance(embedding_blob, bytes)
                or len(embedding_blob) != 512 * 4
                or hashlib.sha256(embedding_blob).hexdigest()
                != row["speaker_embedding_sha256"]
            ):
                raise RuntimeError("Dataset feature blob size or SHA-256 mismatch")
        for row in assessment_rows:
            evidence_json = row.get("evidence_json")
            if (
                not isinstance(evidence_json, str)
                or canonical_json(json.loads(evidence_json)) != evidence_json
            ):
                raise RuntimeError("Speaker assessment evidence JSON is not canonical")

        with self.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT status, candidate_count FROM dataset_analysis_run WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None or run["status"] != "running":
                raise RuntimeError(f"Dataset analysis run is not writable: {run_id}")
            if int(run["candidate_count"]) != len(feature_rows):
                raise RuntimeError("Dataset feature count differs from run candidates")
            group_assets = [
                str(row["asset_sha256"])
                for row in connection.execute(
                    """
                    SELECT asset_sha256
                    FROM dataset_similarity_group_member
                    WHERE run_id = ?
                    ORDER BY asset_sha256
                    """,
                    (run_id,),
                )
            ]
            if group_assets != feature_assets:
                raise RuntimeError("Dataset features do not cover materialized group assets")

            existing_features = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT asset_sha256, derived_id, standardized_sha256,
                           duration_seconds, fingerprint_blob,
                           fingerprint_dimension, fingerprint_sha256,
                           speaker_embedding_blob, speaker_embedding_dimension,
                           speaker_embedding_sha256
                    FROM dataset_asset_feature
                    WHERE run_id = ?
                    ORDER BY asset_sha256
                    """,
                    (run_id,),
                )
            ]
            if existing_features and existing_features != feature_rows:
                raise RuntimeError("Dataset feature set changed within the analysis run")
            existing_assessments = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT asset_sha256, center_cosine, knn_cosine, knn_k,
                           center_robust_z, knn_robust_z, candidate_outlier,
                           evidence_json
                    FROM dataset_speaker_assessment
                    WHERE run_id = ?
                    ORDER BY asset_sha256
                    """,
                    (run_id,),
                )
            ]
            if existing_assessments and existing_assessments != assessment_rows:
                raise RuntimeError("Speaker assessment set changed within the analysis run")
            action = "cached" if existing_features else "stored"
            if not existing_features:
                connection.executemany(
                    """
                    INSERT INTO dataset_asset_feature (
                        run_id, asset_sha256, derived_id, standardized_sha256,
                        duration_seconds, fingerprint_blob,
                        fingerprint_dimension, fingerprint_sha256,
                        speaker_embedding_blob, speaker_embedding_dimension,
                        speaker_embedding_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            row["asset_sha256"],
                            row["derived_id"],
                            row["standardized_sha256"],
                            row["duration_seconds"],
                            row["fingerprint_blob"],
                            row["fingerprint_dimension"],
                            row["fingerprint_sha256"],
                            row["speaker_embedding_blob"],
                            row["speaker_embedding_dimension"],
                            row["speaker_embedding_sha256"],
                        )
                        for row in feature_rows
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO dataset_speaker_assessment (
                        run_id, asset_sha256, center_cosine, knn_cosine,
                        knn_k, center_robust_z, knn_robust_z,
                        candidate_outlier, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            row["asset_sha256"],
                            row["center_cosine"],
                            row["knn_cosine"],
                            row["knn_k"],
                            row["center_robust_z"],
                            row["knn_robust_z"],
                            row["candidate_outlier"],
                            row["evidence_json"],
                        )
                        for row in assessment_rows
                    ],
                )
            connection.execute(
                "UPDATE dataset_analysis_run SET feature_count = ? WHERE run_id = ?",
                (len(feature_rows), run_id),
            )
        return {
            "action": action,
            "run_id": run_id,
            "feature_count": len(feature_rows),
            "speaker_assessment_count": len(assessment_rows),
            "speaker_candidate_outlier_count": sum(
                int(row["candidate_outlier"]) for row in assessment_rows
            ),
        }

    def record_dataset_similarity_edge_reviews(
        self, rows: Iterable[dict[str, Any]]
    ) -> tuple[int, int]:
        row_list = [dict(row) for row in rows]
        replay_columns = (
            "run_id",
            "left_asset_sha256",
            "right_asset_sha256",
            "evidence_type",
            "review_status",
            "review_note",
            "created_at",
        )
        inserted = 0
        ignored = 0
        with self.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in row_list:
                if row["left_asset_sha256"] >= row["right_asset_sha256"]:
                    raise RuntimeError("Edge review assets are not in canonical order")
                existing = connection.execute(
                    """
                    SELECT * FROM dataset_similarity_edge_review
                    WHERE review_batch_id = ? AND source_row_index = ?
                    """,
                    (row["review_batch_id"], row["source_row_index"]),
                ).fetchone()
                if existing is not None:
                    changed = [
                        column
                        for column in replay_columns
                        if existing[column] != row.get(column)
                    ]
                    if changed:
                        raise RuntimeError(
                            "Edge review batch replay changed row "
                            f"{row['source_row_index']}: {', '.join(changed)}"
                        )
                    row["review_round"] = int(existing["review_round"])
                    ignored += 1
                    continue
                current_round = connection.execute(
                    """
                    SELECT COALESCE(MAX(review_round), 0)
                    FROM dataset_similarity_edge_review
                    WHERE run_id = ? AND left_asset_sha256 = ?
                      AND right_asset_sha256 = ? AND evidence_type = ?
                    """,
                    (
                        row["run_id"],
                        row["left_asset_sha256"],
                        row["right_asset_sha256"],
                        row["evidence_type"],
                    ),
                ).fetchone()[0]
                row["review_round"] = int(current_round) + 1
                connection.execute(
                    """
                    INSERT INTO dataset_similarity_edge_review (
                        review_id, run_id, left_asset_sha256,
                        right_asset_sha256, evidence_type, review_round,
                        review_status, review_note, created_at,
                        review_batch_id, source_row_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["review_id"],
                        row["run_id"],
                        row["left_asset_sha256"],
                        row["right_asset_sha256"],
                        row["evidence_type"],
                        row["review_round"],
                        row["review_status"],
                        row.get("review_note"),
                        row["created_at"],
                        row["review_batch_id"],
                        row["source_row_index"],
                    ),
                )
                inserted += 1
        return inserted, ignored

    def record_dataset_speaker_reviews(
        self, rows: Iterable[dict[str, Any]]
    ) -> tuple[int, int]:
        row_list = [dict(row) for row in rows]
        replay_columns = (
            "run_id",
            "asset_sha256",
            "review_status",
            "review_note",
            "created_at",
        )
        inserted = 0
        ignored = 0
        with self.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in row_list:
                existing = connection.execute(
                    """
                    SELECT * FROM dataset_speaker_review
                    WHERE review_batch_id = ? AND source_row_index = ?
                    """,
                    (row["review_batch_id"], row["source_row_index"]),
                ).fetchone()
                if existing is not None:
                    changed = [
                        column
                        for column in replay_columns
                        if existing[column] != row.get(column)
                    ]
                    if changed:
                        raise RuntimeError(
                            "Speaker review batch replay changed row "
                            f"{row['source_row_index']}: {', '.join(changed)}"
                        )
                    row["review_round"] = int(existing["review_round"])
                    ignored += 1
                    continue
                current_round = connection.execute(
                    """
                    SELECT COALESCE(MAX(review_round), 0)
                    FROM dataset_speaker_review
                    WHERE run_id = ? AND asset_sha256 = ?
                    """,
                    (row["run_id"], row["asset_sha256"]),
                ).fetchone()[0]
                row["review_round"] = int(current_round) + 1
                connection.execute(
                    """
                    INSERT INTO dataset_speaker_review (
                        review_id, run_id, asset_sha256, review_round,
                        review_status, review_note, created_at,
                        review_batch_id, source_row_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["review_id"],
                        row["run_id"],
                        row["asset_sha256"],
                        row["review_round"],
                        row["review_status"],
                        row.get("review_note"),
                        row["created_at"],
                        row["review_batch_id"],
                        row["source_row_index"],
                    ),
                )
                inserted += 1
        return inserted, ignored

    def load_review_decisions_by_batch(
        self, *, export_batch_id: str
    ) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM review_decision
                WHERE export_batch_id = ?
                ORDER BY source_row_index
                """,
                (export_batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_review_decision_history(
        self,
        *,
        asset_sha256: str,
        benchmark_id: str,
        benchmark_version: int,
    ) -> list[dict[str, Any]]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM review_decision
                WHERE asset_sha256 = ? AND benchmark_id = ?
                  AND benchmark_version = ?
                ORDER BY review_round, created_at
                """,
                (asset_sha256, benchmark_id, benchmark_version),
            ).fetchall()
        return [dict(row) for row in rows]

    def next_review_rounds(self, *, benchmark_id: str, benchmark_version: int) -> dict[str, int]:
        """Round number each asset's next decision should carry (max + 1, or 1)."""
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT asset_sha256, MAX(review_round) AS review_round
                FROM review_decision
                WHERE benchmark_id = ? AND benchmark_version = ?
                GROUP BY asset_sha256
                """,
                (benchmark_id, benchmark_version),
            ).fetchall()
        current = {str(row["asset_sha256"]): int(row["review_round"]) for row in rows}
        return {sha256: round + 1 for sha256, round in current.items()}

    def record_slice9_reference_reviews(
        self, rows: Iterable[dict[str, Any]]
    ) -> tuple[int, int]:
        """Append Slice 9 reference decisions with deterministic batch replay.

        Slice 9 reviews are distinct from ASR transcript decisions: they record
        whether an already frozen audio candidate is suitable for a named pool.
        The table is created lazily so older catalogs remain readable without a
        schema-version bump; rows are never updated in place.
        """
        row_list = [dict(row) for row in rows]
        replay_columns = (
            "selection_id",
            "selection_version",
            "asset_sha256",
            "pool_id",
            "selection_kind",
            "review_status",
            "review_note",
            "created_at",
            "provenance_json",
        )
        inserted = 0
        ignored = 0
        with self.session() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS slice9_reference_review (
                    review_id TEXT PRIMARY KEY,
                    selection_id TEXT NOT NULL,
                    selection_version INTEGER NOT NULL CHECK (selection_version >= 1),
                    asset_sha256 TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    selection_kind TEXT NOT NULL CHECK (selection_kind IN ('selected', 'boundary')),
                    review_round INTEGER NOT NULL CHECK (review_round >= 1),
                    review_status TEXT NOT NULL CHECK (review_status IN ('approved', 'rejected', 'uncertain')),
                    review_note TEXT,
                    created_at TEXT NOT NULL,
                    review_batch_id TEXT NOT NULL,
                    source_row_index INTEGER NOT NULL CHECK (source_row_index >= 0),
                    provenance_json TEXT NOT NULL,
                    UNIQUE (selection_id, selection_version, asset_sha256, pool_id, review_round),
                    UNIQUE (review_batch_id, source_row_index),
                    FOREIGN KEY (asset_sha256) REFERENCES asset(sha256)
                )
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            for row in row_list:
                existing = connection.execute(
                    """
                    SELECT * FROM slice9_reference_review
                    WHERE review_batch_id = ? AND source_row_index = ?
                    """,
                    (row["review_batch_id"], row["source_row_index"]),
                ).fetchone()
                if existing is not None:
                    changed = [
                        column
                        for column in replay_columns
                        if existing[column] != row.get(column)
                    ]
                    if changed:
                        raise RuntimeError(
                            "Slice 9 review batch replay changed row "
                            f"{row['source_row_index']}: {', '.join(changed)}"
                        )
                    row["review_round"] = int(existing["review_round"])
                    ignored += 1
                    continue
                asset_exists = connection.execute(
                    "SELECT 1 FROM asset WHERE sha256 = ?", (row["asset_sha256"],)
                ).fetchone()
                if asset_exists is None:
                    raise RuntimeError(
                        f"Slice 9 review references unknown asset: {row['asset_sha256']}"
                    )
                current_round = connection.execute(
                    """
                    SELECT COALESCE(MAX(review_round), 0)
                    FROM slice9_reference_review
                    WHERE selection_id = ? AND selection_version = ?
                      AND asset_sha256 = ? AND pool_id = ?
                    """,
                    (
                        row["selection_id"],
                        row["selection_version"],
                        row["asset_sha256"],
                        row["pool_id"],
                    ),
                ).fetchone()[0]
                row["review_round"] = int(current_round) + 1
                connection.execute(
                    """
                    INSERT INTO slice9_reference_review (
                        review_id, selection_id, selection_version,
                        asset_sha256, pool_id, selection_kind, review_round,
                        review_status, review_note, created_at, review_batch_id,
                        source_row_index, provenance_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["review_id"],
                        row["selection_id"],
                        row["selection_version"],
                        row["asset_sha256"],
                        row["pool_id"],
                        row["selection_kind"],
                        row["review_round"],
                        row["review_status"],
                        row.get("review_note"),
                        row["created_at"],
                        row["review_batch_id"],
                        row["source_row_index"],
                        row["provenance_json"],
                    ),
                )
                inserted += 1
        return inserted, ignored

    def load_slice9_reference_reviews(
        self, *, selection_id: str, selection_version: int
    ) -> list[dict[str, Any]]:
        """Load the latest Slice 9 decision per asset/pool."""
        with self.session() as connection:
            table_exists = connection.execute(
                """SELECT 1 FROM sqlite_master WHERE type='table' AND name='slice9_reference_review'"""
            ).fetchone()
            if table_exists is None:
                return []
            rows = connection.execute(
                """
                SELECT current.*
                FROM slice9_reference_review AS current
                JOIN (
                    SELECT selection_id, selection_version, asset_sha256, pool_id,
                           MAX(review_round) AS review_round
                    FROM slice9_reference_review
                    WHERE selection_id = ? AND selection_version = ?
                    GROUP BY selection_id, selection_version, asset_sha256, pool_id
                ) AS latest
                  ON latest.selection_id = current.selection_id
                 AND latest.selection_version = current.selection_version
                 AND latest.asset_sha256 = current.asset_sha256
                 AND latest.pool_id = current.pool_id
                 AND latest.review_round = current.review_round
                ORDER BY current.pool_id, current.selection_kind, current.source_row_index
                """,
                (selection_id, selection_version),
            ).fetchall()
        return [dict(row) for row in rows]
