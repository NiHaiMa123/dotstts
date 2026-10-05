from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.asr_benchmark import (
    character_edit_distance,
    load_asr_benchmark_config,
    normalize_asr_text,
    prepare_asr_benchmark,
)
from dots_tts_lab.asr_evaluation import evaluate_asr_output
from dots_tts_lab.ingest import run_ingest
from dots_tts_lab.standardize_runner import run_standardization


class AsrTextMetricTests(unittest.TestCase):
    def test_normalization_preserves_words_but_removes_case_space_punctuation(self) -> None:
        config = load_asr_benchmark_config()
        self.assertEqual(normalize_asr_text("ＡBC，你 好！", config), "abc你好")
        self.assertEqual(normalize_asr_text("一百2", config), "一百2")
        self.assertEqual(normalize_asr_text("语气～", config), "语气")

    def test_character_edit_distance(self) -> None:
        self.assertEqual(character_edit_distance("符玄", "符玄"), 0)
        self.assertEqual(character_edit_distance("太卜司", "太仆司"), 1)
        self.assertEqual(character_edit_distance("星穹列车", "星穹"), 2)


class AsrBenchmarkFixture(unittest.TestCase):
    """Four synthetic clips, ingested, standardized, and ready to benchmark.

    Shared with the comparison tests, which need the same catalog state plus
    more than one evaluated backend.
    """

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.inbox = self.base / "data" / "inbox"
        self.raw = self.base / "data" / "raw" / "sha256"
        self.work = self.base / "data" / "work" / "standardized"
        self.catalog = self.base / "data" / "catalog" / "catalog.sqlite"
        self.reports = self.base / "data" / "reports"
        self.output = self.base / "data" / "benchmarks" / "asr" / "test_v1"
        emotions = (
            ("中立_neutral", "太卜司正在演算。"),
            ("开心_happy", "星穹列车来啦！"),
            ("生气_angry", "青雀，不许偷懒！"),
            ("难过_sad", "本座也会难过。"),
        )
        for index, (emotion, text) in enumerate(emotions):
            path = self.inbox / "角色甲" / emotion / f"【{emotion}】{text}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            rate = 44_100
            time = np.arange(rate, dtype=np.float64) / rate
            sf.write(
                str(path),
                0.1 * np.sin(2 * np.pi * (220 + index * 20) * time),
                rate,
                subtype="PCM_16",
            )
        run_ingest(
            self.inbox,
            catalog_path=self.catalog,
            raw_dir=self.raw,
            inventory_report_dir=self.reports / "inventory",
            report_dir=self.reports / "ingest",
        )
        run_standardization(
            catalog_path=self.catalog,
            raw_dir=self.raw,
            output_dir=self.work,
            report_dir=self.reports / "standardization",
        )
        with closing(sqlite3.connect(self.catalog)) as connection:
            assets = connection.execute(
                """
                SELECT r.asset_sha256, s.emotion_weak_label
                FROM raw_object r JOIN source_location s
                  ON s.asset_sha256 = r.asset_sha256
                ORDER BY r.asset_sha256
                """
            ).fetchall()
        quality_items = [
            {
                "asset_sha256": asset_sha256,
                "emotion_weak_label": emotion,
                "decision": "pass",
                "reasons": [],
            }
            for asset_sha256, emotion in assets
        ]
        self.quality_report = self.reports / "quality" / "quality.json"
        self.quality_report.parent.mkdir(parents=True)
        self.quality_report.write_text(
            json.dumps({"summary": {}, "assets": quality_items}, ensure_ascii=False),
            encoding="utf-8",
        )
        base_config = load_asr_benchmark_config().model_dump(mode="json")
        base_config.update(
            {
                "benchmark_id": "test_asr",
                "benchmark_version": 1,
                "selection_parent_version": None,
                "sample_count": 4,
                "emotion_targets": {emotion: 1 for emotion, _ in emotions},
                "include_all_quality_review": False,
            }
        )
        self.config = self.base / "benchmark.json"
        self.config.write_text(
            json.dumps(base_config, ensure_ascii=False), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def prepare(self, config_path: Path | None = None):
        return prepare_asr_benchmark(
            catalog_path=self.catalog,
            config_path=config_path or self.config,
            quality_report_path=self.quality_report,
            output_dir=self.output,
        )

    def manifest_entries(self, prepared: dict) -> list[dict]:
        return [
            json.loads(line)
            for line in Path(prepared["reports"]["manifest_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def write_backend_result(
        self,
        *,
        backend_id: str,
        hypotheses: dict[str, str],
        model_license: str = "test",
        runtime_seconds: float = 0.01,
        runtime_metadata: dict | None = None,
    ) -> Path:
        """Serialize a synthetic backend run in the same shape as the real script."""
        inference = {"language": "zh", "beam_size": 1}
        inference_text = json.dumps(
            inference, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        path = self.base / f"backend_{backend_id}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend_id": backend_id,
                    "backend_version": "1",
                    "model_id": f"synthetic/{backend_id}",
                    "model_revision": "abc",
                    "model_license": model_license,
                    "inference_config": inference,
                    "inference_config_sha256": hashlib.sha256(
                        inference_text.encode("utf-8")
                    ).hexdigest(),
                    "manifest_path": str((self.output / "manifest.jsonl").resolve()),
                    "manifest_sha256": hashlib.sha256(
                        (self.output / "manifest.jsonl").read_bytes()
                    ).hexdigest(),
                    "wall_runtime_seconds": 0.4,
                    "runtime_metadata": runtime_metadata
                    or {"model_load_seconds": 0.1},
                    "capabilities": {
                        "timestamps_available": False,
                        "timestamp_type": "none",
                        "timestamps_enabled": False,
                    },
                    "results": [
                        {
                            "asset_sha256": asset_sha256,
                            "hypothesis": hypothesis,
                            "runtime_seconds": runtime_seconds,
                            "detected_language": "zh",
                            "metadata": {},
                            "status": "ok",
                        }
                        for asset_sha256, hypothesis in hypotheses.items()
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path


class AsrBenchmarkPreparationTests(AsrBenchmarkFixture):
    def test_registers_exact_truth_and_freezes_stable_manifest(self) -> None:
        first = self.prepare()
        manifest = Path(first["reports"]["manifest_jsonl"])
        self.assertEqual(first["ground_truth_registered_count"], 4)
        self.assertEqual(first["sample_count"], 4)
        self.assertEqual(
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            first["manifest_sha256"],
        )
        entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.assertIn("太卜司正在演算。", {item["reference_exact"] for item in entries})

        second = self.prepare()
        self.assertEqual(second["ground_truth_registered_count"], 0)
        self.assertEqual(second["ground_truth_cached_count"], 4)
        self.assertEqual(second["manifest_sha256"], first["manifest_sha256"])
        with closing(sqlite3.connect(self.catalog)) as connection:
            truth_count = connection.execute(
                "SELECT count(*) FROM transcript_ground_truth"
            ).fetchone()[0]
            item_count = connection.execute(
                "SELECT count(*) FROM asr_benchmark_item"
            ).fetchone()[0]
        self.assertEqual((truth_count, item_count), (4, 4))

    def test_benchmark_config_drift_is_blocked(self) -> None:
        first = self.prepare()
        manifest = Path(first["reports"]["manifest_jsonl"])
        frozen_bytes = manifest.read_bytes()
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["normalization_casefold"] = False
        drift = self.base / "drift.json"
        drift.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "without a version bump"):
            self.prepare(drift)
        self.assertEqual(manifest.read_bytes(), frozen_bytes)

    def test_evaluation_rejects_config_and_manifest_drift(self) -> None:
        prepared = self.prepare()
        entries = self.manifest_entries(prepared)
        exact = {item["asset_sha256"]: item["reference_exact"] for item in entries}
        result_path = self.write_backend_result(backend_id="exact", hypotheses=exact)

        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["normalization_casefold"] = False
        drift_config = self.base / "evaluation_drift.json"
        drift_config.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "frozen config"):
            evaluate_asr_output(
                result_path=result_path,
                catalog_path=self.catalog,
                benchmark_config_path=drift_config,
                report_root=self.reports / "asr",
            )

        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_payload["manifest_sha256"] = "0" * 64
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
            evaluate_asr_output(
                result_path=result_path,
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                report_root=self.reports / "asr",
            )

    def test_v2_can_inherit_the_exact_v1_selection(self) -> None:
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload.update({"benchmark_version": 1, "selection_parent_version": None})
        parent_config = self.base / "parent_v1.json"
        parent_config.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        parent = prepare_asr_benchmark(
            catalog_path=self.catalog,
            config_path=parent_config,
            quality_report_path=self.quality_report,
            output_dir=self.base / "parent_v1",
        )

        payload.update(
            {
                "benchmark_version": 2,
                "selection_parent_version": 1,
                "remove_characters": ["~", "演"],
            }
        )
        child_config = self.base / "child_v2.json"
        child_config.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        child = prepare_asr_benchmark(
            catalog_path=self.catalog,
            config_path=child_config,
            quality_report_path=self.quality_report,
            output_dir=self.base / "child_v2",
        )
        parent_assets = {
            json.loads(line)["asset_sha256"]
            for line in Path(parent["reports"]["manifest_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
        }
        child_assets = {
            json.loads(line)["asset_sha256"]
            for line in Path(child["reports"]["manifest_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
        }
        self.assertEqual(child_assets, parent_assets)
        self.assertEqual(child["selection_parent_version"], 1)

    def test_exact_backend_output_evaluates_to_zero_cer(self) -> None:
        prepared = self.prepare()
        entries = self.manifest_entries(prepared)
        result = self.write_backend_result(
            backend_id="synthetic",
            hypotheses={
                item["asset_sha256"]: item["reference_exact"] for item in entries
            },
        )
        summary = evaluate_asr_output(
            result_path=result,
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            report_root=self.reports / "asr",
        )
        self.assertEqual(summary["aggregate_cer"], 0.0)
        self.assertEqual(summary["exact_match_count"], 4)
        with closing(sqlite3.connect(self.catalog)) as connection:
            result_count = connection.execute(
                "SELECT count(*) FROM asr_result"
            ).fetchone()[0]
        self.assertEqual(result_count, 4)

    def test_rich_transcription_markers_are_refused(self) -> None:
        prepared = self.prepare()
        entries = self.manifest_entries(prepared)
        result = self.write_backend_result(
            backend_id="emoji",
            hypotheses={
                item["asset_sha256"]: item["reference_exact"] + "\U0001f621"
                for item in entries
            },
        )
        with self.assertRaisesRegex(ValueError, "non-transcript symbols"):
            evaluate_asr_output(
                result_path=result,
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                report_root=self.reports / "asr",
            )
        with closing(sqlite3.connect(self.catalog)) as connection:
            statuses = [
                row[0]
                for row in connection.execute(
                    "SELECT status FROM asr_run WHERE backend_id = 'emoji'"
                )
            ]
            result_count = connection.execute(
                "SELECT count(*) FROM asr_result"
            ).fetchone()[0]
        self.assertEqual(statuses, ["failed"])
        self.assertEqual(result_count, 0)


if __name__ == "__main__":
    unittest.main()
