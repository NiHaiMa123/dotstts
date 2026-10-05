from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from dots_tts_lab.ingest import run_ingest
from dots_tts_lab.quality import (
    DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH,
    DEFAULT_QUALITY_POLICY_PATH,
    QualityPolicy,
    analyze_audio,
    assess_metrics,
    load_quality_analysis_config,
    load_quality_policy,
)
from dots_tts_lab.quality_runner import run_quality


def write_speech_like_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    dc_offset: float = 0.0,
    clipped: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(round(sample_rate * 0.1), dtype=np.float64)
    time = np.arange(round(sample_rate * 0.8), dtype=np.float64) / sample_rate
    amplitude = 1.8 if clipped else 0.2
    speech = amplitude * np.sin(2 * np.pi * 220.0 * time) + dc_offset
    if clipped:
        speech = np.clip(speech, -1.0, 1.0)
    data = np.concatenate((silence, speech, silence))
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


class QualityMetricUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = load_quality_analysis_config(
            DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH
        )
        self.policy = load_quality_policy(DEFAULT_QUALITY_POLICY_PATH)

    def test_signal_metrics_distinguish_clean_peak_silence_and_flat_clipping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            clean_path = base / "clean.wav"
            clipped_path = base / "clipped.wav"
            write_speech_like_wav(clean_path)
            write_speech_like_wav(clipped_path, clipped=True)

            clean = analyze_audio(
                clean_path,
                subtype="PCM_16",
                config=self.analysis,
            )
            clipped = analyze_audio(
                clipped_path,
                subtype="PCM_16",
                config=self.analysis,
            )

        self.assertEqual(clean["status"], "ok")
        self.assertIsNotNone(clean["integrated_loudness_lufs"])
        self.assertAlmostEqual(clean["leading_silence_seconds"], 0.1, delta=0.03)
        self.assertAlmostEqual(clean["trailing_silence_seconds"], 0.1, delta=0.03)
        self.assertEqual(clean["flat_top_run_count"], 0)
        self.assertGreater(clean["digital_silence_frame_ratio"], 0)
        self.assertIsNone(clean["snr_proxy_db"])
        self.assertGreater(clipped["flat_top_run_count"], 0)
        self.assertGreater(clipped["max_flat_top_run_samples"], 2)
        self.assertGreater(clipped["near_peak_sample_ratio"], 0.1)

    def test_policy_routes_threshold_crossings_to_review_not_reject(self) -> None:
        metrics = {
            "status": "ok",
            "sample_rate": 36_000,
            "duration_seconds": 5.0,
            "abs_dc_offset": 0.02,
            "leading_silence_seconds": 0.0,
            "trailing_silence_seconds": 0.7,
            "silence_ratio": 0.1,
            "flat_top_run_count": 1,
            "integrated_loudness_lufs": -20.0,
            "true_peak_estimate_dbtp": -1.0,
            "snr_proxy_db": 20.0,
        }

        assessment = assess_metrics(metrics, policy=self.policy)

        self.assertEqual(assessment["decision"], "review")
        codes = {reason["code"] for reason in assessment["reasons"]}
        self.assertEqual(
            codes,
            {
                "low_sample_rate",
                "dc_offset",
                "long_trailing_silence",
                "possible_hard_clipping",
            },
        )
        rejected = assess_metrics(
            {"status": "error", "error_message": "decode failed"},
            policy=self.policy,
        )
        self.assertEqual(rejected["decision"], "reject")


class QualityPipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.inbox = self.base / "data" / "inbox"
        self.raw = self.base / "data" / "raw" / "sha256"
        self.catalog = self.base / "data" / "catalog" / "catalog.sqlite"
        self.reports = self.base / "data" / "reports"
        source = (
            self.inbox
            / "角色甲"
            / "中立_neutral"
            / "【中立_neutral】质量缓存。.wav"
        )
        write_speech_like_wav(source, sample_rate=44_100)
        run_ingest(
            self.inbox,
            catalog_path=self.catalog,
            raw_dir=self.raw,
            inventory_report_dir=self.reports / "inventory",
            report_dir=self.reports / "ingest",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_metrics(self, *, policy_path: Path | str = DEFAULT_QUALITY_POLICY_PATH):
        return run_quality(
            catalog_path=self.catalog,
            raw_dir=self.raw,
            analysis_config_path=DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH,
            policy_path=policy_path,
            report_dir=self.reports / "quality",
        )

    def test_cached_metrics_allow_policy_reassessment_without_decode(self) -> None:
        first = self.run_metrics()
        self.assertEqual(first["analyzed_count"], 1)
        self.assertEqual(first["cached_count"], 0)

        with mock.patch(
            "dots_tts_lab.quality_runner.analyze_audio",
            side_effect=AssertionError("cached run must not decode"),
        ):
            second = self.run_metrics()
        self.assertEqual(second["analyzed_count"], 0)
        self.assertEqual(second["cached_count"], 1)

        policy = load_quality_policy(DEFAULT_QUALITY_POLICY_PATH).model_dump(
            mode="json"
        )
        policy["policy_version"] = 2
        policy["max_integrated_loudness_lufs"] = -30.0
        policy_path = self.base / "policy_v2.yaml"
        policy_path.write_text(
            json.dumps(policy, ensure_ascii=False),
            encoding="utf-8",
        )
        with mock.patch(
            "dots_tts_lab.quality_runner.analyze_audio",
            side_effect=AssertionError("policy change must not decode"),
        ):
            third = self.run_metrics(policy_path=policy_path)
        self.assertEqual(third["analyzed_count"], 0)
        self.assertEqual(third["cached_count"], 1)
        self.assertEqual(third["review_count"], 1)

        with closing(sqlite3.connect(self.catalog)) as connection:
            metric_count = connection.execute(
                "SELECT count(*) FROM audio_quality_metric"
            ).fetchone()[0]
            policy_count = connection.execute(
                "SELECT count(*) FROM quality_policy"
            ).fetchone()[0]
            assessment_count = connection.execute(
                "SELECT count(*) FROM audio_quality_assessment"
            ).fetchone()[0]
        self.assertEqual(metric_count, 1)
        self.assertEqual(policy_count, 2)
        self.assertEqual(assessment_count, 2)

    def test_analysis_config_drift_is_blocked(self) -> None:
        self.run_metrics()
        analysis = load_quality_analysis_config(
            DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH
        ).model_dump(mode="json")
        analysis["silence_threshold_dbfs"] = -45.0
        path = self.base / "analysis_drift.yaml"
        path.write_text(json.dumps(analysis), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "without a version bump"):
            run_quality(
                catalog_path=self.catalog,
                raw_dir=self.raw,
                analysis_config_path=path,
                policy_path=DEFAULT_QUALITY_POLICY_PATH,
                report_dir=self.reports / "quality",
            )

    def test_corrupt_raw_is_rejected_and_recorded_as_analysis_error(self) -> None:
        raw_file = next(self.raw.rglob("*.wav"))
        raw_file.write_bytes(b"not an audio file")

        summary = self.run_metrics()

        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["reject_count"], 1)
        self.assertEqual(summary["reason_counts"], {"analysis_error": 1})

    def test_report_failure_marks_quality_run_failed_without_metric_commit(
        self,
    ) -> None:
        with mock.patch(
            "dots_tts_lab.quality_runner.write_quality_reports",
            side_effect=OSError("synthetic quality report failure"),
        ):
            with self.assertRaisesRegex(OSError, "synthetic quality report failure"):
                self.run_metrics()

        with closing(sqlite3.connect(self.catalog)) as connection:
            status = connection.execute(
                "SELECT status FROM quality_run ORDER BY started_at DESC LIMIT 1"
            ).fetchone()[0]
            metric_count = connection.execute(
                "SELECT count(*) FROM audio_quality_metric"
            ).fetchone()[0]
        self.assertEqual(status, "failed")
        self.assertEqual(metric_count, 0)


if __name__ == "__main__":
    unittest.main()
