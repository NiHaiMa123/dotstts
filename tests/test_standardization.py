from __future__ import annotations

import hashlib
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
from dots_tts_lab.standardization import (
    DEFAULT_STANDARDIZATION_CONFIG_PATH,
    conservative_trim,
    load_standardization_config,
    standardize_array,
)
from dots_tts_lab.standardize_runner import run_standardization


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sine(sample_rate: int, seconds: float, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * 440.0 * time)


class StandardizationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_standardization_config(
            DEFAULT_STANDARDIZATION_CONFIG_PATH
        )

    def test_36k_and_441k_resample_to_expected_48k_length(self) -> None:
        for source_rate in (36_000, 44_100):
            with self.subTest(source_rate=source_rate):
                source = sine(source_rate, 1.0)
                output, details = standardize_array(
                    source, source_rate, self.config
                )
                self.assertEqual(len(output), 48_000)
                self.assertEqual(details["leading_samples_removed"], 0)
                self.assertEqual(details["trailing_samples_removed"], 0)

    def test_stereo_is_mixed_by_mean(self) -> None:
        left = sine(48_000, 1.0, amplitude=0.2)
        right = sine(48_000, 1.0, amplitude=0.4)
        output, _ = standardize_array(
            np.column_stack((left, right)), 48_000, self.config
        )
        np.testing.assert_allclose(output, (left + right) / 2.0, atol=1e-12)

    def test_only_long_edge_silence_is_trimmed_and_padding_is_retained(self) -> None:
        rate = 48_000
        speech = sine(rate, 1.0)
        long_edges = np.concatenate(
            (np.zeros(round(0.7 * rate)), speech, np.zeros(round(0.8 * rate)))
        )
        trimmed, leading, trailing = conservative_trim(
            long_edges, rate, self.config
        )
        self.assertGreater(leading, round(0.55 * rate))
        self.assertGreater(trailing, round(0.65 * rate))
        self.assertAlmostEqual(len(trimmed) / rate, 1.2, delta=0.04)

        short_edges = np.concatenate(
            (np.zeros(round(0.4 * rate)), speech, np.zeros(round(0.4 * rate)))
        )
        untrimmed, leading, trailing = conservative_trim(
            short_edges, rate, self.config
        )
        self.assertEqual(len(untrimmed), len(short_edges))
        self.assertEqual((leading, trailing), (0, 0))

    def test_peak_protection_only_attenuates(self) -> None:
        quiet, quiet_details = standardize_array(
            sine(48_000, 1.0, amplitude=0.2), 48_000, self.config
        )
        loud, loud_details = standardize_array(
            sine(48_000, 1.0, amplitude=0.99), 48_000, self.config
        )
        self.assertEqual(quiet_details["gain_applied_db"], 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(quiet))), 0.2, places=6)
        self.assertLess(loud_details["gain_applied_db"], 0.0)
        self.assertLessEqual(float(np.max(np.abs(loud))), 10 ** (-1 / 20) + 1e-9)


class StandardizationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.inbox = self.base / "data" / "inbox"
        self.raw = self.base / "data" / "raw" / "sha256"
        self.work = self.base / "data" / "work" / "standardized"
        self.catalog = self.base / "data" / "catalog" / "catalog.sqlite"
        self.reports = self.base / "data" / "reports"
        self.source = (
            self.inbox
            / "角色甲"
            / "中立_neutral"
            / "【中立_neutral】标准化测试。.wav"
        )
        self.source.parent.mkdir(parents=True)
        rate = 44_100
        data = np.concatenate(
            (np.zeros(round(0.1 * rate)), sine(rate, 0.8), np.zeros(round(0.1 * rate)))
        )
        sf.write(str(self.source), data, rate, subtype="PCM_16")
        run_ingest(
            self.inbox,
            catalog_path=self.catalog,
            raw_dir=self.raw,
            inventory_report_dir=self.reports / "inventory",
            report_dir=self.reports / "ingest",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_pipeline(self, **overrides):
        arguments = {
            "catalog_path": self.catalog,
            "raw_dir": self.raw,
            "output_dir": self.work,
            "config_path": DEFAULT_STANDARDIZATION_CONFIG_PATH,
            "report_dir": self.reports / "standardization",
        }
        arguments.update(overrides)
        return run_standardization(**arguments)

    def test_pcm24_output_cache_and_raw_immutability(self) -> None:
        raw_file = next(self.raw.rglob("*.wav"))
        before = sha256(raw_file)
        first = self.run_pipeline()
        output = next(self.work.rglob("*.wav"))
        info = sf.info(str(output))
        self.assertEqual(first["built_count"], 1)
        self.assertEqual(info.samplerate, 48_000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_24")
        self.assertEqual(sha256(raw_file), before)
        self.assertLessEqual(first["max_output_true_peak_estimate_dbtp"], -0.999)

        with mock.patch(
            "dots_tts_lab.standardize_runner.sf.read",
            side_effect=AssertionError("cache hit must not decode"),
        ):
            second = self.run_pipeline()
        self.assertEqual(second["built_count"], 0)
        self.assertEqual(second["cached_count"], 1)
        self.assertEqual(sha256(raw_file), before)

    def test_corrupt_derived_artifact_is_rebuilt(self) -> None:
        self.run_pipeline()
        output = next(self.work.rglob("*.wav"))
        output.write_bytes(b"x" * output.stat().st_size)
        result = self.run_pipeline()
        self.assertEqual(result["rebuilt_count"], 1)
        self.assertEqual(sf.info(str(output)).subtype, "PCM_24")

    def test_config_drift_requires_version_bump(self) -> None:
        self.run_pipeline()
        payload = load_standardization_config(
            DEFAULT_STANDARDIZATION_CONFIG_PATH
        ).model_dump(mode="json")
        payload["target_true_peak_dbtp"] = -2.0
        drift = self.base / "drift.yaml"
        drift.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "without a version bump"):
            self.run_pipeline(config_path=drift)

    def test_report_failure_marks_run_failed_without_catalog_artifact(self) -> None:
        with mock.patch(
            "dots_tts_lab.standardize_runner.write_standardization_reports",
            side_effect=OSError("synthetic report failure"),
        ):
            with self.assertRaisesRegex(OSError, "synthetic report failure"):
                self.run_pipeline()
        self.assertEqual(len(list(self.work.rglob("*.wav"))), 1)
        with closing(sqlite3.connect(self.catalog)) as connection:
            run_status = connection.execute(
                "SELECT status FROM standardization_run ORDER BY started_at DESC LIMIT 1"
            ).fetchone()[0]
            derived_count = connection.execute(
                "SELECT count(*) FROM derived_audio"
            ).fetchone()[0]
        self.assertEqual(run_status, "failed")
        self.assertEqual(derived_count, 0)


if __name__ == "__main__":
    unittest.main()
