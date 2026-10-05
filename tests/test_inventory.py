from __future__ import annotations

import json
import os
import sqlite3
import struct
import tempfile
import unittest
import wave
from contextlib import closing
from pathlib import Path
from unittest import mock

from dots_tts_lab.catalog import Catalog, SCHEMA_VERSION
from dots_tts_lab.inventory import run_inventory


def write_test_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 0.1,
    amplitude: int = 1_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(sample_rate * duration_seconds)
    frames = b"".join(
        struct.pack("<h", amplitude if index % 2 == 0 else -amplitude)
        for index in range(frame_count)
    )
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


class InventoryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.inbox = self.base / "data" / "inbox"
        self.catalog_path = self.base / "data" / "catalog" / "catalog.sqlite"
        self.report_dir = self.base / "data" / "reports" / "inventory"
        self.inbox.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_scan(self) -> dict:
        return run_inventory(
            self.inbox,
            catalog_path=self.catalog_path,
            report_dir=self.report_dir,
        )

    def test_empty_inventory_is_successful_and_idempotent(self) -> None:
        first = self.run_scan()
        second = self.run_scan()

        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(first["discovered_count"], 0)
        self.assertEqual(second["unchanged_count"], 0)
        self.assertTrue((self.report_dir / "inventory.json").is_file())
        self.assertTrue((self.report_dir / "inventory.csv").is_file())
        self.assertTrue((self.report_dir / "inventory.html").is_file())

    def test_inventory_tracks_unicode_idempotency_change_move_and_missing(self) -> None:
        neutral = self.inbox / "角色甲" / "中立_neutral"
        angry = self.inbox / "角色甲" / "生气_angry"
        first_path = neutral / "【中立_neutral】你好，世界。.wav"
        second_path = angry / "【生气_angry】别再打扰我！.wav"
        write_test_wav(first_path, sample_rate=44_100)
        write_test_wav(second_path, sample_rate=36_000)

        first = self.run_scan()
        self.assertEqual(first["discovered_count"], 2)
        self.assertEqual(first["added_count"], 2)
        self.assertEqual(first["readable_count"], 2)
        self.assertEqual(first["emotion_counts"]["中立_neutral"], 1)
        self.assertEqual(first["sample_rate_counts"], {"36000": 1, "44100": 1})

        second = self.run_scan()
        self.assertEqual(second["unchanged_count"], 2)
        self.assertEqual(second["added_count"], 0)

        moved_path = neutral / "【中立_neutral】你好，新的世界。.wav"
        first_path.rename(moved_path)
        write_test_wav(second_path, sample_rate=36_000, amplitude=2_000)
        third = self.run_scan()
        self.assertEqual(third["moved_count"], 1)
        self.assertEqual(third["changed_count"], 1)
        self.assertEqual(third["missing_count"], 0)

        moved_path.unlink()
        fourth = self.run_scan()
        self.assertEqual(fourth["missing_count"], 1)
        self.assertEqual(fourth["unchanged_count"], 1)

        with closing(sqlite3.connect(self.catalog_path)) as connection:
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            asset_count = connection.execute("SELECT count(*) FROM asset").fetchone()[0]
            missing_count = connection.execute(
                """
                SELECT count(*) FROM source_location
                WHERE availability_status = 'missing'
                """
            ).fetchone()[0]
        self.assertEqual(schema_version, SCHEMA_VERSION)
        self.assertEqual(asset_count, 3)
        self.assertEqual(missing_count, 2)

    def test_corrupt_and_unparsed_audio_is_reported_not_hidden(self) -> None:
        bad_path = self.inbox / "角色甲" / "未知" / "没有规范前缀.wav"
        bad_path.parent.mkdir(parents=True)
        bad_path.write_bytes(b"not a wav")

        summary = self.run_scan()

        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual(summary["discovered_count"], 1)
        self.assertEqual(summary["readable_count"], 0)
        self.assertEqual(summary["error_count"], 1)
        report = json.loads(
            (self.report_dir / "inventory.json").read_text(encoding="utf-8")
        )
        record = report["files"][0]
        self.assertEqual(record["parse_status"], "error")
        self.assertEqual(record["probe_status"], "error")
        self.assertIn("filename does not match", record["scan_error"])

    def test_directory_and_filename_emotions_must_match(self) -> None:
        path = (
            self.inbox
            / "角色甲"
            / "开心_happy"
            / "【生气_angry】标签冲突也必须进入目录。.wav"
        )
        write_test_wav(path)
        summary = self.run_scan()

        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["review_required_count"], 1)
        report = json.loads(
            (self.report_dir / "inventory.json").read_text(encoding="utf-8")
        )
        record = report["files"][0]
        self.assertEqual(record["metadata_status"], "review_required")
        self.assertEqual(record["emotion_weak_label"], "开心_happy")
        self.assertEqual(record["filename_emotion_weak_label"], "生气_angry")
        self.assertIn("does not match", record["metadata_review_reason"])

    @unittest.skipUnless(os.name == "nt", "Windows path diagnostic")
    def test_near_legacy_windows_path_limit_is_reported(self) -> None:
        parent = self.inbox / "角色甲" / "中立_neutral"
        filename = "【中立_neutral】较长路径测试.wav"
        path = parent / filename
        while len(str(path.resolve())) < 245:
            parent = parent / ("层级路径" * 5)
            path = parent / filename
        write_test_wav(path)

        summary = self.run_scan()

        self.assertEqual(summary["discovered_count"], 1)
        self.assertGreaterEqual(summary["path_warning_count"], 1)

    def test_report_failure_marks_run_failed(self) -> None:
        path = self.inbox / "角色甲" / "中立_neutral" / "【中立_neutral】测试。.wav"
        write_test_wav(path)

        with mock.patch(
            "dots_tts_lab.inventory.write_inventory_reports",
            side_effect=OSError("synthetic report failure"),
        ):
            with self.assertRaisesRegex(OSError, "synthetic report failure"):
                self.run_scan()

        with closing(sqlite3.connect(self.catalog_path)) as connection:
            status, error_message = connection.execute(
                """
                SELECT status, error_message FROM ingest_run
                ORDER BY started_at DESC LIMIT 1
                """
            ).fetchone()
            location_count = connection.execute(
                "SELECT count(*) FROM source_location"
            ).fetchone()[0]
        self.assertEqual(status, "failed")
        self.assertIn("synthetic report failure", error_message)
        self.assertEqual(location_count, 0)


class CatalogTests(unittest.TestCase):
    def test_failed_run_is_never_marked_successful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(Path(directory) / "catalog.sqlite")
            catalog.initialize()
            run_id = catalog.begin_run(
                root_path="D:/example",
                root_key="d:/example",
                started_at="2026-08-26T00:00:00+00:00",
            )
            catalog.fail_run(
                run_id=run_id,
                finished_at="2026-08-26T00:00:01+00:00",
                error_message="synthetic failure",
            )
            row = catalog.get_run(run_id)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_message"], "synthetic failure")


if __name__ == "__main__":
    unittest.main()
