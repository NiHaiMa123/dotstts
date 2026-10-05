from __future__ import annotations

import hashlib
import shutil
import sqlite3
import struct
import tempfile
import unittest
import wave
from contextlib import closing
from pathlib import Path
from unittest import mock

from dots_tts_lab.ingest import run_ingest


def write_test_wav(path: Path, *, amplitude: int = 1_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(
        struct.pack("<h", amplitude if index % 2 == 0 else -amplitude)
        for index in range(1_600)
    )
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(frames)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ImmutableIngestIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.inbox = self.base / "data" / "inbox"
        self.raw = self.base / "data" / "raw" / "sha256"
        self.catalog = self.base / "data" / "catalog" / "catalog.sqlite"
        self.inventory_reports = self.base / "data" / "reports" / "inventory"
        self.ingest_reports = self.base / "data" / "reports" / "ingest"
        self.inbox.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_import(self) -> dict:
        return run_ingest(
            self.inbox,
            catalog_path=self.catalog,
            raw_dir=self.raw,
            inventory_report_dir=self.inventory_reports,
            report_dir=self.ingest_reports,
        )

    def test_copy_is_deduplicated_idempotent_and_survives_source_deletion(
        self,
    ) -> None:
        first_source = (
            self.inbox / "角色甲" / "中立_neutral" / "【中立_neutral】你好。.wav"
        )
        second_source = (
            self.inbox / "角色甲" / "开心_happy" / "【开心_happy】重复内容。.wav"
        )
        write_test_wav(first_source)
        second_source.parent.mkdir(parents=True)
        shutil.copyfile(first_source, second_source)
        expected_sha256 = file_sha256(first_source)

        first = self.run_import()
        raw_files = list(self.raw.rglob("*.wav"))
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(first["discovered_count"], 2)
        self.assertEqual(first["unique_asset_count"], 1)
        self.assertEqual(first["imported_count"], 1)
        self.assertEqual(first["reused_count"], 0)
        self.assertEqual(len(raw_files), 1)
        self.assertEqual(file_sha256(raw_files[0]), expected_sha256)
        self.assertIn(expected_sha256[:2], raw_files[0].parts)

        second = self.run_import()
        self.assertEqual(second["imported_count"], 0)
        self.assertEqual(second["reused_count"], 1)
        self.assertEqual(len(list(self.raw.rglob("*.wav"))), 1)

        first_source.unlink()
        second_source.unlink()
        third = self.run_import()
        self.assertEqual(third["discovered_count"], 0)
        self.assertTrue(raw_files[0].is_file())
        self.assertEqual(file_sha256(raw_files[0]), expected_sha256)

        with closing(sqlite3.connect(self.catalog)) as connection:
            raw_count = connection.execute(
                "SELECT count(*) FROM raw_object"
            ).fetchone()[0]
            source_name_count = connection.execute(
                "SELECT count(DISTINCT original_name) FROM import_item"
            ).fetchone()[0]
        self.assertEqual(raw_count, 1)
        self.assertEqual(source_name_count, 2)

    def test_interrupted_copy_removes_partial_and_next_run_recovers(self) -> None:
        source = (
            self.inbox / "角色甲" / "中立_neutral" / "【中立_neutral】中断恢复。.wav"
        )
        write_test_wav(source)

        def fail_copy(source_file, destination_file):
            destination_file.write(source_file.read(64))
            raise OSError("synthetic interruption")

        with mock.patch("dots_tts_lab.ingest._copy_stream", side_effect=fail_copy):
            failed = self.run_import()

        self.assertEqual(failed["status"], "completed_with_errors")
        self.assertEqual(failed["error_count"], 1)
        self.assertEqual(list(self.raw.rglob("*.partial")), [])
        self.assertEqual(list(self.raw.rglob("*.wav")), [])

        sha256 = file_sha256(source)
        orphan = self.raw / sha256[:2] / f".{sha256}.crashed.partial"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"incomplete")
        recovered = self.run_import()
        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["imported_count"], 1)
        self.assertEqual(recovered["orphan_partial_count"], 1)
        self.assertFalse(orphan.exists())
        self.assertEqual(len(list(self.raw.rglob("*.wav"))), 1)

    def test_keyboard_interrupt_marks_run_failed_without_final_file(self) -> None:
        source = (
            self.inbox / "角色甲" / "中立_neutral" / "【中立_neutral】用户取消。.wav"
        )
        write_test_wav(source)

        with mock.patch(
            "dots_tts_lab.ingest._copy_stream", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.run_import()

        self.assertEqual(list(self.raw.rglob("*.partial")), [])
        self.assertEqual(list(self.raw.rglob("*.wav")), [])
        with closing(sqlite3.connect(self.catalog)) as connection:
            status = connection.execute(
                "SELECT status FROM import_run ORDER BY started_at DESC LIMIT 1"
            ).fetchone()[0]
        self.assertEqual(status, "failed")

    def test_existing_raw_corruption_is_reported_and_not_overwritten(self) -> None:
        source = (
            self.inbox / "角色甲" / "中立_neutral" / "【中立_neutral】完整性。.wav"
        )
        write_test_wav(source)
        first = self.run_import()
        self.assertEqual(first["status"], "succeeded")
        raw_file = next(self.raw.rglob("*.wav"))
        original_size = raw_file.stat().st_size
        raw_file.write_bytes(b"x" * original_size)

        second = self.run_import()

        self.assertEqual(second["status"], "completed_with_errors")
        self.assertEqual(second["error_count"], 1)
        self.assertEqual(raw_file.read_bytes(), b"x" * original_size)

    def test_report_failure_leaves_verified_object_recoverable(self) -> None:
        source = (
            self.inbox / "角色甲" / "中立_neutral" / "【中立_neutral】报告失败。.wav"
        )
        write_test_wav(source)

        with mock.patch(
            "dots_tts_lab.ingest.write_ingest_reports",
            side_effect=OSError("synthetic report failure"),
        ):
            with self.assertRaisesRegex(OSError, "synthetic report failure"):
                self.run_import()

        self.assertEqual(len(list(self.raw.rglob("*.wav"))), 1)
        with closing(sqlite3.connect(self.catalog)) as connection:
            failed_status = connection.execute(
                "SELECT status FROM import_run ORDER BY started_at DESC LIMIT 1"
            ).fetchone()[0]
            raw_count = connection.execute(
                "SELECT count(*) FROM raw_object"
            ).fetchone()[0]
        self.assertEqual(failed_status, "failed")
        self.assertEqual(raw_count, 0)

        recovered = self.run_import()
        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["imported_count"], 0)
        self.assertEqual(recovered["reused_count"], 1)
        with closing(sqlite3.connect(self.catalog)) as connection:
            raw_count = connection.execute(
                "SELECT count(*) FROM raw_object"
            ).fetchone()[0]
        self.assertEqual(raw_count, 1)


if __name__ == "__main__":
    unittest.main()
