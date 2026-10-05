from __future__ import annotations

import json
import sqlite3
import struct
import tempfile
import unittest
import wave
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from dots_tts_lab.inventory import run_inventory
from dots_tts_lab.metadata import MetadataProfile, parse_metadata


def profile_payload(*, version: int = 1) -> dict:
    return {
        "schema_version": 1,
        "profile_id": "test_speaker_emotion_filename",
        "profile_version": version,
        "speaker_from": "level_1_directory",
        "emotion_from": "level_2_directory",
        "filename_pattern": r"^【(?P<emotion>[^】]+)】(?P<text>.+)$",
        "filename_emotion_group": "emotion",
        "transcript_group": "text",
        "trim_values": True,
        "conflict_policy": "review_required",
    }


def write_test_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(struct.pack("<h", 500) for _ in range(800))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(frames)


class MetadataProfileUnitTests(unittest.TestCase):
    def test_unknown_fields_bad_selectors_regex_and_groups_are_rejected(self) -> None:
        payload = profile_payload()
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            MetadataProfile.model_validate(payload, strict=True)

        payload = profile_payload()
        payload["speaker_from"] = "parent"
        with self.assertRaisesRegex(ValidationError, "level_1_directory"):
            MetadataProfile.model_validate(payload, strict=True)

        payload = profile_payload()
        payload["filename_pattern"] = "(unclosed"
        with self.assertRaisesRegex(ValidationError, "invalid filename_pattern"):
            MetadataProfile.model_validate(payload, strict=True)

        payload = profile_payload()
        payload["filename_pattern"] = r"^(?P<text>.+)$"
        with self.assertRaisesRegex(ValidationError, "emotion"):
            MetadataProfile.model_validate(payload, strict=True)

    def test_directory_levels_are_profile_driven(self) -> None:
        payload = profile_payload()
        payload["speaker_from"] = "level_2_directory"
        payload["emotion_from"] = "level_1_directory"
        profile = MetadataProfile.model_validate(payload, strict=True)

        parsed = parse_metadata(
            Path("开心_happy/角色乙/【开心_happy】配置决定层级.wav"),
            profile=profile,
        )

        self.assertEqual(parsed["metadata_status"], "ok")
        self.assertEqual(parsed["speaker_id"], "角色乙")
        self.assertEqual(parsed["emotion_weak_label"], "开心_happy")
        self.assertEqual(parsed["transcript_candidate"], "配置决定层级")


class MetadataProfileCatalogTests(unittest.TestCase):
    def test_profile_drift_is_blocked_and_version_bump_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inbox = base / "inbox"
            source = inbox / "角色甲" / "中立_neutral" / "【中立_neutral】版本。.wav"
            write_test_wav(source)
            catalog_path = base / "catalog.sqlite"
            profile_path = base / "profile.yaml"
            report_dir = base / "reports"

            payload = profile_payload(version=1)
            profile_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            first = run_inventory(
                inbox,
                catalog_path=catalog_path,
                report_dir=report_dir,
                metadata_profile_path=profile_path,
            )
            self.assertEqual(first["status"], "succeeded")

            payload["trim_values"] = False
            profile_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "without a version bump"):
                run_inventory(
                    inbox,
                    catalog_path=catalog_path,
                    report_dir=report_dir,
                    metadata_profile_path=profile_path,
                )

            payload["profile_version"] = 2
            profile_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            second = run_inventory(
                inbox,
                catalog_path=catalog_path,
                report_dir=report_dir,
                metadata_profile_path=profile_path,
            )
            self.assertEqual(second["status"], "succeeded")

            with closing(sqlite3.connect(catalog_path)) as connection:
                profile_count = connection.execute(
                    "SELECT count(*) FROM metadata_profile"
                ).fetchone()[0]
                parse_count = connection.execute(
                    "SELECT count(*) FROM metadata_parse"
                ).fetchone()[0]
                latest_version = connection.execute(
                    """
                    SELECT parser_profile_version FROM source_location
                    WHERE availability_status = 'available'
                    """
                ).fetchone()[0]
            self.assertEqual(profile_count, 2)
            self.assertEqual(parse_count, 2)
            self.assertEqual(latest_version, 2)


if __name__ == "__main__":
    unittest.main()
