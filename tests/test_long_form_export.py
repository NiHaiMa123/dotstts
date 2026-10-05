from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from dots_tts_lab.long_form_export import export_long_form_dataset


class LongFormExportTests(unittest.TestCase):
    def test_export_is_verified_normal_only_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "segments").mkdir(parents=True)
            manifest_rows = []
            for index in range(2):
                segment_id = str(index + 1) * 64
                audio = source / "segments" / f"{segment_id}.wav"
                sf.write(audio, np.zeros(96000, dtype=np.float64), 48000, subtype="PCM_24")
                audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
                manifest_rows.append(
                    {
                        "segment_id": segment_id,
                        "derived_relative_path": f"segments/{segment_id}.wav",
                        "derived_audio_sha256": audio_hash,
                        "duration_seconds": 2.0,
                        "source_start_frame": index * 96000,
                        "source_end_frame": (index + 1) * 96000,
                        "source_spans": [{"source_start_frame": index * 96000, "source_end_frame": (index + 1) * 96000}],
                        "parent_segment_id": "a" * 64,
                    }
                )
            manifest = {
                "source_sha256": "b" * 64,
                "config_sha256": "c" * 64,
                "segments": manifest_rows,
            }
            manifest_path = source / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            decisions = [
                {
                    "segment_id": manifest_rows[0]["segment_id"],
                    "decision": "keep",
                    "confirmed_style": "normal",
                    "confirmed_text": "测试文本",
                    "review_batch_id": "batch",
                    "payload_sha256": "d" * 64,
                },
                {
                    "segment_id": manifest_rows[1]["segment_id"],
                    "decision": "keep",
                    "confirmed_style": "whisper",
                    "confirmed_text": "耳语文本",
                    "review_batch_id": "batch",
                    "payload_sha256": "d" * 64,
                },
            ]
            snapshot = {
                "source_sha256": manifest["source_sha256"],
                "config_sha256": manifest["config_sha256"],
                "manifest_sha256": manifest_hash,
                "complete": True,
                "reference_segment_id": manifest_rows[0]["segment_id"],
                "decisions": decisions,
            }
            snapshot_path = source / "review.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            config_path = root / "export.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "config_id": "long_form_dataset_fragment",
                        "config_version": 1,
                        "voice_id": "test",
                        "speaker_id": "测试",
                        "emotion_primary": "中立_neutral",
                        "accepted_styles": ["normal"],
                        "output_root": "unused",
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            target = root / "export"
            first = export_long_form_dataset(
                manifest_path, snapshot_path, config_path=config_path, target=target
            )
            second = export_long_form_dataset(
                manifest_path, snapshot_path, config_path=config_path, target=target
            )
            self.assertEqual((first["action"], second["action"]), ("built", "cached"))
            self.assertEqual(first["item_count"], 1)
            exported = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["rows"][0]["text"], "测试文本")
            excluded = json.loads((target / "excluded.json").read_text(encoding="utf-8"))
            self.assertEqual(excluded[0]["confirmed_style"], "whisper")


if __name__ == "__main__":
    unittest.main()
