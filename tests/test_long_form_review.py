from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dots_tts_lab.long_form_review import (
    apply_long_form_review,
    build_long_form_review,
    build_long_form_review_index,
)


class LongFormReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "segments").mkdir()
        self.segment_ids = ["a" * 64, "b" * 64]
        self.manifest = self.root / "manifest.json"
        segments = [self.segment(index) for index in range(2)]
        for index, segment in enumerate(segments):
            audio = self.root / segment["derived_relative_path"]
            audio.write_bytes(f"synthetic audio {index}".encode())
            segment["derived_audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
        payload = {
            "source_sha256": "c" * 64,
            "source_relative_path": "source.wav",
            "config_sha256": "d" * 64,
            "asr_status": "succeeded",
            "source_scan": {"sample_rate_hz": 48000},
            "segments": segments,
        }
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def segment(self, index: int) -> dict:
        return {
            "segment_id": self.segment_ids[index],
            "derived_relative_path": f"segments/{self.segment_ids[index]}.wav",
            "source_start_frame": index * 48000,
            "duration_seconds": 3.0,
            "asr_candidate_text": f"文本{index}",
            "asr_mean_word_probability": 0.9,
            "style_suggestion": "normal",
            "style_cluster_id": "style_000",
            "speaker_cluster_id": "speaker_000",
            "speaker_cluster_size": 2,
            "speaker_cluster_is_dominant": True,
            "snr_proxy_db": 20.0,
            "stereo_correlation": 0.9,
            "pan_standard_deviation": 0.01,
            "boundary_reason": "vad_region",
            "review_reasons": ["target_reference_required"],
        }

    def decisions_payload(self, batch_id: str = "batch-1") -> dict:
        return {
            "schema_version": 1,
            "review_batch_id": batch_id,
            "source_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "reference_segment_id": self.segment_ids[0],
            "decisions": [
                {
                    "segment_id": self.segment_ids[0],
                    "decision": "keep",
                    "confirmed_style": "normal",
                    "confirmed_text": "确认文本",
                    "note": "",
                }
            ],
        }

    def test_review_page_has_persistent_save_status_and_relative_audio(self) -> None:
        result = build_long_form_review(self.manifest)
        content = Path(result["review_path"]).read_text(encoding="utf-8")
        self.assertIn("保存本条", content)
        self.assertIn("本条状态", content)
        self.assertIn("localStorage", content)
        self.assertIn("segments/", content)
        self.assertNotIn(str(self.root.resolve()), content)

    def test_partial_review_is_append_only_idempotent_and_incomplete(self) -> None:
        decisions = self.root / "decisions.json"
        decisions.write_text(json.dumps(self.decisions_payload()), encoding="utf-8")
        first = apply_long_form_review(self.manifest, decisions)
        second = apply_long_form_review(self.manifest, decisions)
        self.assertEqual(first["action"], "applied")
        self.assertEqual(second["action"], "cached")
        self.assertEqual(first["reviewed_count"], 1)
        self.assertFalse(first["complete"])
        self.assertEqual(len(Path(first["log_path"]).read_text(encoding="utf-8").splitlines()), 1)

    def test_reference_must_be_kept_and_text_confirmed(self) -> None:
        payload = self.decisions_payload()
        payload["decisions"][0]["decision"] = "reject"
        decisions = self.root / "invalid.json"
        decisions.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference_segment_id"):
            apply_long_form_review(self.manifest, decisions)

    def test_review_index_distinguishes_complete_and_pending_sources(self) -> None:
        work = self.root / "work"
        pending_root = work / "sources" / "aa" / ("a" * 64) / "refinements" / "cfg"
        complete_root = work / "sources" / "bb" / ("b" * 64) / "refinements" / "cfg"
        for index, target in enumerate((pending_root, complete_root)):
            target.mkdir(parents=True)
            payload = {
                "source_sha256": ("a" if index == 0 else "b") * 64,
                "source_relative_path": f"source-{index}.mp3",
                "config_sha256": "c" * 64,
                "segment_count": 1,
                "refinement": {"reserve_count": index, "auto_excluded_count": 2},
                "segments": [{"duration_seconds": 2.0}],
            }
            manifest = target / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            (target / "review.html").write_text("review", encoding="utf-8")
            if index == 1:
                reviews = target / "reviews"
                reviews.mkdir()
                snapshot = {
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "complete": True,
                    "reviewed_count": 1,
                }
                (reviews / "review_snapshot.json").write_text(
                    json.dumps(snapshot), encoding="utf-8"
                )
        result = build_long_form_review_index(
            work, output_path=self.root / "review_index.html"
        )
        self.assertEqual(result["pending_source_count"], 1)
        self.assertEqual(result["complete_source_count"], 1)
        page = (self.root / "review_index.html").read_text(encoding="utf-8")
        self.assertIn("待审核 1 个源", page)
        self.assertIn("已完成", page)
