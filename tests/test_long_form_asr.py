from __future__ import annotations

import unittest

from dots_tts_lab.long_form_asr import (
    attach_clip_asr_candidate,
    refine_region_boundaries,
    validated_timestamp_words,
)
from dots_tts_lab.long_form_contract import load_long_form_config


class LongFormAsrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_long_form_config()

    @staticmethod
    def metadata(words: list[dict]) -> dict:
        return {
            "timestamp_segments": [
                {
                    "start": 0.0,
                    "end": max((word["end"] for word in words), default=0.0),
                    "text": "".join(word["text"] for word in words),
                    "words": words,
                }
            ]
        }

    def test_timestamp_validation_rejects_non_monotonic_words(self) -> None:
        metadata = self.metadata(
            [
                {"start": 1.0, "end": 2.0, "text": "前", "probability": 0.9},
                {"start": 0.5, "end": 1.0, "text": "后", "probability": 0.9},
            ]
        )
        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            validated_timestamp_words(metadata, audio_duration_seconds=3.0)

    def test_long_region_is_split_at_word_boundaries(self) -> None:
        words = [
            {"start": float(index), "end": float(index + 1), "text": str(index), "probability": 0.9}
            for index in range(30)
        ]
        pieces = refine_region_boundaries(
            source_start_frame=48000,
            source_end_frame=31 * 48000,
            source_sample_rate_hz=48000,
            words=words,
            config=self.config,
        )
        self.assertGreaterEqual(len(pieces), 3)
        self.assertTrue(all(2.0 <= piece["duration_seconds"] <= 15.0 for piece in pieces))
        self.assertTrue(all(piece["boundary_reason"] == "asr_word_boundary" for piece in pieces))
        self.assertTrue(all(piece["asr_candidate_text"] for piece in pieces))

    def test_missing_boundary_words_fails_closed(self) -> None:
        pieces = refine_region_boundaries(
            source_start_frame=0,
            source_end_frame=31 * 48000,
            source_sample_rate_hz=48000,
            words=[],
            config=self.config,
        )
        self.assertEqual(pieces, [])

    def test_asr_text_stays_pending_for_human_confirmation(self) -> None:
        words = [
            {"start": 0.0, "end": 1.0, "text": "测试", "probability": 0.8}
        ]
        result = attach_clip_asr_candidate(
            {"duration_seconds": 2.0},
            hypothesis="测试",
            metadata=self.metadata(words),
        )
        self.assertEqual(result["asr_candidate_text"], "测试")
        self.assertEqual(result["text_review_status"], "pending")
        self.assertIsNone(result["human_confirmed_text"])

