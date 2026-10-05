from __future__ import annotations

import unittest

from dots_tts_lab.long_form_refinement import (
    build_refinement_units,
    load_refinement_config,
    split_timestamp_segments,
)


class LongFormRefinementTests(unittest.TestCase):
    def test_semantic_gap_makes_three_groups_but_keeps_middle_clause_together(self) -> None:
        segments = [
            {"start": 0.0, "end": 0.54, "text": "我回来了", "words": []},
            {"start": 4.2, "end": 4.52, "text": "怎么", "words": []},
            {"start": 5.76, "end": 6.82, "text": "就你一个人在家", "words": []},
            {"start": 9.24, "end": 9.9, "text": "我爸呢", "words": []},
        ]
        groups = split_timestamp_segments(segments, semantic_gap_seconds=1.5)
        self.assertEqual(len(groups), 3)
        self.assertEqual("".join(item["text"] for item in groups[1]), "怎么就你一个人在家")

    def test_internal_long_pause_becomes_multiple_traceable_source_spans(self) -> None:
        config = load_refinement_config()
        parent = {
            "segment_id": "a" * 64,
            "source_start_frame": 48000,
            "source_end_frame": 48000 * 12,
            "style_suggestion": "normal",
            "speaker_cluster_is_dominant": True,
        }
        asr = {
            "asset_sha256": "a" * 64,
            "status": "ok",
            "metadata": {
                "timestamp_segments": [
                    {
                        "start": 0.0,
                        "end": 0.54,
                        "text": "我回来了",
                        "words": [{"text": "我回来了", "probability": 0.9}],
                    },
                    {
                        "start": 4.2,
                        "end": 4.52,
                        "text": "怎么",
                        "words": [{"text": "怎么", "probability": 0.9}],
                    },
                    {
                        "start": 5.76,
                        "end": 6.82,
                        "text": "就你一个人在家",
                        "words": [{"text": "就你一个人在家", "probability": 0.9}],
                    },
                    {
                        "start": 9.24,
                        "end": 9.9,
                        "text": "我爸呢",
                        "words": [{"text": "我爸呢", "probability": 0.9}],
                    },
                ]
            },
        }
        units = build_refinement_units(
            [parent], [asr], config=config, source_sample_rate_hz=48000
        )
        self.assertEqual(len(units), 3)
        self.assertEqual(len(units[1]["source_spans"]), 2)
        self.assertLess(units[1]["estimated_output_seconds"], 2.1)
        self.assertEqual(units[1]["asr_candidate_text"], "怎么就你一个人在家")
        self.assertLess(units[0]["estimated_output_seconds"], config.sentence.minimum_output_seconds)


if __name__ == "__main__":
    unittest.main()
