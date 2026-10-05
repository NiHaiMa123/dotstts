from __future__ import annotations

import unittest
from pathlib import Path

from dots_tts_lab.long_form_batch import (
    assemble_sentence_candidates,
    build_strict_review_html,
    dedupe_by_text,
    rank_candidates,
)
from dots_tts_lab.long_form_strict_gate import load_strict_gate_config

ROOT = Path(__file__).resolve().parents[1]


def _span(start: int, end: int, **kw) -> dict:
    span = {
        "source_start_frame": start,
        "source_end_frame": end,
        "duration_seconds": (end - start) / 48000.0,
        "status": "candidate",
        "boundary_clean": True,
        "overlaps_quarantine": False,
        "primary_text": kw.pop("text", "你好世界"),
        "text_agreement": {"status": kw.pop("agreement", "match")},
    }
    span.update(kw)
    return span


class AssembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_strict_gate_config()
        self.transcript = {
            "source_sha256": "a" * 64,
            "sentences": [
                _span(0, 48000 * 4),
                _span(48000 * 5, 48000 * 6, status="too_short"),
                _span(48000 * 10, 48000 * 15, overlaps_quarantine=True),
            ],
        }
        self.regions = {
            (0, 48000 * 4): 0,
            (48000 * 5, 48000 * 6): 1,
            (48000 * 10, 48000 * 15): 2,
        }

    def test_quarantine_span_fails_g6(self) -> None:
        candidates = assemble_sentence_candidates(
            self.transcript,
            routes_by_region={},
            identity_by_region={},
            style_event_by_region={},
            gate_config=self.gate,
            region_of_span=self.regions,
        )
        by_span = {c["source_start_frame"]: c for c in candidates}
        self.assertIn("G6", by_span[48000 * 10]["failed_gates"])
        self.assertEqual(by_span[48000 * 10]["disposition"], "reject")
        # too_short span fails G7
        self.assertIn("G7", by_span[48000 * 5]["failed_gates"])
        # clean candidate span: no fails but unknowns → quarantine in
        # calibration mode (thresholds uncalibrated)
        self.assertEqual(by_span[0]["disposition"], "quarantine")
        self.assertTrue(by_span[0]["unknown_gates"])

    def test_ranking_prefers_agreed_in_range(self) -> None:
        candidates = assemble_sentence_candidates(
            {
                "source_sha256": "a" * 64,
                "sentences": [
                    _span(0, 48000 * 4, text="b句", agreement="text_mismatch"),
                    _span(48000 * 6, 48000 * 10, text="a句"),
                ],
            },
            routes_by_region={},
            identity_by_region={},
            style_event_by_region={},
            gate_config=self.gate,
            region_of_span={},
        )
        ranked = rank_candidates(candidates)
        by_id = {c["candidate_id"]: c for c in candidates}
        first = by_id[ranked[0]]
        self.assertEqual(first["primary_text"], "a句")

    def test_dedupe_removes_duplicate_text(self) -> None:
        candidates = assemble_sentence_candidates(
            {
                "source_sha256": "a" * 64,
                "sentences": [
                    _span(0, 48000 * 4, text="同一句"),
                    _span(48000 * 10, 48000 * 14, text="同一句"),
                    _span(48000 * 20, 48000 * 24, text="另一句"),
                ],
            },
            routes_by_region={},
            identity_by_region={},
            style_event_by_region={},
            gate_config=self.gate,
            region_of_span={},
        )
        ranked = rank_candidates(candidates)
        kept = dedupe_by_text(ranked, candidates)
        self.assertEqual(len(kept), 2)

    def test_review_page_renders_items(self) -> None:
        html = build_strict_review_html(
            title="t",
            source_sha256="a" * 64,
            items=[
                {
                    "candidate_id": "c1",
                    "raw_audio": "/x.wav",
                    "routed_audio": None,
                    "primary_text": "你好",
                    "secondary_text": "你好",
                    "duration_seconds": 4.0,
                    "disposition": "quarantine",
                    "failed_gates": [],
                    "unknown_gates": ["G3"],
                    "overlaps_quarantine": False,
                    "boundary_clean": True,
                    "event_risk_max": 0.2,
                    "route": "raw",
                    "route_reasons": [],
                }
            ],
            save_name="strict-review-test",
            batch_sha256="b" * 64,
        )
        self.assertIn("strict_review", html)
        self.assertIn("c1", html)
        self.assertIn("/x.wav", html)


if __name__ == "__main__":
    unittest.main()
