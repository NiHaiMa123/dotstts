from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.slice9_candidates import load_slice9_selection_config
from dots_tts_lab.slice9_diversity import _normalized_text, _text_similarity
from dots_tts_lab.slice9_freeze import _write_frozen
from dots_tts_lab.slice9_normalization import _pool_normalize
from dots_tts_lab.slice9_sensitivity import _overlap_metrics, _variant_weights


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/reports/datasets/fuxuan_v1/analysis"


class Slice9UnitTests(unittest.TestCase):
    def test_config_and_weight_contract(self) -> None:
        config = load_slice9_selection_config(ROOT / "configs/lab/datasets/fuxuan_slice9_v1.yaml")
        self.assertEqual(config["input"]["split"], "train")
        self.assertAlmostEqual(sum(config["score"]["weights"].values()), 1.0)
        self.assertEqual(config["ranking"]["global_rank"], "forbidden")

    def test_pool_quantile_normalization_preserves_missing(self) -> None:
        normalized, stats = _pool_normalize([1.0, None, 3.0, 100.0], lower=0.05, upper=0.95, higher_is_better=True)
        self.assertIsNone(normalized[1])
        self.assertEqual(stats["missing_count"], 1)
        self.assertGreater(normalized[2], normalized[0])
        inverted, _ = _pool_normalize([1.0, 3.0], lower=0.05, upper=0.95, higher_is_better=False)
        self.assertGreater(inverted[0], inverted[1])

    def test_text_similarity_and_normalization_are_deterministic(self) -> None:
        self.assertEqual(_normalized_text("你好， 世界！"), "你好世界")
        self.assertEqual(_text_similarity("abc", "abc"), 1.0)
        self.assertEqual(_text_similarity("", "abc"), 0.0)
        self.assertEqual(_overlap_metrics(["a", "b"], ["b", "c"])["overlap_count"], 1)

    def test_weight_variant_renormalizes(self) -> None:
        base = {"objective_quality": 0.3, "speaker_similarity": 0.25, "text_provenance": 0.15, "silence": 0.1, "duration": 0.05, "phonological_coverage": 0.15}
        variant = _variant_weights(base, "objective_quality", ("text_provenance", "phonological_coverage"))
        self.assertAlmostEqual(sum(variant.values()), 1.0)
        self.assertGreater(variant["objective_quality"], base["objective_quality"])

    def test_real_slice9_reports_are_consistent(self) -> None:
        pool = json.loads((ANALYSIS / "slice9_pool_candidates_v1.json").read_text(encoding="utf-8"))
        ranking = json.loads((ANALYSIS / "slice9_ranking_v1.json").read_text(encoding="utf-8"))
        diversity = json.loads((ANALYSIS / "slice9_diversity_rerank_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(pool["candidate_count"], 219)
        self.assertEqual(pool["hard_gate_decision_counts"]["pass"], 216)
        for pool_id, details in ranking["pools"].items():
            scores = [item["composite_score"] for item in details["items"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual(len(diversity["pools"][pool_id]["selected"]), diversity["pools"][pool_id]["top_k"])

    def test_catalog_slice9_review_is_append_only_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(Path(directory) / "catalog.sqlite")
            catalog.initialize()
            asset = "a" * 64
            with catalog.session() as connection:
                connection.execute(
                    """
                    INSERT INTO asset (sha256, size_bytes, format, subtype, sample_rate,
                        channels, frames, duration_seconds, probe_status,
                        probe_error_type, probe_error_message, first_seen_at, last_seen_at)
                    VALUES (?, 1, 'wav', 'PCM_16', 48000, 1, 48000, 1.0,
                        'ok', NULL, NULL, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                    """,
                    (asset,),
                )
            row = {
                "review_id": "review-1", "selection_id": "test", "selection_version": 1,
                "asset_sha256": asset, "pool_id": "main_neutral", "selection_kind": "selected",
                "review_status": "approved", "review_note": "ok", "created_at": "2026-01-01T00:00:00Z",
                "review_batch_id": "batch-1", "source_row_index": 0, "provenance_json": "{}",
            }
            self.assertEqual(catalog.record_slice9_reference_reviews([row]), (1, 0))
            self.assertEqual(catalog.record_slice9_reference_reviews([dict(row)]), (0, 1))
            self.assertEqual(len(catalog.load_slice9_reference_reviews(selection_id="test", selection_version=1)), 1)

    def test_frozen_artifact_rejects_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            self.assertTrue(_write_frozen(path, b"one"))
            self.assertFalse(_write_frozen(path, b"one"))
            with self.assertRaises(RuntimeError):
                _write_frozen(path, b"two")


if __name__ == "__main__":
    unittest.main()

