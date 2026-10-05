from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

import yaml

from dots_tts_lab.asr_benchmark import load_asr_benchmark_config
from dots_tts_lab.asr_comparison import (
    DEFAULT_ASR_LEXICON_PATH,
    compare_asr_backends,
    load_asr_lexicon,
    normalized_terms,
    pairwise_spread,
    punctuation_profile,
    term_recall,
)
from dots_tts_lab.asr_evaluation import evaluate_asr_output, residual_symbols
from test_asr_benchmark import AsrBenchmarkFixture


class LexiconTests(unittest.TestCase):
    def test_shipped_lexicon_loads(self) -> None:
        lexicon = load_asr_lexicon()
        self.assertEqual(lexicon.lexicon_id, "fuxuan_domain")
        self.assertIn("符玄", lexicon.named_entities)
        self.assertIn("寸功不竟", lexicon.archaic_terms)

    def test_duplicate_and_cross_category_terms_are_rejected(self) -> None:
        payload = yaml.safe_load(
            DEFAULT_ASR_LEXICON_PATH.read_text(encoding="utf-8")
        )
        duplicated = dict(payload, named_entities=["符玄", "符玄"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            type(load_asr_lexicon()).model_validate(duplicated, strict=True)
        overlapping = dict(
            payload, named_entities=["符玄"], archaic_terms=["符玄"]
        )
        with self.assertRaisesRegex(ValueError, "both categories"):
            type(load_asr_lexicon()).model_validate(overlapping, strict=True)

    def test_terms_colliding_after_normalization_are_rejected(self) -> None:
        config = load_asr_benchmark_config()
        self.assertEqual(normalized_terms(["符玄"], config), {"符玄": "符玄"})
        with self.assertRaisesRegex(ValueError, "share a normalized form"):
            normalized_terms(["太卜司", "太卜司。"], config)
        with self.assertRaisesRegex(ValueError, "empty string"):
            normalized_terms(["，"], config)


class TextSignalTests(unittest.TestCase):
    def test_punctuation_profile_counts_only_punctuation(self) -> None:
        self.assertEqual(punctuation_profile("你好，世界。"), {"，": 1, "。": 1})
        self.assertEqual(punctuation_profile("没有标点"), {})

    def test_residual_symbols_flags_emoji_but_not_chinese(self) -> None:
        self.assertEqual(residual_symbols("本座也会难过"), {})
        self.assertEqual(residual_symbols("本座\U0001f621"), {"\U0001f621": 1})

    def test_pairwise_spread_is_zero_for_agreement(self) -> None:
        self.assertEqual(pairwise_spread(["符玄"]), 0.0)
        self.assertEqual(pairwise_spread(["符玄", "符玄"]), 0.0)
        self.assertAlmostEqual(pairwise_spread(["符玄", "浮玄"]), 0.5)

    def test_term_recall_counts_reference_occurrences_only(self) -> None:
        results = [
            {"reference_normalized": "太卜司正在演算", "hypothesis_normalized": "太卜司正在演算"},
            {"reference_normalized": "青雀不许青雀偷懒", "hypothesis_normalized": "青雀不许轻雀偷懒"},
            {"reference_normalized": "无关句子", "hypothesis_normalized": "太卜司"},
        ]
        recall = term_recall(results, {"太卜司": "太卜司", "青雀": "青雀"})
        self.assertEqual(recall["occurrence_count"], 3)
        self.assertEqual(recall["hit_count"], 2)
        self.assertEqual(recall["recall"], 2 / 3)
        self.assertEqual(recall["missed_terms"], {"青雀": 1})


class ComparisonTests(AsrBenchmarkFixture):
    def evaluate(self, backend_id: str, hypotheses: dict[str, str]) -> dict:
        return evaluate_asr_output(
            result_path=self.write_backend_result(
                backend_id=backend_id, hypotheses=hypotheses
            ),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            report_root=self.reports / "asr",
        )

    def compare(self, **overrides):
        comparison_config = self.base / "comparison.json"
        payload = {
            "schema_version": 1,
            "comparison_id": "test_selection",
            "comparison_version": 1,
            "selected_backend_id": "exact",
            "selection_rationale": "Exact is the deliberately selected synthetic backend.",
            "consensus_max_pairwise_cer": 0.10,
            "high_cer_delta_min": 0.20,
            "selection_max_aggregate_cer": 0.10,
            "selection_max_realtime_factor": 0.5,
        }
        payload.update(overrides)
        comparison_config.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return compare_asr_backends(
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            comparison_config_path=comparison_config,
            lexicon_path=DEFAULT_ASR_LEXICON_PATH,
            report_dir=self.reports / "asr" / "comparison",
        )

    def seed_two_backends(self) -> list[dict]:
        entries = self.manifest_entries(self.prepare())
        exact = {
            item["asset_sha256"]: item["reference_exact"] for item in entries
        }
        typo = {
            asset_sha256: text.replace("太卜司", "太仆司")
            for asset_sha256, text in exact.items()
        }
        self.evaluate("exact", exact)
        self.evaluate("typo", typo)
        return entries

    def test_ranks_backends_and_queues_the_named_entity_miss(self) -> None:
        self.seed_two_backends()
        summary = self.compare()
        self.assertEqual(summary["backend_count"], 2)
        self.assertEqual(summary["recommended_backend"], "exact")
        self.assertEqual(summary["qualified_backends"], ["exact", "typo"])
        report = json.loads(
            Path(summary["reports"]["json"]).read_text(encoding="utf-8")
        )
        by_id = {item["backend_id"]: item for item in report["backends"]}
        self.assertEqual(by_id["exact"]["cer_rank"], 1)
        self.assertEqual(by_id["exact"]["aggregate_cer"], 0.0)
        self.assertEqual(by_id["exact"]["named_entities"]["recall"], 1.0)
        self.assertGreater(by_id["typo"]["aggregate_cer"], 0.0)
        self.assertEqual(
            by_id["typo"]["named_entities"]["missed_terms"], {"太卜司": 1}
        )
        queue = report["review_queue"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["flags"], ["named_entity_miss"])
        self.assertEqual(queue[0]["priority"], "medium")
        self.assertEqual(queue[0]["missing_named_entities"], ["太卜司"])
        self.assertEqual(queue[0]["unanimous_missing_named_entities"], [])

    def test_unanimous_miss_and_consensus_conflict_raise_priority(self) -> None:
        entries = self.manifest_entries(self.prepare())
        wrong = {
            item["asset_sha256"]: item["reference_exact"].replace("太卜司", "太仆司")
            for item in entries
        }
        self.evaluate("left", wrong)
        self.evaluate("right", wrong)
        summary = self.compare(selected_backend_id="left")
        report = json.loads(
            Path(summary["reports"]["json"]).read_text(encoding="utf-8")
        )
        entry = next(
            item for item in report["review_queue"] if item["ordinal"] == 1
        )
        self.assertEqual(entry["priority"], "high")
        self.assertIn("unanimous_named_entity_miss", entry["flags"])
        self.assertIn("consensus_conflicts_reference", entry["flags"])
        self.assertEqual(entry["backend_pairwise_spread"], 0.0)
        self.assertEqual(entry["unanimous_missing_named_entities"], ["太卜司"])

    def test_gate_failure_removes_a_backend_from_selection(self) -> None:
        self.seed_two_backends()
        summary = self.compare(selection_max_aggregate_cer=0.01)
        self.assertEqual(summary["qualified_backends"], ["exact"])
        self.assertEqual(summary["recommended_backend"], "exact")
        report = json.loads(
            Path(summary["reports"]["json"]).read_text(encoding="utf-8")
        )
        by_id = {item["backend_id"]: item for item in report["backends"]}
        self.assertEqual(by_id["typo"]["gate_failures"], ["aggregate_cer"])
        self.assertFalse(by_id["typo"]["meets_selection_gates"])

    def test_configured_selection_must_pass_the_gates(self) -> None:
        self.seed_two_backends()
        with self.assertRaisesRegex(RuntimeError, "is not qualified"):
            self.compare(
                selected_backend_id="typo",
                selection_max_aggregate_cer=0.01,
            )

    def test_superseded_run_is_ignored(self) -> None:
        entries = self.seed_two_backends()
        corrected = {
            item["asset_sha256"]: item["reference_exact"] for item in entries
        }
        self.evaluate("typo", corrected)
        summary = self.compare()
        report = json.loads(
            Path(summary["reports"]["json"]).read_text(encoding="utf-8")
        )
        by_id = {item["backend_id"]: item for item in report["backends"]}
        self.assertEqual(by_id["typo"]["aggregate_cer"], 0.0)
        self.assertEqual(report["summary"]["review_queue_count"], 0)
        with closing(sqlite3.connect(self.catalog)) as connection:
            run_count = connection.execute(
                "SELECT count(*) FROM asr_run WHERE backend_id = 'typo'"
            ).fetchone()[0]
        self.assertEqual(run_count, 2)

    def test_single_backend_cannot_be_compared(self) -> None:
        entries = self.manifest_entries(self.prepare())
        self.evaluate(
            "only",
            {item["asset_sha256"]: item["reference_exact"] for item in entries},
        )
        with self.assertRaisesRegex(RuntimeError, "at least two"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
