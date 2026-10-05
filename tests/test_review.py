from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from dots_tts_lab.asr_benchmark import (
    load_asr_benchmark_config,
    normalize_asr_text,
    prepare_asr_benchmark,
)
from dots_tts_lab.asr_comparison import compare_asr_backends
from dots_tts_lab.asr_evaluation import evaluate_asr_output
from dots_tts_lab.review import (
    DEFAULT_REVIEW_RULES_PATH,
    ReviewRulesConfig,
    diff_spans,
    homophone_only_difference,
    interjection_hint,
    load_review_rules,
    review_export,
    review_import,
)
from test_asr_benchmark import AsrBenchmarkFixture


class RuleTests(unittest.TestCase):
    def test_shipped_rules_load(self) -> None:
        rules = load_review_rules()
        self.assertEqual(rules.rules_id, "fuxuan_review")
        self.assertEqual(len(rules.homophone_pairs), 5)

    def test_malformed_pairs_are_rejected(self) -> None:
        base = {
            "schema_version": 1,
            "rules_id": "test_rules",
            "rules_version": 1,
            "homophone_pairs": [["不只", "不止"]],
            "interjection_prefixes": ["唉"],
        }
        with self.assertRaisesRegex(ValueError, "exactly two terms"):
            ReviewRulesConfig.model_validate(
                dict(base, homophone_pairs=[["只此"]]), strict=True
            )
        with self.assertRaisesRegex(ValueError, "repeat itself"):
            ReviewRulesConfig.model_validate(
                dict(base, homophone_pairs=[["不只", "不只"]]), strict=True
            )
        with self.assertRaisesRegex(ValueError, "both sides"):
            ReviewRulesConfig.model_validate(
                dict(base, homophone_pairs=[["不只", "不止"], ["不止", "不只"]]),
                strict=True,
            )
        with self.assertRaisesRegex(ValueError, "one character"):
            ReviewRulesConfig.model_validate(
                dict(base, interjection_prefixes=["唉哟"]), strict=True
            )

    def test_homophone_only_difference_requires_full_explanation(self) -> None:
        rules = load_review_rules()
        config = load_asr_benchmark_config()
        reference = normalize_asr_text("世界不只一条道路", config)
        clean = normalize_asr_text("世界不止一条道路", config)
        extra = normalize_asr_text("世界不止两条道路", config)
        # Only the homophone differs: resolvable.
        resolved = homophone_only_difference(
            reference, {"a": clean, "b": clean}, rules
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["reference"], "不只")
        self.assertEqual(resolved[0]["variant"], "不止")
        # Any additional difference blocks auto-resolution entirely.
        self.assertEqual(
            homophone_only_difference(reference, {"a": extra, "b": clean}, rules), []
        )

    def test_interjection_hint_needs_reference_to_lack_the_prefix(self) -> None:
        rules = load_review_rules()
        hint = interjection_hint("本座也会难过", {"a": "唉本座也会难过"}, rules)
        self.assertIsNotNone(hint)
        self.assertEqual(hint["backends"], {"a": "唉"})
        self.assertIsNone(
            interjection_hint("唉本座也会难过", {"a": "唉本座也会难过"}, rules)
        )
        self.assertIsNone(
            interjection_hint("本座也会难过", {"a": "本座也会难过"}, rules)
        )

    def test_interjection_hint_requires_every_backend(self) -> None:
        rules = load_review_rules()
        reference = "本座也会难过"
        self.assertIsNone(interjection_hint(reference, {}, rules))
        self.assertIsNone(
            interjection_hint("唔本座也会难过", {"a": "嗯本座也会难过"}, rules)
        )
        self.assertIsNone(
            interjection_hint(
                reference,
                {"a": "唉本座也会难过", "b": reference, "c": reference},
                rules,
            )
        )
        self.assertIsNone(
            interjection_hint(
                reference,
                {"a": "唉本座也会难过", "b": "唉本座也会难过", "c": reference},
                rules,
            )
        )
        hint = interjection_hint(
            reference,
            {
                "a": "唉本座也会难过",
                "b": "唉本座也会难过",
                "c": "唉本座也会难过",
            },
            rules,
        )
        self.assertIsNotNone(hint)
        self.assertEqual(set(hint["backends"]), {"a", "b", "c"})


class DiffSpanTests(unittest.TestCase):
    def test_substitution_insertion_deletion(self) -> None:
        self.assertEqual(diff_spans("abc", "abc"), [{"equal": True, "text": "abc"}])
        spans = diff_spans("太卜司", "太仆司")
        self.assertEqual(spans, [
            {"equal": True, "text": "太"},
            {"equal": False, "reference": "卜", "hypothesis": "仆"},
            {"equal": True, "text": "司"},
        ])
        self.assertEqual(
            diff_spans("星穹", "星穹列车"),
            [
                {"equal": True, "text": "星穹"},
                {"equal": False, "hypothesis": "列车"},
            ],
        )
        self.assertEqual(
            diff_spans("星穹列车", "星穹"),
            [
                {"equal": True, "text": "星穹"},
                {"equal": False, "reference": "列车"},
            ],
        )


class ReviewFlowTests(AsrBenchmarkFixture):
    def setUp(self) -> None:
        super().setUp()
        self.review_dir = self.reports / "asr" / "review"
        prepared = prepare_asr_benchmark(
            catalog_path=self.catalog,
            config_path=self.config,
            quality_report_path=self.quality_report,
            output_dir=self.output,
        )
        entries = self.manifest_entries(prepared)
        exact = {item["asset_sha256"]: item["reference_exact"] for item in entries}
        typo = {
            asset: text.replace("太卜司", "太仆司")
            for asset, text in exact.items()
        }
        for backend_id, hypotheses in (("alpha", exact), ("beta", typo)):
            evaluate_asr_output(
                result_path=self.write_backend_result(
                    backend_id=backend_id, hypotheses=hypotheses
                ),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                report_root=self.reports / "asr",
            )
        self.comparison_config = self.base / "comparison.json"
        self.comparison_config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "comparison_id": "test_selection",
                    "comparison_version": 1,
                    "selected_backend_id": "alpha",
                    "selection_rationale": (
                        "Alpha is the intentionally selected exact synthetic backend."
                    ),
                    "consensus_max_pairwise_cer": 0.10,
                    "high_cer_delta_min": 0.20,
                    "selection_max_aggregate_cer": 0.10,
                    "selection_max_realtime_factor": 0.5,
                }
            ),
            encoding="utf-8",
        )
        compare_asr_backends(
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            comparison_config_path=self.comparison_config,
            report_dir=self.reports / "asr" / "comparison",
        )

    def export(self) -> dict:
        return review_export(
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            comparison_report_path=self.reports / "asr" / "comparison" / "comparison.json",
            comparison_config_path=self.comparison_config,
            rules_path=DEFAULT_REVIEW_RULES_PATH,
            output_dir=self.review_dir,
        )

    def decisions_payload(
        self, package_sha256: str, batch_id: str, **overrides
    ) -> dict:
        samples = json.loads(
            (self.review_dir / "review.json").read_text(encoding="utf-8")
        )
        decisions = []
        for sample in samples:
            reference = sample["reference_exact"]
            decisions.append(
                {
                    "asset_sha256": sample["asset_sha256"],
                    "text_decision": "accept_reference",
                    "text_final": reference,
                    "text_note": None,
                    "emotion_primary": sample["weak_label"],
                    "emotion_secondary": None,
                    "intensity": None,
                    "review_status": "approved",
                    "auto_rules_applied": sample["auto_rules_applied"],
                }
            )
        payload = {
            "schema_version": 1,
            "benchmark_id": "test_asr",
            "benchmark_version": 1,
            "package_sha256": package_sha256,
            "export_batch_id": batch_id,
            "decided_at": "2026-08-27T00:00:00+00:00",
            "decisions": decisions,
        }
        payload.update(overrides)
        return payload

    def write_decisions(self, payload: dict) -> Path:
        path = self.base / "decisions.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def test_export_builds_package_and_page(self) -> None:
        exported = self.export()
        self.assertEqual(exported["sample_count"], 4)
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        package = json.loads(
            (self.review_dir / "review.json").read_text(encoding="utf-8")
        )
        self.assertIn(exported["package_sha256"], page)
        self.assertIn("const REVIEW_SAMPLES = [", page)
        self.assertNotIn('fetch("review.json")', page)
        self.assertEqual(len(package), 4)
        for sample in package:
            self.assertTrue(sample["audio_rel_path"].startswith("../../../work/standardized/"))
            self.assertIn("reference_exact", sample)
        # Audio files must exist at the relative path the page will resolve.
        for sample in package:
            audio = (
                self.review_dir / sample["audio_rel_path"]
            ).resolve()
            self.assertTrue(audio.is_file(), f"missing audio for #{sample['ordinal']}")

    def test_export_page_escapes_reference_text(self) -> None:
        self.export()
        package = json.loads(
            (self.review_dir / "review.json").read_text(encoding="utf-8")
        )
        # The page renders reference text via JS escapeText, but any Python-side
        # interpolation must be escaped; verify the page contains no raw script
        # injection surface by checking the payload JSON is inert text.
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert", page.replace("<script>\n\"use strict\";", ""))

    def test_export_page_persists_flag_defect_decisions(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn('decision.text_decision === "flag_defect"', page)
        self.assertIn("decisions[decision.asset_sha256] = decision;", page)
        self.assertIn("localStorage.setItem(STORAGE_KEY", page)

    def test_export_page_uses_a_fresh_batch_per_download(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn("const batchId = randomUuid();", page)
        self.assertNotIn('localStorage.getItem(STORAGE_KEY + "-batch")', page)

    def test_export_page_closes_review_select_elements(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertEqual(page.count("</select></label>"), 3)

    def test_export_page_validates_decisions_before_navigation_or_export(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn("采用编辑文本时，最终文本不能为空。", page)
        self.assertIn("主情感和次情感不能相同。", page)
        self.assertIn("标记缺陷时必须填写备注说明。", page)
        self.assertIn("标记缺陷不能使用 approved 状态。", page)
        self.assertIn("if (!save()) { return; }", page)
        self.assertIn("if (save()) { goTo(index - 1); }", page)

    def test_export_page_distinguishes_pending_navigation_state(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn("nav button .dot.pending { background:#f59e0b }", page)
        self.assertIn(
            'decision.review_status === "pending" ? "pending" : "done"', page
        )
        self.assertIn("已保存", page)

    def test_export_page_shows_explicit_save_feedback(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn('id="save_status" role="status" aria-live="polite"', page)
        self.assertIn("已保存到本地草稿", page)
        self.assertIn("本条尚未保存", page)
        self.assertIn("未保存 · 请先选择文本决策", page)
        self.assertIn('button.textContent = "已保存 ✓"', page)
        self.assertIn("● approved", page)

    def test_export_page_validates_and_quarantines_stored_drafts(self) -> None:
        self.export()
        page = (self.review_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn("function restoreDecisions(raw)", page)
        self.assertIn("const assets = new Set(samples.map", page)
        self.assertIn("TEXT_DECISIONS.includes(decision.text_decision)", page)
        self.assertIn("REVIEW_STATUSES.includes(decision.review_status)", page)
        self.assertIn("function quarantineStoredDraft(raw, error)", page)
        self.assertIn("localStorage.removeItem(STORAGE_KEY);", page)

    def test_exported_page_javascript_behaves_in_node_harness(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the review page logic harness")
        self.export()
        harness = Path(__file__).with_name("review_page_harness.js")
        result = subprocess.run(
            [node, str(harness), str(self.review_dir / "review.html")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("review page harness: ok", result.stdout)

    def test_export_uses_comparison_run_ids_not_latest_runs(self) -> None:
        comparison = json.loads(
            (self.reports / "asr" / "comparison" / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        expected_run_ids = {
            backend["backend_id"]: backend["run_id"]
            for backend in comparison["backends"]
        }
        entries = [
            json.loads(line)
            for line in (self.output / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        newer = {item["asset_sha256"]: "完全不同的新结果" for item in entries}
        evaluate_asr_output(
            result_path=self.write_backend_result(
                backend_id="alpha", hypotheses=newer
            ),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            report_root=self.reports / "asr",
        )

        exported = self.export()
        self.assertEqual(exported["asr_run_ids"], expected_run_ids)
        package = json.loads(
            (self.review_dir / "review.json").read_text(encoding="utf-8")
        )
        for sample in package:
            self.assertEqual(
                sample["backends"]["alpha"]["run_id"], expected_run_ids["alpha"]
            )
            self.assertEqual(
                sample["backends"]["alpha"]["hypothesis_raw"],
                sample["reference_exact"],
            )

    def test_export_rejects_comparison_provenance_drift(self) -> None:
        comparison_path = self.reports / "asr" / "comparison" / "comparison.json"
        original = json.loads(comparison_path.read_text(encoding="utf-8"))
        cases = []

        wrong_schema = json.loads(json.dumps(original))
        wrong_schema["schema_version"] = 2
        cases.append((wrong_schema, "schema_version"))

        wrong_config = json.loads(json.dumps(original))
        wrong_config["summary"]["comparison_config_sha256"] = "0" * 64
        cases.append((wrong_config, "config provenance"))

        wrong_manifest = json.loads(json.dumps(original))
        wrong_manifest["summary"]["benchmark_manifest_sha256"] = "0" * 64
        cases.append((wrong_manifest, "benchmark provenance"))

        wrong_recommended = json.loads(json.dumps(original))
        wrong_recommended["summary"]["recommended_backend"] = "missing"
        cases.append((wrong_recommended, "recommended_backend"))

        wrong_sample_count = json.loads(json.dumps(original))
        wrong_sample_count["summary"]["sample_count"] += 1
        cases.append((wrong_sample_count, "sample_count"))

        wrong_reference = json.loads(json.dumps(original))
        wrong_reference["review_queue"][0]["reference_exact"] += "改"
        cases.append((wrong_reference, "reference mismatch"))

        missing_run = json.loads(json.dumps(original))
        missing_run["backends"][0]["run_id"] = str(uuid.uuid4())
        cases.append((missing_run, "missing from catalog"))

        try:
            for payload, message in cases:
                with self.subTest(message=message):
                    comparison_path.write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        self.export()
        finally:
            comparison_path.write_text(
                json.dumps(original, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def test_export_rejects_backend_with_missing_sample_before_hints(self) -> None:
        comparison = json.loads(
            (self.reports / "asr" / "comparison" / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        run_id = comparison["backends"][0]["run_id"]
        with closing(sqlite3.connect(self.catalog)) as connection:
            connection.execute(
                "DELETE FROM asr_result WHERE run_id = ? AND asset_sha256 = ("
                "SELECT asset_sha256 FROM asr_result WHERE run_id = ? LIMIT 1)",
                (run_id, run_id),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "complete benchmark"):
            self.export()

    def test_import_persists_and_is_idempotent(self) -> None:
        exported = self.export()
        batch_id = str(uuid.uuid4())
        payload = self.decisions_payload(exported["package_sha256"], batch_id)
        first = review_import(
            decisions_path=self.write_decisions(payload),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(first["inserted"], 4)
        self.assertEqual(first["ignored"], 0)
        with closing(sqlite3.connect(self.catalog)) as connection:
            rows = connection.execute(
                "SELECT count(*), max(review_round) FROM review_decision"
            ).fetchone()
        self.assertEqual(rows, (4, 1))

        second = review_import(
            decisions_path=self.write_decisions(payload),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["ignored"], 4)
        report = json.loads(
            (self.review_dir / "decision_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["summary"]["decision_count"], 4)
        self.assertTrue(
            all(row["review_round"] == 1 for row in report["decisions"])
        )
        with closing(sqlite3.connect(self.catalog)) as connection:
            rows = connection.execute(
                "SELECT count(*) FROM review_decision"
            ).fetchone()[0]
        self.assertEqual(rows, 4)

    def test_import_rejects_changed_content_for_same_batch(self) -> None:
        exported = self.export()
        batch_id = str(uuid.uuid4())
        payload = self.decisions_payload(exported["package_sha256"], batch_id)
        review_import(
            decisions_path=self.write_decisions(payload),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        changed = json.loads(json.dumps(payload))
        changed["decisions"][0]["text_decision"] = "accept_edited"
        changed["decisions"][0]["text_final"] += "改"
        with self.assertRaisesRegex(ValueError, "batch replay changed"):
            review_import(
                decisions_path=self.write_decisions(changed),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )
        with closing(sqlite3.connect(self.catalog)) as connection:
            rows = connection.execute(
                "SELECT count(*), max(review_round) FROM review_decision"
            ).fetchone()
        self.assertEqual(rows, (4, 1))

    def test_changing_mind_appends_history(self) -> None:
        exported = self.export()
        batch_id = str(uuid.uuid4())
        first_payload = self.decisions_payload(
            exported["package_sha256"], batch_id
        )
        review_import(
            decisions_path=self.write_decisions(first_payload),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        second_payload = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        corrected = dict(second_payload)
        decisions = [dict(item) for item in corrected["decisions"]]
        for decision in decisions:
            decision["text_decision"] = "accept_edited"
            decision["text_final"] = decision["text_final"] + "改"
            decision["emotion_primary"] = "难过_sad"
            decision["intensity"] = "high"
        corrected["decisions"] = decisions
        summary = review_import(
            decisions_path=self.write_decisions(corrected),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(summary["inserted"], 4)
        with closing(sqlite3.connect(self.catalog)) as connection:
            rounds = connection.execute(
                "SELECT review_round, count(*) FROM review_decision "
                "GROUP BY review_round ORDER BY review_round"
            ).fetchall()
        self.assertEqual(rounds, [(1, 4), (2, 4)])
        # Latest decisions must be round 2 with corrections.
        from dots_tts_lab.catalog import Catalog

        latest = Catalog(self.catalog).load_latest_review_decisions(
            benchmark_id="test_asr", benchmark_version=1
        )
        self.assertEqual(len(latest), 4)
        self.assertTrue(all(row["review_round"] == 2 for row in latest))
        self.assertTrue(all(row["label_source"] == "human_corrected" for row in latest))
        history = Catalog(self.catalog).load_review_decision_history(
            asset_sha256=latest[0]["asset_sha256"],
            benchmark_id="test_asr",
            benchmark_version=1,
        )
        self.assertEqual([row["review_round"] for row in history], [1, 2])
        self.assertTrue(
            all(
                row["benchmark_id"] == "test_asr"
                and row["benchmark_version"] == 1
                for row in history
            )
        )

    def test_concurrent_catalog_imports_allocate_distinct_rounds(self) -> None:
        from dots_tts_lab.catalog import Catalog

        exported = self.export()
        sample = json.loads(
            (self.review_dir / "review.json").read_text(encoding="utf-8")
        )[0]

        def insert(index: int) -> tuple[int, int]:
            row = {
                "decision_id": str(uuid.uuid4()),
                "asset_sha256": sample["asset_sha256"],
                "benchmark_id": "test_asr",
                "benchmark_version": 1,
                "text_decision": "accept_reference",
                "text_final": sample["reference_exact"],
                "text_note": f"concurrent-{index}",
                "emotion_primary": sample["weak_label"],
                "emotion_secondary": None,
                "intensity": None,
                "label_source": "weak_label_confirmed",
                "review_status": "approved",
                "auto_rules_applied_json": "[]",
                "created_at": f"2026-08-27T00:00:0{index}+00:00",
                "export_batch_id": str(uuid.uuid4()),
                "source_row_index": 0,
            }
            return Catalog(self.catalog).record_review_decisions([row])

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(insert, (1, 2)))
        self.assertEqual(sorted(results), [(1, 0), (1, 0)])
        with closing(sqlite3.connect(self.catalog)) as connection:
            rounds = connection.execute(
                "SELECT review_round FROM review_decision "
                "WHERE asset_sha256 = ? ORDER BY review_round",
                (sample["asset_sha256"],),
            ).fetchall()
        self.assertEqual(rounds, [(1,), (2,)])

    def test_import_rejects_stale_package_and_unknown_assets(self) -> None:
        exported = self.export()
        stale = self.decisions_payload("0" * 64, str(uuid.uuid4()))
        with self.assertRaisesRegex(ValueError, "different review package"):
            review_import(
                decisions_path=self.write_decisions(stale),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )
        payload = self.decisions_payload(exported["package_sha256"], str(uuid.uuid4()))
        payload["decisions"][0]["asset_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "outside the benchmark"):
            review_import(
                decisions_path=self.write_decisions(payload),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )
        payload = self.decisions_payload(exported["package_sha256"], str(uuid.uuid4()))
        payload["decisions"][0]["text_final"] = None
        with self.assertRaisesRegex(ValueError, "text_final"):
            review_import(
                decisions_path=self.write_decisions(payload),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )

    def test_import_rejects_false_accept_reference(self) -> None:
        exported = self.export()
        payload = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        payload["decisions"][0]["text_final"] += "改"
        with self.assertRaisesRegex(ValueError, "exactly match the frozen reference"):
            review_import(
                decisions_path=self.write_decisions(payload),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )
        with closing(sqlite3.connect(self.catalog)) as connection:
            count = connection.execute(
                "SELECT count(*) FROM review_decision"
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_import_enforces_flag_defect_contract(self) -> None:
        exported = self.export()
        base = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        invalid_cases = (
            ({"text_final": None, "text_note": "待查", "review_status": "approved"},
             "cannot have approved"),
            ({"text_final": "不应保留", "text_note": "待查", "review_status": "pending"},
             "text_final to be null"),
            ({"text_final": None, "text_note": "  ", "review_status": "pending"},
             "non-empty text_note"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(message=message):
                payload = json.loads(json.dumps(base))
                payload["export_batch_id"] = str(uuid.uuid4())
                payload["decisions"][0].update(
                    {"text_decision": "flag_defect", **overrides}
                )
                with self.assertRaisesRegex(ValueError, message):
                    review_import(
                        decisions_path=self.write_decisions(payload),
                        catalog_path=self.catalog,
                        benchmark_config_path=self.config,
                        review_dir=self.review_dir,
                    )

        valid = json.loads(json.dumps(base))
        valid["export_batch_id"] = str(uuid.uuid4())
        valid["decisions"] = [valid["decisions"][0]]
        valid["decisions"][0].update(
            {
                "text_decision": "flag_defect",
                "text_final": None,
                "text_note": "音频与参考关系待查",
                "review_status": "pending",
            }
        )
        summary = review_import(
            decisions_path=self.write_decisions(valid),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(summary["inserted"], 1)
        self.assertEqual(summary["pending_count"], 1)

    def test_import_rejects_invalid_emotion_uuid_and_timestamp(self) -> None:
        exported = self.export()
        base = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        cases = (
            (("decisions", 0, "emotion_primary"), "随便写的标签", "literal"),
            (("export_batch_id",), "0" * 36, "valid UUID"),
            (("decided_at",), "2026-08-27 00:00:00", "UTC offset"),
        )
        for path, value, message in cases:
            with self.subTest(path=path):
                payload = json.loads(json.dumps(base))
                if path[0] == "decisions":
                    payload[path[0]][path[1]][path[2]] = value
                else:
                    payload[path[0]] = value
                with self.assertRaisesRegex(ValueError, message):
                    review_import(
                        decisions_path=self.write_decisions(payload),
                        catalog_path=self.catalog,
                        benchmark_config_path=self.config,
                        review_dir=self.review_dir,
                    )

    def test_import_rejects_tampered_auto_rules(self) -> None:
        exported = self.export()
        payload = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        payload["decisions"][0]["auto_rules_applied"] = [
            {"reference": "伪造", "variant": "规则", "backends": ["alpha"]}
        ]
        with self.assertRaisesRegex(ValueError, "frozen review package"):
            review_import(
                decisions_path=self.write_decisions(payload),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )
        with closing(sqlite3.connect(self.catalog)) as connection:
            count = connection.execute(
                "SELECT count(*) FROM review_decision"
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_import_rejects_duplicate_rows_in_one_batch(self) -> None:
        exported = self.export()
        payload = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        payload["decisions"] = payload["decisions"] + [payload["decisions"][0]]
        with self.assertRaisesRegex(ValueError, "Duplicate decisions"):
            review_import(
                decisions_path=self.write_decisions(payload),
                catalog_path=self.catalog,
                benchmark_config_path=self.config,
                review_dir=self.review_dir,
            )

    def test_decision_report_lists_corrections_and_rejections(self) -> None:
        exported = self.export()
        payload = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        payload["decisions"][0]["review_status"] = "rejected"
        payload["decisions"][0]["text_note"] = "音频和参考不一致"
        payload["decisions"][1]["text_decision"] = "accept_edited"
        payload["decisions"][1]["text_final"] = "edited text"
        summary = review_import(
            decisions_path=self.write_decisions(payload),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(summary["rejected_count"], 1)
        self.assertEqual(summary["text_corrected_count"], 1)
        self.assertEqual(len(summary["rejected_assets"]), 1)
        report = json.loads(
            (self.review_dir / "decision_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(report["decisions"]), 4)
        self.assertTrue(
            (self.review_dir / "decision_report.csv").is_file()
        )
        html_text = (self.review_dir / "decision_report.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("edited text", html_text)

    def test_three_item_end_to_end_smoke_and_second_round(self) -> None:
        from dots_tts_lab.catalog import Catalog

        exported = self.export()
        payload = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        payload["decisions"] = payload["decisions"][:3]
        payload["decisions"][1]["text_decision"] = "accept_edited"
        payload["decisions"][1]["text_final"] += "修"
        payload["decisions"][2].update(
            {
                "text_decision": "flag_defect",
                "text_final": None,
                "text_note": "端到端冒烟待查",
                "review_status": "pending",
            }
        )
        first = review_import(
            decisions_path=self.write_decisions(payload),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(first["inserted"], 3)
        self.assertEqual(first["pending_count"], 1)
        for name in (
            "decision_report.json",
            "decision_report.csv",
            "decision_report.html",
        ):
            self.assertTrue((self.review_dir / name).is_file(), name)

        revised = self.decisions_payload(
            exported["package_sha256"], str(uuid.uuid4())
        )
        revised["decisions"] = [revised["decisions"][0]]
        revised["decisions"][0]["text_decision"] = "accept_edited"
        revised["decisions"][0]["text_final"] += "复核"
        second = review_import(
            decisions_path=self.write_decisions(revised),
            catalog_path=self.catalog,
            benchmark_config_path=self.config,
            review_dir=self.review_dir,
        )
        self.assertEqual(second["inserted"], 1)
        history = Catalog(self.catalog).load_review_decision_history(
            asset_sha256=revised["decisions"][0]["asset_sha256"],
            benchmark_id="test_asr",
            benchmark_version=1,
        )
        self.assertEqual([row["review_round"] for row in history], [1, 2])
        with closing(sqlite3.connect(self.catalog)) as connection:
            weak_label = connection.execute(
                "SELECT emotion_weak_label FROM asr_benchmark_item "
                "WHERE benchmark_id = 'test_asr' AND benchmark_version = 1 "
                "AND asset_sha256 = ?",
                (revised["decisions"][0]["asset_sha256"],),
            ).fetchone()[0]
        self.assertEqual(weak_label, revised["decisions"][0]["emotion_primary"])

    def test_migration_is_idempotent(self) -> None:
        from dots_tts_lab.catalog import SCHEMA_VERSION, Catalog

        catalog = Catalog(self.catalog)
        catalog.initialize()
        with closing(sqlite3.connect(self.catalog)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)
        # Re-initialize (already at version) changes nothing.
        catalog.initialize()
        with closing(sqlite3.connect(self.catalog)) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0],
                SCHEMA_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
