from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_transcript import (
    compare_hypotheses,
    extract_sentence_spans,
    group_regions_for_asr,
    load_transcript_config,
    normalize_zh_text,
    transcribe_source,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_transcript_tests"
SR = 48000


def _config():
    base = load_transcript_config()
    paths = base.paths.model_copy(
        update={
            "work_root": "data/work/tmp_transcript_tests/work",
            "report_root": "data/work/tmp_transcript_tests/reports",
        }
    )
    return base.model_copy(update={"paths": paths})


def _fake_primary(segments_words: list[list[dict]], hypothesis: str):
    def run(audio_dir: Path, rows: list[dict]) -> dict:
        return {
            "results": [
                {
                    "asset_sha256": row["asset_sha256"],
                    "hypothesis": hypothesis,
                    "metadata": {
                        "timestamp_segments": [
                            {"start": 0.0, "end": 1.0, "text": hypothesis,
                             "words": words}
                            for words in segments_words
                        ]
                    },
                }
                for row in rows
            ]
        }
    return run


def _fake_secondary(texts: list[str]):
    def run(audio_dir: Path, rows: list[dict]) -> dict:
        return {
            "results": [
                {"asset_sha256": row["asset_sha256"],
                 "hypothesis": texts[min(i, len(texts) - 1)]}
                for i, row in enumerate(rows)
            ]
        }
    return run


def _words(text: str, start: float, per_char: float = 0.4, gap: float = 0.05):
    out = []
    t = start
    for ch in text:
        out.append({"start": t, "end": t + per_char, "text": ch})
        t += per_char + gap
    return out


class NormalizeTests(unittest.TestCase):
    def test_normalization_strips_punct_and_folds(self) -> None:
        self.assertEqual(
            normalize_zh_text(" 你好，世界！ABC "), "你好世界abc"
        )

    def test_digits_not_folded(self) -> None:
        self.assertNotEqual(normalize_zh_text("三个"), normalize_zh_text("3个"))


class CompareTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        r = compare_hypotheses("今天天气真好。", "今天天气真好")
        self.assertEqual(r["status"], "match")

    def test_digit_disagreement(self) -> None:
        r = compare_hypotheses("他买了3个苹果", "他买了三个苹果")
        self.assertEqual(r["status"], "digit_disagreement")

    def test_coverage_gap(self) -> None:
        r = compare_hypotheses("你好", "")
        self.assertEqual(r["status"], "coverage_gap")

    def test_text_mismatch(self) -> None:
        r = compare_hypotheses("今天天气真好", "今天心情真好")
        self.assertEqual(r["status"], "text_mismatch")


class GroupRegionsTests(unittest.TestCase):
    def test_merge_usable_only_and_break_on_quarantine(self) -> None:
        regions = [
            {"source_start_frame": 0, "source_end_frame": 100, "status": "usable"},
            {"source_start_frame": 120, "source_end_frame": 200, "status": "usable"},
            {"source_start_frame": 210, "source_end_frame": 300,
             "status": "prefilter_quarantine"},
            {"source_start_frame": 310, "source_end_frame": 400, "status": "usable"},
        ]
        groups = group_regions_for_asr(
            regions, merge_gap_seconds=2.0, max_seconds=34.0, sample_rate=10
        )
        self.assertEqual(
            groups, [{"start": 0, "end": 200}, {"start": 310, "end": 400}]
        )


class SentenceSpanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _config().sentence

    def test_split_on_sentence_punctuation(self) -> None:
        words = _words("今天天气真好。" , 0.0) + _words("明天再去玩。", 4.0)
        spans = extract_sentence_spans(
            words,
            segment_start_frame=0,
            segment_end_frame=100 * SR,
            sample_rate=SR,
            config=self.cfg,
        )
        statuses = [s["status"] for s in spans]
        self.assertTrue(all(s in {"candidate", "too_short"} for s in statuses))

    def test_overlong_sentence_splits_at_pause(self) -> None:
        # 40s of words with a 1s pause at 20s → split into two candidates
        first = _words("啊" * 45, 0.0, per_char=0.4, gap=0.02)  # ~19s
        start2 = first[-1]["end"] + 1.0
        second = _words("吧" * 45, start2, per_char=0.4, gap=0.02)
        spans = extract_sentence_spans(
            first + second,
            segment_start_frame=0,
            segment_end_frame=100 * SR,
            sample_rate=SR,
            config=self.cfg,
        )
        candidates = [s for s in spans if s["status"] == "candidate"]
        self.assertGreaterEqual(len(candidates), 2)

    def test_no_words_yields_empty(self) -> None:
        self.assertEqual(
            extract_sentence_spans(
                [],
                segment_start_frame=0,
                segment_end_frame=SR,
                sample_rate=SR,
                config=self.cfg,
            ),
            [],
        )


class TranscribeSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP.mkdir(parents=True, exist_ok=True)
        self.config = _config()
        self.source = TMP / "src.wav"
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(10 * SR) / SR)
        sf.write(
            str(self.source),
            np.stack([tone.astype(np.float32)] * 2, axis=1),
            SR,
            subtype="PCM_16",
        )
        self.regions = [
            {
                "region_index": 0,
                "source_start_frame": 0,
                "source_end_frame": 10 * SR,
                "status": "usable",
            }
        ]

    def tearDown(self) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def test_dual_asr_agreement_flow(self) -> None:
        words = _words("今天天气真的很好啊。", 0.5, per_char=0.5, gap=0.02)
        primary = _fake_primary([words], "今天天气真的很好啊")
        secondary = _fake_secondary(["今天天气真的很好啊"])
        result = transcribe_source(
            self.source,
            self.regions,
            config=self.config,
            primary_fn=primary,
            secondary_fn=secondary,
        )
        self.assertEqual(result["status"], "completed")
        manifest = json.loads(
            Path(result["transcript_path"]).read_text(encoding="utf-8")
        )
        candidates = [
            s for s in manifest["sentences"] if s["status"] == "candidate"
        ]
        self.assertTrue(candidates)
        self.assertEqual(
            candidates[0]["text_agreement"]["status"], "match"
        )
        self.assertFalse(candidates[0]["candidate_text_is_confirmed"])
        self.assertTrue(candidates[0]["requires_final_recheck"])

    def test_disagreement_stays_unconfirmed(self) -> None:
        words = _words("他买了3个苹果吃。", 0.5, per_char=0.5, gap=0.02)
        primary = _fake_primary([words], "他买了3个苹果吃")
        secondary = _fake_secondary(["他买了三个苹果吃"])
        result = transcribe_source(
            self.source,
            self.regions,
            config=self.config,
            primary_fn=primary,
            secondary_fn=secondary,
        )
        manifest = json.loads(
            Path(result["transcript_path"]).read_text(encoding="utf-8")
        )
        disagreement = [
            s for s in manifest["sentences"] if s["status"] == "digit_disagreement"
        ][0]
        self.assertEqual(
            disagreement["text_agreement"]["status"], "digit_disagreement"
        )
        self.assertFalse(disagreement["candidate_text_is_confirmed"])

    def test_resume_cached(self) -> None:
        words = _words("今天天气真的很好啊。", 0.5, per_char=0.5, gap=0.02)
        primary = _fake_primary([words], "今天天气真的很好啊")
        secondary = _fake_secondary(["今天天气真的很好啊"])
        first = transcribe_source(
            self.source, self.regions, config=self.config,
            primary_fn=primary, secondary_fn=secondary,
        )
        second = transcribe_source(
            self.source, self.regions, config=self.config,
            primary_fn=primary, secondary_fn=secondary,
        )
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "cached")


if __name__ == "__main__":
    unittest.main()
