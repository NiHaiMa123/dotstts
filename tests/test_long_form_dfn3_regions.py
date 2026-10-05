from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_dfn3 import load_dfn3_config
from dots_tts_lab.long_form_dfn3_regions import (
    classify_sentence_spans,
    enhance_source_regions,
    plan_windows,
    sentence_window,
    verify_delay_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_dfn3_region_tests"
SR = 48000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def _config():
    base = load_dfn3_config()
    output = base.output.model_copy(
        update={"work_root": "data/work/tmp_dfn3_region_tests/work"}
    )
    return base.model_copy(update={"output": output})


def _identity_enhancer(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32)


class PlanWindowsTests(unittest.TestCase):
    def test_single_window_clamps_context_at_file_edges(self) -> None:
        windows = plan_windows(
            100,
            200,
            core_max_frames=1000,
            context_max_frames=50,
            total_source_frames=250,
        )
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertEqual((w.core_start_frame, w.core_end_frame), (100, 200))
        self.assertEqual(w.context_start_frame, 50)
        self.assertEqual(w.context_end_frame, 250)
        self.assertEqual(w.core_output_bounds(), (50, 150))

    def test_cores_tile_contiguously_and_context_bounded(self) -> None:
        region = (1000, 1000 + 3 * 500 + 123)
        windows = plan_windows(
            *region,
            core_max_frames=500,
            context_max_frames=50,
            total_source_frames=100000,
        )
        self.assertEqual(len(windows), 4)
        cursor = region[0]
        for w in windows:
            self.assertEqual(w.core_start_frame, cursor)
            self.assertLessEqual(w.core_frames, 500)
            self.assertLessEqual(
                w.core_start_frame - w.context_start_frame, 50
            )
            self.assertLessEqual(
                w.context_end_frame - w.core_end_frame, 50
            )
            cursor = w.core_end_frame
        self.assertEqual(cursor, region[1])

    def test_rejects_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            plan_windows(
                200, 100, core_max_frames=10, context_max_frames=0,
                total_source_frames=300,
            )


class SpanPlacementTests(unittest.TestCase):
    def test_sentence_inside_core_is_placed(self) -> None:
        windows = plan_windows(
            0, 1000, core_max_frames=400, context_max_frames=10,
            total_source_frames=1000,
        )
        placements = classify_sentence_spans([(50, 120), (390, 410)], windows)
        self.assertEqual(placements[0].window_index, 0)
        self.assertIsNone(placements[1].window_index)  # crosses 400 boundary

    def test_seam_sentence_gets_dedicated_window_or_quarantine(self) -> None:
        w = sentence_window(
            390, 410, core_max_frames=400, context_max_frames=10,
            total_source_frames=1000,
        )
        self.assertIsNotNone(w)
        self.assertEqual((w.core_start_frame, w.core_end_frame), (390, 410))
        self.assertIsNone(
            sentence_window(
                0, 500, core_max_frames=400, context_max_frames=10,
                total_source_frames=1000,
            )
        )


class DelayAlignmentTests(unittest.TestCase):
    def test_identity_enhancer_has_zero_residual(self) -> None:
        self.assertEqual(verify_delay_alignment(_identity_enhancer), 0)

    def test_delayed_enhancer_reports_lag(self) -> None:
        def delayed(audio: np.ndarray) -> np.ndarray:
            out = np.zeros_like(audio)
            out[100:] = audio[:-100]
            return out

        self.assertEqual(verify_delay_alignment(delayed), 100)

    def test_length_mismatch_raises(self) -> None:
        def short(audio: np.ndarray) -> np.ndarray:
            return audio[:-5]

        with self.assertRaises(RuntimeError):
            verify_delay_alignment(short)


class EnhanceSourceRegionsTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP.mkdir(parents=True, exist_ok=True)
        self.config = _config()
        self.source = TMP / "source.wav"
        rng = np.random.RandomState(0)
        tone = 0.1 * np.sin(
            2 * np.pi * 220 * np.arange(4 * SR) / SR
        ).astype(np.float32)
        stereo = np.stack([tone + 0.001 * rng.randn(4 * SR),
                           tone + 0.001 * rng.randn(4 * SR)], axis=1)
        _write_wav(self.source, stereo.astype(np.float32))
        self.regions = [
            {
                "region_index": 0,
                "source_start_frame": SR,
                "source_end_frame": 2 * SR,
            },
            {
                "region_index": 1,
                "source_start_frame": int(2.5 * SR),
                "source_end_frame": int(3.5 * SR),
            },
        ]

    def tearDown(self) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def test_identity_enhance_writes_bounded_output(self) -> None:
        result = enhance_source_regions(
            self.source,
            self.regions,
            config=self.config,
            enhance_fn=_identity_enhancer,
            spatial_gate={0: True, 1: True},
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["enhanced_regions"], [0, 1])
        records = json.loads(
            Path(result["records_path"]).read_text(encoding="utf-8")
        )
        for row in records["regions"]:
            self.assertEqual(row["status"], "enhanced")
            self.assertEqual(row["channel_route"], "safe_mean")
            out = Path(result["run_dir"]) / row["output"]["relative_path"]
            info = sf.info(str(out))
            self.assertEqual(info.samplerate, SR)
            self.assertEqual(info.channels, 1)
            self.assertEqual(
                info.frames,
                row["core_end_frame"] - row["core_start_frame"],
            )

    def test_stereo_without_spatial_evidence_is_blocked_not_downmixed(self) -> None:
        result = enhance_source_regions(
            self.source,
            self.regions,
            config=self.config,
            enhance_fn=_identity_enhancer,
            spatial_gate=None,  # no evidence → fail closed
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["blocked_regions"], [0, 1])
        records = json.loads(
            Path(result["records_path"]).read_text(encoding="utf-8")
        )
        for row in records["regions"]:
            self.assertEqual(row["status"], "blocked_spatial")
        self.assertEqual(list(Path(result["run_dir"]).glob("region_*.wav")), [])

    def test_resume_skips_committed_regions(self) -> None:
        calls = {"n": 0}

        def counting(audio: np.ndarray) -> np.ndarray:
            calls["n"] += 1
            return np.asarray(audio, dtype=np.float32)

        enhance_source_regions(
            self.source,
            [self.regions[0]],
            config=self.config,
            enhance_fn=counting,
            spatial_gate={0: True},
        )
        windows_first_run = calls["n"]
        enhance_source_regions(
            self.source,
            self.regions,  # region 0 already committed
            config=self.config,
            enhance_fn=counting,
            spatial_gate={0: True, 1: True},
        )
        self.assertGreater(calls["n"], windows_first_run)
        state = json.loads(
            (Path(
                ROOT
                / self.config.output.work_root
            ).rglob("state.json").__next__().read_text(encoding="utf-8"))
        )
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_regions"], [0, 1])

    def test_backend_failure_blocks_and_never_falls_back_to_raw(self) -> None:
        def broken(_audio: np.ndarray) -> np.ndarray:
            raise RuntimeError("dfn3 crashed")

        with self.assertRaises(RuntimeError):
            enhance_source_regions(
                self.source,
                [self.regions[0]],
                config=self.config,
                enhance_fn=broken,
                spatial_gate={0: True},
            )
        work = Path(ROOT / self.config.output.work_root)
        self.assertEqual(list(work.rglob("region_*.wav")), [])
        state = json.loads(
            work.rglob("state.json").__next__().read_text(encoding="utf-8")
        )
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed_regions"], [])

    def test_output_mismatch_blocks(self) -> None:
        def wrong_length(audio: np.ndarray) -> np.ndarray:
            return np.asarray(audio[:-7], dtype=np.float32)

        with self.assertRaises(RuntimeError):
            enhance_source_regions(
                self.source,
                [self.regions[0]],
                config=self.config,
                enhance_fn=wrong_length,
                spatial_gate={0: True},
            )

    def test_long_region_splits_into_bounded_cores(self) -> None:
        spans = []
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(40 * SR) / SR)
        mono = np.stack([tone.astype(np.float32)] * 2, axis=1)
        long_source = TMP / "long.wav"
        _write_wav(long_source, mono)
        region = [
            {"region_index": 0, "source_start_frame": 0, "source_end_frame": 35 * SR}
        ]
        result = enhance_source_regions(
            long_source,
            region,
            config=self.config,
            enhance_fn=_identity_enhancer,
            spatial_gate={0: True},
        )
        records = json.loads(
            Path(result["records_path"]).read_text(encoding="utf-8")
        )
        windows = records["regions"][0]["windows"]
        self.assertEqual(len(windows), 2)  # 35s → 30s + 5s cores
        for w in windows:
            self.assertLessEqual(
                w["core_end_frame"] - w["core_start_frame"], 30 * SR
            )
            self.assertLessEqual(
                w["core_start_frame"] - w["context_start_frame"], 2 * SR
            )
            self.assertLessEqual(
                w["context_end_frame"] - w["core_end_frame"], 2 * SR
            )
            spans.append((w["core_start_frame"], w["core_end_frame"]))
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 35 * SR)


if __name__ == "__main__":
    unittest.main()
