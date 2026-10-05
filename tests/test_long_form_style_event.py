from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_style_event import (
    NoEventBackend,
    StyleAnchors,
    analyze_source_regions,
    load_style_event_config,
    region_style_features,
    score_style_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_style_event_tests"
SR = 48000


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _anchors() -> StyleAnchors:
    # normal anchors cluster near tone-like features; whisper negatives have
    # high flatness / low periodicity; roleplay similar level, different pitch
    normal = np.tile(
        np.array([-20.0, 0.05, 0.02, 0.9, 220.0, -60.0, 0.05, 0.98, 0.2]),
        (4, 1),
    ) + np.linspace(-0.5, 0.5, 4)[:, None] * 0.01
    whisper = np.tile(
        np.array([-30.0, 0.2, 0.4, 0.1, np.nan, -60.0, 0.05, 0.98, 0.5]),
        (2, 1),
    )
    return StyleAnchors(
        normal_matrix=normal,
        negative_matrices={"whispered_text": whisper},
        pack_sha256="f" * 64,
    )


class FakeEventBackend:
    def __init__(self, risk_max: float = 0.05) -> None:
        self.risk_max = risk_max
        self.calls = 0

    def status(self) -> str:
        return "ready"

    def tag(self, mono, *, sample_rate: int):
        self.calls += 1
        return {
            "status": "ok",
            "top_labels": [{"label": "Speech", "score": 0.9}],
            "risk_scores": {
                "whisper": {"max": self.risk_max, "at_seconds": 0.5}
            },
            "speech_score_median": 0.8,
            "speech_score_max": 0.95,
        }


class StyleEventConfigTests(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = load_style_event_config()
        self.assertEqual(config.config_id, "long_form_style_event_v1")
        self.assertEqual(config.events.backend, "panns_cnn14")
        self.assertGreater(len(config.events.risk_label_patterns), 10)


class RegionStyleFeatureTests(unittest.TestCase):
    def test_tone_features(self) -> None:
        stereo = np.stack([_tone(3.0), _tone(3.0)], axis=1)
        f = region_style_features(stereo, sample_rate=SR)
        self.assertGreater(f["periodicity"], 0.5)
        # autocorr may lock onto a subharmonic; accept any integer divisor of 220
        ratio = 220.0 / f["pitch_median_hz"]
        self.assertAlmostEqual(ratio, round(ratio), delta=0.02)
        self.assertAlmostEqual(f["stereo_correlation"], 1.0, places=5)
        self.assertLess(f["spectral_flatness_median"], 0.1)

    def test_decorrelated_stereo_features(self) -> None:
        rng = np.random.RandomState(0)
        stereo = np.stack(
            [rng.uniform(-0.2, 0.2, 3 * SR), rng.uniform(-0.2, 0.2, 3 * SR)],
            axis=1,
        ).astype(np.float32)
        f = region_style_features(stereo, sample_rate=SR)
        self.assertLess(f["stereo_correlation"], 0.1)


class ScoreStyleEvidenceTests(unittest.TestCase):
    def test_normal_like_scores_close(self) -> None:
        features = {
            "level_dbfs": -20.0,
            "silence_ratio": 0.05,
            "spectral_flatness_median": 0.02,
            "periodicity": 0.9,
            "pitch_median_hz": 220.0,
            "side_to_mid_db": -60.0,
            "pan_standard_deviation": 0.05,
            "stereo_correlation": 0.98,
            "transients_per_second": 0.2,
        }
        evidence = score_style_evidence(features, _anchors())
        self.assertLess(evidence["normal_mean_z"], 1.0)
        self.assertEqual(evidence["nearest_negative_kind"], "whispered_text")

    def test_whisper_like_is_far_from_normal(self) -> None:
        features = {
            "level_dbfs": -30.0,
            "silence_ratio": 0.2,
            "spectral_flatness_median": 0.4,
            "periodicity": 0.1,
            "pitch_median_hz": None,
            "side_to_mid_db": -60.0,
            "pan_standard_deviation": 0.05,
            "stereo_correlation": 0.98,
            "transients_per_second": 0.5,
        }
        evidence = score_style_evidence(features, _anchors())
        self.assertGreater(evidence["normal_mean_z"], 1.0)
        self.assertLess(
            evidence["negative_distance_by_kind"]["whispered_text"], 1.0
        )


class AnalyzeSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP.mkdir(parents=True, exist_ok=True)
        base = load_style_event_config()
        paths = base.paths.model_copy(
            update={
                "work_root": "data/work/tmp_style_event_tests/work",
                "report_root": "data/work/tmp_style_event_tests/reports",
            }
        )
        self.config = base.model_copy(update={"paths": paths})
        self.source = TMP / "src.wav"
        _write = np.stack([_tone(4.0), _tone(4.0)], axis=1)
        sf.write(str(self.source), _write, SR, subtype="PCM_16")
        self.regions = [
            {
                "region_index": 0,
                "source_start_frame": 0,
                "source_end_frame": 4 * SR,
                "status": "usable",
                "reasons": [],
                "spatial_risk_fraction": 0.0,
                "transients_per_second": 0.1,
            }
        ]

    def tearDown(self) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def test_end_to_end_with_event_backend(self) -> None:
        result = analyze_source_regions(
            self.source,
            self.regions,
            config=self.config,
            anchors=_anchors(),
            event_backend=FakeEventBackend(),
        )
        self.assertEqual(result["status"], "completed")
        manifest = json.loads(
            Path(result["evidence_path"]).read_text(encoding="utf-8")
        )
        row = manifest["regions"][0]
        self.assertEqual(row["status"], "evidence_complete")
        self.assertIn("normal_mean_z", row["style_evidence"])
        self.assertEqual(row["event_evidence"]["status"], "ok")
        self.assertEqual(row["prefilter_spatial_risk_fraction"], 0.0)

    def test_missing_backend_marks_evidence_missing(self) -> None:
        result = analyze_source_regions(
            self.source,
            self.regions,
            config=self.config,
            anchors=_anchors(),
            event_backend=NoEventBackend(),
        )
        manifest = json.loads(
            Path(result["evidence_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["regions"][0]["status"], "evidence_missing")
        self.assertEqual(manifest["event_backend"], "unavailable")

    def test_resume_cached(self) -> None:
        first = analyze_source_regions(
            self.source,
            self.regions,
            config=self.config,
            anchors=_anchors(),
            event_backend=FakeEventBackend(),
        )
        second = analyze_source_regions(
            self.source,
            self.regions,
            config=self.config,
            anchors=_anchors(),
            event_backend=FakeEventBackend(),
        )
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "cached")


class PannsBackendSmokeTests(unittest.TestCase):
    def test_real_backend_tags_speech_if_weights_present(self) -> None:
        checkpoint = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 3e8:
            self.skipTest("PANNs checkpoint not downloaded")
        from dots_tts_lab.long_form_style_event import PannsEventBackend

        config = load_style_event_config()
        backend = PannsEventBackend(config.events)
        result = backend.tag(_tone(2.0), sample_rate=SR)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["top_labels"])
        self.assertIn("whisper", result["risk_scores"])


if __name__ == "__main__":
    unittest.main()
