from __future__ import annotations

import unittest

import numpy as np

from dots_tts_lab.long_form_contract import load_long_form_config
from dots_tts_lab.long_form_features import analyze_segment_samples


RATE = 16000


def tone(seconds: float, frequency: float = 220.0, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(round(RATE * seconds), dtype=np.float64) / RATE
    return amplitude * np.sin(2.0 * np.pi * frequency * time)


class LongFormFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_long_form_config()

    def analyze(self, samples: np.ndarray) -> dict:
        return analyze_segment_samples(
            samples,
            sample_rate=RATE,
            config=self.config,
            activity_start_threshold_dbfs=-40.0,
            activity_continue_threshold_dbfs=-45.0,
        )

    def test_safe_correlated_stereo_can_be_mean_mixed(self) -> None:
        left = tone(3.0)
        result = self.analyze(np.column_stack((left, left * 0.9)))
        self.assertEqual(result["recommended_channel_strategy"], "mean")
        self.assertGreater(result["stereo_correlation"], 0.99)
        self.assertEqual(result["spatial_review_reasons"], [])

    def test_inverted_stereo_requires_review(self) -> None:
        left = tone(3.0)
        result = self.analyze(np.column_stack((left, -left)))
        self.assertEqual(result["recommended_channel_strategy"], "review")
        self.assertIn("low_stereo_correlation", result["spatial_review_reasons"])
        self.assertIn("side_dominant_audio", result["spatial_review_reasons"])

    def test_moving_pan_requires_review(self) -> None:
        source = tone(4.0)
        midpoint = len(source) // 2
        left_gain = np.concatenate((np.ones(midpoint), np.full(len(source) - midpoint, 0.05)))
        right_gain = np.concatenate((np.full(midpoint, 0.05), np.ones(len(source) - midpoint)))
        result = self.analyze(np.column_stack((source * left_gain, source * right_gain)))
        self.assertIn("moving_stereo_position", result["spatial_review_reasons"])

    def test_near_clipping_and_silence_are_routed_to_review(self) -> None:
        clipped = np.ones(round(RATE * 0.2), dtype=np.float64)
        samples = np.concatenate((np.zeros(round(RATE * 2.5)), clipped, tone(0.3)))
        result = self.analyze(samples)
        self.assertIn("near_clipping", result["quality_review_reasons"])
        self.assertIn("high_silence_ratio", result["quality_review_reasons"])

    def test_overlap_proxy_is_explicitly_uncalibrated(self) -> None:
        result = self.analyze(tone(3.0))
        self.assertIn("overlap_risk_proxy", result)
        self.assertFalse(result["overlap_proxy_is_calibrated"])

