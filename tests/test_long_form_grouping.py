from __future__ import annotations

import unittest

import numpy as np

from dots_tts_lab.long_form_contract import load_long_form_config
from dots_tts_lab.long_form_grouping import (
    cluster_speaker_embeddings,
    cluster_style_features,
    estimate_pitch_periodicity,
    suggest_styles,
)


class LongFormGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_long_form_config()

    def test_speaker_clusters_are_deterministic_and_fail_closed_without_reference(self) -> None:
        vectors = [
            np.asarray([1.0, 0.0]),
            np.asarray([0.99, 0.01]),
            np.asarray([0.0, 1.0]),
        ]
        rows = cluster_speaker_embeddings(
            vectors,
            durations=[5.0, 6.0, 2.0],
            cosine_link_threshold=0.9,
            knn_k=2,
            minimum_cluster_size=2,
            reference_vector=None,
            reference_minimum_cosine=0.75,
        )
        self.assertEqual(rows[0]["speaker_cluster_id"], rows[1]["speaker_cluster_id"])
        self.assertNotEqual(rows[0]["speaker_cluster_id"], rows[2]["speaker_cluster_id"])
        self.assertTrue(all(row["target_speaker_status"] == "reference_required" for row in rows))

    def test_reference_marks_match_but_not_different_cluster(self) -> None:
        vectors = [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])]
        rows = cluster_speaker_embeddings(
            vectors,
            durations=[5.0, 5.0],
            cosine_link_threshold=0.9,
            knn_k=1,
            minimum_cluster_size=2,
            reference_vector=np.asarray([1.0, 0.0]),
            reference_minimum_cosine=0.75,
        )
        self.assertEqual(rows[0]["target_speaker_status"], "target_match")
        self.assertEqual(rows[1]["target_speaker_status"], "target_review")

    def test_pitch_periodicity_distinguishes_tone_from_noise(self) -> None:
        rate = 16000
        time = np.arange(rate * 2, dtype=np.float64) / rate
        tone = np.sin(2.0 * np.pi * 200.0 * time).astype(np.float32)
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 0.2, len(tone)).astype(np.float32)
        tonal = estimate_pitch_periodicity(tone, sample_rate=rate, analysis_channel=0)
        noisy = estimate_pitch_periodicity(noise, sample_rate=rate, analysis_channel=0)
        self.assertGreater(tonal["periodicity"], noisy["periodicity"])
        self.assertAlmostEqual(tonal["pitch_median_hz"], 200.0, delta=10.0)

    def test_style_groups_are_deterministic(self) -> None:
        rows = [self.style_row(level=-20.0 - index) for index in range(12)]
        first = cluster_style_features(rows, maximum_groups=3, minimum_group_size=3)
        second = cluster_style_features(rows, maximum_groups=3, minimum_group_size=3)
        self.assertEqual(first, second)
        self.assertLessEqual(len(set(first)), 3)

    def test_no_reference_prevents_normal_auto_accept(self) -> None:
        rows = [self.style_row(level=-20.0), self.style_row(level=-30.0)]
        speaker = [
            {
                "speaker_cluster_is_dominant": True,
                "speaker_cluster_is_small": False,
                "target_speaker_status": "reference_required",
            },
            {
                "speaker_cluster_is_dominant": True,
                "speaker_cluster_is_small": False,
                "target_speaker_status": "reference_required",
            },
        ]
        suggestions = suggest_styles(rows, config=self.config, speaker_groups=speaker)
        self.assertTrue(all(item["requires_review"] for item in suggestions))

    @staticmethod
    def style_row(level: float) -> dict:
        return {
            "level_dbfs": level,
            "snr_proxy_db": 20.0,
            "silence_ratio": 0.05,
            "spectral_flatness_median": 0.001,
            "periodicity": 0.8,
            "pitch_median_hz": 200.0,
            "pan_standard_deviation": 0.01,
            "side_to_mid_db": -20.0,
            "spatial_review_reasons": [],
        }

