from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_contract import load_long_form_config
from dots_tts_lab.long_form_segmentation import segment_long_audio


RATE = 16000


def tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(round(RATE * seconds), dtype=np.float64) / RATE
    return amplitude * np.sin(2.0 * np.pi * 220.0 * time)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(RATE * seconds), dtype=np.float64)


class LongFormSegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = load_long_form_config()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, name: str, audio: np.ndarray) -> Path:
        path = self.root / name
        sf.write(path, audio, RATE, subtype="PCM_16")
        return path

    def test_two_utterances_are_split_at_long_pause(self) -> None:
        path = self.write(
            "two.wav",
            np.concatenate((silence(1.0), tone(3.0), silence(1.0), tone(3.0), silence(1.0))),
        )
        result = segment_long_audio(path, config=self.config)
        self.assertEqual(result["candidate_segment_count"], 2)
        self.assertTrue(all(2.0 <= item["duration_seconds"] <= 15.0 for item in result["segments"]))
        self.assertTrue(all(item["boundary_reason"] == "vad_region" for item in result["segments"]))

    def test_short_pause_is_merged(self) -> None:
        path = self.write(
            "merged.wav",
            np.concatenate((silence(1.0), tone(2.0), silence(0.2), tone(2.0), silence(1.0))),
        )
        result = segment_long_audio(path, config=self.config)
        self.assertEqual(result["candidate_segment_count"], 1)
        self.assertGreater(result["segments"][0]["duration_seconds"], 4.0)

    def test_long_continuous_region_is_bounded_and_routed_to_review(self) -> None:
        path = self.write("long.wav", np.concatenate((silence(1.0), tone(31.0), silence(1.0))))
        result = segment_long_audio(path, config=self.config)
        self.assertGreaterEqual(result["candidate_segment_count"], 3)
        self.assertTrue(all(2.0 <= item["duration_seconds"] <= 15.0 for item in result["segments"]))
        self.assertTrue(all(item["review_required"] for item in result["segments"]))
        self.assertTrue(all(item["boundary_reason"] == "forced_max_duration_split" for item in result["segments"]))

    def test_subminimum_region_is_reported_but_not_emitted(self) -> None:
        path = self.write("short.wav", np.concatenate((silence(1.0), tone(0.5), silence(1.0))))
        result = segment_long_audio(path, config=self.config)
        self.assertEqual(result["candidate_segment_count"], 0)
        self.assertEqual(result["dropped_short_region_count"], 1)

