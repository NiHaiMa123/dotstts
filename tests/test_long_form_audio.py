from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_audio import iter_audio_blocks, scan_long_audio
from dots_tts_lab.long_form_contract import load_long_form_config


class LongFormAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = load_long_form_config()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_stereo(self, name: str, *, seconds: float, inverted: bool = False) -> Path:
        rate = 48_000
        time = np.arange(round(rate * seconds), dtype=np.float64) / rate
        left = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
        right = -left if inverted else left * 0.8
        path = self.root / name
        sf.write(path, np.column_stack((left, right)), rate, subtype="PCM_16")
        return path

    def test_blocks_are_bounded_and_cover_source_exactly(self) -> None:
        path = self.write_stereo("bounded.wav", seconds=5.25)
        blocks = list(iter_audio_blocks(path, block_seconds=2.0))
        self.assertEqual([block.start_frame for block in blocks], [0, 96000, 192000])
        self.assertEqual(blocks[-1].end_frame, round(5.25 * 48000))
        self.assertTrue(all(block.samples.shape[0] <= 96000 for block in blocks))
        self.assertTrue(all(block.samples.dtype == np.float32 for block in blocks))

    def test_streaming_scan_hash_metrics_and_memory_bound(self) -> None:
        path = self.write_stereo("scan.wav", seconds=5.25)
        result = scan_long_audio(path, config=self.config)
        self.assertEqual(result["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["decoded_frames"], round(5.25 * 48000))
        self.assertLessEqual(
            result["maximum_decoded_block_frames"],
            round(self.config.decode.block_seconds * 48000),
        )
        self.assertGreater(result["stereo_correlation"], 0.99)
        self.assertEqual(result["spatial_review_reasons"], [])

    def test_inverted_stereo_is_routed_to_spatial_review(self) -> None:
        path = self.write_stereo("inverted.wav", seconds=2.0, inverted=True)
        result = scan_long_audio(path, config=self.config)
        self.assertLess(result["stereo_correlation"], -0.99)
        self.assertIn("low_stereo_correlation", result["spatial_review_reasons"])

    def test_invalid_block_size_fails(self) -> None:
        path = self.write_stereo("invalid.wav", seconds=1.0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            list(iter_audio_blocks(path, block_seconds=0.0))

