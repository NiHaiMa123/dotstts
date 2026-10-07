from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dots_tts import runtime as runtime_module
from dots_tts.runtime import DotsTtsRuntime


def _bare_runtime(sample_rate: int = 48_000) -> DotsTtsRuntime:
    instance = DotsTtsRuntime.__new__(DotsTtsRuntime)
    instance.sample_rate = sample_rate
    instance._prompt_audio_cache = OrderedDict()
    return instance


class PromptAudioCacheTests(unittest.TestCase):
    def test_same_file_loads_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prompt = Path(temporary) / "prompt.wav"
            prompt.write_bytes(b"fake-wav")
            instance = _bare_runtime()
            loaded = np.full(48_000, 0.1, dtype=np.float32)

            with mock.patch.object(
                runtime_module.librosa, "load", return_value=(loaded, 48_000)
            ) as load_mock, mock.patch.object(
                runtime_module.librosa.effects,
                "trim",
                side_effect=lambda audio, top_db: (audio, None),
            ), mock.patch.object(
                runtime_module,
                "high_quality_resample",
                side_effect=lambda audio, orig_sr, target_sr: audio,
            ):
                first = instance._load_prompt_audio(str(prompt))
                second = instance._load_prompt_audio(str(prompt))

            self.assertEqual(load_mock.call_count, 1)
            self.assertIs(first, second)
            self.assertEqual(len(instance._prompt_audio_cache), 1)

    def test_modified_file_is_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prompt = Path(temporary) / "prompt.wav"
            prompt.write_bytes(b"fake-wav")
            instance = _bare_runtime()
            loaded = np.full(48_000, 0.1, dtype=np.float32)

            def fake_load(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
                return loaded, 48_000

            with mock.patch.object(
                runtime_module.librosa, "load", side_effect=fake_load
            ) as load_mock, mock.patch.object(
                runtime_module.librosa.effects,
                "trim",
                side_effect=lambda audio, top_db: (audio, None),
            ), mock.patch.object(
                runtime_module,
                "high_quality_resample",
                side_effect=lambda audio, orig_sr, target_sr: audio,
            ):
                instance._load_prompt_audio(str(prompt))
                import os

                os.utime(prompt, ns=(0, 2_000_000_000))
                instance._load_prompt_audio(str(prompt))

            self.assertEqual(load_mock.call_count, 2)
            self.assertEqual(len(instance._prompt_audio_cache), 2)

    def test_missing_file_error_is_not_cached(self) -> None:
        instance = _bare_runtime()
        with self.assertRaises(Exception):
            instance._load_prompt_audio("definitely/missing.wav")
        self.assertEqual(len(instance._prompt_audio_cache), 0)


if __name__ == "__main__":
    unittest.main()
