from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dots_tts_lab.long_form_pipeline import (
    recover_long_form_work_root,
    run_long_form_pipeline,
)


class FakeEncoder:
    def __call__(self, audio: torch.Tensor, *, audio_lengths: torch.Tensor) -> torch.Tensor:
        del audio_lengths
        output = torch.zeros((audio.shape[0], 512), dtype=torch.float32)
        output[:, 0] = 1.0
        output[:, 1] = torch.mean(torch.abs(audio), dim=1)
        return output


class LongFormPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input = self.root / "input"
        self.work = self.root / "work"
        self.reports = self.root / "reports"
        self.input.mkdir()
        rate = 48000
        time = np.arange(rate * 3, dtype=np.float64) / rate
        tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
        audio = np.concatenate((np.zeros(rate), tone, np.zeros(rate), tone, np.zeros(rate)))
        sf.write(self.input / "long.wav", np.column_stack((audio, audio * 0.9)), rate)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_pipeline(self, **overrides):
        arguments = {
            "work_root": self.work,
            "report_root": self.reports,
            "skip_asr": True,
            "encoder": FakeEncoder(),
        }
        arguments.update(overrides)
        return run_long_form_pipeline(self.input, **arguments)

    def test_pipeline_materializes_traceable_pcm24_candidates_and_caches(self) -> None:
        first = self.run_pipeline()
        self.assertEqual(first["source_count"], 1)
        self.assertEqual(first["segment_count"], 2)
        self.assertTrue(Path(first["sources"][0]["manifest_path"]).is_file())
        manifest_path = next(self.work.rglob("manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["asr_status"], "skipped")
        self.assertTrue(all(item["text_review_status"] == "pending" for item in manifest["segments"]))
        for item in manifest["segments"]:
            path = manifest_path.parent / item["derived_relative_path"]
            info = sf.info(path)
            self.assertEqual((info.samplerate, info.channels, info.subtype), (48000, 1, "PCM_24"))
        second = self.run_pipeline()
        self.assertEqual(second["sources"][0]["cache_action"], "cached")

    def test_startup_removes_partial_and_marks_running_state_interrupted(self) -> None:
        partial = self.work / "sources" / "x" / "left.partial"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"partial")
        state = partial.parent / "state.json"
        state.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        result = recover_long_form_work_root(self.work)
        self.assertEqual(result["deleted_partial_count"], 1)
        self.assertEqual(result["marked_interrupted_count"], 1)
        self.assertFalse(partial.exists())
        self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["status"], "interrupted")
