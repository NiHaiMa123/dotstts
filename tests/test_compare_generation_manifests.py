from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
from scripts import compare_generation_manifests


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompareGenerationManifestsTests(unittest.TestCase):
    def test_reports_waveform_and_runtime_differences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_audio = root / "left.wav"
            right_audio = root / "right.wav"
            sf.write(left_audio, np.array([0.0, 0.25, -0.25]), 48_000, subtype="FLOAT")
            sf.write(right_audio, np.array([0.0, 0.25, -0.20]), 48_000, subtype="FLOAT")
            left_manifest = root / "left.jsonl"
            right_manifest = root / "right.jsonl"
            left_manifest.write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "ok",
                        "output_path": "left.wav",
                        "output_sha256": sha256(left_audio),
                        "rtf": 0.5,
                        "peak_torch_cuda_bytes": 1024**3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            right_manifest.write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "ok",
                        "output_path": "right.wav",
                        "output_sha256": sha256(right_audio),
                        "rtf": 0.25,
                        "peak_torch_cuda_bytes": 2 * 1024**3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(compare_generation_manifests, "ROOT", root):
                report = compare_generation_manifests.compare(
                    left_manifest, right_manifest
                )
            self.assertEqual(report["common_successful_count"], 1)
            self.assertEqual(report["sample_count_equal_count"], 1)
            self.assertEqual(report["output_hash_equal_count"], 0)
            self.assertAlmostEqual(report["right_to_left_mean_rtf_ratio"], 0.5)
            self.assertEqual(report["right"]["peak_torch_cuda_gib"], 2.0)
            self.assertLess(report["minimum_overlap_waveform_cosine"], 1.0)
            self.assertGreater(report["maximum_overlap_rmse"], 0.0)

    def test_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio.wav"
            sf.write(audio, np.zeros(8), 48_000)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "ok",
                        "output_path": "audio.wav",
                        "output_sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(compare_generation_manifests, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    compare_generation_manifests.compare(manifest, manifest)


if __name__ == "__main__":
    unittest.main()
