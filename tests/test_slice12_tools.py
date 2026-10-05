from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Slice12ContainerManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("build_slice12_container_manifests.py")

    def test_windows_frozen_path_is_rebased_to_container(self) -> None:
        relative = self.module.standardized_relative_path(
            r"C:\repo\data\work\standardized\ab\sample.wav"
        )
        self.assertEqual(relative, PurePosixPath("ab/sample.wav"))

    def test_paths_outside_standardized_root_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.module.standardized_relative_path(r"D:\other\sample.wav")


class Slice12RegressionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("run_slice12_regression_gate.py")

    def test_summary_uses_explicit_mean_and_max_aggregation(self) -> None:
        metrics = {
            "item_count": 2,
            "item_error_count": 0,
            "items": [
                {"status": "ok", "cer": 0.0, "speaker_cosine": 0.8, "quality_decision": "pass", "rtf": 0.5},
                {"status": "ok", "cer": 0.2, "speaker_cosine": 0.6, "quality_decision": "review", "rtf": 1.0},
            ],
        }
        generation = [{"peak_torch_cuda_bytes": 2 * 1024 ** 3}, {"peak_torch_cuda_bytes": 3 * 1024 ** 3}]
        summary = self.module.summarize(metrics, generation)
        self.assertAlmostEqual(summary["mean_cer"], 0.1)
        self.assertAlmostEqual(summary["mean_speaker_cosine"], 0.7)
        self.assertAlmostEqual(summary["quality_pass_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_rtf"], 0.75)
        self.assertAlmostEqual(summary["inference_peak_cuda_gib"], 3.0)

    def test_missing_metrics_fail_gate_instead_of_becoming_zero(self) -> None:
        result = self.module.check_max("example", None, 1.0)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
