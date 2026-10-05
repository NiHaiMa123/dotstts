from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Slice10MetricRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluate = load_script("evaluate_slice10_outputs.py")
        cls.blind = load_script("apply_slice10_blind_review.py")

    def test_zero_is_not_replaced_by_missing_default(self) -> None:
        for module in (self.evaluate, self.blind):
            self.assertEqual(module.numeric_or_default(0.0, 1.0), 0.0)
            self.assertEqual(module.numeric_or_default(None, 1.0), 1.0)


class Slice10AsrInterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.asr = load_script("run_slice10_asr.py")

    def test_configured_asr_python_is_resolved_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "lab" / "slice10" / "benchmark.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                "metrics:\n  asr_python: data/work/asr/Scripts/python.exe\n",
                encoding="utf-8",
            )
            expected = (root / "data/work/asr/Scripts/python.exe").resolve()
            self.assertEqual(self.asr.configured_asr_python(config_path), expected)

    def test_missing_asr_python_setting_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs" / "lab" / "slice10" / "benchmark.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("metrics: {}\n", encoding="utf-8")
            self.assertIsNone(self.asr.configured_asr_python(config_path))


if __name__ == "__main__":
    unittest.main()
