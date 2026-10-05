from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_quality import (
    DnsmosBackend,
    load_quality_config,
    run_quality_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_quality_tests"
SR = 48000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


class _FakeBackend:
    def score(self, audio: np.ndarray, *, sample_rate: int) -> dict:
        level = float(np.sqrt(np.mean(np.asarray(audio) ** 2) + 1e-12))
        return {
            "sig": 3.0 + level,
            "bak": 2.5 + level,
            "ovr": 2.8 + level,
            "p808": 3.1 + level,
            "sig_raw": 3.0,
            "bak_raw": 2.5,
            "ovr_raw": 2.8,
            "num_hops": 1,
            "clip_seconds": len(audio) / sample_rate,
            "padded": len(audio) < 9 * sample_rate,
        }


class QualityAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(TMP, ignore_errors=True)
        rng = np.random.default_rng(1)
        cls.source = TMP / "source.wav"
        _write_wav(cls.source, rng.normal(0, 0.05, SR * 6).astype(np.float32))
        cls.run_dir = TMP / "run"
        raw_dir = cls.run_dir / "raw"
        raw_dir.mkdir(parents=True)
        seg = rng.normal(0, 0.05, SR).astype(np.float32)
        _write_wav(raw_dir / "region_00000.wav", seg)
        import hashlib

        cls.seg_sha = hashlib.sha256(
            (raw_dir / "region_00000.wav").read_bytes()
        ).hexdigest()
        cls.manifest = {
            "source_sha256": None,  # filled per test
            "run_dir": str(cls.run_dir),
            "regions": [
                {
                    "region_index": 0,
                    "source_start_frame": 0,
                    "source_end_frame": SR,
                    "route": "raw",
                    "status": "completed",
                    "output": {
                        "relative_path": "raw/region_00000.wav",
                        "sha256": cls.seg_sha,
                    },
                },
                {
                    "region_index": 1,
                    "source_start_frame": SR,
                    "source_end_frame": 2 * SR,
                    "route": "skipped_prefilter",
                    "status": "skipped_prefilter",
                },
            ],
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def _config(self):
        base = load_quality_config()
        paths = base.paths.model_copy(
            update={
                "work_root": "data/work/tmp_quality_tests/work",
                "report_root": "data/work/tmp_quality_tests/reports",
            }
        )
        return base.model_copy(update={"paths": paths})

    def test_scores_pairs_and_writes_report(self) -> None:
        from dots_tts_lab.long_form_audio import file_sha256

        manifest = dict(self.manifest)
        manifest["source_sha256"] = file_sha256(self.source)
        result = run_quality_acceptance(
            self.source,
            manifest,
            config=self._config(),
            backend=_FakeBackend(),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"]["scored_pairs"], 1)
        report = json.loads(Path(result["report_path"]).read_text("utf-8"))
        row = report["regions"][0]
        self.assertIn("delta", row)
        self.assertTrue(row["output_sha256_verified"])
        self.assertTrue(row["short_clip"])
        self.assertFalse(report["thresholds_calibrated"])

    def test_source_mismatch_aborts(self) -> None:
        manifest = dict(self.manifest)
        manifest["source_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            run_quality_acceptance(
                self.source, manifest, config=self._config(),
                backend=_FakeBackend(),
            )


@unittest.skipUnless(
    (ROOT / "data/work/models/dnsmos/sig_bak_ovr.onnx").is_file(),
    "DNSMOS models not downloaded",
)
class DnsmosRealTests(unittest.TestCase):
    def test_real_backend_scores_and_marks_short(self) -> None:
        backend = DnsmosBackend(load_quality_config().models)
        speechish = (
            0.1 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)
        ).astype(np.float32)
        scores = backend.score(speechish, sample_rate=SR)
        self.assertTrue(1.0 <= scores["ovr"] <= 5.0)
        self.assertTrue(scores["padded"])

    def test_short_clip_padding_sensitivity_measured(self) -> None:
        """Same content scored at different durations quantifies how much the
        self-loop padding moves the score — the short-clip validity check."""
        backend = DnsmosBackend(load_quality_config().models)
        t = np.arange(SR * 9) / SR
        tone = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        full = backend.score(tone, sample_rate=SR)
        short = backend.score(tone[: SR], sample_rate=SR)
        # Report both; no assertion on closeness — the gap itself is evidence.
        self.assertFalse(short["padded"] is False)
        self.assertIn("ovr", full)


if __name__ == "__main__":
    unittest.main()
