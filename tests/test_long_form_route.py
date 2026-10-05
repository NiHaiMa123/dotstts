from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.long_form_dfn3 import load_dfn3_config
from dots_tts_lab.long_form_route import (
    RouteConfig,
    decide_route,
    load_route_config,
    run_routes,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_route_tests"
SR = 48000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def _config() -> RouteConfig:
    base = load_route_config()
    output = base.output.model_copy(
        update={"work_root": "data/work/tmp_route_tests/work"}
    )
    return base.model_copy(update={"output": output})


def _dfn3_config():
    base = load_dfn3_config()
    output = base.output.model_copy(
        update={"work_root": "data/work/tmp_route_tests/dfn3"}
    )
    return base.model_copy(update={"output": output})


def _region(index: int, start: int, end: int, status: str = "usable") -> dict:
    return {
        "region_index": index,
        "source_start_frame": start,
        "source_end_frame": end,
        "status": status,
    }


def _noise_evidence(score: float) -> dict:
    return {"noise_scores": {"hiss": {"max": score, "at_seconds": 0.0}}}


class DecideRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.routing = _config().routing

    def test_quarantined_region_is_skipped(self) -> None:
        route, reasons, _ = decide_route(
            _region(0, 0, 100, status="prefilter_quarantine"),
            event_evidence=None,
            downmix_allowed=True,
            source_channels=2,
            config=self.routing,
        )
        self.assertEqual(route, "skipped_prefilter")

    def test_stereo_without_permission_blocks(self) -> None:
        route, reasons, _ = decide_route(
            _region(0, 0, 100),
            event_evidence=None,
            downmix_allowed=False,
            source_channels=2,
            config=self.routing,
        )
        self.assertEqual(route, "blocked_spatial")

    def test_noise_above_threshold_routes_dfn3(self) -> None:
        route, _, noise = decide_route(
            _region(0, 0, 100),
            event_evidence=_noise_evidence(0.9),
            downmix_allowed=True,
            source_channels=2,
            config=self.routing,
        )
        self.assertEqual(route, "dfn3_denoised")
        self.assertEqual(noise, 0.9)

    def test_missing_noise_evidence_defaults_raw(self) -> None:
        route, reasons, _ = decide_route(
            _region(0, 0, 100),
            event_evidence=None,
            downmix_allowed=True,
            source_channels=1,
            config=self.routing,
        )
        self.assertEqual(route, "raw")
        self.assertIn("noise_evidence_missing_default_raw", reasons)

    def test_low_noise_routes_raw(self) -> None:
        route, _, _ = decide_route(
            _region(0, 0, 100),
            event_evidence=_noise_evidence(0.05),
            downmix_allowed=True,
            source_channels=2,
            config=self.routing,
        )
        self.assertEqual(route, "raw")


class RunRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(TMP, ignore_errors=True)
        rng = np.random.default_rng(0)
        cls.source = TMP / "source.wav"
        # stereo source: same content both channels
        mono = rng.normal(0, 0.05, SR * 4).astype(np.float32)
        _write_wav(cls.source, np.stack([mono, mono], axis=1))
        cls.config = _config()
        cls.dfn3 = _dfn3_config()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def test_raw_and_dfn3_routes_materialize(self) -> None:
        regions = [
            _region(0, 0, SR),          # raw (low noise)
            _region(1, SR, 2 * SR),     # dfn3 (high noise)
            _region(2, 2 * SR, 3 * SR, status="prefilter_quarantine"),
        ]
        evidence = {0: _noise_evidence(0.0), 1: _noise_evidence(0.9)}
        gate = {0: True, 1: True, 2: True}
        result = run_routes(
            self.source,
            regions,
            config=self.config,
            dfn3_config=self.dfn3,
            enhance_fn=lambda x: np.asarray(x, dtype=np.float32) * 0.5,
            event_evidence_by_region=evidence,
            spatial_gate=gate,
        )
        self.assertEqual(result["route_counts"]["raw"], 1)
        self.assertEqual(result["route_counts"]["dfn3_denoised"], 1)
        self.assertEqual(result["route_counts"]["skipped_prefilter"], 1)
        manifest = json.loads(
            Path(result["manifest_path"]).read_text(encoding="utf-8")
        )
        rows = {r["region_index"]: r for r in manifest["regions"]}
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["output"]["frames"], SR)
        raw_path = Path(result["run_dir"]) / rows[0]["output"]["relative_path"]
        self.assertTrue(raw_path.is_file())
        self.assertEqual(rows[1]["status"], "completed")
        dfn3_path = Path(result["run_dir"]) / rows[1]["output"]["relative_path"]
        data, rate = sf.read(str(dfn3_path), dtype="float32")
        self.assertEqual(data.shape, (SR,))
        self.assertEqual(rate, SR)

    def test_dfn3_route_without_backend_aborts(self) -> None:
        regions = [_region(9, 0, SR)]
        with self.assertRaises(RuntimeError):
            run_routes(
                self.source,
                regions,
                config=self.config.model_copy(
                    update={
                        "routing": self.config.routing.model_copy(
                            update={"dfn3_noise_threshold": 0.0}
                        )
                    }
                ),
                dfn3_config=self.dfn3,
                enhance_fn=None,
                event_evidence_by_region={9: _noise_evidence(0.9)},
                spatial_gate={9: True},
            )

    def test_resume_skips_completed(self) -> None:
        regions = [_region(20, 0, SR), _region(21, SR, 2 * SR)]
        evidence = {20: _noise_evidence(0.0), 21: _noise_evidence(0.0)}
        gate = {20: True, 21: True}
        first = run_routes(
            self.source,
            regions,
            config=self.config,
            dfn3_config=self.dfn3,
            enhance_fn=None,
            event_evidence_by_region=evidence,
            spatial_gate=gate,
        )
        second = run_routes(
            self.source,
            regions,
            config=self.config,
            dfn3_config=self.dfn3,
            enhance_fn=None,
            event_evidence_by_region=evidence,
            spatial_gate=gate,
        )
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")


if __name__ == "__main__":
    unittest.main()
