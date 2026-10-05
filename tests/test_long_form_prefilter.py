from __future__ import annotations

import json
import shutil
import unittest
import wave
from pathlib import Path

import numpy as np

from dots_tts_lab.long_form_prefilter import (
    PrefilterConfig,
    PrefilterPaths,
    build_speech_regions,
    load_prefilter_config,
    prefilter_completed,
    prefilter_source,
    run_prefilter,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_prefilter_tests"
SR = 48000


class EnergyFakeVAD:
    """Deterministic stand-in for Silero: speech = window RMS above threshold."""

    def __init__(self, rms_threshold: float = 0.01) -> None:
        self.rms_threshold = rms_threshold
        self.reset_calls = 0
        self.calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def probabilities(self, samples_16k_mono: np.ndarray) -> np.ndarray:
        self.calls += 1
        samples = np.asarray(samples_16k_mono, dtype=np.float32).reshape(-1)
        window = 512
        count = samples.size // window + (1 if samples.size % window else 0)
        padded = np.zeros(count * window, dtype=np.float32)
        padded[: samples.size] = samples
        frames = padded.reshape(count, window)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        return (rms > self.rms_threshold).astype(np.float32) * 0.9


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(samples.shape[1] if samples.ndim > 1 else 1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _test_config() -> PrefilterConfig:
    base = load_prefilter_config()
    return base.model_copy(
        update={
            "paths": PrefilterPaths(
                work_root="data/work/tmp_prefilter_tests/work",
                report_root="data/work/tmp_prefilter_tests/reports",
            )
        }
    )


class PrefilterConfigTests(unittest.TestCase):
    def test_config_loads_and_hashes(self) -> None:
        config = load_prefilter_config()
        self.assertEqual(config.config_id, "long_form_prefilter_v1")
        self.assertEqual(config.vad.backend, "silero_onnx")
        self.assertEqual(len(config.config_sha256()), 64)
        self.assertEqual(
            config.config_sha256(), load_prefilter_config().config_sha256()
        )

    def test_config_rejects_inbox_paths(self) -> None:
        with self.assertRaises(Exception):
            PrefilterPaths(
                work_root="data/inbox/prefilter",
                report_root="data/work/long_form/prefilter",
            )


class BuildSpeechRegionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vad = load_prefilter_config().vad
        self.hop = self.vad.window_samples / self.vad.sample_rate_hz

    def test_merges_gaps_and_drops_short_segments(self) -> None:
        probs = np.zeros(200, dtype=np.float32)
        probs[10:40] = 0.9
        probs[45:80] = 0.9  # 5-window gap (< merge gap 0.35s/0.032s ≈ 11 windows)
        probs[150:155] = 0.9  # too short: 5 windows ≈ 0.16s < min 0.3s
        regions = build_speech_regions(
            probs, window_hop_seconds=self.hop, config=self.vad
        )
        self.assertEqual(regions, [(10, 80)])

    def test_splits_long_region_at_probability_valley(self) -> None:
        probs = np.full(2000, 0.9, dtype=np.float32)
        valley = 1000
        probs[valley - 2 : valley + 3] = 0.0
        regions = build_speech_regions(
            probs, window_hop_seconds=self.hop, config=self.vad
        )
        # 2000 windows × 32ms = 64s > 34s cap → split near the valley
        self.assertGreaterEqual(len(regions), 2)
        for start, end in regions:
            self.assertLessEqual(end - start, round(34.0 / self.hop) + 2)
        boundaries = [r[0] for r in regions[1:]]
        self.assertTrue(any(abs(b - valley) <= 8 for b in boundaries))

    def test_empty_input(self) -> None:
        self.assertEqual(
            build_speech_regions(
                np.zeros(0, dtype=np.float32),
                window_hop_seconds=self.hop,
                config=self.vad,
            ),
            [],
        )


class PrefilterSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP.mkdir(parents=True, exist_ok=True)
        self.config = _test_config()

    def tearDown(self) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def _speech_silence_wav(self) -> Path:
        silence = np.zeros(round(1.0 * SR), dtype=np.float32)
        speech = _tone(3.0)
        tail = np.zeros(round(1.0 * SR), dtype=np.float32)
        mono = np.concatenate([silence, speech, tail])
        stereo = np.stack([mono, mono], axis=1)
        path = TMP / "speech.wav"
        _write_wav(path, stereo)
        return path

    def test_regions_written_and_resume_cached(self) -> None:
        path = self._speech_silence_wav()
        detector = EnergyFakeVAD()
        result = prefilter_source(path, config=self.config, detector=detector)
        self.assertEqual(result["status"], "completed")
        regions = json.loads(Path(result["regions_path"]).read_text(encoding="utf-8"))
        self.assertEqual(regions["usable_region_count"], 1)
        region = regions["regions"][0]
        self.assertEqual(region["status"], "usable")
        # tone spans seconds 1..4 → ~96000 frames at 48k
        self.assertAlmostEqual(region["source_start_frame"], SR, delta=SR // 4)
        self.assertAlmostEqual(region["source_end_frame"], 4 * SR, delta=SR // 4)
        calls_before = detector.calls
        second = prefilter_source(path, config=self.config, detector=detector)
        self.assertEqual(second["status"], "cached")
        self.assertEqual(detector.calls, calls_before)
        self.assertTrue(
            prefilter_completed(
                ROOT / self.config.paths.work_root,
                result["source_sha256"],
                self.config,
            )
        )

    def test_config_change_invalidates_cache(self) -> None:
        path = self._speech_silence_wav()
        detector = EnergyFakeVAD()
        first = prefilter_source(path, config=self.config, detector=detector)
        other = self.config.model_copy(
            update={"config_version": 2, "config_id": "long_form_prefilter_v1b"}
        )
        self.assertFalse(
            prefilter_completed(
                ROOT / self.config.paths.work_root,
                first["source_sha256"],
                other,
            )
        )

    def test_failed_run_is_not_resumed_as_completed(self) -> None:
        path = self._speech_silence_wav()

        class BrokenVAD(EnergyFakeVAD):
            def probabilities(self, samples):  # type: ignore[override]
                raise RuntimeError("simulated crash")

        with self.assertRaises(RuntimeError):
            prefilter_source(path, config=self.config, detector=BrokenVAD())
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = json.loads(
            (
                ROOT
                / self.config.paths.work_root
                / digest[:2]
                / digest
                / "state.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(state["status"], "failed")
        self.assertFalse(
            prefilter_completed(
                ROOT / self.config.paths.work_root, digest, self.config
            )
        )

    def test_decorrelated_stereo_quarantined(self) -> None:
        rng = np.random.RandomState(0)
        left = rng.uniform(-0.3, 0.3, 5 * SR).astype(np.float32)
        right = rng.uniform(-0.3, 0.3, 5 * SR).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        path = TMP / "binaural_noise.wav"
        _write_wav(path, stereo)
        result = prefilter_source(path, config=self.config, detector=EnergyFakeVAD())
        regions = json.loads(Path(result["regions_path"]).read_text(encoding="utf-8"))
        self.assertGreaterEqual(regions["quarantine_region_count"], 1)
        self.assertIn("spatial_risk_region", regions["regions"][0]["reasons"])

    def test_transient_dense_region_quarantined(self) -> None:
        speech = _tone(5.0)
        rng = np.random.RandomState(1)
        clicks = np.zeros_like(speech)
        burst = (rng.uniform(-0.9, 0.9, 48)).astype(np.float32)  # ~1ms burst
        for i in range(0, clicks.size - 48, SR // 4):  # 4 clicks/sec > 2.0 limit
            clicks[i : i + 48] = burst
        mono = np.clip(speech + clicks, -1.0, 1.0)
        stereo = np.stack([mono, mono], axis=1)
        path = TMP / "clicks.wav"
        _write_wav(path, stereo)
        result = prefilter_source(path, config=self.config, detector=EnergyFakeVAD())
        regions = json.loads(Path(result["regions_path"]).read_text(encoding="utf-8"))
        self.assertGreaterEqual(regions["quarantine_region_count"], 1)
        self.assertIn("transient_dense_region", regions["regions"][0]["reasons"])

    def test_run_prefilter_resets_detector_per_source(self) -> None:
        first = self._speech_silence_wav()
        second = TMP / "speech2.wav"
        shutil.copy(first, second)
        detector = EnergyFakeVAD()
        result = run_prefilter([first, second], config=self.config, detector=detector)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(detector.reset_calls, 2)
        self.assertEqual(len(result["sources"]), 2)


if __name__ == "__main__":
    unittest.main()
