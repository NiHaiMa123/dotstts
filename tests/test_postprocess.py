from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scripts import postprocess_generation_manifest

from dots_tts_lab.postprocess import (
    DEFAULT_FUXUAN_VOICE_POLISH_CONFIG_PATH,
    EdgeTrimConfig,
    VoicePolishConfig,
    apply_fuxuan_voice_polish,
    load_voice_polish_config,
    measure_integrated_loudness,
    safe_edge_trim,
)
from dots_tts_lab.standardization import true_peak_estimate


def config() -> EdgeTrimConfig:
    return EdgeTrimConfig.model_validate(
        {
            "schema_version": 1,
            "config_id": "test_edge_trim",
            "config_version": 1,
            "silence_frame_ms": 20.0,
            "silence_hop_ms": 10.0,
            "silence_threshold_dbfs": -50.0,
            "trim_trigger_seconds": 0.5,
            "preserve_leading_seconds": 0.12,
            "preserve_trailing_seconds": 0.16,
            "fade_seconds": 0.005,
            "output_subtype": "PCM_24",
        },
        strict=True,
    )


def tone(rate: int, seconds: float) -> np.ndarray:
    x = np.arange(round(rate * seconds), dtype=np.float64) / rate
    return 0.2 * np.sin(2 * np.pi * 220 * x)


class SafeEdgeTrimTests(unittest.TestCase):
    def test_long_edges_are_trimmed_with_padding(self) -> None:
        rate = 48_000
        source = np.concatenate((np.zeros(rate), tone(rate, 1.0), np.zeros(rate)))
        output, details = safe_edge_trim(source, rate, config())
        self.assertGreater(details["leading_samples_removed"], round(0.8 * rate))
        self.assertGreater(details["trailing_samples_removed"], round(0.8 * rate))
        self.assertAlmostEqual(len(output) / rate, 1.28, delta=0.04)
        self.assertEqual(output[0], 0.0)
        self.assertEqual(output[-1], 0.0)

    def test_short_edges_are_byte_equivalent_in_memory(self) -> None:
        rate = 48_000
        source = np.concatenate((np.zeros(round(0.4 * rate)), tone(rate, 1.0)))
        output, details = safe_edge_trim(source, rate, config())
        np.testing.assert_array_equal(output, source)
        self.assertEqual(details["leading_samples_removed"], 0)

    def test_all_silent_audio_is_not_destroyed(self) -> None:
        source = np.zeros(48_000, dtype=np.float64)
        output, details = safe_edge_trim(source, 48_000, config())
        np.testing.assert_array_equal(output, source)
        self.assertTrue(details["all_silent"])

    def test_operation_is_idempotent(self) -> None:
        rate = 48_000
        source = np.concatenate((np.zeros(rate), tone(rate, 1.0), np.zeros(rate)))
        first, _ = safe_edge_trim(source, rate, config())
        second, details = safe_edge_trim(first, rate, config())
        np.testing.assert_array_equal(second, first)
        self.assertEqual(details["leading_samples_removed"], 0)
        self.assertEqual(details["trailing_samples_removed"], 0)

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mono"):
            safe_edge_trim(np.zeros((10, 2)), 48_000, config())


class FuxuanVoicePolishTests(unittest.TestCase):
    def test_default_profile_is_versioned_and_loads_strictly(self) -> None:
        profile = load_voice_polish_config(DEFAULT_FUXUAN_VOICE_POLISH_CONFIG_PATH)
        self.assertEqual(profile.config_id, "fuxuan_voice_polish_roughness_control_v2")
        self.assertEqual(profile.config_version, 2)
        self.assertEqual(len(profile.canonical_sha256()), 64)

    def test_nearfield_chain_preserves_shape_and_true_peak_ceiling(self) -> None:
        rate = 48_000
        time = np.arange(rate * 2, dtype=np.float64) / rate
        source = 0.9 * np.sin(2.0 * np.pi * 240.0 * time)
        source += 0.08 * np.sin(2.0 * np.pi * 7_000.0 * time)

        output, details = apply_fuxuan_voice_polish(source, rate)

        self.assertEqual(output.shape, source.shape)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(details["profile"], "fuxuan_voice_polish_roughness_control_v2")
        self.assertLessEqual(true_peak_estimate(output, 4), 10.0 ** (-1.0 / 20.0) + 1e-12)
        self.assertGreater(details["deesser"]["maximum_reduction_db"], 0.0)
        self.assertLessEqual(details["deesser"]["maximum_reduction_db"], 1.25)

    def test_parallel_compression_brings_quiet_detail_closer(self) -> None:
        rate = 48_000
        time = np.arange(rate * 2, dtype=np.float64) / rate
        quiet = 0.04 * np.sin(2.0 * np.pi * 440.0 * time)
        loud = 0.4 * np.sin(2.0 * np.pi * 440.0 * time)
        source = np.concatenate((quiet, loud))

        output, _ = apply_fuxuan_voice_polish(
            source,
            rate,
            include_brightness=False,
        )

        quiet_slice = slice(rate, rate * 2)
        loud_slice = slice(rate * 3, rate * 4)
        source_ratio = np.sqrt(np.mean(source[quiet_slice] ** 2)) / np.sqrt(
            np.mean(source[loud_slice] ** 2)
        )
        output_ratio = np.sqrt(np.mean(output[quiet_slice] ** 2)) / np.sqrt(
            np.mean(output[loud_slice] ** 2)
        )
        self.assertGreater(output_ratio, source_ratio * 1.05)

    def test_v2_dynamic_bands_reduce_hot_harshness_and_target_loudness(self) -> None:
        profile = load_voice_polish_config(
            Path("configs/lab/postprocess/fuxuan_voice_polish_roughness_control_v2.yaml")
        )
        rate = 48_000
        time = np.arange(rate * 3, dtype=np.float64) / rate
        source = 0.12 * np.sin(2.0 * np.pi * 240.0 * time)
        source += 0.10 * np.sin(2.0 * np.pi * 3_100.0 * time)
        source += 0.06 * np.sin(2.0 * np.pi * 7_200.0 * time)

        output, details = apply_fuxuan_voice_polish(source, rate, profile)

        self.assertEqual(details["profile"], "fuxuan_voice_polish_roughness_control_v2")
        self.assertEqual(len(details["dynamic_eq_bands"]), 2)
        self.assertGreater(
            details["dynamic_eq_bands"][0]["maximum_reduction_db"], 0.5
        )
        self.assertGreater(
            details["dynamic_eq_bands"][1]["maximum_reduction_db"], 0.25
        )
        self.assertAlmostEqual(
            pyln.Meter(rate).integrated_loudness(output), -17.0, delta=0.15
        )


def polish_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "config_id": "test_polish",
        "config_version": 1,
        "brightness_crossover_hz": 8000.0,
        "brightness_gain_db": 0.0,
        "body_eq": {"center_hz": 210.0, "gain_db": 0.0, "q": 0.75},
        "presence_eq": {"center_hz": 4000.0, "gain_db": 0.0, "q": 1.0},
        "parallel_compressor": {
            "threshold_dbfs": -24.0,
            "ratio": 3.0,
            "attack_ms": 12.0,
            "release_ms": 140.0,
            "makeup_gain_db": 0.0,
            "mix": 0.10,
            "gate_dbfs": -45.0,
            "frame_ms": 10.0,
        },
        "deesser": {
            "low_hz": 5000.0,
            "high_hz": 10000.0,
            "threshold_dbfs": -28.0,
            "ratio": 2.0,
            "maximum_reduction_db": 1.0,
            "attack_ms": 5.0,
            "release_ms": 80.0,
            "frame_ms": 10.0,
        },
        "match_input_loudness": False,
        "target_loudness_lufs": -17.0,
        "maximum_loudness_adjustment_db": 6.0,
        "true_peak_ceiling_dbtp": -1.5,
        "true_peak_oversample": 4,
    }
    payload.update(overrides)
    return payload


class PromptLoudnessCalibrationTests(unittest.TestCase):
    def test_calibration_block_parses(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                prompt_loudness_calibration={"maximum_gain_db": 6.0}
            ),
            strict=True,
        )
        self.assertIsNotNone(profile.prompt_loudness_calibration)
        self.assertEqual(profile.prompt_loudness_calibration.maximum_gain_db, 6.0)

    def test_calibration_rejects_match_input_loudness(self) -> None:
        with self.assertRaisesRegex(Exception, "match_input_loudness"):
            VoicePolishConfig.model_validate(
                polish_payload(
                    match_input_loudness=True,
                    target_loudness_lufs=None,
                    prompt_loudness_calibration={"maximum_gain_db": 6.0},
                ),
                strict=True,
            )

    def test_unused_calibration_keeps_legacy_hash_contract(self) -> None:
        config = VoicePolishConfig.model_validate(polish_payload(), strict=True)
        payload = config.model_dump(mode="json")
        payload.pop("prompt_loudness_calibration")
        payload.pop("low_cut_hz")
        payload.pop("dynamic_eq_bands")
        payload.pop("exciter")
        payload.pop("ambience")
        payload.pop("stabilizer")
        expected = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(config.canonical_sha256(), expected)

    def test_fixed_gain_replaces_loudness_matching(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                brightness_gain_db=0.0,
                prompt_loudness_calibration={"maximum_gain_db": 6.0},
            ),
            strict=True,
        )
        rate = 48_000
        source = tone(rate, 2.0)

        neutral, _ = apply_fuxuan_voice_polish(
            source, rate, profile, calibration_gain_db=0.0
        )
        boosted, details = apply_fuxuan_voice_polish(
            source, rate, profile, calibration_gain_db=3.0
        )

        self.assertEqual(details["loudness_mode"], "prompt_calibrated")
        self.assertEqual(details["calibration_gain_db"], 3.0)
        self.assertEqual(details["loudness_adjustment_db"], 3.0)
        loudness_delta = measure_integrated_loudness(
            boosted, rate
        ) - measure_integrated_loudness(neutral, rate)
        self.assertAlmostEqual(loudness_delta, 3.0, delta=0.2)

    def test_none_gain_falls_back_to_target_loudness(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                prompt_loudness_calibration={"maximum_gain_db": 6.0}
            ),
            strict=True,
        )
        rate = 48_000
        output, details = apply_fuxuan_voice_polish(
            tone(rate, 2.0), rate, profile
        )
        self.assertEqual(details["loudness_mode"], "target_lufs")
        self.assertIsNone(details["calibration_gain_db"])
        self.assertAlmostEqual(
            pyln.Meter(rate).integrated_loudness(output), -17.0, delta=0.3
        )


class ExciterTests(unittest.TestCase):
    def test_exciter_block_parses(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                exciter={
                    "source_low_hz": 4000.0,
                    "drive": 4.0,
                    "bias": 0.3,
                    "harmonic_low_hz": 8000.0,
                    "harmonic_high_hz": 16000.0,
                    "mix": 0.10,
                }
            ),
            strict=True,
        )
        self.assertIsNotNone(profile.exciter)
        self.assertEqual(profile.exciter.mix, 0.10)

    def test_exciter_rejects_harmonic_below_source(self) -> None:
        with self.assertRaisesRegex(Exception, "above source_low_hz"):
            VoicePolishConfig.model_validate(
                polish_payload(
                    exciter={
                        "source_low_hz": 8000.0,
                        "drive": 4.0,
                        "bias": 0.3,
                        "harmonic_low_hz": 6000.0,
                        "harmonic_high_hz": 16000.0,
                        "mix": 0.10,
                    }
                ),
                strict=True,
            )

    def test_unused_exciter_keeps_legacy_hash_contract(self) -> None:
        config = VoicePolishConfig.model_validate(polish_payload(), strict=True)
        payload = config.model_dump(mode="json")
        for key in (
            "prompt_loudness_calibration",
            "low_cut_hz",
            "dynamic_eq_bands",
            "exciter",
            "ambience",
            "stabilizer",
        ):
            payload.pop(key)
        expected = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(config.canonical_sha256(), expected)

    def test_exciter_adds_targeted_hf_energy(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                exciter={
                    "source_low_hz": 4000.0,
                    "drive": 4.0,
                    "bias": 0.3,
                    "harmonic_low_hz": 8000.0,
                    "harmonic_high_hz": 16000.0,
                    "mix": 0.10,
                }
            ),
            strict=True,
        )
        rate = 48_000
        x = np.arange(rate, dtype=np.float64) / rate
        source = 0.15 * np.sin(2 * np.pi * 5000.0 * x)

        def band_energy(sig: np.ndarray) -> float:
            from scipy.signal import butter, sosfiltfilt

            band = sosfiltfilt(
                butter(2, [8000.0, 16000.0], btype="bandpass", fs=rate, output="sos"),
                sig,
            )
            return float(np.sqrt(np.mean(band**2)))

        clean, _ = apply_fuxuan_voice_polish(
            source, rate, profile.model_copy(update={"exciter": None})
        )
        excited, details = apply_fuxuan_voice_polish(source, rate, profile)

        self.assertGreater(band_energy(excited), band_energy(clean) * 1.2)
        self.assertEqual(details["exciter"]["mix"], 0.10)
        self.assertIsNotNone(details["exciter"]["harmonic_level_dbfs"])

    def test_exciter_keeps_silence_silent(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                exciter={
                    "source_low_hz": 4000.0,
                    "drive": 4.0,
                    "bias": 0.3,
                    "harmonic_low_hz": 8000.0,
                    "harmonic_high_hz": 16000.0,
                    "mix": 0.10,
                }
            ),
            strict=True,
        )
        rate = 48_000
        source = np.concatenate((np.zeros(rate // 2), tone(rate, 0.5)))
        output, _ = apply_fuxuan_voice_polish(source, rate, profile)
        self.assertLess(float(np.abs(output[: rate // 4]).max()), 1e-3)


class AmbienceTests(unittest.TestCase):
    def _ambience_payload(self) -> dict[str, object]:
        return {
            "rt60_ms": 220.0,
            "pre_delay_ms": 12.0,
            "band_low_hz": 250.0,
            "band_high_hz": 7000.0,
            "mix": 0.08,
            "tail": True,
        }

    def test_ambience_block_parses(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(ambience=self._ambience_payload()),
            strict=True,
        )
        self.assertIsNotNone(profile.ambience)
        self.assertEqual(profile.ambience.mix, 0.08)

    def test_ambience_rejects_inverted_band(self) -> None:
        with self.assertRaisesRegex(Exception, "band_high_hz"):
            VoicePolishConfig.model_validate(
                polish_payload(
                    ambience={**self._ambience_payload(), "band_high_hz": 200.0}
                ),
                strict=True,
            )

    def test_ambience_appends_decay_tail(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(ambience=self._ambience_payload()),
            strict=True,
        )
        rate = 48_000
        source = np.concatenate((np.zeros(rate // 2), tone(rate, 0.5)))
        output, details = apply_fuxuan_voice_polish(source, rate, profile)
        self.assertGreater(len(output), len(source))
        self.assertGreater(details["ambience"]["tail_samples_added"], 0)
        # A decay tail should leave audible energy after the dry signal ends.
        tail = output[len(source) + rate // 20 : len(source) + rate // 10]
        self.assertGreater(float(np.abs(tail).max()), 1e-4)

    def test_ambience_off_preserves_length(self) -> None:
        profile = VoicePolishConfig.model_validate(polish_payload(), strict=True)
        rate = 48_000
        output, details = apply_fuxuan_voice_polish(tone(rate, 0.5), rate, profile)
        self.assertEqual(len(output), int(rate * 0.5))
        self.assertEqual(details["ambience"]["mix"], 0.0)


class StabilizerTests(unittest.TestCase):
    def _stabilizer_payload(self) -> dict[str, object]:
        return {
            "low_hz": 400.0,
            "high_hz": 9000.0,
            "frames": 9,
            "strength": 0.8,
        }

    def _flutter_db(self, audio: np.ndarray, rate: int) -> float:
        hop = rate // 96
        frames = 1 + (len(audio) - 2048) // hop
        envelope = np.empty(frames)
        band_lo, band_hi = int(500 * 2048 / rate), int(4000 * 2048 / rate)
        for index in range(frames):
            window = audio[index * hop : index * hop + 2048] * np.hanning(2048)
            envelope[index] = np.abs(np.fft.rfft(window))[band_lo:band_hi].sum()
        envelope_db = 20.0 * np.log10(envelope + 1e-9)
        return float(np.abs(np.diff(envelope_db)).mean())

    def test_stabilizer_block_parses(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(stabilizer=self._stabilizer_payload()),
            strict=True,
        )
        self.assertIsNotNone(profile.stabilizer)
        self.assertEqual(profile.stabilizer.frames, 9)
        self.assertEqual(profile.stabilizer.strength, 0.8)

    def test_stabilizer_rejects_even_frames(self) -> None:
        with self.assertRaisesRegex(Exception, "frames must be odd"):
            VoicePolishConfig.model_validate(
                polish_payload(
                    stabilizer={**self._stabilizer_payload(), "frames": 8}
                ),
                strict=True,
            )

    def test_stabilizer_rejects_inverted_band(self) -> None:
        with self.assertRaisesRegex(Exception, "high_hz"):
            VoicePolishConfig.model_validate(
                polish_payload(
                    stabilizer={**self._stabilizer_payload(), "high_hz": 300.0}
                ),
                strict=True,
            )

    def test_stabilizer_damps_envelope_warble(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(
                brightness_gain_db=0.0,
                stabilizer=self._stabilizer_payload(),
            ),
            strict=True,
        )
        rate = 48_000
        t = np.arange(rate) / rate
        warble = 1.0 + 0.45 * np.sin(2.0 * np.pi * 23.0 * t)
        source = tone(rate, 1.0) * warble
        stabilized, details = apply_fuxuan_voice_polish(
            source, rate, profile, calibration_gain_db=0.0
        )
        self.assertLess(self._flutter_db(stabilized, rate), self._flutter_db(source, rate))
        self.assertEqual(details["stabilizer"]["frames"], 9)

    def test_stabilizer_preserves_length_and_quiet(self) -> None:
        profile = VoicePolishConfig.model_validate(
            polish_payload(stabilizer=self._stabilizer_payload()),
            strict=True,
        )
        rate = 48_000
        source = tone(rate, 0.5)
        output, _ = apply_fuxuan_voice_polish(
            source, rate, profile, calibration_gain_db=0.0
        )
        self.assertEqual(len(output), len(source))
        silent, _ = apply_fuxuan_voice_polish(
            np.zeros(rate // 4), rate, profile, calibration_gain_db=0.0
        )
        self.assertEqual(float(np.abs(silent).max()), 0.0)


class PostprocessManifestTests(unittest.TestCase):
    def test_runner_preserves_source_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.wav"
            samples = np.concatenate(
                (np.zeros(48_000), tone(48_000, 1.0), np.zeros(48_000))
            )
            sf.write(source, samples, 48_000, subtype="PCM_24")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "source.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "ok",
                        "output_path": "raw.wav",
                        "output_sha256": source_hash,
                        "generation_seconds": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "config_id: test_edge_trim",
                        "config_version: 1",
                        "silence_frame_ms: 20.0",
                        "silence_hop_ms: 10.0",
                        "silence_threshold_dbfs: -50.0",
                        "trim_trigger_seconds: 0.5",
                        "preserve_leading_seconds: 0.12",
                        "preserve_trailing_seconds: 0.16",
                        "fade_seconds: 0.005",
                        "output_subtype: PCM_24",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_manifest = root / "processed.jsonl"
            with mock.patch.object(postprocess_generation_manifest, "ROOT", root):
                summary = postprocess_generation_manifest.run(
                    manifest,
                    root / "processed",
                    output_manifest,
                    config_path,
                )
            self.assertEqual(summary["trimmed_count"], 1)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            output_row = json.loads(output_manifest.read_text(encoding="utf-8"))
            output = root / output_row["output_path"]
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            self.assertLess(output_row["sample_count"], len(samples))


if __name__ == "__main__":
    unittest.main()
