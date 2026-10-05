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
    apply_fuxuan_voice_polish,
    load_voice_polish_config,
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
