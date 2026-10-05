from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import yaml

from dots_tts_lab import release
from dots_tts_lab.postprocess import load_edge_trim_config
from dots_tts_lab.release import ReleasePresetConfig, process_audio, true_peak_dbtp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tone(sample_rate: int, seconds: float, amplitude: float = 0.2) -> np.ndarray:
    times = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * 220.0 * times)


class ReleaseProcessingTests(unittest.TestCase):
    def test_trim_loudness_and_peak_safe_gain(self) -> None:
        sample_rate = 48_000
        with tempfile.TemporaryDirectory() as temporary:
            edge_path = Path(temporary) / "edge.yaml"
            edge_path.write_text(
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
            preset = ReleasePresetConfig.model_validate(
                {
                    "description": "test",
                    "edge_trim": True,
                    "loudness_normalization": True,
                    "target_loudness_lufs": -5.0,
                    "maximum_gain_db": 12.0,
                    "safety_limiter": True,
                    "true_peak_ceiling_dbtp": -1.0,
                    "formats": ["wav"],
                },
                strict=True,
            )
            source = np.concatenate(
                (np.zeros(sample_rate), tone(sample_rate, 1.0, 0.9), np.zeros(sample_rate))
            )
            output, details = process_audio(
                source,
                sample_rate,
                preset,
                edge_trim_config=load_edge_trim_config(edge_path),
            )
            self.assertLess(len(output), len(source))
            self.assertTrue(details["limited_by_true_peak"])
            self.assertLessEqual(true_peak_dbtp(output), -1.0 + 1e-9)


class ReleaseExporterTests(unittest.TestCase):
    def test_export_is_idempotent_traceable_and_format_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            samples = np.concatenate(
                (np.zeros(48_000), tone(48_000, 1.0), np.zeros(48_000))
            )
            sf.write(source, samples, 48_000, subtype="PCM_24")
            manifest = root / "source.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "job_id": "job-1",
                        "ordinal": 1,
                        "sentence_id": "sentence",
                        "text": "测试。",
                        "purpose": "test",
                        "seed": 1,
                        "request_id": "request",
                        "asset_sha256": "a" * 64,
                        "audio_relative_path": "reference.wav",
                        "audio_sha256": "b" * 64,
                        "prompt_text": "参考文本。",
                        "prompt_text_sha256": "c" * 64,
                        "speaker_id": "speaker",
                        "pool_ids": ["pool"],
                        "model_id": "model",
                        "model_revision": "revision",
                        "model_adapter": {"training_step": 400},
                        "precision": "bfloat16",
                        "runtime_options": {"optimize": True},
                        "sampling_options": {"num_steps": 10},
                        "output_path": "source.wav",
                        "output_sha256": sha256(source),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            acceptance = root / "acceptance.json"
            acceptance.write_text('{"accepted": true}\n', encoding="utf-8")
            edge = root / "edge.yaml"
            edge.write_text(
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
            edge_hash = load_edge_trim_config(edge).canonical_sha256()
            config = root / "release.yaml"
            payload = {
                "schema_version": 1,
                "config_id": "test_release",
                "config_version": 1,
                "source": {
                    "generation_manifest": "source.jsonl",
                    "generation_manifest_sha256": sha256(manifest),
                    "acceptance_report": "acceptance.json",
                    "acceptance_required": True,
                },
                "model": {"model_id": "model"},
                "postprocess": {
                    "edge_trim_config": "edge.yaml",
                    "edge_trim_config_sha256": edge_hash,
                },
                "presets": {
                    "raw": {
                        "description": "raw",
                        "edge_trim": False,
                        "loudness_normalization": False,
                        "target_loudness_lufs": None,
                        "maximum_gain_db": None,
                        "safety_limiter": False,
                        "true_peak_ceiling_dbtp": None,
                        "formats": ["wav"],
                    },
                    "release": {
                        "description": "release",
                        "edge_trim": True,
                        "loudness_normalization": True,
                        "target_loudness_lufs": -16.0,
                        "maximum_gain_db": 12.0,
                        "safety_limiter": True,
                        "true_peak_ceiling_dbtp": -1.0,
                        "formats": ["wav", "flac", "mp3"],
                    },
                },
                "formats": {
                    "wav": {"container": "WAV", "subtype": "PCM_24"},
                    "flac": {"container": "FLAC", "subtype": "PCM_24"},
                    "mp3": {
                        "container": "MP3",
                        "subtype": "MPEG_LAYER_III",
                        "bitrate_mode": "VARIABLE",
                        "compression_level": 0.1,
                    },
                },
                "output_root": "exports",
            }
            config.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with mock.patch.object(release, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "require run_label"):
                    release.export_preset(config, "raw", limit=1)
                raw_summary = release.export_preset(config, "raw")
                release.export_preset(config, "release")
                first_hashes = {
                    path.relative_to(root).as_posix(): sha256(path)
                    for path in (root / "exports").rglob("*")
                    if path.is_file()
                }
                release_summary = release.export_preset(config, "release")
                second_hashes = {
                    path.relative_to(root).as_posix(): sha256(path)
                    for path in (root / "exports").rglob("*")
                    if path.is_file()
                }
            self.assertEqual(raw_summary["output_count"], 1)
            self.assertEqual(release_summary["output_count"], 3)
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(
                sha256(root / "exports/raw/job-1.wav"), sha256(source)
            )
            for extension in ("wav", "flac", "mp3"):
                audio_path = root / f"exports/release/job-1.{extension}"
                sidecar_path = audio_path.with_suffix(audio_path.suffix + ".json")
                decoded, sample_rate = sf.read(
                    str(audio_path), dtype="float64", always_2d=False
                )
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                self.assertEqual(sample_rate, 48_000)
                self.assertLessEqual(true_peak_dbtp(decoded), -1.0 + 1e-3)
                self.assertEqual(sidecar["source_audio_sha256"], sha256(source))
                self.assertEqual(sidecar["output"]["sha256"], sha256(audio_path))
                self.assertEqual(sidecar["model"]["sampling_options"]["num_steps"], 10)


if __name__ == "__main__":
    unittest.main()
