from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.fuxuan_batch import (
    BatchCancelled,
    run_batch,
    split_text,
)
from dots_tts_lab.postprocess import (
    DEFAULT_BRIGHTNESS_GAIN_DB,
    EdgeTrimConfig,
    apply_default_brightness,
)
from dots_tts_lab.standardization import true_peak_estimate


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "audio": np.full(4_800, 0.1, dtype=np.float32),
            "sample_rate": 48_000,
        }


def edge_trim_config() -> EdgeTrimConfig:
    return EdgeTrimConfig.model_validate(
        {
            "schema_version": 1,
            "config_id": "test_fuxuan_batch_trim",
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


class TextSplittingTests(unittest.TestCase):
    def test_keeps_terminal_punctuation_and_paragraph_boundaries(self) -> None:
        self.assertEqual(
            split_text("第一句。第二句！\n第三句没有句号"),
            ["第一句。", "第二句！", "第三句没有句号"],
        )

    def test_long_sentence_prefers_soft_boundary(self) -> None:
        result = split_text("甲" * 12 + "，" + "乙" * 15 + "。", max_chars=20)
        self.assertEqual(result, ["甲" * 12 + "，", "乙" * 15 + "。"])


class BatchGenerationTests(unittest.TestCase):
    def test_multiple_txt_files_only_leave_final_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            (input_dir / "nested").mkdir(parents=True)
            (input_dir / "first.txt").write_text("第一句。第二句！", encoding="utf-8")
            (input_dir / "nested/second.TXT").write_text("第三句。", encoding="utf-8")
            (input_dir / "ignored.md").write_text("不处理。", encoding="utf-8")
            runtime = FakeRuntime()

            summary = run_batch(
                runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                edge_trim_config=edge_trim_config(),
                pause_ms=50,
            )

            self.assertEqual(summary["input_count"], 2)
            self.assertEqual(len(summary["generated"]), 2)
            self.assertEqual(summary["errors"], [])
            self.assertEqual(len(runtime.calls), 3)
            self.assertEqual(
                [call["text"] for call in runtime.calls],
                ["第一句。", "第二句！", "第三句。"],
            )
            self.assertTrue(all(call["num_steps"] == 16 for call in runtime.calls))
            self.assertEqual(
                sorted(
                    path.relative_to(output_dir).as_posix()
                    for path in output_dir.rglob("*")
                ),
                ["first.wav", "nested", "nested/second.wav"],
            )
            first_audio, sample_rate = sf.read(output_dir / "first.wav")
            self.assertEqual(sample_rate, 48_000)
            self.assertEqual(first_audio.size, 12_000)
            self.assertEqual(
                summary["generated"][0]["processing"]["profile"],
                "fuxuan_voice_polish_roughness_control_v2",
            )
            self.assertFalse(any("partial" in path.name for path in output_dir.rglob("*")))

            second_summary = run_batch(
                runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                edge_trim_config=edge_trim_config(),
                pause_ms=50,
            )
            self.assertEqual(second_summary["generated"], [])
            self.assertEqual(second_summary["skipped"], ["first.txt", "nested/second.TXT"])
            self.assertEqual(len(runtime.calls), 3)

    def test_cancellation_stops_before_writing_a_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "cancel.txt").write_text("第一句。第二句。", encoding="utf-8")
            runtime = FakeRuntime()

            with self.assertRaises(BatchCancelled):
                run_batch(
                    runtime,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    edge_trim_config=edge_trim_config(),
                    cancelled=lambda: bool(runtime.calls),
                )

            self.assertFalse((output_dir / "cancel.wav").exists())
            self.assertFalse(any(output_dir.rglob("*.partial.wav")))


class BrightnessProcessingTests(unittest.TestCase):
    def test_brightens_high_band_without_changing_length_or_pitch(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        low = 0.15 * np.sin(2.0 * np.pi * 500.0 * time)
        high = 0.15 * np.sin(2.0 * np.pi * 6_000.0 * time)
        audio = low + high

        processed, details = apply_default_brightness(audio, sample_rate)

        frequencies = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
        original_spectrum = np.abs(np.fft.rfft(audio))
        processed_spectrum = np.abs(np.fft.rfft(processed))
        low_bin = int(np.argmin(np.abs(frequencies - 500.0)))
        high_bin = int(np.argmin(np.abs(frequencies - 6_000.0)))
        relative_gain_db = 20.0 * np.log10(
            (processed_spectrum[high_bin] / original_spectrum[high_bin])
            / (processed_spectrum[low_bin] / original_spectrum[low_bin])
        )

        self.assertEqual(processed.size, audio.size)
        self.assertAlmostEqual(relative_gain_db, DEFAULT_BRIGHTNESS_GAIN_DB, delta=0.08)
        self.assertEqual(details["profile"], "E_half_brightness_same_pitch")

    def test_true_peak_safety_reduces_hot_audio(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        audio = 0.99 * np.sin(2.0 * np.pi * 6_000.0 * time)

        processed, details = apply_default_brightness(audio, sample_rate)

        ceiling = 10.0 ** (-1.0 / 20.0)
        self.assertLessEqual(true_peak_estimate(processed, 4), ceiling + 1e-12)
        self.assertLess(float(details["peak_safety_attenuation_db"]), 0.0)


if __name__ == "__main__":
    unittest.main()
