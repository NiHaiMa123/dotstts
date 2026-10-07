from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from dots_tts_lab.fuxuan_batch import (
    BatchCancelled,
    _calibrate_prompt_loudness,
    run_batch,
    soften_emphasis_punctuation,
    split_text,
)
from dots_tts_lab.postprocess import (
    DEFAULT_BRIGHTNESS_GAIN_DB,
    EdgeTrimConfig,
    VoicePolishConfig,
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


class SineRuntime(FakeRuntime):
    def __init__(self, amplitude: float) -> None:
        super().__init__()
        self.amplitude = amplitude

    def generate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        rate = 48_000
        time = np.arange(rate, dtype=np.float64) / rate
        return {
            "audio": (self.amplitude * np.sin(2 * np.pi * 440.0 * time)).astype(
                np.float32
            ),
            "sample_rate": rate,
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


class EmphasisSofteningTests(unittest.TestCase):
    def test_demotes_exclamation_to_period(self) -> None:
        self.assertEqual(soften_emphasis_punctuation("你敢！胡说！！"), "你敢。胡说。")

    def test_keeps_question_mark_when_stripping_emphasis(self) -> None:
        self.assertEqual(soften_emphasis_punctuation("什么？！你说？"), "什么？你说？")

    def test_collapses_break_runs_left_by_softening(self) -> None:
        self.assertEqual(soften_emphasis_punctuation("走！，也好"), "走。也好")

    def test_halfwidth_exclamation_is_softened(self) -> None:
        self.assertEqual(soften_emphasis_punctuation("wait! stop!!"), "wait. stop.")

    def test_run_batch_softens_segment_text_before_generate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "line.txt").write_text("开火！冷静点？", encoding="utf-8")
            runtime = FakeRuntime()

            run_batch(
                runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                edge_trim_config=edge_trim_config(),
                pause_ms=50,
                soften_emphasis=True,
            )

            self.assertEqual(runtime.calls[0]["text"], "开火。")
            self.assertEqual(runtime.calls[1]["text"], "冷静点？")

    def test_run_batch_leaves_punctuation_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            (input_dir / "line.txt").write_text("开火！", encoding="utf-8")
            runtime = FakeRuntime()

            run_batch(
                runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                edge_trim_config=edge_trim_config(),
                pause_ms=50,
            )

            self.assertEqual(runtime.calls[0]["text"], "开火！")


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


def calibrated_polish_config() -> VoicePolishConfig:
    return VoicePolishConfig.model_validate(
        {
            "schema_version": 1,
            "config_id": "test_calibrated_polish",
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
            "prompt_loudness_calibration": {"maximum_gain_db": 6.0},
            "maximum_loudness_adjustment_db": 6.0,
            "true_peak_ceiling_dbtp": -1.5,
            "true_peak_oversample": 4,
        },
        strict=True,
    )


class PromptLoudnessCalibrationBatchTests(unittest.TestCase):
    def test_outputs_are_anchored_to_reference_loudness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            output_dir = root / "output"
            (input_dir / "line.txt").write_text("第一句。第二句！", encoding="utf-8")
            rate = 48_000
            time = np.arange(rate * 2, dtype=np.float64) / rate
            reference = 0.12 * np.sin(2.0 * np.pi * 220.0 * time)
            prompt_wav = root / "prompt.wav"
            sf.write(prompt_wav, reference, rate, subtype="PCM_24")
            runtime = SineRuntime(amplitude=0.2)

            summary = run_batch(
                runtime,
                input_dir=input_dir,
                output_dir=output_dir,
                edge_trim_config=edge_trim_config(),
                pause_ms=0,
                prompt_audio_path=prompt_wav,
                prompt_text="参考台词。",
                voice_polish_config=calibrated_polish_config(),
            )

            calibration = summary["loudness_calibration"]
            self.assertEqual(calibration["status"], "ok")
            self.assertLess(calibration["calibration_gain_db"], 0.0)
            self.assertEqual(len(summary["generated"]), 1)
            processing = summary["generated"][0]["processing"]
            self.assertEqual(processing["loudness_mode"], "prompt_calibrated")
            self.assertEqual(
                processing["calibration_gain_db"],
                calibration["calibration_gain_db"],
            )
            output, output_rate = sf.read(output_dir / "line.wav")
            reference_lufs = pyln.Meter(rate).integrated_loudness(reference)
            self.assertAlmostEqual(
                pyln.Meter(output_rate).integrated_loudness(output),
                reference_lufs,
                delta=0.5,
            )
            # One probe segment plus the two real segments were rendered.
            self.assertEqual(len(runtime.calls), 3)
            self.assertEqual(runtime.calls[0]["text"], "参考台词。")

    def test_missing_prompt_audio_falls_back_to_target_loudness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gain, report = _calibrate_prompt_loudness(
                SineRuntime(amplitude=0.2),
                prompt_audio_path=root / "missing.wav",
                prompt_text="参考台词。",
                voice_polish_config=calibrated_polish_config(),
                edge_trim_config=edge_trim_config(),
                base_seed=42,
                pause_ms=0,
                max_chars=120,
                num_steps=16,
                language="chinese",
                template_name="tts",
                normalize_text=False,
                soften_emphasis=False,
                speaker_scale=1.5,
                ode_method="euler",
                guidance_scale=1.2,
                cancelled=None,
            )
            self.assertIsNone(gain)
            self.assertEqual(report["status"], "reference_unavailable")


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
