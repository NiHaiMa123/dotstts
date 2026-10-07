from __future__ import annotations

import unittest

import numpy as np

from dots_tts_lab.grit_diagnostics import (
    band_relative_energy_db,
    compute_grit_metrics,
    voiced_frame_mask,
)


def _tone(rate: float, seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return 0.1 * np.sin(2.0 * np.pi * freq * t)


def _noise(rate: float, seconds: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.05 * rng.standard_normal(int(rate * seconds))


class BandRelativeEnergyTests(unittest.TestCase):
    def test_tone_in_band_has_positive_relative_energy(self) -> None:
        rate = 48_000
        spectra = np.abs(np.fft.rfft(_tone(rate, 1.0, freq=6000.0))) ** 2
        freqs = np.fft.rfftfreq(rate, 1.0 / rate)
        rel = band_relative_energy_db(spectra, freqs, (4000.0, 8000.0))
        self.assertGreater(rel, 0.0)

    def test_tone_below_band_has_negative_relative_energy(self) -> None:
        rate = 48_000
        spectra = np.abs(np.fft.rfft(_tone(rate, 1.0, freq=500.0))) ** 2
        freqs = np.fft.rfftfreq(rate, 1.0 / rate)
        rel = band_relative_energy_db(spectra, freqs, (8000.0, 12000.0))
        self.assertLess(rel, 0.0)


class VoicedFrameSelectionTests(unittest.TestCase):
    def test_mask_is_deterministic(self) -> None:
        rate = 48_000
        audio = _tone(rate, 1.0, freq=150.0)
        first, _ = voiced_frame_mask(audio, rate)
        second, _ = voiced_frame_mask(audio, rate)
        np.testing.assert_array_equal(first, second)

    def test_sustained_tone_marks_frames_voiced(self) -> None:
        rate = 48_000
        voiced, _ = voiced_frame_mask(_tone(rate, 1.5, freq=150.0), rate)
        self.assertGreater(np.count_nonzero(voiced), 0)


class GritMetricTests(unittest.TestCase):
    def test_noise_flatter_than_harmonic_comb_in_air_band(self) -> None:
        rate = 48_000
        n = rate
        comb = np.zeros(n)
        comb[:: rate // 200] = 0.2  # 200 Hz impulse train → harmonic comb
        flat_noise = compute_grit_metrics(_noise(rate, 1.0), rate)
        flat_comb = compute_grit_metrics(comb, rate)
        self.assertGreater(
            flat_noise["flatness_8000_12000_median"],
            flat_comb["flatness_8000_12000_median"],
        )

    def test_warbled_tone_has_higher_temporal_delta_than_steady(self) -> None:
        rate = 48_000
        t = np.arange(rate) / rate
        steady = _tone(rate, 1.0, freq=1000.0)
        warbled = steady * (1.0 + 0.6 * np.sin(2.0 * np.pi * 25.0 * t))
        m_steady = compute_grit_metrics(steady, rate)
        m_warbled = compute_grit_metrics(warbled, rate)
        self.assertGreater(
            m_warbled["temporal_delta_2k_9k_median_db"],
            m_steady["temporal_delta_2k_9k_median_db"],
        )

    def test_metric_keys_present(self) -> None:
        rate = 48_000
        metrics = compute_grit_metrics(
            _tone(rate, 0.5) + _noise(rate, 0.5) * 0.02, rate
        )
        for key in (
            "duration_s", "rms_dbfs", "integrated_lufs", "true_peak_dbtp",
            "band_4000_8000_rel_db", "band_8000_12000_rel_db",
            "band_12000_18000_rel_db",
            "flatness_4000_8000_mean", "flatness_8000_12000_p90",
            "crest_4000_8000_median", "entropy_8000_12000_median",
            "temporal_delta_2k_9k_median_db", "temporal_delta_2k_9k_p90_db",
        ):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
