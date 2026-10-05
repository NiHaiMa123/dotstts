from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dots_tts_lab.long_form_identity import (
    ReferenceVectors,
    build_speaker_timeline,
    embed_region_windows,
    identify_source,
    load_identity_config,
    score_region_identity,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "work" / "tmp_identity_tests"
SR = 48000


class FreqFakeEncoder:
    """Deterministic stand-in: maps dominant frequency to a unit vector so
    different 'voices' (tone frequencies) get near-orthogonal embeddings."""

    def __call__(self, audio, audio_lengths=None) -> torch.Tensor:
        x = audio.reshape(-1).detach().cpu().numpy().astype(np.float64)
        spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        freq = float(np.argmax(spectrum)) * SR / x.size
        v = np.cos(
            2 * np.pi * np.arange(512) * (round(freq / 20.0) / 512.0)
        )
        return torch.from_numpy(v.reshape(1, 512).astype(np.float32))


def _ref_vector_for(freq_hz: float) -> np.ndarray:
    v = np.cos(
        2 * np.pi * np.arange(512) * (round(freq_hz / 20.0) / 512.0)
    ).astype(np.float64)
    return v / np.linalg.norm(v)


def _tone(seconds: float, freq: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _config():
    base = load_identity_config()
    paths = base.paths.model_copy(
        update={
            "work_root": "data/work/tmp_identity_tests/work",
            "report_root": "data/work/tmp_identity_tests/reports",
        }
    )
    return base.model_copy(update={"paths": paths})


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, SR, subtype="PCM_16")


class IdentityConfigTests(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = load_identity_config()
        self.assertEqual(config.config_id, "long_form_identity_v1")
        self.assertEqual(len(config.config_sha256()), 64)
        self.assertEqual(config.encoder.model_family, "campplus")


class TimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _config()
        self.encoder = FreqFakeEncoder()

    def _windows(self, samples: np.ndarray) -> list:
        return embed_region_windows(
            samples,
            region_start_frame=0,
            sample_rate=SR,
            config=self.config,
            encoder=self.encoder,
        )

    def test_single_speaker_region_yields_one_turn(self) -> None:
        windows = self._windows(_tone(6.0, 220.0))
        turns, protos, diag = build_speaker_timeline(
            windows, config=self.config, prototypes=[]
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].speaker_label, "speaker_000")
        self.assertEqual(diag["voiced_window_count"], len(windows))

    def test_speaker_change_inside_region_detected(self) -> None:
        audio = np.concatenate([_tone(3.0, 220.0), _tone(3.0, 1400.0)])
        windows = self._windows(audio)
        turns, protos, _ = build_speaker_timeline(
            windows, config=self.config, prototypes=[]
        )
        labels = {t.speaker_label for t in turns}
        self.assertEqual(len(labels), 2)

    def test_prototype_labels_persist_across_regions(self) -> None:
        protos: list[np.ndarray] = []
        w1 = self._windows(_tone(4.0, 220.0))
        turns1, protos, _ = build_speaker_timeline(
            w1, config=self.config, prototypes=protos
        )
        w2 = self._windows(_tone(4.0, 220.0))
        turns2, protos, _ = build_speaker_timeline(
            w2, config=self.config, prototypes=protos
        )
        self.assertEqual(turns1[0].speaker_label, turns2[0].speaker_label)

    def test_low_energy_windows_not_embedded(self) -> None:
        audio = np.concatenate(
            [_tone(3.0, 220.0), np.zeros(3 * SR, dtype=np.float32)]
        )
        windows = self._windows(audio)
        self.assertTrue(any(w.vector is None for w in windows))
        self.assertTrue(any(w.vector is not None for w in windows))


class ScoreRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _config()
        self.encoder = FreqFakeEncoder()
        self.references = ReferenceVectors(
            normal=np.stack([_ref_vector_for(220.0)]),
            negatives_by_kind={
                "same_actor_roleplay": np.stack([_ref_vector_for(1400.0)])
            },
            pack_sha256="f" * 64,
        )

    def test_matching_reference_scores_high(self) -> None:
        windows = embed_region_windows(
            _tone(6.0, 220.0),
            region_start_frame=0,
            sample_rate=SR,
            config=self.config,
            encoder=self.encoder,
        )
        turns, _, _ = build_speaker_timeline(
            windows, config=self.config, prototypes=[]
        )
        record = score_region_identity(
            windows, turns, self.references, config=self.config
        )
        self.assertEqual(record["status"], "single_speaker_candidate")
        self.assertGreater(record["target_cosine_min"], 0.95)
        self.assertLess(
            record["negative_cosine_max_by_kind"]["same_actor_roleplay"], 0.5
        )

    def test_non_target_speaker_scores_low(self) -> None:
        windows = embed_region_windows(
            _tone(6.0, 1400.0),
            region_start_frame=0,
            sample_rate=SR,
            config=self.config,
            encoder=self.encoder,
        )
        turns, _, _ = build_speaker_timeline(
            windows, config=self.config, prototypes=[]
        )
        record = score_region_identity(
            windows, turns, self.references, config=self.config
        )
        self.assertLess(record["target_cosine_median"], 0.5)
        self.assertGreater(
            record["negative_cosine_max_by_kind"]["same_actor_roleplay"], 0.9
        )

    def test_silent_region_reports_insufficient_speech(self) -> None:
        windows = embed_region_windows(
            np.zeros(4 * SR, dtype=np.float32),
            region_start_frame=0,
            sample_rate=SR,
            config=self.config,
            encoder=self.encoder,
        )
        record = score_region_identity(
            windows, [], self.references, config=self.config
        )
        self.assertEqual(record["status"], "insufficient_speech")

    def test_speaker_change_region_flagged(self) -> None:
        audio = np.concatenate([_tone(3.0, 220.0), _tone(3.0, 1400.0)])
        windows = embed_region_windows(
            audio,
            region_start_frame=0,
            sample_rate=SR,
            config=self.config,
            encoder=self.encoder,
        )
        turns, _, _ = build_speaker_timeline(
            windows, config=self.config, prototypes=[]
        )
        record = score_region_identity(
            windows, turns, self.references, config=self.config
        )
        self.assertEqual(record["status"], "speaker_change_suspect")
        self.assertTrue(record["speaker_change_suspect"])


class IdentifySourceTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP.mkdir(parents=True, exist_ok=True)
        self.config = _config()
        self.encoder = FreqFakeEncoder()
        self.references = ReferenceVectors(
            normal=np.stack([_ref_vector_for(220.0)]),
            negatives_by_kind={},
            pack_sha256="f" * 64,
        )
        self.source = TMP / "two_speakers.wav"
        audio = np.concatenate([_tone(4.0, 220.0), _tone(4.0, 1400.0)])
        _write_wav(self.source, np.stack([audio, audio], axis=1))

    def tearDown(self) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def _regions(self) -> list[dict]:
        return [
            {
                "region_index": 0,
                "source_start_frame": 0,
                "source_end_frame": 4 * SR,
                "status": "usable",
            },
            {
                "region_index": 1,
                "source_start_frame": 4 * SR,
                "source_end_frame": 8 * SR,
                "status": "usable",
            },
        ]

    def test_end_to_end_and_resume(self) -> None:
        result = identify_source(
            self.source,
            self._regions(),
            config=self.config,
            encoder=self.encoder,
            references=self.references,
        )
        self.assertEqual(result["status"], "completed")
        manifest = json.loads(
            Path(result["identity_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["regions"]), 2)
        self.assertEqual(
            manifest["regions"][0]["status"], "single_speaker_candidate"
        )
        second = identify_source(
            self.source,
            self._regions(),
            config=self.config,
            encoder=self.encoder,
            references=self.references,
        )
        self.assertEqual(second["status"], "cached")


if __name__ == "__main__":
    unittest.main()
