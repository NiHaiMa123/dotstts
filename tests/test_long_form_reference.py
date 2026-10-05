from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from dots_tts_lab.long_form_reference import (
    NegativeExample,
    ReferenceClip,
    ReferencePack,
    propose_reference_clips,
)


def _clip(index: int, duration: float = 5.0) -> dict:
    frames = int(duration * 48000)
    return {
        "clip_id": f"{index:064x}",
        "audio_relative_path": f"audio/{index:064x}.wav",
        "audio_sha256": f"{index + 100:064x}",
        "source_sha256": "a" * 64,
        "source_sample_rate_hz": 48000,
        "source_spans": [
            {"source_start_frame": index * 4800000, "source_end_frame": index * 4800000 + frames}
        ],
        "duration_seconds": duration,
        "text_sha256": f"{index + 200:064x}",
        "confirmed_review_batch_id": "batch-1",
        "confirmed_at": "2026-09-13T00:00:00Z",
    }


def _negative(kind: str, index: int) -> dict:
    payload = _clip(index)
    payload.pop("confirmed_review_batch_id")
    payload.pop("confirmed_at")
    payload["kind"] = kind
    payload["note"] = "fixture"
    return payload


def _pack_payload(status: str = "confirmed") -> dict:
    return {
        "schema_version": 1,
        "pack_id": "long_form_reference_suisui",
        "pack_version": 1,
        "voice_id": "suisui",
        "status": status,
        "clips": [_clip(i) for i in range(6)],
        "negatives": [
            _negative("same_actor_roleplay", 50),
            _negative("whispered_text", 51),
            _negative("nearfield_binaural", 52),
            _negative("event_over_speech", 53),
        ],
        "created_from_review_batches": ["batch-1"],
        "created_at": "2026-09-13T00:00:00Z",
        "labeling_spec": "docs/long-form-labeling-v1.md",
    }


class ReferencePackTests(unittest.TestCase):
    def test_confirmed_pack_validates_and_hashes_stable(self) -> None:
        pack = ReferencePack.model_validate(_pack_payload(), strict=True)
        self.assertEqual(pack.status, "confirmed")
        self.assertEqual(
            pack.pack_sha256(),
            ReferencePack.model_validate(_pack_payload(), strict=True).pack_sha256(),
        )

    def test_confirmed_pack_requires_hard_negative_coverage(self) -> None:
        payload = _pack_payload()
        payload["negatives"] = payload["negatives"][:2]
        with self.assertRaisesRegex(ValidationError, "hard negatives"):
            ReferencePack.model_validate(payload, strict=True)

    def test_draft_pack_allows_missing_negatives(self) -> None:
        payload = _pack_payload(status="draft_pending_confirmation")
        payload["negatives"] = []
        pack = ReferencePack.model_validate(payload, strict=True)
        self.assertEqual(pack.status, "draft_pending_confirmation")

    def test_clip_count_bounds(self) -> None:
        payload = _pack_payload()
        payload["clips"] = payload["clips"][:5]
        with self.assertRaises(ValidationError):
            ReferencePack.model_validate(payload, strict=True)
        payload = _pack_payload()
        payload["clips"] = [_clip(i) for i in range(13)]
        with self.assertRaises(ValidationError):
            ReferencePack.model_validate(payload, strict=True)

    def test_clip_duration_contract(self) -> None:
        payload = _pack_payload()
        payload["clips"][0]["duration_seconds"] = 2.0
        payload["clips"][0]["source_spans"] = [
            {"source_start_frame": 0, "source_end_frame": 96000}
        ]
        with self.assertRaisesRegex(ValidationError, "3-10 second"):
            ReferencePack.model_validate(payload, strict=True)


class ProposeReferenceClipsTests(unittest.TestCase):
    def _manifest(self, durations: list[float], tmp: Path) -> Path:
        rows = []
        for i, duration in enumerate(durations):
            rows.append(
                {
                    "audio_relative_path": f"audio/{i}.wav",
                    "audio_sha256": f"{i:064x}",
                    "source_sha256": "a" * 64,
                    "source_sample_rate_hz": 48000,
                    "source_start_frame": i * 48000 * 60,
                    "source_end_frame": i * 48000 * 60 + int(duration * 48000),
                    "source_spans": [
                        {
                            "source_start_frame": i * 48000 * 60,
                            "source_end_frame": i * 48000 * 60 + int(duration * 48000),
                        }
                    ],
                    "duration_seconds": duration,
                    "text_sha256": f"{i + 500:064x}",
                    "style": "normal",
                    "review_batch_id": "batch-1",
                }
            )
        path = tmp / "manifest.json"
        path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return path

    def test_proposal_filters_duration_and_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest([1.5, 4.0, 6.0, 11.0], Path(tmp))
            proposal = propose_reference_clips(manifest)
            durations = sorted(row["duration_seconds"] for row in proposal)
            self.assertEqual(durations, [4.0, 6.0])

    def test_proposal_caps_count_and_binds_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest([4.0 + 0.1 * i for i in range(20)], Path(tmp))
            proposal = propose_reference_clips(manifest, max_clips=12)
            self.assertLessEqual(len(proposal), 12)
            for row in proposal:
                self.assertRegex(row["audio_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(row["source_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
