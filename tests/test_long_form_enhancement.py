from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dots_tts_lab.long_form_enhancement import (
    discover_candidates,
    load_denoise_ab_config,
    select_ab_cases,
)


class LongFormEnhancementTests(unittest.TestCase):
    def test_default_config_is_conservative_and_consistent(self) -> None:
        config = load_denoise_ab_config()
        self.assertEqual(config.enhancement["model_name"], "DeepFilterNet3")
        self.assertLessEqual(float(config.enhancement["attenuation_limit_db"]), 8.0)
        self.assertFalse(config.enhancement["post_filter"])
        self.assertEqual(
            config.selection["total_cases"],
            config.selection["high_noise_floor_cases"]
            + config.selection["medium_noise_floor_cases"]
            + config.selection["clean_control_cases"],
        )

    def test_backend_version_is_pinned_for_api_compatibility(self) -> None:
        config = load_denoise_ab_config()
        self.assertEqual(config.enhancement["package_version"], "0.5.6")

    def test_selection_is_deterministic_and_source_balanced(self) -> None:
        config = load_denoise_ab_config()
        rows = []
        for source in range(6):
            for index in range(12):
                if index < 4:
                    floor = -40.0 - index * 0.1
                elif index < 9:
                    floor = -45.0 - index * 0.1
                else:
                    floor = -55.0 - index * 0.1
                rows.append(
                    {
                        "segment_id": f"{source:02d}{index:02d}",
                        "source_sha256": f"source-{source}",
                        "noise_floor_proxy_dbfs": floor,
                    }
                )
        first = select_ab_cases(rows, config)
        second = select_ab_cases(rows, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertGreaterEqual(len({row["source_sha256"] for row in first}), 6)
        self.assertEqual(len({row["segment_id"] for row in first}), 20)

    def test_discovery_skips_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            refinement = root / "aa" / ("a" * 64) / "refinements" / "cfg"
            (refinement / "segments").mkdir(parents=True)
            audio = refinement / "segments/a.wav"
            audio.write_bytes(b"audio")
            manifest = refinement / "manifest.json"
            payload = {
                "source_sha256": "a" * 64,
                "source_relative_path": "source.wav",
                "segmentation": {
                    "activity_start_threshold_dbfs": -45.0,
                    "activity_continue_threshold_dbfs": -49.0,
                },
                "segments": [
                    {
                        "segment_id": "b" * 64,
                        "derived_relative_path": "segments/a.wav",
                        "noise_floor_proxy_dbfs": -40.0,
                    }
                ],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            pending = discover_candidates(root)
            self.assertEqual(len(pending), 1)
            reviews = refinement / "reviews"
            reviews.mkdir()
            import hashlib

            snapshot = {
                "complete": True,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
            (reviews / "review_snapshot.json").write_text(
                json.dumps(snapshot), encoding="utf-8"
            )
            self.assertEqual(discover_candidates(root), [])


if __name__ == "__main__":
    unittest.main()
