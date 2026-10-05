from __future__ import annotations

import unittest

import yaml
from pydantic import ValidationError

from dots_tts_lab.long_form_splits import (
    DEFAULT_SPLIT_CONFIG_PATH,
    SplitConfig,
    SplitRecord,
    check_split_isolation,
    load_split_config,
)


def _record(
    candidate_id: str,
    source_sha256: str,
    start: int,
    end: int,
    text_sha256: str | None = None,
    pair_group: str | None = None,
) -> SplitRecord:
    return SplitRecord(
        candidate_id=candidate_id,
        source_sha256=source_sha256,
        source_start_frame=start,
        source_end_frame=end,
        source_sample_rate_hz=48000,
        text_sha256=text_sha256,
        pair_group=pair_group,
    )


class SplitConfigTests(unittest.TestCase):
    def test_real_split_config_loads_and_covers_all_sources(self) -> None:
        config = load_split_config()
        self.assertEqual(len(config.assignments), 7)
        splits = {a.split for a in config.assignments}
        self.assertEqual(splits, {"reference", "development", "holdout"})
        holdout = [a for a in config.assignments if a.split == "holdout"]
        self.assertEqual(len(holdout), 2)
        reference = [a for a in config.assignments if a.split == "reference"]
        self.assertEqual(len(reference), 1)

    def test_duplicate_source_assignment_fails(self) -> None:
        payload = yaml.safe_load(DEFAULT_SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))
        duplicate = dict(payload["assignments"][0])
        duplicate["split"] = "development"
        payload["assignments"].append(duplicate)
        with self.assertRaisesRegex(ValidationError, "assigned to both"):
            SplitConfig.model_validate(payload, strict=True)

    def test_missing_holdout_fails(self) -> None:
        payload = yaml.safe_load(DEFAULT_SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))
        for assignment in payload["assignments"]:
            if assignment["split"] == "holdout":
                assignment["split"] = "development"
        with self.assertRaisesRegex(ValidationError, "holdout"):
            SplitConfig.model_validate(payload, strict=True)


class SplitIsolationTests(unittest.TestCase):
    def _config(self) -> SplitConfig:
        return SplitConfig.model_validate(
            {
                "schema_version": 1,
                "config_id": "test_split",
                "config_version": 1,
                "assignments": [
                    {
                        "source_sha256": "a" * 64,
                        "source_relative_path": "x/a.mp3",
                        "split": "development",
                        "reason": "test",
                    },
                    {
                        "source_sha256": "b" * 64,
                        "source_relative_path": "x/b.mp3",
                        "split": "holdout",
                        "reason": "test",
                    },
                    {
                        "source_sha256": "c" * 64,
                        "source_relative_path": "x/c.mp3",
                        "split": "reference",
                        "reason": "test",
                    },
                ],
            },
            strict=True,
        )

    def test_non_adjacent_records_pass(self) -> None:
        config = self._config()
        records = [
            _record("c1", "a" * 64, 0, 48000),
            _record("c2", "b" * 64, 0, 48000, text_sha256="1" * 64),
            _record("c3", "b" * 64, 48000 * 100, 48000 * 101, text_sha256="2" * 64),
        ]
        self.assertEqual(check_split_isolation(records, config), [])

    def test_duplicate_text_across_splits_flagged(self) -> None:
        config = self._config()
        records = [
            _record("c1", "a" * 64, 0, 48000, text_sha256="9" * 64),
            _record("c2", "b" * 64, 0, 48000, text_sha256="9" * 64),
        ]
        violations = check_split_isolation(records, config)
        self.assertEqual(len(violations), 1)
        self.assertIn("duplicate_text", violations[0])

    def test_raw_enhanced_pair_across_splits_flagged(self) -> None:
        config = self._config()
        records = [
            _record("raw-1", "a" * 64, 0, 48000, pair_group="pair-1"),
            _record("dfn3-1", "b" * 64, 0, 48000, pair_group="pair-1"),
        ]
        violations = check_split_isolation(records, config)
        self.assertEqual(len(violations), 1)
        self.assertIn("paired_raw_enhanced_group", violations[0])


if __name__ == "__main__":
    unittest.main()
