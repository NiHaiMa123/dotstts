from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from dots_tts_lab.long_form_contract import (
    DEFAULT_LONG_FORM_CONFIG_PATH,
    LongFormConfig,
    LongFormSegmentRecord,
    load_long_form_config,
    segment_identity,
)


class LongFormContractTests(unittest.TestCase):
    def test_default_config_is_fail_closed_and_hash_is_stable(self) -> None:
        first = load_long_form_config()
        second = load_long_form_config(DEFAULT_LONG_FORM_CONFIG_PATH)
        self.assertEqual(first.config_sha256(), second.config_sha256())
        self.assertEqual(first.style.accepted_styles, ["normal"])
        self.assertTrue(first.speaker.require_reference_for_auto_keep)
        self.assertTrue(first.text.asr_is_candidate_only)
        self.assertLessEqual(first.segmentation.hard_maximum_seconds, 15.0)
        self.assertEqual(first.decode.max_parallel_sources, 1)

    def test_unknown_config_field_fails(self) -> None:
        payload = yaml.safe_load(
            DEFAULT_LONG_FORM_CONFIG_PATH.read_text(encoding="utf-8")
        )
        payload["unsafe_auto_accept"] = True
        with self.assertRaises(ValidationError):
            LongFormConfig.model_validate(payload, strict=True)

    def test_invalid_duration_order_and_output_under_inbox_fail(self) -> None:
        payload = yaml.safe_load(
            DEFAULT_LONG_FORM_CONFIG_PATH.read_text(encoding="utf-8")
        )
        payload["segmentation"]["minimum_seconds"] = 13.0
        with self.assertRaisesRegex(ValidationError, "minimum <= preferred"):
            LongFormConfig.model_validate(payload, strict=True)

        payload = yaml.safe_load(
            DEFAULT_LONG_FORM_CONFIG_PATH.read_text(encoding="utf-8")
        )
        payload["paths"]["work_root"] = "data/inbox/generated"
        with self.assertRaisesRegex(ValidationError, "cannot be written under"):
            LongFormConfig.model_validate(payload, strict=True)

    def test_segment_identity_binds_source_boundaries_and_config(self) -> None:
        config_hash = load_long_form_config().config_sha256()
        arguments = {
            "source_sha256": "a" * 64,
            "source_start_frame": 48000,
            "source_end_frame": 144000,
            "config_sha256": config_hash,
            "implementation_version": 1,
        }
        first = segment_identity(**arguments)
        self.assertEqual(first, segment_identity(**arguments))
        arguments["source_end_frame"] += 1
        self.assertNotEqual(first, segment_identity(**arguments))

    def test_approved_segment_requires_confirmed_text(self) -> None:
        common = {
            "schema_version": 1,
            "segment_id": "a" * 64,
            "source_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "source_start_frame": 0,
            "source_end_frame": 96000,
            "source_sample_rate_hz": 48000,
            "duration_seconds": 2.0,
            "style": "normal",
            "review_reasons": [],
        }
        with self.assertRaisesRegex(ValidationError, "human_confirmed_text"):
            LongFormSegmentRecord(status="approved", **common)
        record = LongFormSegmentRecord(
            status="approved", human_confirmed_text="测试文本", **common
        )
        self.assertEqual(record.status, "approved")

    def test_config_loader_rejects_non_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            path.write_text("- invalid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a YAML mapping"):
                load_long_form_config(path)

