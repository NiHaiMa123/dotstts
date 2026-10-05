from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from dots_tts_lab.long_form_dfn3 import (
    DEFAULT_DFN3_CONFIG_PATH,
    Dfn3ProductionConfig,
    Dfn3VerificationError,
    compute_run_hash,
    load_dfn3_config,
    verify_backend,
    verify_model_files,
)


def _payload() -> dict:
    return yaml.safe_load(DEFAULT_DFN3_CONFIG_PATH.read_text(encoding="utf-8"))


class Dfn3ConfigTests(unittest.TestCase):
    def test_production_config_loads_and_hashes_stable(self) -> None:
        config = load_dfn3_config()
        self.assertEqual(config.backend.model_name, "DeepFilterNet3")
        self.assertEqual(config.backend.package_version, "0.5.6")
        self.assertFalse(config.backend.post_filter)
        self.assertTrue(config.backend.compensate_delay)
        self.assertEqual(config.backend.attenuation_limit_db, 8.0)
        self.assertEqual(config.chunking.core_max_seconds, 30.0)
        self.assertEqual(config.chunking.context_max_seconds, 2.0)
        self.assertEqual(
            config.config_sha256(), load_dfn3_config().config_sha256()
        )

    def test_post_filter_and_delay_compensation_are_pinned(self) -> None:
        payload = _payload()
        payload["backend"]["post_filter"] = True
        with self.assertRaises(ValidationError):
            Dfn3ProductionConfig.model_validate(payload, strict=True)
        payload = _payload()
        payload["backend"]["compensate_delay"] = False
        with self.assertRaises(ValidationError):
            Dfn3ProductionConfig.model_validate(payload, strict=True)

    def test_chunk_bounds_are_capped(self) -> None:
        payload = _payload()
        payload["chunking"]["core_max_seconds"] = 45.0
        with self.assertRaises(ValidationError):
            Dfn3ProductionConfig.model_validate(payload, strict=True)
        payload = _payload()
        payload["chunking"]["context_max_seconds"] = 5.0
        with self.assertRaises(ValidationError):
            Dfn3ProductionConfig.model_validate(payload, strict=True)

    def test_output_cannot_live_under_inbox(self) -> None:
        payload = _payload()
        payload["output"]["work_root"] = "data/inbox/dfn3"
        with self.assertRaisesRegex(ValidationError, "inbox"):
            Dfn3ProductionConfig.model_validate(payload, strict=True)

    def test_silent_raw_fallback_is_not_expressible(self) -> None:
        payload = _payload()
        payload["failure_policy"]["on_backend_failure"] = "fallback_to_raw"
        with self.assertRaises(ValidationError):
            Dfn3ProductionConfig.model_validate(payload, strict=True)


class Dfn3VerificationTests(unittest.TestCase):
    def test_real_cache_matches_pinned_fingerprints(self) -> None:
        config = load_dfn3_config()
        cache = Path(__file__).resolve().parents[1] / config.model_files.cache_root
        if not (cache / config.model_files.weights_relative_path).is_file():
            self.skipTest("deepfilternet model cache not present")
        report = verify_backend(config)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["files"]["weights"]["status"], "verified")

    def test_hash_mismatch_fails_loudly(self) -> None:
        payload = _payload()
        payload["model_files"]["weights_sha256"] = "0" * 64
        config = Dfn3ProductionConfig.model_validate(payload, strict=True)
        cache = Path(__file__).resolve().parents[1] / config.model_files.cache_root
        if not (cache / config.model_files.weights_relative_path).is_file():
            self.skipTest("deepfilternet model cache not present")
        with self.assertRaisesRegex(Dfn3VerificationError, "mismatch"):
            verify_model_files(config)

    def test_missing_weights_fails_loudly(self) -> None:
        payload = _payload()
        payload["model_files"]["cache_root"] = "data/work/models/nonexistent-cache"
        config = Dfn3ProductionConfig.model_validate(payload, strict=True)
        with self.assertRaisesRegex(Dfn3VerificationError, "missing"):
            verify_model_files(config)

    def test_absolute_cache_root_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _payload()
            payload["model_files"]["cache_root"] = str(Path(tmp).as_posix())
            with self.assertRaises(ValidationError):
                Dfn3ProductionConfig.model_validate(payload, strict=True)

    def test_wrong_package_version_fails(self) -> None:
        config = load_dfn3_config()
        cache = Path(__file__).resolve().parents[1] / config.model_files.cache_root
        if not (cache / config.model_files.weights_relative_path).is_file():
            self.skipTest("deepfilternet model cache not present")
        with self.assertRaisesRegex(Dfn3VerificationError, "pinned"):
            verify_backend(config, package_version="0.6.0")

    def test_run_hash_binds_source_region_and_config(self) -> None:
        config = load_dfn3_config()
        arguments = {
            "source_sha256": "a" * 64,
            "core_start_frame": 0,
            "core_end_frame": 1440000,
            "channel_route": "safe_mean",
            "implementation_version": 1,
        }
        first = compute_run_hash(config, **arguments)
        self.assertEqual(first, compute_run_hash(config, **arguments))
        arguments["core_end_frame"] += 48000
        self.assertNotEqual(first, compute_run_hash(config, **arguments))


if __name__ == "__main__":
    unittest.main()
