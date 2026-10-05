from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dots_tts_lab.voice_registry import (
    DEFAULT_VOICE_REGISTRY_PATH,
    load_voice_registry,
)


class VoiceRegistryTests(unittest.TestCase):
    def test_current_registry_loads_and_validates_every_bound_artifact(self) -> None:
        registry = load_voice_registry(DEFAULT_VOICE_REGISTRY_PATH)
        profile = registry.get(registry.default_model_id)

        self.assertEqual(profile.model_id, "fuxuan_step400_v1")
        self.assertEqual(profile.adapter.training_step, 400)
        self.assertEqual(profile.generation.num_steps, 16)
        self.assertIn("roughness_control_v2", profile.postprocess.voice_polish_config)
        self.assertEqual(len(profile.canonical_sha256()), 64)

    def test_registry_rejects_a_profile_path_outside_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry.yaml"
            registry.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "registry_id: test_registry",
                        "registry_version: 1",
                        "default_model_id: demo_model",
                        "profiles:",
                        "  - ../outside.yaml",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_voice_registry(
                    registry,
                    project_root=root,
                    validate_artifacts=False,
                )


if __name__ == "__main__":
    unittest.main()
