from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dots_tts_lab.fuxuan_webui import (
    DEFAULT_MODEL_ID,
    INDEX_HTML,
    FuxuanWebApplication,
    StateStore,
    cleanup_stale_artifacts,
    recover_and_start_session,
)
from dots_tts_lab.voice_registry import LoadedVoiceRegistry, VoiceProfile, load_voice_registry


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "audio": np.full(4_800, 0.1, dtype=np.float32),
            "sample_rate": 48_000,
        }


class WebUIStateTests(unittest.TestCase):
    def test_recovery_records_interruption_and_only_removes_temporary_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_dir = root / "state"
            output_dir = root / "output"
            state_dir.mkdir()
            output_dir.mkdir()
            work_file = state_dir / "work/chunk.wav"
            work_file.parent.mkdir()
            work_file.write_bytes(b"temporary")
            partial = output_dir / ".sample.partial.wav"
            partial.write_bytes(b"partial")
            final = output_dir / "sample.wav"
            final.write_bytes(b"final")

            store = StateStore(state_dir / "state.json", state_dir / "history.jsonl")
            store.replace(
                {
                    "schema_version": 1,
                    "session_id": "old-session",
                    "job_id": "old-job",
                    "status": "generating",
                    "message": "running",
                    "progress": 42.0,
                }
            )

            state = recover_and_start_session(
                store,
                state_dir=state_dir,
                output_dir=output_dir,
            )

            self.assertEqual(state["status"], "idle")
            self.assertEqual(state["recovery"]["previous_status"], "generating")
            self.assertEqual(state["recovery"]["previous_job_id"], "old-job")
            self.assertEqual(state["recovery"]["cleaned_artifact_count"], 2)
            self.assertFalse(work_file.exists())
            self.assertFalse(partial.exists())
            self.assertEqual(final.read_bytes(), b"final")
            history = [
                json.loads(line)
                for line in (state_dir / "history.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertIn("interrupted", [entry["status"] for entry in history])

    def test_invalid_state_transition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = StateStore(root / "state.json", root / "history.jsonl")
            store.replace({"status": "idle", "message": "ready"})
            with self.assertRaisesRegex(RuntimeError, "idle -> completed"):
                store.transition("completed", "invalid")

    def test_cleanup_does_not_remove_completed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_dir = root / "state"
            output_dir = root / "output"
            state_dir.mkdir()
            output_dir.mkdir()
            completed = output_dir / "keep.wav"
            completed.write_bytes(b"keep")

            removed = cleanup_stale_artifacts(state_dir, output_dir)

            self.assertEqual(removed, [])
            self.assertTrue(completed.exists())
            self.assertTrue((state_dir / "work").is_dir())


class WebUIPageTests(unittest.TestCase):
    def test_page_keeps_the_requested_primary_controls(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")
        self.assertEqual(html.count("<select"), 1)
        self.assertEqual(html.count('id="generate"'), 1)
        self.assertIn("state.models", html)
        self.assertIn("profile_sha256", html)
        self.assertIn('id="open-output"', html)
        self.assertIn("/api/heartbeat", html)
        self.assertIn("/api/close", html)


class WebUIGenerationTests(unittest.TestCase):
    def test_worker_reaches_completed_through_recorded_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            state_dir = root / "state"
            input_dir.mkdir()
            (input_dir / "demo.txt").write_text("测试一句。", encoding="utf-8")
            store = StateStore(state_dir / "state.json", state_dir / "history.jsonl")
            recover_and_start_session(
                store,
                state_dir=state_dir,
                output_dir=output_dir,
            )
            app = FuxuanWebApplication(
                store,
                input_dir=input_dir,
                output_dir=output_dir,
            )
            fake_runtime = FakeRuntime()
            app._runtime = fake_runtime

            accepted, _ = app.start_generation(DEFAULT_MODEL_ID)
            self.assertTrue(accepted)
            assert app._worker is not None
            app._worker.join(timeout=10.0)

            self.assertFalse(app._worker.is_alive())
            state = app.public_status()
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["progress"], 100.0)
            self.assertEqual(state["outputs"], ["demo.wav"])
            self.assertTrue(state["can_open_output"])
            self.assertTrue((output_dir / "demo.wav").is_file())
            profile = load_voice_registry().get(DEFAULT_MODEL_ID)
            self.assertEqual(fake_runtime.calls[0]["prompt_text"], profile.prompt.text)
            self.assertEqual(
                fake_runtime.calls[0]["num_steps"], profile.generation.num_steps
            )
            self.assertEqual(
                fake_runtime.calls[0]["speaker_scale"],
                profile.generation.speaker_scale,
            )
            self.assertEqual(
                fake_runtime.calls[0]["guidance_scale"],
                profile.generation.guidance_scale,
            )
            self.assertEqual(
                fake_runtime.calls[0]["template_name"],
                profile.generation.template_name,
            )
            self.assertEqual(
                fake_runtime.calls[0]["normalize_text"],
                profile.generation.normalize_text,
            )
            statuses = [
                json.loads(line)["status"]
                for line in (state_dir / "history.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for required in (
                "preparing",
                "loading_model",
                "generating",
                "finalizing",
                "completed",
            ):
                self.assertIn(required, statuses)

    def test_switching_model_releases_old_runtime_and_loads_selected_profile(self) -> None:
        registry = load_voice_registry()
        first = registry.get(registry.default_model_id)
        payload = first.model_dump(mode="python")
        payload["model_id"] = "second_voice_v1"
        payload["display_name"] = "Second voice"
        payload["base_model"]["path"] = "pretrained_models/second-base"
        payload["adapter"]["path"] = "models/second-adapter"
        second = VoiceProfile.model_validate(payload, strict=True)
        multi = LoadedVoiceRegistry(
            root=registry.root,
            spec=registry.spec,
            profiles={first.model_id: first, second.model_id: second},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = StateStore(root / "state.json", root / "history.jsonl")
            store.replace({"status": "idle", "message": "ready"})
            app = FuxuanWebApplication(store, registry=multi)
            old_runtime = object()
            new_runtime = object()
            app._runtime = old_runtime
            app._runtime_model_id = first.model_id

            with (
                mock.patch(
                    "dots_tts_lab.fuxuan_webui.validate_voice_profile_artifacts"
                ) as validate,
                mock.patch(
                    "dots_tts_lab.fuxuan_webui.load_runtime",
                    return_value=new_runtime,
                ) as loader,
            ):
                loaded = app._load_selected_runtime(second.model_id)

            self.assertIs(loaded, new_runtime)
            self.assertEqual(app._runtime_model_id, second.model_id)
            validate.assert_called_once_with(multi, second)
            self.assertEqual(
                loader.call_args.kwargs["base_model"],
                (registry.root / "pretrained_models/second-base").resolve(),
            )
            self.assertEqual(
                loader.call_args.kwargs["adapter"],
                (registry.root / "models/second-adapter").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
