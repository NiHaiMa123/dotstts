from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

import dots_tts_lab.long_form_pipeline as pipeline
import dots_tts_lab.long_form_refinement as refinement
import test_long_form_pipeline as pipeline_fixture
import test_long_form_review as review_fixture
from dots_tts_lab.long_form_contract import LongFormConfig, load_long_form_config
from dots_tts_lab.long_form_export import _load_inputs, export_long_form_dataset
from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.long_form_render import BoundaryRepairPolicy, render_repaired_spans
from dots_tts_lab.long_form_review import apply_long_form_review, build_long_form_review


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = review_fixture.LongFormReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.batch = self.root / "batch.json"

    def apply(self, batch_id="one", complete=False):
        payload = self.fixture.decisions_payload(batch_id)
        if complete:
            payload["decisions"].append({**payload["decisions"][0],
                                         "segment_id": self.fixture.segment_ids[1],
                                         "confirmed_text": "second confirmed text"})
        write_json(self.batch, payload)
        return apply_long_form_review(self.fixture.manifest, self.batch)

    def test_changed_audio_manifest_cannot_inherit_old_approval(self):
        self.apply(complete=True)
        manifest = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        row = manifest["segments"][1]
        audio = self.root / row["derived_relative_path"]
        audio.write_bytes(b"changed audio")
        row["derived_audio_sha256"] = sha(audio)
        write_json(self.fixture.manifest, manifest)
        before = (self.root / "reviews/decisions.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "another manifest"):
            self.apply("new-partial")
        self.assertEqual(before, (self.root / "reviews/decisions.jsonl").read_bytes())

    def test_unrecorded_audio_drift_is_rejected_before_writing(self):
        (self.root / "segments" / f"{self.fixture.segment_ids[0]}.wav").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "audio hash drift"):
            self.apply()
        self.assertFalse((self.root / "reviews").exists())

    def test_snapshot_decisions_bind_manifest_and_audio_and_are_idempotent(self):
        first = self.apply(complete=True)
        second = apply_long_form_review(self.fixture.manifest, self.batch)
        self.assertTrue(first["complete"])
        self.assertEqual(second["action"], "cached")
        for decision in first["decisions"]:
            self.assertEqual(decision["manifest_sha256"], sha(self.fixture.manifest))
            self.assertEqual(len(decision["derived_audio_sha256"]), 64)
        snapshot = self.root / "reviews/review_snapshot.json"
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload["decisions"][0]["manifest_sha256"] = "0" * 64
        write_json(snapshot, payload)
        with self.assertRaisesRegex(ValueError, "decision binding"):
            _load_inputs(self.fixture.manifest, snapshot)

    def make_legacy(self):
        self.apply(complete=True)
        log = self.root / "reviews/decisions.jsonl"
        fields = {"source_sha256", "config_sha256", "manifest_sha256", "derived_audio_sha256"}
        events = [{key: value for key, value in json.loads(line).items() if key not in fields}
                  for line in log.read_text(encoding="utf-8").splitlines()]
        for event in events:
            event["schema_version"] = 1
        log.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        snapshot = self.root / "reviews/review_snapshot.json"
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload.update(schema_version=1, decisions=events)
        write_json(snapshot, payload)
        return log.read_bytes()

    def test_legacy_matching_snapshot_can_anchor_unchanged_decisions(self):
        before = self.make_legacy()
        result = self.apply("new-partial")
        self.assertTrue(result["complete"])
        self.assertTrue((self.root / "reviews/decisions.jsonl").read_bytes().startswith(before))
        self.assertTrue(all(row["manifest_sha256"] == sha(self.fixture.manifest) for row in result["decisions"]))
        self.assertTrue(self.apply("third-partial")["complete"])

    def test_legacy_drift_or_missing_snapshot_cannot_be_rebound(self):
        self.make_legacy()
        snapshot = self.root / "reviews/review_snapshot.json"
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = "f" * 64
        write_json(snapshot, payload)
        with self.assertRaisesRegex(ValueError, "legacy review snapshot"):
            self.apply("new-partial")
        snapshot.unlink()
        with self.assertRaisesRegex(ValueError, "no binding snapshot"):
            self.apply("new-partial")


class PipelineRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = pipeline_fixture.LongFormPipelineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    @staticmethod
    def asr_payload(rows, fail_ids=()):
        return {"backend_id": "fake", "backend_version": "1", "model_id": "fake",
                "model_revision": "1", "inference_config_sha256": "d" * 64,
                "results": [
                    {"asset_sha256": row["segment_id"],
                     "status": "failed" if row["segment_id"] in fail_ids else "ok",
                     "hypothesis": "valid text",
                     "metadata": {"timestamp_segments": [
                         {"start": 0.1, "end": 2.0, "text": "valid text", "words": [
                             {"start": 0.1, "end": 2.0, "text": "valid text", "probability": 0.9}]}]}}
                    for row in rows]}

    def run_pipeline(self, **kwargs):
        return self.fixture.run_pipeline(skip_asr=False, **kwargs)

    def test_partial_asr_retries_only_failed_rows_without_rematerializing(self):
        calls = []
        def asr(**kwargs):
            rows = kwargs["segment_rows"]
            calls.append([row["segment_id"] for row in rows])
            return self.asr_payload(rows, calls[0][-1:] if len(calls) == 1 else ())
        with patch.object(pipeline, "_run_timestamp_asr", side_effect=asr):
            first = self.run_pipeline()
            self.assertEqual(first["sources"][0]["asr_status"], "partial")
            with patch.object(pipeline, "_materialize_segments", side_effect=AssertionError("must reuse audio")):
                second = self.run_pipeline()
                third = self.run_pipeline()
        self.assertEqual(calls[1], calls[0][-1:])
        self.assertEqual(len(calls), 2)
        self.assertEqual(second["sources"][0]["asr_status"], "succeeded")
        self.assertEqual(third["sources"][0]["cache_action"], "cached")
        manifest_path = Path(second["sources"][0]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        aggregate = json.loads(manifest_path.with_name("asr_timestamp_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(aggregate["results"]), 2)
        self.assertEqual(manifest["asr_failure_count"], 0)
        self.assertTrue(all("asr_failed" not in row["review_reasons"] for row in manifest["segments"]))

    def test_all_failed_and_missing_results_are_not_cached_as_success(self):
        def asr(**kwargs):
            return self.asr_payload([])
        with patch.object(pipeline, "_run_timestamp_asr", side_effect=asr) as runner:
            first = self.run_pipeline()
            second = self.run_pipeline()
        self.assertEqual(first["sources"][0]["asr_status"], "failed")
        self.assertEqual(second["sources"][0]["asr_failure_count"], 2)
        self.assertEqual(runner.call_count, 2)

    def test_backend_crash_resumes_materialized_audio(self):
        with patch.object(pipeline, "_run_timestamp_asr", side_effect=RuntimeError("backend stopped")):
            with self.assertRaisesRegex(RuntimeError, "backend stopped"):
                self.run_pipeline()
        with patch.object(pipeline, "_materialize_segments", side_effect=AssertionError("must reuse audio")), \
             patch.object(pipeline, "_run_timestamp_asr", side_effect=lambda **kw: self.asr_payload(kw["segment_rows"])):
            resumed = self.run_pipeline()
        self.assertEqual(resumed["sources"][0]["cache_action"], "resumed")
        self.assertEqual(resumed["sources"][0]["asr_status"], "succeeded")

    def test_version_change_and_force_preserve_prior_manifests(self):
        first = self.fixture.run_pipeline()
        path = Path(first["sources"][0]["manifest_path"])
        before = path.read_bytes()
        with patch.object(pipeline, "LONG_FORM_PIPELINE_IMPLEMENTATION_VERSION", 99):
            upgraded = self.fixture.run_pipeline()
        forced = self.fixture.run_pipeline(force=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len({first["sources"][0]["manifest_path"], upgraded["sources"][0]["manifest_path"], forced["sources"][0]["manifest_path"]}), 3)

    def test_work_or_report_inside_input_rejected_before_recovery(self):
        sentinel = self.fixture.input / "existing.partial"
        sentinel.write_bytes(b"keep")
        for field in ("work_root", "report_root"):
            with self.subTest(field=field), patch.object(pipeline, "recover_long_form_work_root") as recovery:
                with self.assertRaisesRegex(ValueError, "protected input"):
                    self.fixture.run_pipeline(**{field: self.fixture.input})
                recovery.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_changed_asr_settings_get_new_namespace(self):
        config = load_long_form_config().model_dump(mode="json")
        backend = self.fixture.root / "backend.yaml"
        backend.write_text("settings: one\n", encoding="utf-8")
        config["text"]["timestamp_backend_config"] = str(backend)
        config_path = self.fixture.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        first = self.fixture.run_pipeline(config_path=config_path)
        backend.write_text("settings: two\n", encoding="utf-8")
        second = self.fixture.run_pipeline(config_path=config_path)
        self.assertNotEqual(first["sources"][0]["manifest_path"], second["sources"][0]["manifest_path"])


class OutputPathTests(unittest.TestCase):
    def test_windows_drive_paths_cannot_pass_relative_config_validation(self):
        for invalid in ("C:/inbox/output", "C:relative", "\\\\server\\share\\output"):
            payload = load_long_form_config().model_dump(mode="json")
            payload["paths"]["work_root"] = invalid
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                LongFormConfig.model_validate(payload)

    def test_ancestor_output_and_resolved_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input/source.wav"
            source.parent.mkdir()
            source.write_bytes(b"original")
            for target in (root, root / "input/../input/output"):
                with self.assertRaisesRegex(ValueError, "protected input"):
                    validate_output_path(target, protected_inputs=(source.parent,))

    def test_review_and_export_guard_outputs_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected input"):
                build_long_form_review(manifest, output_path=manifest)
            with self.assertRaisesRegex(ValueError, "protected input"):
                export_long_form_dataset(manifest, root / "snapshot.json", target=root)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "{}")


class BoundaryRepairTests(unittest.TestCase):
    def test_nonzero_edges_and_internal_silence_joins_are_declicked_and_traceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            rate = 48000
            values = np.full((rate * 8, 2), 0.2, dtype=np.float32)
            sf.write(source, values, rate, subtype="FLOAT")
            before = sha(source)
            spans = [{"source_start_frame": rate, "source_end_frame": rate * 3},
                     {"source_start_frame": rate * 5, "source_end_frame": rate * 7}]
            output, actual_rate, trace = render_repaired_spans(
                source, spans, join_silence_seconds=0.08, policy=BoundaryRepairPolicy())
            self.assertEqual(actual_rate, rate)
            self.assertTrue(np.all(output[0] == 0))
            self.assertTrue(np.all(output[-1] == 0))
            self.assertLess(float(np.max(np.abs(np.diff(output[:, 0])))), 0.001)
            self.assertEqual(len(output), rate * 4 + round(0.08 * rate))
            for mapping in trace["mappings"]:
                start, end = mapping["output_start_frame"], mapping["output_end_frame"]
                self.assertEqual(end - start, mapping["source_end_frame"] - mapping["source_start_frame"])
                np.testing.assert_array_equal(output[start + mapping["fade_in_frames"]:end - mapping["fade_out_frames"]], np.float32(0.2))
            self.assertEqual(before, sha(source))

    def test_zero_crossing_search_is_bounded_and_deterministic_at_file_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tone.wav"
            rate = 44100
            wave = (0.3 * np.sin(2 * np.pi * 193 * np.arange(rate * 3) / rate + 0.7)).astype(np.float32)
            sf.write(source, wave, rate, subtype="FLOAT")
            spans = [{"source_start_frame": 0, "source_end_frame": len(wave)}]
            first, _, trace = render_repaired_spans(source, spans, join_silence_seconds=0.08, policy=BoundaryRepairPolicy())
            second, _, other = render_repaired_spans(source, spans, join_silence_seconds=0.08, policy=BoundaryRepairPolicy())
            np.testing.assert_array_equal(first, second)
            self.assertEqual(trace, other)
            mapping = trace["mappings"][0]
            self.assertLessEqual(abs(mapping["source_start_frame"]), round(0.005 * rate))
            self.assertLessEqual(abs(mapping["source_end_frame"] - len(wave)), round(0.005 * rate))
            self.assertEqual(first[0, 0], 0)
            self.assertEqual(first[-1, 0], 0)


class RefinementMemoryTests(unittest.TestCase):
    def test_excluded_candidate_pcm_does_not_accumulate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.zeros(96000), 48000)
            config = load_long_form_config()
            manifest = root / "manifest.json"
            write_json(manifest, {"asr_status": "succeeded", "source_sha256": sha(source),
                                 "source_relative_path": source.name, "config_sha256": config.config_sha256(),
                                 "source_scan": {"sample_rate_hz": 48000}, "segments": [],
                                 "segmentation": {"activity_start_threshold_dbfs": -30.0,
                                                  "activity_continue_threshold_dbfs": -34.0}})
            write_json(root / "asr_timestamp_results.json", {"results": []})
            refs, peak = [], [0]
            def read(*args, **kwargs):
                samples = np.zeros((96000, 2), dtype=np.float32)
                refs.append(weakref.ref(samples))
                peak[0] = max(peak[0], sum(ref() is not None for ref in refs))
                return samples, 48000
            units = [{"unit_id": str(i), "parent": {}, "source_start_frame": i * 96000,
                      "source_spans": [{"source_start_frame": 0, "source_end_frame": 96000}]} for i in range(64)]
            with patch.object(refinement, "build_refinement_units", return_value=units), \
                 patch.object(refinement, "_initial_exclusion_reasons", new=lambda *args: []), \
                 patch.object(refinement, "_read_composite", new=read), \
                 patch.object(refinement, "analyze_segment_samples", new=lambda *args, **kwargs: {"duration_seconds": 2.0}), \
                 patch.object(refinement, "_quality_exclusion_reasons", return_value=["exclude"]):
                result = refinement.refine_long_form_manifest(manifest, source)
            self.assertEqual(result["auto_excluded_count"], 64)
            self.assertLessEqual(peak[0], 1)
            self.assertTrue(all(ref() is None for ref in refs))
            before = Path(result["manifest_path"]).read_bytes()
            with patch.object(refinement, "_read_composite", side_effect=AssertionError("cache must not render")):
                cached = refinement.refine_long_form_manifest(manifest, source)
            self.assertEqual(cached["cache_action"], "cached")
            self.assertEqual(before, Path(result["manifest_path"]).read_bytes())


class RefinementIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.wav"
        self.manifest_path = self.root / "manifest.json"
        rate = 44100
        time = np.arange(rate * 4) / rate
        audio = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        audio[~(((time >= 0.25) & (time < 1.3)) | ((time >= 2.1) & (time < 3.3)))] = 0.0
        sf.write(self.source, audio, rate, subtype="PCM_24")
        parent = {"segment_id": "a" * 64, "source_start_frame": 0, "source_end_frame": rate * 4,
                  "style_suggestion": "normal", "speaker_cluster_is_dominant": True,
                  "speaker_cluster_id": "speaker_0", "speaker_cluster_size": 10,
                  "speaker_center_cosine": 0.95, "style_cluster_id": "style_0", "snr_proxy_db": 30.0}
        manifest = {"asr_status": "succeeded", "source_sha256": sha(self.source),
                    "source_relative_path": self.source.name, "config_sha256": load_long_form_config().config_sha256(),
                    "source_scan": {"sample_rate_hz": rate, "channels": 1}, "segments": [parent],
                    "segmentation": {"activity_start_threshold_dbfs": -30.0, "activity_continue_threshold_dbfs": -34.0}}
        write_json(self.manifest_path, manifest)
        self.asr_path = self.root / "asr_timestamp_results.json"
        write_json(self.asr_path, {"results": [{"asset_sha256": "a" * 64, "status": "ok", "metadata": {
            "timestamp_segments": [
                {"start": start, "end": end, "text": text,
                 "words": [{"start": start, "end": end, "text": text, "probability": 0.95}]}
                for start, end, text in ((0.25, 1.3, "这是一段"), (2.1, 3.3, "测试语音"))]}}]})

    def refine(self, **kwargs):
        return refinement.refine_long_form_manifest(self.manifest_path, self.source, **kwargs)

    def test_real_pcm_refinement_review_and_export_preserve_boundary_trace(self):
        original_hash = sha(self.source)
        result = self.refine()
        self.assertEqual(result["shortlist_count"], 1)
        manifest_path = Path(result["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = manifest["segments"][0]
        self.assertEqual(len(row["source_spans"]), 2)
        self.assertGreaterEqual(row["duration_seconds"], 2.0)
        rendered, rate = sf.read(manifest_path.parent / row["derived_relative_path"])
        self.assertEqual(rate, 48000)
        self.assertEqual((rendered[0], rendered[-1]), (0, 0))
        self.assertEqual(row["boundary_repair"]["frames_before_resample"],
                         sum(span["source_end_frame"] - span["source_start_frame"] for span in row["source_spans"])
                         + row["boundary_repair"]["join_silence_frames"])
        batch_path = self.root / "decisions.json"
        write_json(batch_path, {"review_batch_id": "synthetic", "source_sha256": manifest["source_sha256"],
                               "config_sha256": manifest["config_sha256"], "manifest_sha256": sha(manifest_path),
                               "reference_segment_id": row["segment_id"], "decisions": [{
                                   "segment_id": row["segment_id"], "decision": "keep", "confirmed_style": "normal",
                                   "confirmed_text": "这是一段测试语音"}]})
        apply_long_form_review(manifest_path, batch_path)
        exported = export_long_form_dataset(manifest_path, manifest_path.parent / "reviews/review_snapshot.json", target=self.root / "export")
        self.assertEqual(exported["item_count"], 1)
        exported_manifest = json.loads((self.root / "export/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(exported_manifest["rows"][0]["boundary_repair"], row["boundary_repair"])
        self.assertEqual(sha(self.source), original_hash)
        cached = self.refine()
        self.assertEqual(cached["cache_action"], "cached")
        self.assertEqual(cached["manifest_path"], str(manifest_path))

    def test_changed_asr_creates_new_refinement_and_explicit_old_target_is_refused(self):
        first = self.refine()
        first_path = Path(first["manifest_path"])
        before = first_path.read_bytes()
        asr = json.loads(self.asr_path.read_text(encoding="utf-8"))
        asr["results"][0]["metadata"]["timestamp_segments"][0]["text"] = "新的文本"
        write_json(self.asr_path, asr)
        second = self.refine()
        self.assertNotEqual(first["manifest_path"], second["manifest_path"])
        with self.assertRaisesRegex(RuntimeError, "other inputs"):
            self.refine(output_root=first_path.parent)
        self.assertEqual(first_path.read_bytes(), before)

    def test_output_root_cannot_replace_source_or_parent_manifest(self):
        before = self.manifest_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "protected input"):
            self.refine(output_root=self.root)
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_new_duration_contract_is_enforced_even_if_review_says_keep(self):
        from dots_tts_lab.long_form_export import _selected_rows, load_long_form_export_config
        manifest = {"segments": [{"segment_id": "one", "duration_seconds": 1.8}],
                    "refinement": {"minimum_output_seconds": 2.0, "maximum_output_seconds": 15.0}}
        snapshot = {"decisions": [{"segment_id": "one", "decision": "keep", "confirmed_style": "normal", "confirmed_text": "text"}]}
        with self.assertRaisesRegex(RuntimeError, "duration contract"):
            _selected_rows(manifest, snapshot, config=load_long_form_export_config())
