from __future__ import annotations

import json
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from dots_tts.models.dots_tts.config import LoraArtifactConfig
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.training.checkpoint import (
    TRAINABLE_MODEL_FILENAME,
    TRAINABLE_MODEL_METADATA_FILENAME,
    load_trainable_model_artifact,
    save_trainable_model_artifact,
)
from dots_tts.training.peft import configure_lora_modules, merge_lora_modules
from dots_tts.training.runtime_metrics import write_training_runtime_metrics


class _Velocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.output_layer = nn.Linear(4, 2)


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_field_predictor = _Velocity()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _Core()
        self.other = nn.Linear(4, 4)
        self.config = SimpleNamespace(lora=None)


class _FakePeftConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _fake_inject(_config, module):
    module.q_proj.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
    module.q_proj.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
    return module


class _MergeableLoraLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.merged = False

    def merge(self, *, safe_merge: bool) -> None:
        self.merged = safe_merge
        with torch.no_grad():
            self.base_layer.weight.add_(1.0)

    def get_base_layer(self) -> nn.Module:
        return self.base_layer


class _NoDeviceRuntime(DotsTtsRuntime):
    def __init__(self, model, pretrained_path, **kwargs) -> None:
        self.model = model
        self.pretrained_path = pretrained_path
        self.kwargs = kwargs


class LoraTrainingTests(unittest.TestCase):
    def test_lora_merge_replaces_wrappers_with_merged_base_layers(self) -> None:
        model = _Model()
        wrapper = _MergeableLoraLayer()
        original = wrapper.base_layer.weight.detach().clone()
        model.core.velocity_field_predictor.q_proj = wrapper
        count = merge_lora_modules(model)
        merged = model.core.velocity_field_predictor.q_proj
        self.assertEqual(count, 1)
        self.assertIsInstance(merged, nn.Linear)
        self.assertTrue(torch.allclose(merged.weight, original + 1.0))
        self.assertFalse(any(hasattr(module, "lora_A") for module in model.modules()))

    def test_training_injection_freezes_base_and_keeps_adapter_trainable(self) -> None:
        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = _FakePeftConfig
        fake_peft.inject_adapter_in_model = _fake_inject
        config = LoraArtifactConfig(
            target_modules=["q_proj"],
            rank=2,
            alpha=4,
            dropout=0.05,
            train_output_layer=True,
        )
        model = _Model()
        with patch.dict(sys.modules, {"peft": fake_peft}):
            summary = configure_lora_modules(model, config, for_training=True)

        self.assertGreater(summary["lora_parameters"], 0)
        self.assertFalse(model.other.weight.requires_grad)
        self.assertFalse(model.core.velocity_field_predictor.q_proj.weight.requires_grad)
        self.assertTrue(model.core.velocity_field_predictor.q_proj.lora_A["default"].weight.requires_grad)
        self.assertTrue(model.core.velocity_field_predictor.output_layer.weight.requires_grad)
        self.assertEqual(model.config.lora, config)

    def test_lora_config_is_part_of_model_artifact_contract(self) -> None:
        config = LoraArtifactConfig(
            target_modules=["q_proj", "v_proj"],
            rank=8,
            alpha=16,
        )
        self.assertEqual(config.to_declared_dict()["rank"], 8)
        self.assertEqual(config.to_declared_dict()["target_scope"], "core.velocity_field_predictor")

    def test_adapter_checkpoint_saves_and_restores_only_trainable_delta(self) -> None:
        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = _FakePeftConfig
        fake_peft.inject_adapter_in_model = _fake_inject
        config = LoraArtifactConfig(
            target_modules=["q_proj"],
            rank=2,
            alpha=4,
            train_output_layer=True,
        )
        source = _Model()
        target = _Model()
        with patch.dict(sys.modules, {"peft": fake_peft}):
            configure_lora_modules(source, config, for_training=True)
            configure_lora_modules(target, config, for_training=True)

        with torch.no_grad():
            for parameter in source.parameters():
                if parameter.requires_grad:
                    parameter.fill_(0.25)
            for parameter in target.parameters():
                if parameter.requires_grad:
                    parameter.zero_()

        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            save_trainable_model_artifact(source, model_dir)
            self.assertTrue((model_dir / TRAINABLE_MODEL_FILENAME).is_file())
            self.assertTrue((model_dir / TRAINABLE_MODEL_METADATA_FILENAME).is_file())
            self.assertFalse((model_dir / "model.safetensors").exists())
            metadata = load_trainable_model_artifact(
                target,
                model_dir,
                require_matching_trainable_parameters=True,
            )

        self.assertEqual(metadata["parameter_count"], 26)
        for name, parameter in target.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.all(parameter == 0.25), name)
        self.assertFalse(torch.all(target.other.weight == 0.25))

    def test_runtime_metrics_are_structured_and_zero_step_resume_preserves_them(self) -> None:
        with TemporaryDirectory() as directory:
            write_training_runtime_metrics(
                output_dir=directory,
                status="succeeded",
                start_global_step=2,
                end_global_step=3,
                runtime_started_at=time.perf_counter() - 0.01,
                device=torch.device("cpu"),
            )
            metrics_path = Path(directory) / "training_runtime_metrics.json"
            original = metrics_path.read_text(encoding="utf-8")
            metrics = json.loads(original)

            self.assertEqual(metrics["schema_version"], 1)
            self.assertEqual(metrics["status"], "succeeded")
            self.assertEqual(metrics["start_global_step"], 2)
            self.assertEqual(metrics["end_global_step"], 3)
            self.assertEqual(metrics["completed_optimizer_steps"], 1)
            self.assertGreater(metrics["elapsed_seconds"], 0)
            self.assertEqual(
                metrics["cuda"],
                {
                    "available": False,
                    "device": None,
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                    "total_memory_bytes": None,
                },
            )

            write_training_runtime_metrics(
                output_dir=directory,
                status="succeeded",
                start_global_step=3,
                end_global_step=3,
                runtime_started_at=time.perf_counter(),
                device=torch.device("cpu"),
            )
            self.assertEqual(metrics_path.read_text(encoding="utf-8"), original)

    def test_runtime_loads_compact_delta_on_declared_lora_architecture(self) -> None:
        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = _FakePeftConfig
        fake_peft.inject_adapter_in_model = _fake_inject
        config = LoraArtifactConfig(
            target_modules=["q_proj"],
            rank=2,
            alpha=4,
            train_output_layer=True,
        )
        source = _Model()
        target = _Model()
        with patch.dict(sys.modules, {"peft": fake_peft}):
            configure_lora_modules(source, config, for_training=True)
            with torch.no_grad():
                for parameter in source.parameters():
                    if parameter.requires_grad:
                        parameter.fill_(0.375)
            with TemporaryDirectory() as directory:
                root = Path(directory)
                artifact_dir = root / "artifact"
                save_trainable_model_artifact(source, artifact_dir)
                with patch(
                    "dots_tts.runtime.DotsTtsModel.from_pretrained",
                    return_value=target,
                ):
                    runtime = _NoDeviceRuntime.from_pretrained_with_trainable_delta(
                        str(root),
                        artifact_dir,
                        precision="bfloat16",
                    )

        self.assertIs(runtime.model, target)
        self.assertEqual(runtime.model.config.lora, config)
        self.assertTrue(
            torch.all(
                runtime.model.core.velocity_field_predictor.q_proj.lora_A[
                    "default"
                ].weight
                == 0.375
            )
        )
        self.assertTrue(
            torch.all(
                runtime.model.core.velocity_field_predictor.output_layer.weight
                == 0.375
            )
        )


if __name__ == "__main__":
    unittest.main()
