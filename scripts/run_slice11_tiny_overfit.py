#!/usr/bin/env python3
"""Run the Slice11 parameter-efficient tiny overfit smoke test.

This intentionally trains only the DiT output layer.  It validates the real
training batch/forward/loss path and checkpoint round-trip without claiming a
full base/SOAR fine-tune on a 16 GB card.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dots_tts.config.data import DataConfig  # noqa: E402
from dots_tts.data.collator import PadCollator  # noqa: E402
from dots_tts.data.pipelines.tts_pipeline import BasicTtsPipeline  # noqa: E402
from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: E402
from dots_tts.training import losses as loss_ops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lab/slice11/tiny_overfit_v1.yaml")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def freeze_for_smoke(model: DotsTtsModel) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output_layer = model.core.velocity_field_predictor.output_layer
    for parameter in output_layer.parameters():
        parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters after smoke freeze policy.")
    return trainable


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 8 <= len(rows) <= 32:
        raise ValueError(f"Tiny manifest must contain 8..32 rows, got {len(rows)}")
    return rows


def make_batches(model: DotsTtsModel, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data_cfg = DataConfig.model_validate(
        {
            "sources": [
                {
                    "name": "slice11_tiny",
                    "adapter": {"params": {"manifest_path": "unused.jsonl"}},
                }
            ],
            "train_audio_sample_rate": int(model.config.vocoder.sample_rate),
            "audio_samples_per_llm_token": int(model.hop_size) * int(model.config.patch_size),
            "num_tokens_per_epoch": 1,
            "num_workers": 0,
            "pin_memory": False,
            "max_audio_seconds_in_batch": 10.0,
            "max_text_tokens_in_batch": 1024,
            "max_samples_per_batch": 1,
            "bucketing_pool_size": 8,
        }
    )
    pipeline = BasicTtsPipeline(model.tokenizer, data_cfg)
    collator = PadCollator(model.tokenizer)
    processed = [pipeline.process_sample(row) for row in rows]
    return [collator([row]) for row in processed]


def scalar_loss(loss_terms: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    numerators, denominators = loss_ops.collapse_loss_terms(loss_terms)
    # The tiny smoke has no AppConfig object; all three terms use unit weight.
    loss = torch.zeros((), device=next(iter(loss_terms.values())).loss.device)
    metrics: dict[str, float] = {}
    for name, numerator in numerators.items():
        denominator = denominators[name]
        value = numerator / denominator.clamp_min(1.0)
        loss = loss + value
        metrics[name] = float(value.detach().float().item())
    metrics["loss"] = float(loss.detach().float().item())
    return loss, metrics


def snapshot_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def save_adapter_checkpoint(path: Path, model: DotsTtsModel, optimizer: torch.optim.Optimizer, step: int, seed: int) -> None:
    state = {
        "schema_version": 1,
        "experiment_id": "slice11_tiny_overfit_v1",
        "base_model_path": "models/dots.tts-mf-1step",
        "step": int(step),
        "seed": int(seed),
        "trainable_state": {
            name: tensor.detach().cpu()
            for name, tensor in model.named_parameters()
            if tensor.requires_grad
        },
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> int:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = (ROOT / cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    manifest_path = (ROOT / cfg["manifest_path"]).resolve()
    rows = load_rows(manifest_path)
    model = DotsTtsModel.from_pretrained(str((ROOT / cfg["pretrained_model_path"]).resolve()))
    model.to(device)
    trainable = freeze_for_smoke(model)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.0)
    batches = make_batches(model, rows)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        batch = move_batch(batches[(step - 1) % len(batches)], device)
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss_terms = model(model.prepare_training_batch(batch))
            loss, metrics = scalar_loss(loss_terms)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        metrics.update(
            {
                "step": step,
                "grad_norm": float(grad_norm.detach().float().item()),
                "fid": batch["fids"][0],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        del batch, loss_terms, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_path = out_dir / f"checkpoint-step{int(args.steps):08d}.pt"
    save_adapter_checkpoint(checkpoint_path, model, optimizer, int(args.steps), seed)
    summary = {
        "schema_version": 1,
        "status": "succeeded",
        "experiment_id": "slice11_tiny_overfit_v1",
        "base_model_path": str((ROOT / cfg["pretrained_model_path"]).resolve()),
        "manifest_path": str(manifest_path),
        "sample_count": len(rows),
        "steps": int(args.steps),
        "seed": seed,
        "device": str(device),
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "cuda_memory": snapshot_cuda(),
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "tiny_overfit_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "succeeded", "summary": str(out_dir / 'tiny_overfit_metrics.json')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
