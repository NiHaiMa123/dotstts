"""Checkpoint helpers for distributed dots_tts training.

This module persists not only model/optimizer/scheduler state, but also
rank-local RNG state and data-loader progress so resumed training can continue
from the same point with minimal drift.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

TRAINABLE_MODEL_FILENAME = "trainable_model.safetensors"
TRAINABLE_MODEL_METADATA_FILENAME = "trainable_model.json"


def _checkpoint_dir(log_dir: str, step: int) -> Path:
    """Return the canonical directory name for a training step checkpoint."""
    return Path(log_dir) / f"checkpoint-{step:08d}"


def _checkpoint_entries(log_dir: str) -> list[tuple[int, Path]]:
    """List valid ``checkpoint-*`` directories sorted by step number."""
    entries = []
    for path in Path(log_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            entries.append((int(suffix), path))
    return sorted(entries)


def resolve_latest_train_checkpoint(log_dir: str) -> Path:
    """Resolve the checkpoint directory that should be used for resume.

    Preference order:
    1. ``<log_dir>/latest`` symlink, if present.
    2. The numerically largest ``checkpoint-*`` directory.
    """
    latest_path = Path(log_dir) / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        return latest_path.resolve(strict=True)

    entries = _checkpoint_entries(log_dir)
    if not entries:
        raise FileNotFoundError(
            f"No checkpoint found under {log_dir!s}; expected latest or checkpoint-*."
        )
    return entries[-1][1].resolve(strict=True)


def _rng_state() -> dict:
    """Capture Python/NumPy/PyTorch RNG state for deterministic resume."""
    numpy_state = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNG state previously produced by :func:`_rng_state`."""
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _replace_latest_symlink(log_dir: str, save_dir: Path) -> None:
    """Atomically refresh the ``latest`` symlink to point at ``save_dir``."""
    log_path = Path(log_dir)
    link_path = log_path / "latest"
    tmp_link_path = log_path / "latest.tmp"

    if tmp_link_path.is_symlink() or tmp_link_path.is_junction():
        tmp_link_path.unlink()
    elif tmp_link_path.exists():
        shutil.rmtree(tmp_link_path)
    try:
        tmp_link_path.symlink_to(save_dir.name)
    except OSError:
        # Windows without symlink privilege: a directory junction needs none
        # and is transparent to downstream path consumers.
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(tmp_link_path), str(save_dir.resolve())],
            check=True,
            capture_output=True,
        )

    if link_path.is_junction():
        link_path.unlink()
    elif link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    tmp_link_path.rename(link_path)


def _cleanup_old_checkpoints(log_dir: str, keep_max: int) -> None:
    """Delete older checkpoints while keeping the newest ``keep_max`` ones."""
    if keep_max <= 0:
        return
    for _, path in _checkpoint_entries(log_dir)[:-keep_max]:
        shutil.rmtree(path, ignore_errors=True)


def _pack_rank_payload(accelerator, payload: dict, *, payload_name: str) -> dict | None:
    """Collect rank-local payloads onto the main process for checkpointing.

    Some training state is intentionally local to each rank, for example RNG
    state or data-loader shard progress. We therefore gather a per-rank payload
    and store it in the checkpoint as ``{world_size, per_rank}``.
    """
    local_payload = {
        "rank": int(accelerator.process_index),
        "payload": payload,
    }
    if dist.is_available() and dist.is_initialized():
        gathered: list[dict | None] = [None] * int(accelerator.num_processes)
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    if not accelerator.is_main_process:
        return None

    per_rank = {}
    for item in gathered:
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Failed to gather rank-scoped {payload_name} for checkpointing."
            )
        per_rank[str(int(item["rank"]))] = item["payload"]
    return {
        "world_size": len(gathered),
        "per_rank": per_rank,
    }


def _extract_rank_payload(
    accelerator, payload: dict | None, *, payload_name: str
) -> dict:
    """Recover the payload for the current rank from a packed checkpoint blob."""
    if payload is None:
        return {}

    expected_world_size = int(accelerator.num_processes)
    if int(payload["world_size"]) != expected_world_size:
        raise RuntimeError(
            f"Checkpoint {payload_name} payload does not match the current world."
        )

    local_rank = str(int(accelerator.process_index))
    if local_rank not in payload["per_rank"]:
        raise RuntimeError(f"Checkpoint {payload_name} is missing rank {local_rank}.")
    return payload["per_rank"][local_rank]


def _uses_trainable_delta(model) -> bool:
    lora_config = getattr(getattr(model, "config", None), "lora", None)
    return bool(lora_config is not None and getattr(lora_config, "enabled", False))


def save_trainable_model_artifact(model, model_dir: str | Path) -> Path:
    """Save only parameters that can change during adapter training."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainable = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("Adapter checkpoint has no trainable parameters to save")

    weights_path = model_dir / TRAINABLE_MODEL_FILENAME
    save_file(trainable, str(weights_path))
    lora_config = getattr(model.config, "lora", None)
    metadata = {
        "schema_version": 1,
        "format": "dots_tts_trainable_delta",
        "base_model_required": True,
        "lora": (
            lora_config.to_declared_dict()
            if hasattr(lora_config, "to_declared_dict")
            else None
        ),
        "parameter_count": sum(tensor.numel() for tensor in trainable.values()),
        "parameters": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in trainable.items()
        },
    }
    (model_dir / TRAINABLE_MODEL_METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return weights_path


def load_trainable_model_artifact(
    model,
    model_dir: str | Path,
    *,
    require_matching_trainable_parameters: bool = False,
) -> dict:
    """Overlay a trainable-delta artifact onto an already loaded base model."""
    model_dir = Path(model_dir)
    metadata_path = model_dir / TRAINABLE_MODEL_METADATA_FILENAME
    weights_path = model_dir / TRAINABLE_MODEL_FILENAME
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Trainable-delta artifact is incomplete: {model_dir!s}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "dots_tts_trainable_delta":
        raise ValueError(f"Unsupported trainable-delta format in {metadata_path!s}")
    weights = load_file(str(weights_path), device="cpu")
    declared_names = set(metadata.get("parameters") or {})
    if set(weights) != declared_names:
        raise ValueError("Trainable-delta metadata does not match its weight keys")

    model_parameters = dict(model.named_parameters())
    missing_parameters = declared_names - set(model_parameters)
    if missing_parameters:
        raise ValueError(
            "Trainable-delta parameters are absent from the base model: "
            + ", ".join(sorted(missing_parameters))
        )
    for name, tensor in weights.items():
        if tuple(tensor.shape) != tuple(model_parameters[name].shape):
            raise ValueError(f"Trainable-delta shape mismatch for {name}")

    if require_matching_trainable_parameters:
        current_trainable = {
            name for name, parameter in model_parameters.items() if parameter.requires_grad
        }
        if current_trainable != declared_names:
            raise ValueError(
                "Current trainable parameter set does not match the saved adapter checkpoint"
            )

    incompatible = model.load_state_dict(weights, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected trainable-delta keys: "
            + ", ".join(sorted(incompatible.unexpected_keys))
        )
    return metadata


def save_train_checkpoint(
    accelerator,
    model,
    optimizer,
    progress,
    log_dir: str,
    keep_max: int,
    data_state: dict,
    scheduler_state: dict,
) -> None:
    """Save a full resumable training checkpoint.

    Stored artifacts include:
    - full model weights in ``save_pretrained`` format, or a compact trainable
      delta when LoRA is enabled
    - optimizer / scheduler / scaler state
    - training progress counters
    - rank-local RNG state
    - rank-local data pipeline state
    """
    accelerator.wait_for_everyone()
    packed_data_state = _pack_rank_payload(
        accelerator,
        data_state,
        payload_name="data_state",
    )
    packed_rng_state = _pack_rank_payload(
        accelerator,
        _rng_state(),
        payload_name="rng_state",
    )

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_dir = _checkpoint_dir(log_dir, progress.global_step)
        tmp_dir = save_dir.with_name(f"{save_dir.name}.tmp")
        model_dir = tmp_dir / "model"
        scaler = getattr(accelerator, "scaler", None)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write into a temporary directory first so interrupted saves never
            # leave behind a half-written checkpoint that looks valid.
            if _uses_trainable_delta(unwrapped_model):
                save_trainable_model_artifact(unwrapped_model, model_dir)
            else:
                unwrapped_model.save_pretrained(model_dir)

            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            torch.save(scheduler_state, tmp_dir / "scheduler.pt")
            torch.save(
                {} if scaler is None else scaler.state_dict(),
                tmp_dir / "scaler.pt",
            )
            torch.save(packed_rng_state, tmp_dir / "rng_state.pt")
            torch.save(packed_data_state, tmp_dir / "data_state.pt")
            (tmp_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        field.name: int(getattr(progress, field.name))
                        for field in fields(progress)
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if save_dir.exists():
                shutil.rmtree(save_dir)
            tmp_dir.rename(save_dir)
            _replace_latest_symlink(log_dir, save_dir)
            _cleanup_old_checkpoints(log_dir, keep_max)
            accelerator.print(f"Checkpoint saved: {save_dir}")
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    accelerator.wait_for_everyone()


def load_train_checkpoint(
    accelerator,
    model,
    optimizer,
    progress,
    checkpoint_dir: str | Path,
    scheduler,
) -> dict:
    """Restore a checkpoint previously written by :func:`save_train_checkpoint`.

    Returns auxiliary state that the caller usually needs to resume the input
    pipeline and scheduler bookkeeping.
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint model directory not found: {model_dir!s}")

    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    if (model_dir / TRAINABLE_MODEL_METADATA_FILENAME).is_file():
        load_trainable_model_artifact(
            unwrapped_model,
            model_dir,
            require_matching_trainable_parameters=True,
        )
    else:
        unwrapped_model.load_pretrained_weights(model_dir)

    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
    )

    scheduler_payload = torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu")
    scheduler.load_state_dict(scheduler_payload["state_dict"])

    scaler = getattr(accelerator, "scaler", None)
    scaler_state = torch.load(checkpoint_dir / "scaler.pt", map_location="cpu")
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)

    rng_state_payload = torch.load(checkpoint_dir / "rng_state.pt", map_location="cpu")
    _restore_rng_state(
        _extract_rank_payload(
            accelerator,
            rng_state_payload,
            payload_name="rng_state",
        )
    )
    data_state_payload = torch.load(
        checkpoint_dir / "data_state.pt", map_location="cpu"
    )

    trainer_state = json.loads(
        (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    for field in fields(progress):
        setattr(progress, field.name, int(trainer_state[field.name]))

    accelerator.wait_for_everyone()
    return {
        "checkpoint_dir": checkpoint_dir,
        "data_state": _extract_rank_payload(
            accelerator,
            data_state_payload,
            payload_name="data_state",
        ),
        "scheduler_state": scheduler_payload,
    }
