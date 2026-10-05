from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def write_training_runtime_metrics(
    *,
    output_dir: str | Path,
    status: str,
    start_global_step: int,
    end_global_step: int,
    runtime_started_at: float,
    device: torch.device,
    is_main_process: bool = True,
) -> bool:
    """Write runtime metrics without replacing a measured run on zero-step resume."""
    if not is_main_process:
        return False

    output_path = Path(output_dir) / "training_runtime_metrics.json"
    completed_steps = int(end_global_step) - int(start_global_step)
    if completed_steps <= 0 and output_path.is_file():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cuda_metrics = {
        "available": device.type == "cuda",
        "device": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "total_memory_bytes": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_metrics.update(
            {
                "device": properties.name,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "start_global_step": int(start_global_step),
                "end_global_step": int(end_global_step),
                "completed_optimizer_steps": completed_steps,
                "elapsed_seconds": time.perf_counter() - runtime_started_at,
                "cuda": cuda_metrics,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return True
