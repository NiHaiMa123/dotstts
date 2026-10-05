#!/usr/bin/env python3
"""Audit Slice12 model, runtime, dependency and provenance prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_ARTIFACTS = (
    "config.json",
    "latent_stats.pt",
    "llm_config.json",
    "model.safetensors",
    "vocoder.safetensors",
    "speaker_encoder.safetensors",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_inventory(path: Path) -> dict[str, Any]:
    files = {name: sha256_file(path / name) for name in REQUIRED_MODEL_ARTIFACTS}
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "complete": path.is_dir() and all(files.values()),
        "files": files,
    }


def command_probe(command: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def audit(config_path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 2:
        raise ValueError("Slice12 preflight requires experiment contract schema_version 2")

    deployment_cfg = config["models"]["deployment_baseline"]
    control_cfg = config["models"]["training_control"]
    deployment = model_inventory((repo_root / deployment_cfg["local_path"]).resolve())
    training_control = model_inventory((repo_root / control_cfg["local_path"]).resolve())

    required_names = list(config["runtime"]["required_python_dependencies"])
    optional_names = list(config["runtime"]["optional_python_dependencies"])
    host_dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in required_names + optional_names
    }

    docker_cli = shutil.which("docker")
    docker_client = command_probe([docker_cli, "--version"]) if docker_cli else {"ok": False, "error": "docker CLI not found"}
    docker_engine = command_probe([docker_cli, "info", "--format", "{{json .ServerVersion}}"] ) if docker_cli else {"ok": False, "error": "docker CLI not found"}

    cuda = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
        cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        cuda["torch_version"] = torch.__version__
        cuda["cuda_version"] = torch.version.cuda
    except Exception as error:  # pragma: no cover - diagnostic fallback
        cuda["error"] = f"{type(error).__name__}: {error}"

    dataset_files: dict[str, dict[str, Any]] = {}
    for key in ("train_manifest", "validation_manifest", "test_manifest"):
        path = (repo_root / config["dataset"][key]).resolve()
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "readable": False}
        if path.is_file():
            try:
                entry["sha256"] = sha256_file(path)
                entry["readable"] = entry["sha256"] is not None
            except OSError as error:
                entry["error"] = f"{type(error).__name__}: {error}"
        dataset_files[key] = entry

    docker_required = config["runtime"]["mode"] == "docker_wsl2"
    blocking_reasons = []
    if not deployment["complete"]:
        blocking_reasons.append("deployment baseline artifact is incomplete")
    if not training_control["complete"]:
        blocking_reasons.append("official dots.tts-soar training-control artifact is incomplete or absent")
    if not all(item["readable"] for item in dataset_files.values()):
        blocking_reasons.append("one or more frozen dataset manifests are not readable")
    if docker_required and not docker_client["ok"]:
        blocking_reasons.append("Docker CLI is unavailable")
    if docker_required and not docker_engine["ok"]:
        blocking_reasons.append("Docker Desktop Linux engine is not running")

    warnings = []
    missing_host_required = [name for name in required_names if not host_dependencies[name]]
    if docker_required and missing_host_required:
        warnings.append(
            "host Python is missing training packages, but Docker mode installs them in the image: "
            + ", ".join(missing_host_required)
        )
    missing_optional = [name for name in optional_names if not host_dependencies[name]]
    if missing_optional:
        warnings.append(
            "optional acceleration packages are absent and are intentionally unused by the first run: "
            + ", ".join(missing_optional)
        )

    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "status": "ready" if not blocking_reasons else "blocked_missing_training_prerequisites",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "platform": platform.platform(),
        "python": sys.version,
        "runtime_mode": config["runtime"]["mode"],
        "deployment_baseline": deployment,
        "training_control": training_control,
        "dataset_files": dataset_files,
        "host_dependencies_diagnostic_only": host_dependencies,
        "docker": {"cli_path": docker_cli, "client": docker_client, "engine": docker_engine},
        "host_cuda_diagnostic_only": cuda,
        "required_model_artifacts": list(REQUIRED_MODEL_ARTIFACTS),
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lab/slice12/experiment_contract_v2.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    report = audit(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_output = config["outputs"]["preflight_report"]
    output_path = (ROOT / (args.output or configured_output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_path), "blocking_reasons": report["blocking_reasons"], "warnings": report["warnings"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
