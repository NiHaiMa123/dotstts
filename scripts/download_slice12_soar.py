#!/usr/bin/env python3
"""Download the Slice12 SOAR checkpoint at the immutable contract revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "config.json",
    "latent_stats.pt",
    "llm_config.json",
    "model.safetensors",
    "vocoder.safetensors",
    "speaker_encoder.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lab/slice12/experiment_contract_v2.yaml"))
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["models"]["training_control"]
    revision = model.get("revision")
    if not revision or len(str(revision)) != 40:
        raise ValueError("Slice12 SOAR revision must be a full immutable commit SHA")
    destination = (ROOT / model["local_path"]).resolve()
    snapshot_download(
        repo_id=model["model_id"],
        revision=str(revision),
        local_dir=destination,
    )
    missing = [name for name in REQUIRED_FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded SOAR checkpoint is incomplete: {missing}")
    files = {name: {"size": (destination / name).stat().st_size, "sha256": sha256_file(destination / name)} for name in REQUIRED_FILES}
    payload = {
        "schema_version": 1,
        "status": "succeeded",
        "model_id": model["model_id"],
        "revision": str(revision),
        "local_path": str(destination),
        "files": files,
    }
    report = ROOT / "data/reports/datasets/fuxuan_v1/slice12/soar_checkpoint.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "succeeded", "model": str(destination), "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
