from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for config_path in args.config:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        backend = str(config["backend_id"])
        revision = str(config["model_revision"])
        output = (args.output_root / backend / revision).resolve()
        output.mkdir(parents=True, exist_ok=True)
        downloaded = snapshot_download(
            repo_id=str(config["model_id"]),
            revision=revision,
            local_dir=output,
        )
        results.append(
            {
                "backend_id": backend,
                "model_id": config["model_id"],
                "model_revision": revision,
                "local_path": str(Path(downloaded).resolve()),
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
