from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_dfn3 import (
    DEFAULT_DFN3_CONFIG_PATH,
    Dfn3VerificationError,
    load_dfn3_config,
    verify_backend,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the pinned DeepFilterNet3 backend (LF-13A)."
    )
    parser.add_argument("--config", default=str(DEFAULT_DFN3_CONFIG_PATH))
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON output path (default: <report_root>/backend_verification.json)",
    )
    args = parser.parse_args()

    config = load_dfn3_config(args.config)
    package_version: str | None = None
    try:
        import df  # type: ignore

        package_version = getattr(df, "__version__", None)
        if package_version is None:
            try:
                import importlib.metadata

                package_version = importlib.metadata.version("deepfilternet")
            except Exception:
                package_version = None
    except ImportError:
        package_version = None
    image_tag = os.environ.get("DFN3_IMAGE_TAG") or None
    image_id = os.environ.get("DFN3_IMAGE_ID") or None

    try:
        report = verify_backend(
            config, package_version=package_version, image_tag=image_tag
        )
    except Dfn3VerificationError as error:
        report = {"status": "failed", "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0
    report["checked_at_unix"] = time.time()
    report["config_sha256"] = config.config_sha256()
    if image_id:
        report["image_id"] = image_id

    output = Path(args.report).resolve() if args.report else (
        Path(__file__).resolve().parents[1]
        / config.output.report_root
        / "backend_verification.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
