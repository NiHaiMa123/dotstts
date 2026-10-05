from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import soundfile as sf

from dots_tts_lab.standardization import true_peak_estimate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def managed(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    derived_root = args.derived_root.resolve()
    connection = sqlite3.connect(args.catalog)
    connection.row_factory = sqlite3.Row
    raw_rows = connection.execute(
        "SELECT asset_sha256, relative_path, size_bytes FROM raw_object ORDER BY asset_sha256"
    ).fetchall()
    derived_rows = connection.execute(
        """
        SELECT * FROM derived_audio
        WHERE config_id = 'training_audio' AND config_version = 1
        ORDER BY asset_sha256
        """
    ).fetchall()
    connection.close()

    errors: list[dict[str, str]] = []
    max_true_peak = None
    for row in raw_rows:
        try:
            path = managed(raw_root, row["relative_path"])
            if path.stat().st_size != row["size_bytes"]:
                raise ValueError("size mismatch")
            if sha256(path) != row["asset_sha256"]:
                raise ValueError("SHA-256 mismatch")
        except Exception as error:
            errors.append(
                {"kind": "raw", "asset_sha256": row["asset_sha256"], "error": str(error)}
            )

    for row in derived_rows:
        try:
            path = managed(derived_root, row["relative_path"])
            if path.stat().st_size != row["size_bytes"]:
                raise ValueError("size mismatch")
            if sha256(path) != row["output_sha256"]:
                raise ValueError("SHA-256 mismatch")
            info = sf.info(str(path))
            if (
                info.samplerate != 48_000
                or info.channels != 1
                or info.subtype != "PCM_24"
                or info.frames != row["output_frames"]
            ):
                raise ValueError(f"unexpected WAV metadata: {info}")
            audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
            if sample_rate != 48_000:
                raise ValueError("decoded sample rate mismatch")
            peak = true_peak_estimate(audio, 4)
            peak_db = None if peak <= 0 else 20.0 * math.log10(peak)
            if peak_db is not None:
                max_true_peak = peak_db if max_true_peak is None else max(max_true_peak, peak_db)
                if peak_db > -0.999:
                    raise ValueError(f"true peak exceeds tolerance: {peak_db:.6f} dBTP")
        except Exception as error:
            errors.append(
                {
                    "kind": "derived",
                    "asset_sha256": row["asset_sha256"],
                    "error": str(error),
                }
            )

    payload = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "raw_count": len(raw_rows),
        "derived_count": len(derived_rows),
        "raw_sha256_verified_count": len(raw_rows)
        - sum(error["kind"] == "raw" for error in errors),
        "derived_sha256_and_format_verified_count": len(derived_rows)
        - sum(error["kind"] == "derived" for error in errors),
        "max_recomputed_true_peak_estimate_dbtp": max_true_peak,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
