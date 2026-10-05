from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_audio import ensure_48k_derivative, file_sha256
from dots_tts_lab.long_form_assess import assess_source
from dots_tts_lab.long_form_identity import (
    _normalized,
    load_identity_config,
    load_reference_vectors,
)
from dots_tts_lab.speaker_embedding import (
    compute_speaker_embedding,
    load_speaker_encoder,
)

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
COMPOSE = ROOT / "docker/long_form_denoise/compose.yaml"
INCOMING = ROOT / "data/incoming"
VERDICTS = ROOT / "data/reports/long_form/incoming_verdicts"


def _run(cmd: list[str], log) -> bool:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    return subprocess.run(
        cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT
    ).returncode == 0


def _posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _file_cosine(source: Path, regions_path: Path, max_seconds: float = 20.0):
    """Embed ≤20 s of concatenated usable speech; cosine vs the frozen
    normal-reference centroid + per-negative-kind maxima."""
    config = load_identity_config(
        str(ROOT / "configs/lab/long_form/identity_v1.yaml")
    )
    encoder = load_speaker_encoder(config.encoder)
    refs = load_reference_vectors(config, encoder=encoder)
    normal = _normalized(np.mean(refs.normal, axis=0))

    regions = json.loads(regions_path.read_text(encoding="utf-8"))["regions"]
    usable = [
        (int(r["source_start_frame"]), int(r["source_end_frame"]))
        for r in regions if r.get("status") == "usable"
    ]
    if not usable:
        return None, {}
    info = sf.info(str(source))
    rate = int(info.samplerate)
    chunks: list[np.ndarray] = []
    total = 0
    with sf.SoundFile(str(source), mode="r") as handle:
        for start, end in usable:
            if total > max_seconds * rate:
                break
            handle.seek(start)
            block = handle.read(
                end - start, dtype="float32", always_2d=True
            )
            chunks.append(block.mean(axis=1))
            total += len(block)
    mono = np.concatenate(chunks)
    if rate != int(config.encoder.input_sample_rate):
        mono = soxr.resample(
            mono, rate, int(config.encoder.input_sample_rate), quality="VHQ"
        )
    vec = _normalized(
        compute_speaker_embedding(
            mono[: round(max_seconds * config.encoder.input_sample_rate)]
            .astype(np.float32),
            sample_rate=int(config.encoder.input_sample_rate),
            config=config.encoder,
            encoder=encoder,
        )
    )
    cosine = float(vec @ normal)
    negs = {
        kind: max(float(vec @ _normalized(x)) for x in matrix)
        for kind, matrix in refs.negatives_by_kind.items()
    }
    return cosine, negs


def process_one(
    source: Path,
    *,
    full_pipeline: bool,
    on_discard: str,
    log,
) -> dict:
    source = source.resolve()
    original = str(source)
    source, source_hash = ensure_48k_derivative(
        source, work_root=ROOT / "data/work/long_form"
    )
    sha12 = source_hash[:12]
    rel = _posix(source)
    record: dict = {
        "source": original,
        "pipeline_source": str(source),
        "sha256": source_hash,
        "stages": {},
    }

    work = ROOT / "data/work/long_form"
    regions_path = (
        work / "prefilter" / sha12[:2] / source_hash / "regions.json"
    )
    stages = [
        ("prefilter", [
            PY, "scripts/run_long_form_prefilter.py", "--sources", rel]),
        ("style_event", [
            PY, "scripts/run_long_form_style_event.py", "--sources", rel]),
        ("identity", [
            PY, "scripts/run_long_form_identity.py", "--sources", rel]),
        ("transcript", [
            PY, "scripts/run_long_form_transcript.py", "--sources", rel]),
    ]
    for name, cmd in stages:
        ok = _run(cmd, log)
        record["stages"][name] = "ok" if ok else "failed"
        if not ok:
            record["verdict"] = "stage_failed"
            return record

    cosine, negs = _file_cosine(source, regions_path)
    report = assess_source(
        source_hash, file_cosine=cosine, file_negative_cosines=negs
    )
    record["assessment"] = report
    record["verdict"] = report["verdict"]

    if report["verdict"] == "keep" and full_pipeline:
        style_path = (
            work / "style_event" / sha12[:2] / source_hash
            / "style_event.json"
        )
        rows = json.loads(regions_path.read_text(encoding="utf-8"))["regions"]
        gate = {
            str(r["region_index"]):
                float(r.get("spatial_risk_fraction") or 0.0) == 0.0
            for r in rows
        }
        gate_path = VERDICTS / f"spatial_gate_{sha12}.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        ok = _run([
            "docker", "compose", "-f", str(COMPOSE), "run", "--rm",
            "enhancer", "python", "scripts/run_long_form_route.py",
            "--source", rel,
            "--regions", _posix(regions_path),
            "--style-event", _posix(style_path),
            "--spatial-gate", _posix(gate_path),
            "--report", f"data/reports/long_form/incoming_verdicts/route_{sha12}.json",
        ], log)
        record["stages"]["route_dfn3"] = "ok" if ok else "failed"
        if ok:
            routes_path = (
                work / "routes" / sha12[:2] / source_hash / "routes.json"
            )
            ok = _run([
                PY, "scripts/run_long_form_quality.py",
                "--source", rel,
                "--routes", _posix(routes_path),
                "--style-event", _posix(style_path),
            ], log)
            record["stages"]["quality"] = "ok" if ok else "failed"
        if record["stages"].get("quality") == "ok":
            ok = _run([
                PY, "scripts/build_strict_review.py", "--source", rel,
            ], log)
            record["stages"]["strict_review"] = "ok" if ok else "failed"

    if report["verdict"] != "keep":
        original_path = Path(original)
        derivative = (
            source if str(source) != original else None
        )
        if on_discard == "move":
            rejected = INCOMING / "rejected"
            rejected.mkdir(parents=True, exist_ok=True)
            target = rejected / original_path.name
            shutil.move(str(original_path), str(target))
            record["moved_to"] = str(target)
        elif on_discard == "delete":
            original_path.unlink()
            record["deleted"] = True
        if derivative is not None and derivative.is_file():
            derivative.unlink()
            record["derivative_removed"] = True
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-phase incoming-source loop: Phase-A evidence stages "
        "+ file-level timbre check vs the approved reference pack → KEEP/"
        "DISCARD verdict; KEEP optionally continues to route/DFN3/quality/"
        "strict-review. Drop files into data/incoming/."
    )
    parser.add_argument("sources", nargs="*", help="Explicit files; default: "
                        "every audio file directly under data/incoming/.")
    parser.add_argument("--dir", default=None,
                        help="Scan this directory for audio files instead of "
                        "data/incoming/ (avoids shell arg encoding issues).")
    parser.add_argument("--watch", action="store_true",
                        help="Poll data/incoming/ for new files.")
    parser.add_argument("--full", action="store_true",
                        help="Run Phase B (route/DFN3/quality/review) on "
                        "KEEP verdicts.")
    parser.add_argument(
        "--on-discard", choices=["move", "delete", "keep"], default="move",
        help="move → data/incoming/rejected/ (recoverable); delete removes "
        "the source permanently.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    INCOMING.mkdir(parents=True, exist_ok=True)
    VERDICTS.mkdir(parents=True, exist_ok=True)
    suffixes = {".mp3", ".wav", ".flac", ".m4a"}
    done: set[str] = set()

    scan_dir = Path(args.dir).resolve() if args.dir else INCOMING

    def pending() -> list[Path]:
        return [
            p for p in sorted(scan_dir.iterdir())
            if p.is_file() and p.suffix.lower() in suffixes
            and str(p) not in done
        ]

    while True:
        if args.sources:
            batch = [Path(s).resolve() for s in args.sources]
        elif args.dir:
            batch = pending()
        else:
            batch = pending()
        if not batch:
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
            continue
        for source in batch:
            log_path = VERDICTS / f"{file_sha256(source)[:12]}.log"
            with log_path.open("a", encoding="utf-8") as log:
                started = time.time()
                rec = process_one(
                    source,
                    full_pipeline=args.full,
                    on_discard=args.on_discard,
                    log=log,
                )
            rec["elapsed_seconds"] = round(time.time() - started, 1)
            out = VERDICTS / f"{rec['sha256'][:12]}.verdict.json"
            out.write_text(
                json.dumps(rec, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            done.add(str(source))
            print(json.dumps({
                "file": source.name,
                "verdict": rec.get("verdict"),
                "metrics": (rec.get("assessment") or {}).get("metrics"),
                "elapsed_seconds": rec["elapsed_seconds"],
            }, ensure_ascii=False), flush=True)
        if not args.watch:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
