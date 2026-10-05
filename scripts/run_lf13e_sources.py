from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_audio import ensure_48k_derivative, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
COMPOSE = ROOT / "docker/long_form_denoise/compose.yaml"
LOG_DIR = ROOT / "data/reports/long_form/lf13e_runs"


def _run(label: str, cmd: list[str], log) -> bool:
    started = time.time()
    log.write(f"\n===== {label}\n$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.run(
        cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT
    )
    elapsed = round(time.time() - started, 1)
    log.write(f"----- {label}: exit={proc.returncode} {elapsed}s\n")
    log.flush()
    return proc.returncode == 0


def _manifest(work: str, source_hash: str, name: str) -> Path:
    return (
        ROOT / "data/work/long_form" / work
        / source_hash[:2] / source_hash / name
    )


def _posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LF-13E: run the strict pipeline sequentially over the "
        "remaining inbox sources (prefilter → style/event → identity → "
        "transcript → route → container DFN3 → quality → strict review page)."
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Source SHA256 prefixes already processed (skipped).",
    )
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    inbox = ROOT / "data/inbox/岁岁"
    sources = sorted(
        p for p in inbox.iterdir()
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for source in sources:
        # Non-48 kHz sources are normalized to a 48 kHz derivative first;
        # downstream frame coordinates then refer to the derivative.
        source, source_hash = ensure_48k_derivative(
            source, work_root=ROOT / "data/work/long_form"
        )
        sha12 = source_hash[:12]
        if args.only and sha12 not in args.only:
            continue
        if any(source_hash.startswith(s) for s in args.skip):
            continue
        log_path = LOG_DIR / f"{sha12}.log"
        rec = {"source": str(source), "sha256": source_hash, "stages": {}}
        with log_path.open("w", encoding="utf-8") as log:
            src = str(source.relative_to(ROOT)).replace("\\", "/")
            stages = [
                ("prefilter", [
                    PY, "scripts/run_long_form_prefilter.py",
                    "--sources", src,
                ]),
                ("style_event", [
                    PY, "scripts/run_long_form_style_event.py",
                    "--sources", src,
                ]),
                ("identity", [
                    PY, "scripts/run_long_form_identity.py",
                    "--sources", src,
                ]),
                ("transcript", [
                    PY, "scripts/run_long_form_transcript.py",
                    "--sources", src,
                ]),
            ]
            ok = True
            for name, cmd in stages:
                ok = _run(name, cmd, log)
                rec["stages"][name] = "ok" if ok else "failed"
                if not ok:
                    break
            if ok:
                regions = _manifest("prefilter", source_hash, "regions.json")
                style = _manifest("style_event", source_hash, "style_event.json")
                gate_path = LOG_DIR / f"spatial_gate_{sha12}.json"
                rows = json.loads(regions.read_text(encoding="utf-8"))["regions"]
                gate = {
                    str(r["region_index"]):
                        float(r.get("spatial_risk_fraction") or 0.0) == 0.0
                    for r in rows
                }
                gate_path.write_text(json.dumps(gate), encoding="utf-8")
                # run_long_form_route decides AND executes routes; when any
                # region needs DFN3 it requires the df backend, so the whole
                # stage runs inside the pinned enhancer container.
                ok = _run("route_dfn3", [
                    "docker", "compose", "-f", str(COMPOSE),
                    "run", "--rm", "enhancer",
                    "python", "scripts/run_long_form_route.py",
                    "--source", src,
                    "--regions", _posix(regions),
                    "--style-event", _posix(style),
                    "--spatial-gate", _posix(gate_path),
                    "--report", f"data/reports/long_form/lf13e_runs/route_{sha12}.json",
                ], log)
                rec["stages"]["route_dfn3"] = "ok" if ok else "failed"
            if ok:
                routes_manifest = _manifest(
                    "routes", source_hash, "routes.json"
                )
                ok = _run("quality", [
                    PY, "scripts/run_long_form_quality.py",
                    "--source", src,
                    "--routes", _posix(routes_manifest),
                    "--style-event", _posix(style),
                ], log)
                rec["stages"]["quality"] = "ok" if ok else "failed"
            if ok:
                ok = _run("strict_review", [
                    PY, "scripts/build_strict_review.py",
                    "--source", src,
                ], log)
                rec["stages"]["strict_review"] = "ok" if ok else "failed"
        rec["status"] = (
            "completed" if all(v == "ok" for v in rec["stages"].values())
            else "failed"
        )
        results[sha12] = rec
        print(sha12, rec["status"], rec["stages"], flush=True)

    summary = LOG_DIR / "summary.json"
    summary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: v["status"] for k, v in results.items()}, indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
