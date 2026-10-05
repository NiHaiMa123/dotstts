from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_batch import (
    assemble_sentence_candidates,
    build_strict_review_html,
    dedupe_by_text,
    rank_candidates,
)
from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.long_form_strict_gate import (
    apply_batch_budget,
    load_strict_gate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _work_dir(stage: str, source_sha256: str) -> Path:
    return (
        ROOT / "data" / "work" / "long_form" / stage
        / source_sha256[:2] / source_sha256
    )


def _cut_clip(
    source: Path, start: int, end: int, out: Path
) -> None:
    with sf.SoundFile(str(source), mode="r") as f:
        f.seek(start)
        block = f.read(end - start, dtype="float32", always_2d=True)
        rate = int(f.samplerate)
    mono = block.mean(axis=1).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mono, rate, subtype="PCM_16")


def _cut_from_region(
    region_wav: Path, region_start: int, start: int, end: int, out: Path
) -> None:
    data, rate = sf.read(str(region_wav), dtype="float32", always_2d=True)
    seg = data[start - region_start: end - region_start]
    mono = seg.mean(axis=1).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mono, rate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble strict-gate sentence candidates and build the "
        "review page (LF-12H)."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--gate-config", default=None)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    source_hash = file_sha256(source)
    gate_config = load_strict_gate_config(args.gate_config) if args.gate_config else load_strict_gate_config()

    transcript = _load(_work_dir("transcript", source_hash) / "transcript.json")
    routes_path = _work_dir("routes", source_hash) / "routes.json"
    routes = _load(routes_path) if routes_path.is_file() else {"regions": []}
    identity_path = _work_dir("identity", source_hash) / "identity.json"
    identity = _load(identity_path) if identity_path.is_file() else {"regions": []}
    style_path = _work_dir("style_event", source_hash) / "style_event.json"
    style = _load(style_path) if style_path.is_file() else {"regions": []}
    prefilter = _load(_work_dir("prefilter", source_hash) / "regions.json")

    routes_by_region = {
        int(r["region_index"]): r for r in routes.get("regions", [])
    }
    identity_by_region = {
        int(r["region_index"]): r for r in identity.get("regions", [])
    }
    style_by_region = {
        int(r["region_index"]): r for r in style.get("regions", [])
    }

    # map each sentence span to the region containing it
    region_bounds = sorted(
        (int(r["source_start_frame"]), int(r["source_end_frame"]), int(r["region_index"]))
        for r in prefilter.get("regions", [])
    )
    region_of_span: dict[tuple[int, int], int] = {}
    for span in transcript.get("sentences", []):
        s, e = int(span["source_start_frame"]), int(span["source_end_frame"])
        for rs, re_, ri in region_bounds:
            if rs <= s and e <= re_:
                region_of_span[(s, e)] = ri
                break

    candidates = assemble_sentence_candidates(
        transcript,
        routes_by_region=routes_by_region,
        identity_by_region=identity_by_region,
        style_event_by_region=style_by_region,
        gate_config=gate_config,
        region_of_span=region_of_span,
    )
    ranked = dedupe_by_text(rank_candidates(candidates), candidates)
    source_of = {str(c["candidate_id"]): source_hash for c in candidates}
    budget = apply_batch_budget(ranked, source_of, gate_config.batch_budget)
    for c in candidates:
        if c["disposition"] in ("strict_candidate", "calibration_sample"):
            c["budget_disposition"] = budget.get(c["candidate_id"], "reserve_budget")

    # materialise review clips under the served reports tree
    validate_output_path(
        ROOT / "data/reports/long_form/strict_review" / source_hash[:12] / "audio"
    ).mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    by_id = {str(c["candidate_id"]): c for c in candidates}
    for cid in ranked:
        c = by_id[cid]
        s, e = c["source_start_frame"], c["source_end_frame"]
        raw_rel = f"strict_review/{source_hash[:12]}/audio/{cid[:16]}_raw.wav"
        _cut_clip(source, s, e, ROOT / "data/reports/long_form" / raw_rel)
        routed_rel = None
        route_row = routes_by_region.get(c["region_index"]) if c["region_index"] is not None else None
        if (
            route_row
            and route_row.get("status") == "completed"
            and route_row.get("output")
        ):
            region_wav = routes_path.parent / route_row["output"]["relative_path"]
            if region_wav.is_file():
                routed_rel = f"strict_review/{source_hash[:12]}/audio/{cid[:16]}_routed.wav"
                _cut_from_region(
                    region_wav,
                    int(route_row["source_start_frame"]),
                    s, e,
                    ROOT / "data/reports/long_form" / routed_rel,
                )
        span = next(
            x for x in transcript["sentences"]
            if int(x["source_start_frame"]) == s and int(x["source_end_frame"]) == e
        )
        items.append(
            {
                "candidate_id": cid,
                "raw_audio": "/" + f"data/reports/long_form/{raw_rel}".replace("\\", "/"),
                "routed_audio": (
                    "/" + f"data/reports/long_form/{routed_rel}".replace("\\", "/")
                    if routed_rel else None
                ),
                "primary_text": c["primary_text"],
                "secondary_text": span.get("secondary_text"),
                "duration_seconds": c["duration_seconds"],
                "disposition": c["disposition"],
                "budget_disposition": c.get("budget_disposition"),
                "failed_gates": c["failed_gates"],
                "unknown_gates": c["unknown_gates"],
                "overlaps_quarantine": c["overlaps_quarantine"],
                "boundary_clean": c["boundary_clean"],
                "event_risk_max": c["event_risk_max"],
                "route": c["route"],
                "route_reasons": (route_row or {}).get("route_reasons", []),
            }
        )

    batch_sha256 = hashlib.sha256(
        json.dumps(
            {"source": source_hash, "items": [i["candidate_id"] for i in items]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    html = build_strict_review_html(
        title=source.name,
        source_sha256=source_hash,
        items=items,
        save_name=f"strict-review-{source_hash[:12]}",
        batch_sha256=batch_sha256,
    )
    out = ROOT / "data/reports/long_form" / f"strict_review_{source_hash[:12]}.html"
    out.write_text(html, encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "page": str(out),
        "candidates": len(items),
        "strict_candidates": sum(
            1 for i in items if i["disposition"] == "strict_candidate"
        ),
        "batch_sha256": batch_sha256,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
