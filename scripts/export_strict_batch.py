from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dots_tts_lab.long_form_audio import file_sha256
from dots_tts_lab.long_form_paths import validate_output_path
from dots_tts_lab.long_form_render import (
    BoundaryRepairPolicy,
    render_repaired_spans,
)

ROOT = Path(__file__).resolve().parents[1]
STRICT_EXPORT_IMPLEMENTATION_VERSION = 1


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        writer(partial)
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export human-confirmed strict candidates with boundary "
        "repair and full binding metadata."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--decisions",
        required=True,
        help="incoming/strict-review-*.json (confirm/reject decisions)",
    )
    parser.add_argument(
        "--texts",
        required=True,
        help="incoming/strict-review-text-*.json (human-confirmed text)",
    )
    parser.add_argument(
        "--out",
        default="data/work/long_form/exports/strict_v1",
        help="Export directory (repo-relative)",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    source_hash = file_sha256(source)
    sha12 = source_hash[:12]

    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    transcript = json.loads(
        (
            ROOT / "data/work/long_form/transcript" / sha12[:2] / source_hash
            / "transcript.json"
        ).read_text(encoding="utf-8")
    )
    routes_path = (
        ROOT / "data/work/long_form/routes" / sha12[:2] / source_hash
        / "routes.json"
    )
    routes = (
        json.loads(routes_path.read_text(encoding="utf-8"))
        if routes_path.is_file()
        else {"regions": []}
    )
    prefilter = json.loads(
        (
            ROOT / "data/work/long_form/prefilter" / sha12[:2] / source_hash
            / "regions.json"
        ).read_text(encoding="utf-8")
    )

    confirmed = {
        cid for cid, d in decisions.get("decisions", {}).items()
        if d.get("decision") == "confirm"
    }
    text_of = {
        cid: t.get("text_final", "")
        for cid, t in texts.get("texts", {}).items()
    }
    if texts.get("source_sha256") != source_hash:
        raise SystemExit("texts export belongs to a different source")
    missing_text = confirmed - set(text_of)
    if missing_text:
        raise SystemExit(
            f"{len(missing_text)} confirmed candidates lack text_final"
        )

    routes_by_region = {
        int(r["region_index"]): r for r in routes.get("regions", [])
    }
    region_bounds = sorted(
        (
            int(r["source_start_frame"]),
            int(r["source_end_frame"]),
            int(r["region_index"]),
        )
        for r in prefilter.get("regions", [])
    )

    def region_of(s: int, e: int) -> int | None:
        for rs, re_, ri in region_bounds:
            if rs <= s and e <= re_:
                return ri
        return None

    export_dir = validate_output_path(ROOT / args.out)
    audio_dir = export_dir / "audio"
    policy = BoundaryRepairPolicy()

    def cid(span: dict) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "source_sha256": source_hash,
                    "start": span["source_start_frame"],
                    "end": span["source_end_frame"],
                    "text": span.get("primary_text"),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    rows = []
    for span in transcript.get("sentences", []):
        span_id = cid(span)
        if span_id not in confirmed:
            continue
        s, e = int(span["source_start_frame"]), int(span["source_end_frame"])
        ri = region_of(s, e)
        route_row = routes_by_region.get(ri) if ri is not None else None
        routed_wav = None
        if (
            route_row
            and route_row.get("status") == "completed"
            and route_row.get("output")
        ):
            candidate_wav = routes_path.parent / route_row["output"]["relative_path"]
            if candidate_wav.is_file():
                routed_wav = candidate_wav

        if routed_wav is not None:
            offset = int(route_row["source_start_frame"])
            audio, rate, trace = render_repaired_spans(
                routed_wav,
                [{
                    "source_start_frame": s - offset,
                    "source_end_frame": e - offset,
                }],
                join_silence_seconds=0.0,
                policy=policy,
            )
            audio_source = route_row["route"]
            region_start = offset
        else:
            audio, rate, trace = render_repaired_spans(
                source,
                [{"source_start_frame": s, "source_end_frame": e}],
                join_silence_seconds=0.0,
                policy=policy,
            )
            audio_source = "raw"
            region_start = 0

        out_name = f"{span_id[:16]}.wav"
        out_path = audio_dir / out_name
        mono = audio.mean(axis=1) if audio.ndim > 1 else audio

        def write_wav(target: Path, data=mono, sample_rate=rate) -> None:
            sf.write(str(target), data, sample_rate, subtype="PCM_16", format="WAV")

        _atomic_write(out_path, write_wav)
        out_sha = file_sha256(out_path)
        mapping = trace["mappings"][0]
        warnings: list[str] = []
        if span.get("overlaps_quarantine"):
            warnings.append("crosses_prefilter_quarantine")
        if route_row and route_row.get("status") == "blocked_spatial":
            warnings.append("spatial_blocked_region_raw_downmix")
        if span.get("text_agreement", {}).get("status") != "match":
            warnings.append("asr_disagreement_human_resolved")
        rows.append(
            {
                "candidate_id": span_id,
                "source_sha256": source_hash,
                "source_start_frame": region_start
                + int(mapping["source_start_frame"]),
                "source_end_frame": region_start
                + int(mapping["source_end_frame"]),
                "duration_seconds": round(len(mono) / rate, 4),
                "audio": {
                    "relative_path": f"audio/{out_name}",
                    "sha256": out_sha,
                    "sample_rate_hz": rate,
                    "channels": 1,
                },
                "audio_route": audio_source,
                "region_index": ri,
                "text_final": text_of[span_id],
                "text_sha256": hashlib.sha256(
                    text_of[span_id].encode("utf-8")
                ).hexdigest(),
                "text_source": "human_confirmed",
                "span_status": span.get("status"),
                "overlaps_quarantine": span.get("overlaps_quarantine"),
                "boundary_trace": mapping,
                "warnings": warnings,
                "decision": "human_confirmed",
                "batch_sha256": decisions.get("batch_sha256"),
            }
        )

    manifest = {
        "schema_version": 1,
        "implementation_version": STRICT_EXPORT_IMPLEMENTATION_VERSION,
        "source_sha256": source_hash,
        "export": "strict_v1",
        "decisions_sha256": hashlib.sha256(
            Path(args.decisions).read_bytes()
        ).hexdigest(),
        "texts_sha256": hashlib.sha256(Path(args.texts).read_bytes()).hexdigest(),
        "candidate_count": len(rows),
        "candidates": rows,
    }
    manifest_path = export_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        lambda p: p.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        ),
    )
    print(json.dumps({
        "status": "completed",
        "manifest": str(manifest_path),
        "candidates": len(rows),
        "audio_seconds": round(sum(r["duration_seconds"] for r in rows), 1),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
