from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from dots_tts_lab.long_form_reference import (
    ClipBinding,
    NegativeExample,
    ReferenceClip,
    ReferencePack,
    propose_reference_clips,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_propose(args: argparse.Namespace) -> int:
    manifest = Path(args.export_manifest).resolve()
    chosen = propose_reference_clips(
        manifest,
        max_clips=args.max_clips,
        minimum_seconds=args.minimum_seconds,
        maximum_seconds=args.maximum_seconds,
    )
    audio_root = manifest.parent
    for row in chosen:
        audio_path = (audio_root / row["audio_relative_path"]).resolve()
        if not audio_path.is_file() or _sha256(audio_path) != row["audio_sha256"]:
            raise RuntimeError(f"audio hash mismatch or missing: {audio_path}")
        row["audio_absolute_path"] = str(audio_path)
    payload = {
        "schema_version": 1,
        "status": "draft_pending_confirmation",
        "voice_id": args.voice_id,
        "source_export_manifest": str(manifest),
        "labeling_spec": "docs/long-form-labeling-v1.md",
        "note": "Draft only. Each clip must be re-confirmed under the v1 labeling "
        "spec before it may be frozen into a reference pack; historical keep "
        "decisions are not truth under the new standard.",
        "candidates": chosen,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"draft proposal: {len(chosen)} candidates -> {output}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    confirmation = json.loads(Path(args.confirmations).resolve().read_text(encoding="utf-8"))
    if confirmation.get("labeling_spec") != "docs/long-form-labeling-v1.md":
        raise ValueError("confirmations must declare the v1 labeling spec")
    batch_id = confirmation.get("review_batch_id")
    if not batch_id:
        raise ValueError("confirmations require review_batch_id")
    prior_batch_ids = set(confirmation.get("prior_batch_ids", []))
    allowed = set(ClipBinding.model_fields)
    clip_items = []
    for item in confirmation["clips"]:
        item = dict(item)
        origin = item.pop("origin_review_batch_id", None)
        if origin:
            prior_batch_ids.add(origin)
        clip_items.append({key: value for key, value in item.items() if key in allowed})
    clips = [
        ReferenceClip.model_validate(
            {
                **item,
                "confirmed_review_batch_id": batch_id,
                "confirmed_at": confirmation["confirmed_at"],
            },
            strict=True,
        )
        for item in clip_items
    ]
    audio_root = Path(args.audio_root).resolve()
    for clip in clips:
        audio_path = (audio_root / clip.audio_relative_path).resolve()
        if not audio_path.is_file() or _sha256(audio_path) != clip.audio_sha256:
            raise RuntimeError(f"reference audio hash mismatch: {audio_path}")
    negative_items = list(confirmation.get("negatives", []))
    if args.negatives:
        for negatives_path in args.negatives.split(","):
            extra = json.loads(Path(negatives_path).resolve().read_text(encoding="utf-8"))
            negative_items.extend(extra.get("negatives", []))
    negatives = []
    for item in negative_items:
        item = {key: value for key, value in item.items() if key in allowed or key in ("kind", "note")}
        candidate = NegativeExample.model_validate(item, strict=True)
        audio_path = (ROOT / candidate.audio_relative_path).resolve()
        if not audio_path.is_file() or _sha256(audio_path) != candidate.audio_sha256:
            raise RuntimeError(f"negative audio hash mismatch: {audio_path}")
        negatives.append(candidate)
    pack = ReferencePack.model_validate(
        {
            "schema_version": 1,
            "pack_id": args.pack_id,
            "pack_version": args.pack_version,
            "voice_id": args.voice_id,
            "status": "confirmed",
            "clips": [clip.model_dump(mode="json") for clip in clips],
            "negatives": [item.model_dump(mode="json") for item in negatives],
            "created_from_review_batches": sorted({batch_id, *prior_batch_ids}),
            "created_at": confirmation["confirmed_at"],
            "labeling_spec": "docs/long-form-labeling-v1.md",
        },
        strict=True,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            {**pack.model_dump(mode="json"), "pack_sha256": pack.pack_sha256()},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"confirmed pack {pack.pack_id} v{pack.pack_version}: {output}")
    print(f"pack_sha256: {pack.pack_sha256()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the long-form normal-voice reference pack (LF-12A)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser(
        "propose", help="Draft reference candidates from a confirmed export manifest."
    )
    propose.add_argument("--export-manifest", required=True)
    propose.add_argument("--voice-id", default="suisui")
    propose.add_argument("--max-clips", type=int, default=12)
    propose.add_argument("--minimum-seconds", type=float, default=3.0)
    propose.add_argument("--maximum-seconds", type=float, default=10.0)
    propose.add_argument(
        "--output",
        default="data/reports/long_form/reference_pack_proposal_v1.json",
    )
    propose.set_defaults(func=cmd_propose)

    freeze = subparsers.add_parser(
        "freeze", help="Freeze human-confirmed clips into a versioned reference pack."
    )
    freeze.add_argument("--confirmations", required=True)
    freeze.add_argument(
        "--negatives",
        default=None,
        help="Optional negatives JSON exported from the negative scout page.",
    )
    freeze.add_argument("--audio-root", required=True)
    freeze.add_argument("--pack-id", default="long_form_reference_suisui")
    freeze.add_argument("--pack-version", type=int, default=1)
    freeze.add_argument("--voice-id", default="suisui")
    freeze.add_argument(
        "--output",
        default="configs/lab/long_form/reference_pack_suisui_v1.yaml",
    )
    freeze.set_defaults(func=cmd_freeze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
