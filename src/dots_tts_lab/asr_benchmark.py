from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import utc_now
from dots_tts_lab.reports import _atomic_write_text


DEFAULT_ASR_BENCHMARK_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "lab"
    / "asr"
    / "benchmark_v2.yaml"
)


class AsrBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    benchmark_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    benchmark_version: int = Field(ge=1)
    selection_parent_version: int | None = Field(default=None, ge=1)
    sample_count: int = Field(ge=1)
    truth_provenance_kind: Literal["user_confirmed_filename"]
    truth_provenance_note: str = Field(min_length=10)
    standardization_config_id: str
    standardization_config_version: int = Field(ge=1)
    emotion_targets: dict[str, int]
    include_all_quality_review: bool
    include_all_emotion: str
    normalization_unicode_form: Literal["NFC", "NFKC"]
    normalization_casefold: bool
    remove_unicode_categories: list[Literal["P", "Z"]]
    remove_characters: list[str] | None = None
    convert_traditional_to_simplified: Literal[False]
    normalize_numbers: Literal[False]

    @model_validator(mode="after")
    def validate_targets(self) -> AsrBenchmarkConfig:
        if sum(self.emotion_targets.values()) != self.sample_count:
            raise ValueError("emotion target counts must sum to sample_count")
        if any(count < 1 for count in self.emotion_targets.values()):
            raise ValueError("every emotion target must be positive")
        if self.include_all_emotion not in self.emotion_targets:
            raise ValueError("include_all_emotion must have an emotion target")
        if self.selection_parent_version == self.benchmark_version:
            raise ValueError("selection_parent_version must differ from benchmark_version")
        if len(set(self.remove_unicode_categories)) != len(
            self.remove_unicode_categories
        ):
            raise ValueError("remove_unicode_categories cannot contain duplicates")
        remove_characters = self.remove_characters or []
        if any(len(character) != 1 for character in remove_characters):
            raise ValueError("remove_characters entries must be single characters")
        if len(set(remove_characters)) != len(remove_characters):
            raise ValueError("remove_characters cannot contain duplicates")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_asr_benchmark_config(
    path: str | Path = DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
) -> AsrBenchmarkConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"ASR benchmark config must be a YAML mapping: {config_path}")
    return AsrBenchmarkConfig.model_validate(payload, strict=True)


def normalize_asr_text(text: str, config: AsrBenchmarkConfig) -> str:
    normalized = unicodedata.normalize(config.normalization_unicode_form, text)
    if config.normalization_casefold:
        normalized = normalized.casefold()
    removed = set(config.remove_unicode_categories)
    removed_characters = set(config.remove_characters or [])
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[:1] not in removed
        and character not in removed_characters
    )


def character_edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_report_assets(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ValueError(f"Report must contain an assets list: {path}")
    return {str(item["asset_sha256"]): item for item in payload["assets"]}


def _difficulty_scores(
    candidates: list[dict[str, Any]], config: AsrBenchmarkConfig
) -> dict[str, float]:
    normalized = {
        str(candidate["asset_sha256"]): normalize_asr_text(
            str(candidate["text_exact"]), config
        )
        for candidate in candidates
    }
    frequencies = Counter(character for text in normalized.values() for character in text)
    return {
        asset_sha256: len(text) * 0.01
        + sum(1.0 / frequencies[character] for character in text)
        for asset_sha256, text in normalized.items()
    }


def _select_items(
    candidates: list[dict[str, Any]],
    quality: dict[str, dict[str, Any]],
    config: AsrBenchmarkConfig,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, list[dict[str, Any]]] = {}

    def add(candidate: dict[str, Any], reason: dict[str, Any]) -> None:
        asset_sha256 = str(candidate["asset_sha256"])
        selected.setdefault(asset_sha256, candidate)
        reasons.setdefault(asset_sha256, []).append(reason)

    if config.include_all_quality_review:
        for candidate in candidates:
            quality_item = quality[str(candidate["asset_sha256"])]
            if quality_item.get("decision") == "review":
                add(
                    candidate,
                    {
                        "code": "quality_review",
                        "quality_reason_codes": [
                            reason["code"] for reason in quality_item.get("reasons", [])
                        ],
                    },
                )
    for candidate in candidates:
        if candidate["emotion_weak_label"] == config.include_all_emotion:
            add(candidate, {"code": "all_low_resource_emotion"})

    counts = Counter(
        candidate["emotion_weak_label"] for candidate in selected.values()
    )
    for emotion, count in counts.items():
        if count > config.emotion_targets[emotion]:
            raise RuntimeError(
                f"Required selections exceed target for {emotion}: "
                f"required={count}, target={config.emotion_targets[emotion]}"
            )

    difficulty = _difficulty_scores(candidates, config)
    for emotion, target in config.emotion_targets.items():
        ranked = sorted(
            (
                candidate
                for candidate in candidates
                if candidate["emotion_weak_label"] == emotion
            ),
            key=lambda candidate: (
                -difficulty[str(candidate["asset_sha256"])],
                str(candidate["asset_sha256"]),
            ),
        )
        for candidate in ranked:
            if counts[emotion] >= target:
                break
            asset_sha256 = str(candidate["asset_sha256"])
            if asset_sha256 in selected:
                continue
            add(
                candidate,
                {
                    "code": "linguistic_difficulty_fill",
                    "score": difficulty[asset_sha256],
                },
            )
            counts[emotion] += 1

    if len(selected) != config.sample_count:
        raise RuntimeError(
            f"Expected {config.sample_count} benchmark items, selected {len(selected)}"
        )
    emotion_order = {emotion: index for index, emotion in enumerate(config.emotion_targets)}
    ordered = sorted(
        selected.values(),
        key=lambda candidate: (
            emotion_order[str(candidate["emotion_weak_label"])],
            str(candidate["asset_sha256"]),
        ),
    )
    items = []
    for ordinal, candidate in enumerate(ordered, start=1):
        asset_sha256 = str(candidate["asset_sha256"])
        reference = str(candidate["text_exact"])
        items.append(
            {
                "ordinal": ordinal,
                "asset_sha256": asset_sha256,
                "source_relative_path": candidate["source_relative_path"],
                "derived_id": candidate["derived_id"],
                "derived_relative_path": candidate["derived_relative_path"],
                "derived_sha256": candidate["output_sha256"],
                "duration_seconds": candidate["derived_duration_seconds"],
                "speaker_id": candidate["speaker_id"],
                "emotion_weak_label": candidate["emotion_weak_label"],
                "reference_exact": reference,
                "reference_normalized": normalize_asr_text(reference, config),
                "text_sha256": candidate["text_sha256"],
                "selection_reasons": reasons[asset_sha256],
                "selection_reasons_json": json.dumps(
                    reasons[asset_sha256],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "source_quality": quality[asset_sha256],
            }
        )
    return items


def _inherit_selection(
    *,
    catalog: Catalog,
    candidates: list[dict[str, Any]],
    quality: dict[str, dict[str, Any]],
    config: AsrBenchmarkConfig,
) -> list[dict[str, Any]]:
    parent_version = config.selection_parent_version
    if parent_version is None:
        raise ValueError("selection_parent_version is required")
    parent = catalog.load_asr_benchmark(
        benchmark_id=config.benchmark_id,
        benchmark_version=parent_version,
    )
    if parent is None:
        raise RuntimeError(
            f"Selection parent is not registered: "
            f"{config.benchmark_id}@{parent_version}"
        )
    parent_manifest_path = Path(str(parent["manifest_path"])).resolve()
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(
            f"Selection parent manifest is missing: {parent_manifest_path}"
        )
    parent_bytes = parent_manifest_path.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != parent["manifest_sha256"]:
        raise RuntimeError("Selection parent manifest no longer matches its frozen hash")
    parent_items = [
        json.loads(line)
        for line in parent_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(parent_items) != config.sample_count:
        raise RuntimeError(
            f"Selection parent has {len(parent_items)} items, expected "
            f"{config.sample_count}"
        )
    current = {str(candidate["asset_sha256"]): candidate for candidate in candidates}
    inherited = []
    for parent_item in parent_items:
        asset_sha256 = str(parent_item["asset_sha256"])
        candidate = current.get(asset_sha256)
        if candidate is None:
            raise RuntimeError(
                f"Selection parent asset is unavailable in current candidates: "
                f"{asset_sha256}"
            )
        if (
            candidate["output_sha256"] != parent_item["derived_sha256"]
            or candidate["text_sha256"] != parent_item["text_sha256"]
        ):
            raise RuntimeError(
                f"Selection parent asset content drifted: {asset_sha256}"
            )
        reasons = parent_item["selection_reasons"]
        reference = str(candidate["text_exact"])
        inherited.append(
            {
                "ordinal": int(parent_item["ordinal"]),
                "asset_sha256": asset_sha256,
                "source_relative_path": candidate["source_relative_path"],
                "derived_id": candidate["derived_id"],
                "derived_relative_path": candidate["derived_relative_path"],
                "derived_sha256": candidate["output_sha256"],
                "duration_seconds": candidate["derived_duration_seconds"],
                "speaker_id": candidate["speaker_id"],
                "emotion_weak_label": candidate["emotion_weak_label"],
                "reference_exact": reference,
                "reference_normalized": normalize_asr_text(reference, config),
                "text_sha256": candidate["text_sha256"],
                "selection_reasons": reasons,
                "selection_reasons_json": json.dumps(
                    reasons,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "source_quality": quality[asset_sha256],
            }
        )
    return inherited


def _write_benchmark_reports(
    output_dir: Path, summary: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    benchmark_path = output_dir / "benchmark.json"
    csv_path = output_dir / "manifest.csv"
    html_path = output_dir / "benchmark.html"
    manifest_text = "".join(
        json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in items
    )
    _atomic_write_text(manifest_path, manifest_text)
    _atomic_write_text(
        benchmark_path,
        json.dumps(
            {"schema_version": 1, "summary": summary, "assets": items},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    csv_output = io.StringIO(newline="")
    fields = (
        "ordinal",
        "asset_sha256",
        "source_relative_path",
        "derived_relative_path",
        "duration_seconds",
        "emotion_weak_label",
        "reference_exact",
        "reference_normalized",
        "selection_reasons_json",
    )
    writer = csv.DictWriter(csv_output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(items)
    _atomic_write_text(csv_path, csv_output.getvalue())
    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{}</td></tr>".format(
            item["ordinal"],
            html.escape(str(item["emotion_weak_label"])),
            html.escape(str(item["reference_exact"])),
            html.escape(str(item["source_relative_path"])),
            float(item["duration_seconds"]),
            html.escape(item["selection_reasons_json"]),
        )
        for item in items
    )
    summary_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if not isinstance(value, (dict, list))
    )
    _atomic_write_text(
        html_path,
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>dots.tts ASR benchmark</title><style>
body {{ font:14px/1.5 system-ui,sans-serif; margin:2rem; color:#1f2937 }}
table {{ border-collapse:collapse; width:100%; margin-bottom:2rem }}
th,td {{ border:1px solid #d1d5db; padding:.4rem .55rem; text-align:left }}
th {{ background:#f3f4f6 }} code {{ overflow-wrap:anywhere }}
</style></head><body><h1>ASR benchmark</h1><h2>Summary</h2>
<table><tbody>{summary_rows}</tbody></table><h2>Assets</h2><table><thead>
<tr><th>#</th><th>Emotion</th><th>Exact reference</th><th>Source</th><th>Seconds</th><th>Selection</th></tr>
</thead><tbody>{rows}</tbody></table></body></html>""",
    )
    return {
        "manifest_jsonl": str(manifest_path),
        "benchmark_json": str(benchmark_path),
        "manifest_csv": str(csv_path),
        "benchmark_html": str(html_path),
    }


def prepare_asr_benchmark(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    config_path: str | Path = DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    quality_report_path: str | Path = "data/reports/quality/quality.json",
    output_dir: str | Path = "data/benchmarks/asr/fuxuan_v2",
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    resolved_quality_path = Path(quality_report_path).resolve()
    config = load_asr_benchmark_config(resolved_config_path)
    catalog = Catalog(catalog_path)
    catalog.initialize()
    now = utc_now()
    raw_candidates = catalog.load_standardization_candidates()
    truth_records = []
    for candidate in raw_candidates:
        text = candidate.get("transcript_candidate")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                f"Missing filename transcript for asset {candidate['asset_sha256']}"
            )
        truth_records.append(
            {
                "asset_sha256": candidate["asset_sha256"],
                "text_exact": text,
                "text_sha256": _text_sha256(text),
                "provenance_kind": config.truth_provenance_kind,
                "provenance_note": config.truth_provenance_note,
                "confirmed_at": now,
            }
        )
    registered_count, cached_truth_count = catalog.register_filename_ground_truths(
        truth_records
    )
    candidates = catalog.load_asr_preparation_candidates(
        standardization_config_id=config.standardization_config_id,
        standardization_config_version=config.standardization_config_version,
    )
    if len(candidates) != len(raw_candidates):
        raise RuntimeError(
            "Every raw asset must have the selected standardized artifact before "
            f"benchmark preparation: raw={len(raw_candidates)}, derived={len(candidates)}"
        )
    quality = _load_report_assets(resolved_quality_path)
    missing_quality = {str(item["asset_sha256"]) for item in candidates} - quality.keys()
    if missing_quality:
        raise RuntimeError(f"Quality report is missing {len(missing_quality)} assets")
    items = (
        _inherit_selection(
            catalog=catalog,
            candidates=candidates,
            quality=quality,
            config=config,
        )
        if config.selection_parent_version is not None
        else _select_items(candidates, quality, config)
    )
    manifest_payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in items
    )
    manifest_sha256 = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()
    emotion_counts = dict(Counter(item["emotion_weak_label"] for item in items))
    quality_review_count = sum(
        item["source_quality"].get("decision") == "review" for item in items
    )
    summary = {
        "status": "succeeded",
        "benchmark_id": config.benchmark_id,
        "benchmark_version": config.benchmark_version,
        "selection_parent_version": config.selection_parent_version,
        "config_sha256": config.config_sha256(),
        "manifest_sha256": manifest_sha256,
        "created_at": now,
        "ground_truth_total_count": len(truth_records),
        "ground_truth_registered_count": registered_count,
        "ground_truth_cached_count": cached_truth_count,
        "sample_count": len(items),
        "emotion_counts": emotion_counts,
        "quality_review_count": quality_review_count,
        "total_audio_seconds": sum(float(item["duration_seconds"]) for item in items),
        "normalization": {
            "unicode_form": config.normalization_unicode_form,
            "casefold": config.normalization_casefold,
            "remove_unicode_categories": list(config.remove_unicode_categories),
            "remove_characters": list(config.remove_characters or []),
            "traditional_to_simplified": config.convert_traditional_to_simplified,
            "normalize_numbers": config.normalize_numbers,
        },
    }
    catalog.assert_asr_benchmark_compatible(
        benchmark_id=config.benchmark_id,
        benchmark_version=config.benchmark_version,
        config_sha256=config.config_sha256(),
        manifest_sha256=manifest_sha256,
    )
    output_root = Path(output_dir).resolve()
    reports = _write_benchmark_reports(output_root, summary, items)
    catalog.register_asr_benchmark(
        benchmark={
            "benchmark_id": config.benchmark_id,
            "benchmark_version": config.benchmark_version,
            "config_sha256": config.config_sha256(),
            "config_json": config.canonical_json(),
            "manifest_path": reports["manifest_jsonl"],
            "manifest_sha256": manifest_sha256,
            "created_at": now,
        },
        items=items,
    )
    summary["reports"] = reports
    summary["catalog_path"] = str(Path(catalog_path).resolve())
    summary["config_path"] = str(resolved_config_path)
    summary["quality_report_path"] = str(resolved_quality_path)
    return summary
