from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dots_tts_lab.asr_benchmark import (
    DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    AsrBenchmarkConfig,
    character_edit_distance,
    load_asr_benchmark_config,
    normalize_asr_text,
)
from dots_tts_lab.asr_comparison import (
    DEFAULT_ASR_COMPARISON_CONFIG_PATH,
    DEFAULT_ASR_LEXICON_PATH,
    load_asr_comparison_config,
    load_asr_lexicon,
)
from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import utc_now
from dots_tts_lab.reports import _atomic_write_text


_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "lab" / "asr"
DEFAULT_REVIEW_RULES_PATH = _CONFIG_ROOT / "review_rules_v1.yaml"

# The page sits at data/reports/asr/review/ and audio at
# data/work/standardized/<prefix>/<id>.wav; this is the hop count between them.
_AUDIO_REL_HOPS = "../../../work/standardized"

TEXT_DECISIONS = ("accept_reference", "accept_edited", "flag_defect")
REVIEW_STATUSES = ("approved", "rejected", "pending")
INTENSITIES = ("low", "medium", "high")
EMOTION_CHOICES = ("中立_neutral", "开心_happy", "生气_angry", "难过_sad")


class ReviewRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    rules_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    rules_version: int = Field(ge=1)
    homophone_pairs: list[list[str]] = Field(min_length=1)
    interjection_prefixes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rules(self) -> ReviewRulesConfig:
        # YAML gives lists; pydantic strict mode refuses tuple coercion, so pairs
        # arrive as two-element lists and are checked here.
        for pair in self.homophone_pairs:
            if len(pair) != 2:
                raise ValueError(f"A homophone pair needs exactly two terms: {pair!r}")
            if pair[0] == pair[1]:
                raise ValueError("A homophone pair cannot repeat itself")
            if not pair[0].strip() or not pair[1].strip():
                raise ValueError("A homophone pair cannot contain blank terms")
        left = [pair[0] for pair in self.homophone_pairs]
        right = [pair[1] for pair in self.homophone_pairs]
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            raise ValueError("Homophone pair terms must be unique per side")
        if set(left) & set(right):
            raise ValueError(
                "A term cannot appear on both sides of homophone pairs: "
                f"{sorted(set(left) & set(right))}"
            )
        if len(set(self.interjection_prefixes)) != len(self.interjection_prefixes):
            raise ValueError("interjection_prefixes cannot contain duplicates")
        for prefix in self.interjection_prefixes:
            if len(prefix) != 1:
                raise ValueError(f"Interjection prefix must be one character: {prefix!r}")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_review_rules(
    path: str | Path = DEFAULT_REVIEW_RULES_PATH,
) -> ReviewRulesConfig:
    rules_path = Path(path).resolve()
    payload = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Review rules must be a YAML mapping: {rules_path}")
    return ReviewRulesConfig.model_validate(payload, strict=True)


def diff_spans(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    """Character diff as spans, precomputed so the page renders without logic.

    Returns aligned spans: equal runs carry `text`; differing runs carry what
    each side has (`reference` and/or `hypothesis`, absent when that side is
    empty). A Python-side pass keeps the static page a pure view layer, per the
    project rule that UI never owns business logic.
    """
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    matrix = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for column in range(columns):
        matrix[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1]
                + (reference[row - 1] != hypothesis[column - 1]),
            )
    spans: list[dict[str, Any]] = []
    row, column = rows - 1, columns - 1
    while row > 0 or column > 0:
        if (
            row > 0
            and column > 0
            and reference[row - 1] == hypothesis[column - 1]
            and matrix[row][column] == matrix[row - 1][column - 1]
        ):
            _append_span(spans, True, text=reference[row - 1])
            row -= 1
            column -= 1
        elif (
            row > 0 and matrix[row][column] == matrix[row - 1][column] + 1
        ):
            _append_span(spans, False, reference=reference[row - 1])
            row -= 1
        else:
            _append_span(spans, False, hypothesis=hypothesis[column - 1])
            column -= 1
    return spans


def _append_span(
    spans: list[dict[str, Any]], equal: bool, **parts: str
) -> None:
    """Append one character to the span list, merging with the current run.

    The DP backtrack walks backwards, so spans are prepended in run order; the
    caller reverses once at the end. Runs are merged so a
    one-character-per-step backtrack yields contiguous spans the page can
    render without post-processing.
    """
    if spans and spans[0].get("equal") == equal:
        previous = spans[0]
        for key, value in parts.items():
            previous[key] = value + previous.get(key, "")
        return
    span: dict[str, Any] = {"equal": equal}
    span.update(parts)
    spans.insert(0, span)


def homophone_only_difference(
    reference_normalized: str,
    hypotheses_normalized: dict[str, str],
    rules: ReviewRulesConfig,
) -> list[dict[str, str]]:
    """Rules the whole backend panel triggered, keyed by pair.

    A sample is auto-resolvable only when every differing backend's distance to
    the reference is fully explained by the configured homophone pairs — swap
    the variant for the reference spelling and the texts become identical. Any
    residual difference means a human must look.
    """
    triggered: dict[tuple[str, str], set[str]] = {}
    for backend, hypothesis in hypotheses_normalized.items():
        residual = hypothesis
        matched: list[tuple[str, str]] = []
        for reference_form, variant in rules.homophone_pairs:
            if variant in residual:
                residual = residual.replace(variant, reference_form)
                matched.append((reference_form, variant))
        if residual != reference_normalized:
            return []
        for pair in matched:
            triggered.setdefault(pair, set()).add(backend)
    return [
        {
            "reference": reference_form,
            "variant": variant,
            "backends": sorted(backends),
        }
        for (reference_form, variant), backends in sorted(triggered.items())
    ]


def interjection_hint(
    reference_normalized: str,
    hypotheses_normalized: dict[str, str],
    rules: ReviewRulesConfig,
) -> dict[str, Any] | None:
    """Detect a leading interjection every backend heard but the reference lacks.

    Unlike homophones this carries no decision: whether the sound exists must be
    judged by listening, so it only annotates the sample for attention.
    """
    if any(
        reference_normalized.startswith(prefix)
        for prefix in rules.interjection_prefixes
    ):
        return None
    heard: dict[str, str] = {}
    for backend, hypothesis in hypotheses_normalized.items():
        for prefix in rules.interjection_prefixes:
            if hypothesis.startswith(prefix) and not reference_normalized.startswith(
                prefix
            ):
                heard[backend] = prefix
                break
    if not hypotheses_normalized or len(heard) != len(hypotheses_normalized):
        return None
    return {
        "interjections": sorted(set(heard.values())),
        "backends": {backend: prefix for backend, prefix in heard.items()},
    }


def _relative_audio_path(derived_relative_path: str) -> str:
    return f"{_AUDIO_REL_HOPS}/{derived_relative_path}".replace(os.sep, "/")


def review_export(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    benchmark_config_path: str | Path = DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    comparison_report_path: str | Path = "data/reports/asr/comparison/comparison.json",
    comparison_config_path: str | Path = DEFAULT_ASR_COMPARISON_CONFIG_PATH,
    lexicon_path: str | Path = DEFAULT_ASR_LEXICON_PATH,
    rules_path: str | Path = DEFAULT_REVIEW_RULES_PATH,
    output_dir: str | Path = "data/reports/asr/review",
) -> dict[str, Any]:
    benchmark_config = load_asr_benchmark_config(benchmark_config_path)
    comparison_config = load_asr_comparison_config(comparison_config_path)
    lexicon = load_asr_lexicon(lexicon_path)
    rules = load_review_rules(rules_path)
    comparison = json.loads(
        Path(comparison_report_path).resolve().read_text(encoding="utf-8")
    )
    if comparison.get("schema_version") != 1:
        raise ValueError("Unsupported comparison report schema_version")
    summary = comparison.get("summary") or {}
    if (
        summary.get("benchmark_id") != benchmark_config.benchmark_id
        or summary.get("benchmark_version") != benchmark_config.benchmark_version
    ):
        raise ValueError(
            "Comparison report was built for a different benchmark: "
            f"{summary.get('benchmark_id')}@{summary.get('benchmark_version')}"
        )
    expected_summary = {
        "comparison_id": comparison_config.comparison_id,
        "comparison_version": comparison_config.comparison_version,
        "comparison_config_sha256": comparison_config.config_sha256(),
        "lexicon_id": lexicon.lexicon_id,
        "lexicon_version": lexicon.lexicon_version,
        "lexicon_sha256": lexicon.config_sha256(),
    }
    mismatched_summary = [
        key for key, value in expected_summary.items() if summary.get(key) != value
    ]
    if mismatched_summary:
        raise ValueError(
            "Comparison report config provenance mismatch: "
            + ", ".join(mismatched_summary)
        )
    recommended = summary.get("recommended_backend")
    queue_entries = comparison.get("review_queue", [])
    if not isinstance(queue_entries, list):
        raise ValueError("Comparison review_queue must be a list")
    queue_ordinals = [int(entry["ordinal"]) for entry in queue_entries]
    if len(set(queue_ordinals)) != len(queue_ordinals):
        raise ValueError("Comparison review_queue contains duplicate ordinals")
    queue_by_ordinal = {
        int(entry["ordinal"]): entry for entry in queue_entries
    }
    catalog = Catalog(catalog_path)
    catalog.initialize()
    benchmark_record = catalog.load_asr_benchmark(
        benchmark_id=benchmark_config.benchmark_id,
        benchmark_version=benchmark_config.benchmark_version,
    )
    if benchmark_record is None:
        raise ValueError("Benchmark is not registered in the catalog")
    expected_benchmark = {
        "benchmark_config_sha256": benchmark_config.config_sha256(),
        "benchmark_manifest_sha256": benchmark_record["manifest_sha256"],
    }
    mismatched_benchmark = [
        key for key, value in expected_benchmark.items() if summary.get(key) != value
    ]
    if benchmark_record["config_sha256"] != benchmark_config.config_sha256():
        mismatched_benchmark.append("catalog benchmark config_sha256")
    if mismatched_benchmark:
        raise ValueError(
            "Comparison report benchmark provenance mismatch: "
            + ", ".join(mismatched_benchmark)
        )
    items = catalog.load_asr_benchmark_items(
        benchmark_id=benchmark_config.benchmark_id,
        benchmark_version=benchmark_config.benchmark_version,
    )
    item_by_ordinal = {int(item["ordinal"]): item for item in items}
    expected_assets = {str(item["asset_sha256"]) for item in items}
    if summary.get("sample_count") != len(items):
        raise ValueError("Comparison sample_count does not match benchmark items")
    for entry in queue_entries:
        ordinal = int(entry["ordinal"])
        item = item_by_ordinal.get(ordinal)
        if item is None:
            raise ValueError(
                f"Comparison review_queue ordinal is outside benchmark: {ordinal}"
            )
        expected_reference = str(item["text_exact"])
        if (
            entry.get("reference_exact") != expected_reference
            or entry.get("reference_normalized")
            != normalize_asr_text(expected_reference, benchmark_config)
        ):
            raise ValueError(
                f"Comparison review_queue reference mismatch at ordinal {ordinal}"
            )
    backend_entries = comparison.get("backends")
    if not isinstance(backend_entries, list) or not backend_entries:
        raise ValueError("Comparison report has no backend run provenance")
    runs = []
    seen_backend_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    for entry in backend_entries:
        if not isinstance(entry, dict):
            raise ValueError("Comparison backend provenance must be an object")
        backend_id = str(entry.get("backend_id") or "")
        run_id = str(entry.get("run_id") or "")
        if not backend_id or not run_id:
            raise ValueError("Comparison backend provenance needs backend_id and run_id")
        if backend_id in seen_backend_ids or run_id in seen_run_ids:
            raise ValueError("Comparison backend provenance contains duplicates")
        seen_backend_ids.add(backend_id)
        seen_run_ids.add(run_id)
        run = catalog.get_asr_run(run_id=run_id)
        if run is None:
            raise ValueError(f"Comparison ASR run is missing from catalog: {run_id}")
        if (
            run["benchmark_id"] != benchmark_config.benchmark_id
            or run["benchmark_version"] != benchmark_config.benchmark_version
            or run["backend_id"] != backend_id
            or run["status"] != "succeeded"
        ):
            raise ValueError(
                "Comparison ASR run provenance does not match catalog: "
                f"{backend_id}/{run_id}"
            )
        runs.append(run)
    if summary.get("backend_count") != len(runs):
        raise ValueError("Comparison backend_count does not match backend provenance")
    if recommended not in seen_backend_ids:
        raise ValueError("Comparison recommended_backend is not in backend provenance")
    results_by_asset: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        results = catalog.load_asr_results(run_id=str(run["run_id"]))
        result_assets = {str(result["asset_sha256"]) for result in results}
        if result_assets != expected_assets or any(
            result["status"] != "ok" for result in results
        ):
            raise ValueError(
                "Comparison ASR run results do not match the complete benchmark: "
                f"{run['backend_id']}/{run['run_id']}"
            )
        for result in results:
            results_by_asset.setdefault(str(result["asset_sha256"]), {})[
                str(run["backend_id"])
            ] = result
    samples = []
    for item in items:
        ordinal = int(item["ordinal"])
        asset_sha256 = str(item["asset_sha256"])
        per_asset = results_by_asset.get(asset_sha256, {})
        queue_entry = queue_by_ordinal.get(ordinal)
        reference_exact = str(item["text_exact"])
        reference_normalized = normalize_asr_text(reference_exact, benchmark_config)
        hypotheses_normalized = {
            backend: str(result.get("hypothesis_normalized") or "")
            for backend, result in per_asset.items()
        }
        auto_rules = homophone_only_difference(
            reference_normalized, hypotheses_normalized, rules
        )
        interjection = interjection_hint(
            reference_normalized, hypotheses_normalized, rules
        )
        weak_label = str(item["emotion_weak_label"])
        samples.append(
            {
                "ordinal": ordinal,
                "asset_sha256": asset_sha256,
                "audio_rel_path": _relative_audio_path(
                    str(item["derived_relative_path"])
                ),
                "duration_seconds": item["duration_seconds"],
                "weak_label": weak_label,
                "priority": queue_entry["priority"] if queue_entry else None,
                "flags": queue_entry["flags"] if queue_entry else [],
                "reference_exact": reference_exact,
                "reference_normalized": reference_normalized,
                "backends": {
                    backend: {
                        "hypothesis_raw": str(
                            result.get("hypothesis_raw") or ""
                        ),
                        "run_id": str(result["run_id"]),
                        "cer": result.get("cer"),
                        "selected": backend == recommended,
                    }
                    for backend, result in sorted(per_asset.items())
                },
                "selected_backend": recommended,
                "diff_spans": diff_spans(
                    reference_exact,
                    str(
                        per_asset.get(str(recommended), {}).get("hypothesis_raw")
                        or reference_exact
                    ),
                )
                if recommended in per_asset
                else [],
                "auto_rules_applied": auto_rules,
                "interjection_hint": interjection,
            }
        )
    payload = json.dumps(samples, ensure_ascii=False, sort_keys=True)
    package_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_path = output / "review.json"
    page_path = output / "review.html"
    _atomic_write_text(package_path, payload + "\n")
    summary_out = {
        "status": "succeeded",
        "benchmark_id": benchmark_config.benchmark_id,
        "benchmark_version": benchmark_config.benchmark_version,
        "package_sha256": package_sha256,
        "rules_id": rules.rules_id,
        "rules_version": rules.rules_version,
        "rules_sha256": rules.config_sha256(),
        "recommended_backend": recommended,
        "asr_run_ids": {
            str(run["backend_id"]): str(run["run_id"]) for run in runs
        },
        "sample_count": len(samples),
        "queued_count": sum(1 for sample in samples if sample["priority"]),
        "auto_resolved_count": sum(
            1 for sample in samples if sample["auto_rules_applied"]
        ),
        "interjection_hint_count": sum(
            1 for sample in samples if sample["interjection_hint"]
        ),
        "created_at": utc_now(),
        "catalog_path": str(Path(catalog_path).resolve()),
    }
    _atomic_write_text(
        page_path,
        _review_page(
            benchmark_id=benchmark_config.benchmark_id,
            benchmark_version=benchmark_config.benchmark_version,
            package_sha256=package_sha256,
            samples_json=payload,
        ),
    )
    _atomic_write_text(
        output / "export.json",
        json.dumps(summary_out, ensure_ascii=False, indent=2) + "\n",
    )
    summary_out["reports"] = {
        "page": str(page_path),
        "package": str(package_path),
        "summary": str(output / "export.json"),
    }
    return summary_out


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_decision: Literal[
        "accept_reference", "accept_edited", "flag_defect"
    ]
    text_final: str | None
    text_note: str | None = None
    emotion_primary: Literal[
        "中立_neutral", "开心_happy", "生气_angry", "难过_sad"
    ] | None = None
    emotion_secondary: Literal[
        "中立_neutral", "开心_happy", "生气_angry", "难过_sad"
    ] | None = None
    intensity: Literal["low", "medium", "high"] | None = None
    review_status: Literal["approved", "rejected", "pending"]
    auto_rules_applied: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> ReviewDecision:
        if self.text_decision == "flag_defect":
            if self.text_final is not None:
                raise ValueError("flag_defect requires text_final to be null")
            if self.review_status == "approved":
                raise ValueError("flag_defect cannot have approved review_status")
            if self.text_note is None or not self.text_note.strip():
                raise ValueError("flag_defect requires a non-empty text_note")
        elif self.text_final is None or not self.text_final.strip():
            raise ValueError(f"text_final is required for {self.text_decision}")
        if self.emotion_secondary is not None and self.emotion_secondary == self.emotion_primary:
            raise ValueError("emotion_secondary must differ from emotion_primary")
        return self


class ReviewDecisionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    benchmark_id: str
    benchmark_version: int
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_batch_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    decided_at: str
    decisions: list[ReviewDecision] = Field(min_length=1)

    @field_validator("export_batch_id")
    @classmethod
    def validate_export_batch_id(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except ValueError as error:
            raise ValueError("export_batch_id must be a valid UUID") from error
        if str(parsed) != value:
            raise ValueError("export_batch_id must use canonical lowercase UUID form")
        return value

    @field_validator("decided_at")
    @classmethod
    def validate_decided_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("decided_at must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("decided_at must include a UTC offset")
        return value

    @model_validator(mode="after")
    def validate_batch(self) -> ReviewDecisionBatch:
        assets = [decision.asset_sha256 for decision in self.decisions]
        if len(set(assets)) != len(assets):
            duplicates = sorted(
                {asset for asset in assets if assets.count(asset) > 1}
            )
            raise ValueError(f"Duplicate decisions in batch: {duplicates}")
        return self


def review_import(
    *,
    decisions_path: str | Path,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    benchmark_config_path: str | Path = DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    review_dir: str | Path = "data/reports/asr/review",
) -> dict[str, Any]:
    benchmark_config = load_asr_benchmark_config(benchmark_config_path)
    decisions_path = Path(decisions_path).resolve()
    raw = json.loads(decisions_path.read_text(encoding="utf-8"))
    batch = ReviewDecisionBatch.model_validate(raw, strict=True)
    if (
        batch.benchmark_id != benchmark_config.benchmark_id
        or batch.benchmark_version != benchmark_config.benchmark_version
    ):
        raise ValueError(
            "Decision batch targets a different benchmark: "
            f"{batch.benchmark_id}@{batch.benchmark_version}"
        )
    catalog = Catalog(catalog_path)
    catalog.initialize()
    package_path = Path(review_dir).resolve() / "review.json"
    if not package_path.is_file():
        raise FileNotFoundError(
            f"Review package missing; re-run review-export first: {package_path}"
        )
    package_payload = package_path.read_text(encoding="utf-8").strip()
    package_sha256 = hashlib.sha256(
        package_payload.encode("utf-8")
    ).hexdigest()
    if package_sha256 != batch.package_sha256:
        raise ValueError(
            "Decision batch was exported from a different review package; "
            "re-run review-export and redo the review on the fresh page"
        )
    package = json.loads(package_payload)
    package_by_asset = {
        str(sample["asset_sha256"]): sample for sample in package
    }
    package_assets = set(package_by_asset)
    unknown = {decision.asset_sha256 for decision in batch.decisions} - package_assets
    if unknown:
        raise ValueError(
            f"{len(unknown)} decisions reference assets outside the benchmark"
        )
    for decision in batch.decisions:
        package_sample = package_by_asset[decision.asset_sha256]
        if (
            decision.text_decision == "accept_reference"
            and decision.text_final
            != str(package_sample["reference_exact"])
        ):
            raise ValueError(
                "accept_reference text_final must exactly match the frozen "
                f"reference for asset {decision.asset_sha256}"
            )
        if decision.auto_rules_applied != package_sample.get(
            "auto_rules_applied", []
        ):
            raise ValueError(
                "auto_rules_applied does not match the frozen review package "
                f"for asset {decision.asset_sha256}"
            )
    items = catalog.load_asr_benchmark_items(
        benchmark_id=benchmark_config.benchmark_id,
        benchmark_version=benchmark_config.benchmark_version,
    )
    weak_labels = {str(item["asset_sha256"]): str(item["emotion_weak_label"]) for item in items}
    rows = []
    for index, decision in enumerate(batch.decisions):
        rows.append(
            {
                "decision_id": str(uuid.uuid4()),
                "asset_sha256": decision.asset_sha256,
                "benchmark_id": batch.benchmark_id,
                "benchmark_version": batch.benchmark_version,
                "text_decision": decision.text_decision,
                "text_final": decision.text_final,
                "text_note": decision.text_note,
                "emotion_primary": decision.emotion_primary or weak_labels[decision.asset_sha256],
                "emotion_secondary": decision.emotion_secondary,
                "intensity": decision.intensity,
                "label_source": "human_corrected"
                if _emotion_changed(decision, weak_labels[decision.asset_sha256])
                else "weak_label_confirmed",
                "review_status": decision.review_status,
                "auto_rules_applied_json": json.dumps(
                    decision.auto_rules_applied,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "created_at": batch.decided_at,
                "export_batch_id": batch.export_batch_id,
                "source_row_index": index,
            }
        )
    inserted, ignored = catalog.record_review_decisions(rows)
    persisted_rows = catalog.load_review_decisions_by_batch(
        export_batch_id=batch.export_batch_id
    )
    if len(persisted_rows) != len(batch.decisions):
        raise RuntimeError(
            "Persisted review batch row count does not match validated input"
        )
    summary = _decision_report(
        catalog=catalog,
        benchmark_config=benchmark_config,
        batch=batch,
        rows=persisted_rows,
        inserted=inserted,
        ignored=ignored,
        report_dir=Path(review_dir).resolve(),
    )
    summary["decisions_path"] = str(decisions_path)
    summary["catalog_path"] = str(Path(catalog_path).resolve())
    return summary


def _emotion_changed(decision: ReviewDecision, weak_label: str) -> bool:
    return (
        decision.emotion_primary is not None and decision.emotion_primary != weak_label
    ) or decision.emotion_secondary is not None or decision.intensity is not None


def _decision_report(
    *,
    catalog: Catalog,
    benchmark_config: AsrBenchmarkConfig,
    batch: ReviewDecisionBatch,
    rows: list[dict[str, Any]],
    inserted: int,
    ignored: int,
    report_dir: Path,
) -> dict[str, Any]:
    items = catalog.load_asr_benchmark_items(
        benchmark_id=benchmark_config.benchmark_id,
        benchmark_version=benchmark_config.benchmark_version,
    )
    by_asset = {str(item["asset_sha256"]): item for item in items}
    latest = {
        str(row["asset_sha256"]): row
        for row in catalog.load_latest_review_decisions(
            benchmark_id=benchmark_config.benchmark_id,
            benchmark_version=benchmark_config.benchmark_version,
        )
    }
    lines = []
    rejected = []
    emotion_corrections = 0
    text_corrections = 0
    for row in rows:
        item = by_asset[row["asset_sha256"]]
        original = str(item["text_exact"])
        final = row.get("text_final")
        text_changed = final is not None and final != original
        if text_changed:
            text_corrections += 1
        if row["label_source"] == "human_corrected":
            emotion_corrections += 1
        if row["review_status"] == "rejected":
            rejected.append(
                {
                    "ordinal": item["ordinal"],
                    "asset_sha256": row["asset_sha256"],
                    "reason": row.get("text_note"),
                }
            )
        lines.append(
            {
                "ordinal": item["ordinal"],
                "asset_sha256": row["asset_sha256"],
                "review_round": row["review_round"],
                "text_decision": row["text_decision"],
                "text_original": original,
                "text_final": final,
                "text_changed": text_changed,
                "weak_label": str(item["emotion_weak_label"]),
                "emotion_primary": row.get("emotion_primary"),
                "emotion_secondary": row.get("emotion_secondary"),
                "intensity": row.get("intensity"),
                "label_source": row["label_source"],
                "review_status": row["review_status"],
                "text_note": row.get("text_note"),
                "auto_rules_applied": json.loads(row["auto_rules_applied_json"]),
            }
        )
    lines.sort(key=lambda line: line["ordinal"])
    summary = {
        "status": "succeeded",
        "benchmark_id": benchmark_config.benchmark_id,
        "benchmark_version": benchmark_config.benchmark_version,
        "export_batch_id": batch.export_batch_id,
        "decided_at": batch.decided_at,
        "decision_count": len(rows),
        "inserted": inserted,
        "ignored": ignored,
        "text_corrected_count": text_corrections,
        "emotion_corrected_count": emotion_corrections,
        "approved_count": sum(
            1 for row in rows if row["review_status"] == "approved"
        ),
        "rejected_count": sum(
            1 for row in rows if row["review_status"] == "rejected"
        ),
        "pending_count": sum(
            1 for row in rows if row["review_status"] == "pending"
        ),
        "total_decided_assets": len(latest),
        "rejected_assets": rejected,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        report_dir / "decision_report.json",
        json.dumps(
            {"schema_version": 1, "summary": summary, "decisions": lines},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    if lines:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
        _atomic_write_text(report_dir / "decision_report.csv", output.getvalue())
    _atomic_write_text(
        report_dir / "decision_report.html",
        _decision_report_html(summary, lines),
    )
    summary["reports"] = {
        "json": str(report_dir / "decision_report.json"),
        "csv": str(report_dir / "decision_report.csv"),
        "html": str(report_dir / "decision_report.html"),
    }
    return summary


def _decision_report_html(
    summary: dict[str, Any], lines: list[dict[str, Any]]
) -> str:
    summary_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if not isinstance(value, (dict, list))
    )
    # Row cells match the 9 header columns; class carries review_status.
    rows = "\n".join(
        '<tr class="{}"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>'
        "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(line["review_status"])),          # class
            line["ordinal"],                                   # #
            html.escape(str(line["text_decision"])),           # Text decision
            html.escape(str(line["text_original"])),           # Original
            html.escape(str(line["text_final"] or "")),        # Final
            "yes" if line["text_changed"] else "",             # Changed
            html.escape(str(line["weak_label"])),              # Weak label
            html.escape(str(line["emotion_primary"] or "")),   # Primary
            html.escape(str(line["emotion_secondary"] or "")), # Secondary
            html.escape(str(line["review_status"])),            # Status
        )
        for line in lines
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>dots.tts review decisions</title><style>
body {{ font:14px/1.55 system-ui,sans-serif; margin:2rem; color:#1f2937 }}
table {{ border-collapse:collapse; width:100%; margin-bottom:2rem }}
th,td {{ border:1px solid #d1d5db; padding:.4rem .55rem; text-align:left;
         vertical-align:top }}
th {{ background:#f3f4f6 }}
tr.rejected td {{ background:#fef2f2 }}
tr.pending td {{ background:#fffbeb }}
</style></head><body><h1>Review decisions</h1>
<h2>Summary</h2><table><tbody>{summary_rows}</tbody></table>
<h2>Decisions</h2><table><thead>
<tr><th>#</th><th>Text decision</th><th>Original</th><th>Final</th>
<th>Changed</th><th>Weak label</th><th>Primary</th><th>Secondary</th>
<th>Status</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>dots.tts transcript review</title>
<style>
:root {{ --ink:#1f2937; --line:#d1d5db; --muted:#6b7280; --paper:#f9fafb }}
* {{ box-sizing:border-box }}
body {{ font:15px/1.6 system-ui,sans-serif; margin:0; color:var(--ink) }}
header {{ padding:.8rem 1.2rem; border-bottom:1px solid var(--line);
        background:var(--paper); position:sticky; top:0; z-index:2 }}
header h1 {{ margin:0; font-size:1.1rem }}
header .meta {{ color:var(--muted); font-size:.85rem }}
main {{ display:flex; min-height:calc(100vh - 64px) }}
nav {{ width:300px; flex:none; border-right:1px solid var(--line);
      overflow-y:auto; max-height:calc(100vh - 64px); position:sticky;
      top:64px }}
nav .group {{ padding:.5rem .8rem; font-size:.8rem; color:var(--muted);
             border-bottom:1px solid var(--line); background:var(--paper) }}
nav button {{ display:block; width:100%; text-align:left; padding:.45rem .8rem;
             border:0; border-bottom:1px solid #f3f4f6; background:#fff;
             cursor:pointer; font:inherit; color:inherit }}
nav button.active {{ background:#eef2ff; font-weight:600 }}
nav button .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
                  margin-right:.5rem; vertical-align:baseline }}
nav button .dot.undone {{ background:#d1d5db }}
nav button .dot.done {{ background:#16a34a }}
nav button .dot.pending {{ background:#f59e0b }}
nav button .dot.rejected {{ background:#dc2626 }}
nav button .ord {{ color:var(--muted); font-size:.8rem }}
section {{ flex:1; padding:1.2rem 1.6rem; max-width:880px }}
.audio {{ margin-bottom:1rem }}
.audio audio {{ width:100% }}
.badge {{ display:inline-block; padding:.05rem .5rem; border-radius:999px;
         font-size:.75rem; border:1px solid var(--line); margin-left:.4rem }}
.badge.high {{ background:#fef3c7; border-color:#f59e0b }}
.badge.medium {{ background:#f0f9ff; border-color:#38bdf8 }}
.badge.low {{ background:#f8fafc }}
.rule-note {{ background:#f0fdf4; border:1px solid #86efac; padding:.6rem .9rem;
             border-radius:6px; margin:.6rem 0; font-size:.9rem }}
.listen-note {{ background:#fffbeb; border:1px solid #fcd34d; padding:.6rem .9rem;
               border-radius:6px; margin:.6rem 0; font-size:.9rem }}
h2 {{ font-size:1rem; margin:1.2rem 0 .4rem; color:#374151 }}
.reference {{ font-size:1.25rem; line-height:1.9; background:var(--paper);
             padding:.8rem 1rem; border-radius:6px; border:1px solid var(--line) }}
.hyp {{ margin:.4rem 0; padding:.5rem .8rem; background:#fff;
       border:1px solid var(--line); border-radius:6px }}
.hyp .who {{ color:var(--muted); font-size:.8rem }}
.hyp .cer {{ color:var(--muted); font-size:.8rem; margin-left:.6rem }}
.hyp ins {{ background:#fee2e2; text-decoration:none }}
.hyp del {{ background:#f3f4f6; color:var(--muted) }}
.hyp.selected {{ border-color:#6366f1 }}
details.other {{ margin:.4rem 0 }}
details.other .hyp {{ border-style:dashed }}
textarea, select {{ font:inherit; padding:.4rem .6rem; width:100%;
                   border:1px solid var(--line); border-radius:6px }}
textarea {{ min-height:3.2rem; resize:vertical }}
fieldset {{ border:1px solid var(--line); border-radius:8px; margin:.8rem 0;
           padding:.7rem 1rem }}
fieldset legend {{ font-size:.85rem; color:#374151; padding:0 .4rem }}
.radios label {{ display:inline-block; margin-right:1.2rem; cursor:pointer }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:.8rem }}
.actions {{ position:sticky; bottom:0; background:#fff; border-top:1px solid var(--line);
           padding:.8rem 0; display:flex; gap:.6rem; align-items:center }}
.actions button {{ font:inherit; padding:.45rem 1rem; border-radius:6px;
                  border:1px solid var(--line); background:#fff; cursor:pointer }}
.actions button.primary {{ background:#4f46e5; color:#fff; border-color:#4f46e5 }}
.save-status {{ min-width:14rem; font-size:.85rem; color:var(--muted) }}
.save-status.saved {{ color:#15803d; font-weight:600 }}
.save-status.error {{ color:#b91c1c; font-weight:600 }}
.status-legend {{ margin-left:1rem; font-size:.8rem; color:var(--muted) }}
.legend-approved {{ color:#16a34a }}
.legend-pending {{ color:#d97706 }}
.legend-rejected {{ color:#dc2626 }}
.legend-unsaved {{ color:#9ca3af }}
.weak {{ color:var(--muted) }}
.export {{ margin-left:auto }}
.hidden {{ display:none }}
.progress {{ font-size:.85rem; color:var(--muted) }}
</style></head><body>
<header>
<h1>dots.tts transcript review — {title}</h1>
<div class="meta">package <code>{package_sha8}</code> ·
<span class="progress" id="progress"></span>
<span class="status-legend"><span class="legend-approved">● approved</span> ·
<span class="legend-pending">● pending</span> ·
<span class="legend-rejected">● rejected</span> ·
<span class="legend-unsaved">● 未保存</span></span></div>
</header>
<main>
<nav id="nav"></nav>
<section id="panel"></section>
</main>
<script>
"use strict";
const PACKAGE_SHA256 = "{package_sha256}";
const STORAGE_KEY = "dotstts-review-{benchmark_id}-v{benchmark_version}-" + PACKAGE_SHA256.slice(0, 16);
const REVIEW_SAMPLES = {samples_json};
const EMOTIONS = {emotions_json};
const INTENSITIES = ["low", "medium", "high"];
const TEXT_DECISIONS = ["accept_reference", "accept_edited", "flag_defect"];
const REVIEW_STATUSES = ["approved", "rejected", "pending"];
let samples = [];
let index = 0;
let decisions = {{}};

async function load() {{
  samples = REVIEW_SAMPLES;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {{
    try {{ decisions = restoreDecisions(stored); }}
    catch (error) {{ quarantineStoredDraft(stored, error); }}
  }}
  renderNav();
  goTo(firstPending() ?? 0);
}}

function restoreDecisions(raw) {{
  const parsed = JSON.parse(raw);
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {{
    throw new Error("草稿不是决策映射");
  }}
  const assets = new Set(samples.map((sample) => sample.asset_sha256));
  for (const [asset, decision] of Object.entries(parsed)) {{
    if (
      !assets.has(asset) || !decision || typeof decision !== "object" ||
      decision.asset_sha256 !== asset ||
      !TEXT_DECISIONS.includes(decision.text_decision) ||
      !REVIEW_STATUSES.includes(decision.review_status) ||
      (decision.text_decision === "flag_defect"
        ? decision.text_final !== null
        : typeof decision.text_final !== "string" || !decision.text_final.trim()) ||
      (decision.emotion_primary !== null && !EMOTIONS.includes(decision.emotion_primary)) ||
      (decision.emotion_secondary !== null && !EMOTIONS.includes(decision.emotion_secondary)) ||
      (decision.intensity !== null && !INTENSITIES.includes(decision.intensity))
    ) {{
      throw new Error(`草稿包含无效或非当前包资产：${{asset}}`);
    }}
  }}
  return parsed;
}}

function quarantineStoredDraft(raw, error) {{
  const quarantineKey = `${{STORAGE_KEY}}-invalid-${{Date.now()}}`;
  try {{ localStorage.setItem(quarantineKey, raw); }}
  catch (storageError) {{ /* best-effort backup */ }}
  localStorage.removeItem(STORAGE_KEY);
  decisions = {{}};
  alert(`检测到损坏或过期的审核草稿，已隔离并从空白状态恢复。\n${{error.message}}`);
}}

function firstPending() {{
  for (let i = 0; i < samples.length; i += 1) {{
    if (!decisions[samples[i].asset_sha256]) {{ return i; }}
  }}
  return null;
}}

function decisionFor(sample) {{
  return decisions[sample.asset_sha256] || null;
}}

function renderNav() {{
  const nav = document.getElementById("nav");
  const groups = [
    ["high", "高优先级"], ["medium", "中优先级"], ["low", "低优先级"],
    [null, "无标记（至少一家后端完全正确）"],
  ];
  let html = "";
  for (const [priority, label] of groups) {{
    const items = samples
      .map((sample, position) => ({{ sample, position }}))
      .filter((entry) => (entry.sample.priority ?? null) === priority);
    if (!items.length) {{ continue; }}
    html += `<div class="group">${{label}} (${{items.length}})</div>`;
    for (const entry of items) {{
      const decision = decisionFor(entry.sample);
      let dot = "undone";
      if (decision) {{
        dot = decision.review_status === "rejected"
          ? "rejected"
          : decision.review_status === "pending" ? "pending" : "done";
      }}
      html += `<button data-position="${{entry.position}}" class="${
        entry.position === index ? "active" : ""
      }"><span class="dot ${{dot}}"></span>` +
        `#${{entry.sample.ordinal}} ${{escapeText(entry.sample.weak_label)}}` +
        ` <span class="ord">${{entry.sample.duration_seconds.toFixed(1)}}s</span></button>`;
    }}
  }}
  nav.innerHTML = html;
  nav.querySelectorAll("button[data-position]").forEach((button) => {{
    button.addEventListener("click", () => {{
      goTo(parseInt(button.dataset.position, 10));
    }});
  }});
  const saved = samples.filter((sample) => decisions[sample.asset_sha256]).length;
  document.getElementById("progress").textContent =
    `${{saved}}/${{samples.length}} 已保存`;
}}

function escapeText(value) {{
  return String(value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}}

function renderSpans(spans) {{
  let html = "";
  for (const span of spans) {{
    if (span.equal) {{ html += escapeText(span.text); }}
    else {{
      const reference = span.reference ?? "";
      const hypothesis = span.hypothesis ?? "";
      if (reference) {{ html += `<del>${{escapeText(reference)}}</del>`; }}
      if (hypothesis) {{ html += `<ins>${{escapeText(hypothesis)}}</ins>`; }}
    }}
  }}
  return html;
}}

function render() {{
  const sample = samples[index];
  const decision = decisionFor(sample) || {{
    text_decision: sample.auto_rules_applied.length ? "accept_reference" : "",
    text_final: sample.auto_rules_applied.length ? sample.reference_exact : null,
    text_note: "",
    emotion_primary: sample.weak_label,
    emotion_secondary: null,
    intensity: null,
    review_status: "approved",
  }};
  const panel = document.getElementById("panel");
  const priorityBadge = sample.priority
    ? `<span class="badge ${{sample.priority}}">${{
        {{high: "high", medium: "medium", low: "low"}}[sample.priority]
      }}${{sample.priority === "high" ? " 优先" : sample.priority === "medium" ? " 优先" : ""}}</span>`
    : "";
  let notes = "";
  if (sample.auto_rules_applied.length) {{
    const pairs = sample.auto_rules_applied
      .map((rule) => `${{rule.variant}} → ${{rule.reference}}`).join("；");
    notes += `<div class="rule-note">同音异形已按参考自动裁定（${{escapeText(pairs)}}）。` +
      `音频不携带用字信息，人工无需听判；如需推翻请直接改选下方决策。</div>`;
  }}
  if (sample.interjection_hint) {{
    const heard = Object.entries(sample.interjection_hint.backends)
      .map(([backend, prefix]) => `${{backend}}听到「${{prefix}}」`).join("；");
    notes += `<div class="listen-note">多家后端听到参考缺失的句首语气词（${{escapeText(heard)}}）。` +
      `请<b>听音频</b>裁定参考是否需要补语气词。</div>`;
  }}
  let backends = "";
  const sorted = Object.entries(sample.backends)
    .sort((left, right) => (right[1].selected ? 1 : 0) - (left[1].selected ? 1 : 0));
  for (const [backend, info] of sorted) {{
    const body = `<div class="hyp ${{info.selected ? "selected" : ""}}">` +
      `<span class="who">${{escapeText(backend)}}${{info.selected ? " ★" : ""}}</span>` +
      `<span class="cer">CER ${{(info.cer ?? 0).toFixed(4)}}</span>` +
      (info.selected
        ? renderSpans(sample.diff_spans)
        : escapeText(info.hypothesis_raw || "—")) +
      `</div>`;
    if (info.selected) {{ backends += body; }}
    else {{
      backends += `<details class="other"><summary>${{escapeText(backend)}}</summary>${{body}}</details>`;
    }}
  }}
  const emotionOptions = ["", ...EMOTIONS]
    .map((emotion) => `<option value="${{emotion}}" ${
      decision.emotion_primary === emotion ? "selected" : ""
    }>${{emotion || "（空）"}}</option>`).join("");
  const secondaryOptions = ["", ...EMOTIONS]
    .map((emotion) => `<option value="${{emotion}}" ${
      decision.emotion_secondary === emotion ? "selected" : ""
    }>${{emotion || "（无）"}}</option>`).join("");
  const intensityOptions = ["", ...INTENSITIES]
    .map((level) => `<option value="${{level}}" ${
      decision.intensity === level ? "selected" : ""
    }>${{level || "（未定）"}}</option>`).join("");
  panel.innerHTML = `
    <h2>音频 #${{sample.ordinal}} <span class="weak">${{escapeText(sample.weak_label)}}</span>${{priorityBadge}}</h2>
    <div class="audio"><audio controls preload="none" src="${{sample.audio_rel_path}}"></audio></div>
    ${{notes}}
    <h2>参考文本</h2>
    <div class="reference" id="reference">${{escapeText(sample.reference_exact)}}</div>
    <h2>后端假设</h2>
    ${{backends}}
    <h2>文本决策</h2>
    <fieldset><div class="radios">
      <label><input type="radio" name="text_decision" value="accept_reference" ${
        decision.text_decision === "accept_reference" ? "checked" : ""
      }>接受参考</label>
      <label><input type="radio" name="text_decision" value="accept_edited" ${
        decision.text_decision === "accept_edited" ? "checked" : ""
      }>采用编辑文本</label>
      <label><input type="radio" name="text_decision" value="flag_defect" ${
        decision.text_decision === "flag_defect" ? "checked" : ""
      }>标记缺陷待查</label>
    </div>
    <textarea id="text_final" placeholder="最终文本（采用编辑文本时必填；接受参考自动填参考文本）">${
      escapeText(decision.text_final ?? sample.reference_exact)
    }</textarea>
    <textarea id="text_note" placeholder="备注：裁定依据（如：音频中确实有句首「唉」）">${
      escapeText(decision.text_note ?? "")
    }</textarea></fieldset>
    <h2>情感标签</h2>
    <fieldset>
      <p class="weak">原始弱标签：<code>${{escapeText(sample.weak_label)}}</code>（只读，永不覆盖）</p>
      <div class="grid2">
        <label>emotion_primary<select id="emotion_primary">${{emotionOptions}}</select></label>
        <label>emotion_secondary<select id="emotion_secondary">${{secondaryOptions}}</select></label>
        <label>intensity<select id="intensity">${{intensityOptions}}</select></label>
      </div>
    </fieldset>
    <h2>整体状态</h2>
    <fieldset><div class="radios">
      <label><input type="radio" name="review_status" value="approved" ${
        decision.review_status === "approved" ? "checked" : ""
      }>approved</label>
      <label><input type="radio" name="review_status" value="rejected" ${
        decision.review_status === "rejected" ? "checked" : ""
      }>rejected</label>
      <label><input type="radio" name="review_status" value="pending" ${
        decision.review_status === "pending" ? "checked" : ""
      }>pending</label>
    </div></fieldset>
    <div class="actions">
      <button id="prev">← 上一条</button>
      <button id="next">下一条 →</button>
      <button id="next_undecided">下一个未决</button>
      <button class="primary" id="save">保存本条</button>
      <span class="save-status" id="save_status" role="status" aria-live="polite"></span>
      <button class="export" id="export">导出 decisions.json</button>
    </div>`;
  const savedDecision = decisionFor(sample);
  if (savedDecision) {{
    showSaveStatus(
      `已保存到本地草稿 · ${{savedDecision.review_status}}`, "saved"
    );
  }} else {{
    showSaveStatus("本条尚未保存", "unsaved");
  }}
  panel.querySelectorAll("input[name=text_decision]").forEach((input) => {{
    input.addEventListener("change", () => {{
      if (input.value === "accept_reference") {{
        document.getElementById("text_final").value = sample.reference_exact;
      }} else if (input.value === "flag_defect") {{
        document.getElementById("text_final").value = "";
        const pending = document.querySelector(
          'input[name="review_status"][value="pending"]'
        );
        if (pending) {{ pending.checked = true; }}
      }}
    }});
  }});
  panel.querySelector("#prev").addEventListener("click", () => {{
    if (save()) {{ goTo(index - 1); }}
  }});
  panel.querySelector("#next").addEventListener("click", () => {{
    if (save()) {{ goTo(index + 1); }}
  }});
  panel.querySelector("#next_undecided").addEventListener("click", () => {{
    if (!save()) {{ return; }}
    let target = null;
    for (let step = 1; step <= samples.length; step += 1) {{
      const position = (index + step) % samples.length;
      if (!decisions[samples[position].asset_sha256]) {{ target = position; break; }}
    }}
    goTo(target ?? index);
  }});
  panel.querySelector("#save").addEventListener("click", () => {{
    save(true); renderNav();
  }});
  panel.querySelector("#export").addEventListener("click", exportDecisions);
  panel.querySelector("audio").focus();
}}

function collectDecision() {{
  const sample = samples[index];
  const textDecision = document.querySelector(
    "input[name=text_decision]:checked"
  )?.value ?? "";
  const reviewStatus = document.querySelector(
    "input[name=review_status]:checked"
  )?.value ?? "pending";
  const textFinal = document.getElementById("text_final").value.trim();
  return {{
    asset_sha256: sample.asset_sha256,
    text_decision: textDecision,
    text_final: textDecision === "flag_defect" ? null : textFinal,
    text_note: document.getElementById("text_note").value.trim() || null,
    emotion_primary: document.getElementById("emotion_primary").value || null,
    emotion_secondary: document.getElementById("emotion_secondary").value || null,
    intensity: document.getElementById("intensity").value || null,
    review_status: reviewStatus,
    auto_rules_applied: sample.auto_rules_applied,
  }};
}}

function showSaveStatus(message, kind) {{
  const status = document.getElementById("save_status");
  if (!status) {{ return; }}
  status.textContent = message;
  status.className = `save-status ${{kind || "unsaved"}}`;
}}

function save(requireDecision = false) {{
  const decision = collectDecision();
  if (!decision.text_decision) {{
    showSaveStatus("未保存 · 请先选择文本决策", "error");
    return !requireDecision;
  }}
  let error = null;
  if (decision.text_decision === "accept_edited" && !decision.text_final) {{
    error = "采用编辑文本时，最终文本不能为空。";
  }} else if (
    decision.emotion_primary &&
    decision.emotion_primary === decision.emotion_secondary
  ) {{
    error = "主情感和次情感不能相同。";
  }} else if (
    decision.text_decision === "flag_defect" && !decision.text_note
  ) {{
    error = "标记缺陷时必须填写备注说明。";
  }} else if (
    decision.text_decision === "flag_defect" &&
    decision.review_status === "approved"
  ) {{
    error = "标记缺陷不能使用 approved 状态。";
  }}
  if (error) {{
    showSaveStatus(`未保存 · ${{error}}`, "error");
    alert(error);
    return false;
  }}
  decisions[decision.asset_sha256] = decision;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(decisions));
  const savedAt = new Date().toLocaleTimeString([], {{
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  }});
  showSaveStatus(
    `已保存到本地草稿 · ${{decision.review_status}} · ${{savedAt}}`, "saved"
  );
  const button = document.getElementById("save");
  if (button) {{
    button.textContent = "已保存 ✓";
    const savedAsset = decision.asset_sha256;
    setTimeout(() => {{
      if (samples[index]?.asset_sha256 === savedAsset) {{
        const currentButton = document.getElementById("save");
        if (currentButton) {{ currentButton.textContent = "保存本条"; }}
      }}
    }}, 1200);
  }}
  return true;
}}

function randomUuid() {{
  if (crypto.randomUUID) {{ return crypto.randomUUID(); }}
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {{
    const random = Math.random() * 16 | 0;
    const value = character === "x" ? random : (random & 0x3 | 0x8);
    return value.toString(16);
  }});
}}

function exportDecisions() {{
  if (!save()) {{ return; }}
  const decided = Object.values(decisions);
  if (!decided.length) {{
    alert("还没有任何裁定。"); return;
  }}
  const batchId = randomUuid();
  const payload = {{
    schema_version: 1,
    benchmark_id: "{benchmark_id}",
    benchmark_version: {benchmark_version},
    package_sha256: PACKAGE_SHA256,
    export_batch_id: batchId,
    decided_at: new Date().toISOString(),
    decisions: decided,
  }};
  const text = JSON.stringify(payload, null, 2) + "\\n";
  const download = document.createElement("a");
  download.href = URL.createObjectURL(new Blob([text], {{ type: "application/json" }}));
  download.download = `decisions-${{batchId.slice(0, 8)}}.json`;
  download.click();
  URL.revokeObjectURL(download.href);
  const box = document.createElement("textarea");
  box.value = text;
  box.style.position = "fixed"; box.style.opacity = "0";
  document.body.appendChild(box);
  box.select();
  try {{ document.execCommand("copy"); }}
  catch (error) {{ /* copy is a fallback; the download link is primary */ }}
  box.remove();
  alert(`已导出 ${{decided.length}} 条裁定（decisions.json 已下载；若下载被浏览器拦截，JSON 已同时复制到剪贴板，请粘贴保存）。\\n` +
    `每次导出使用新的 batch id；重复导入同一文件是安全的。`);
}}

function goTo(position) {{
  if (position < 0 || position >= samples.length) {{ return; }}
  index = position;
  renderNav();
  render();
  window.scrollTo(0, 0);
}}

load();
</script>
</body></html>
"""


def _review_page(
    *,
    benchmark_id: str,
    benchmark_version: int,
    package_sha256: str,
    samples_json: str,
) -> str:
    # Token replacement avoids interpreting JavaScript template expressions.
    # The template still uses doubled braces inherited from its former
    # str.format implementation, so collapse them only after replacing tokens.
    page = _PAGE_TEMPLATE.replace("{{", "{").replace("}}", "}")
    embedded_samples = (
        samples_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    for key, value in (
        ("{title}", f"{benchmark_id}@v{benchmark_version}"),
        ("{package_sha256}", package_sha256),
        ("{package_sha8}", package_sha256[:8]),
        ("{benchmark_id}", benchmark_id),
        ("{benchmark_version}", str(benchmark_version)),
        ("{samples_json}", embedded_samples),
        ("{emotions_json}", json.dumps(EMOTION_CHOICES)),
    ):
        page = page.replace(key, value)
    return page


