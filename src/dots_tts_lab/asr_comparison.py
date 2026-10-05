from __future__ import annotations

import csv
import hashlib
import html
import io
import itertools
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dots_tts_lab.asr_benchmark import (
    DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    AsrBenchmarkConfig,
    character_edit_distance,
    load_asr_benchmark_config,
    normalize_asr_text,
)
from dots_tts_lab.catalog import Catalog
from dots_tts_lab.inventory import utc_now
from dots_tts_lab.reports import _atomic_write_text


_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "lab" / "asr"
DEFAULT_ASR_COMPARISON_CONFIG_PATH = _CONFIG_ROOT / "comparison_v2.yaml"
DEFAULT_ASR_LEXICON_PATH = _CONFIG_ROOT / "lexicon_fuxuan_v1.yaml"


class AsrComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    comparison_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    comparison_version: int = Field(ge=1)
    selected_backend_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_.-]{2,63}$"
    )
    selection_rationale: str | None = Field(default=None, min_length=20)
    consensus_max_pairwise_cer: float = Field(gt=0.0, le=1.0)
    high_cer_delta_min: float = Field(gt=0.0, le=1.0)
    selection_max_aggregate_cer: float = Field(gt=0.0, le=1.0)
    selection_max_realtime_factor: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_explicit_selection(self) -> AsrComparisonConfig:
        if (self.selected_backend_id is None) != (self.selection_rationale is None):
            raise ValueError(
                "selected_backend_id and selection_rationale must be set together"
            )
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


class AsrLexiconConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    lexicon_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    lexicon_version: int = Field(ge=1)
    named_entities: list[str] = Field(min_length=1)
    archaic_terms: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_terms(self) -> AsrLexiconConfig:
        for field, terms in (
            ("named_entities", self.named_entities),
            ("archaic_terms", self.archaic_terms),
        ):
            if any(not term.strip() for term in terms):
                raise ValueError(f"{field} cannot contain blank terms")
            if len(set(terms)) != len(terms):
                raise ValueError(f"{field} cannot contain duplicates")
        overlap = set(self.named_entities) & set(self.archaic_terms)
        if overlap:
            raise ValueError(f"A term cannot be in both categories: {sorted(overlap)}")
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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


def load_asr_comparison_config(
    path: str | Path = DEFAULT_ASR_COMPARISON_CONFIG_PATH,
) -> AsrComparisonConfig:
    return AsrComparisonConfig.model_validate(
        _load_yaml_mapping(Path(path).resolve()), strict=True
    )


def load_asr_lexicon(
    path: str | Path = DEFAULT_ASR_LEXICON_PATH,
) -> AsrLexiconConfig:
    return AsrLexiconConfig.model_validate(
        _load_yaml_mapping(Path(path).resolve()), strict=True
    )


def normalized_terms(
    terms: list[str], config: AsrBenchmarkConfig
) -> dict[str, str]:
    """Map each term's normalized form back to how the lexicon spells it.

    Terms are matched against normalized hypotheses, so they must go through the
    same normalization. Two terms collapsing to one normalized form would make
    per-term accounting ambiguous, so that is rejected rather than merged.
    """
    mapping: dict[str, str] = {}
    for term in terms:
        normalized = normalize_asr_text(term, config)
        if not normalized:
            raise ValueError(f"Term normalizes to an empty string: {term!r}")
        if normalized in mapping:
            raise ValueError(
                f"Terms {mapping[normalized]!r} and {term!r} share a normalized form"
            )
        mapping[normalized] = term
    return mapping


def punctuation_profile(text: str) -> dict[str, int]:
    """Count punctuation the backend emitted, before normalization strips it.

    CER is computed on punctuation-free text, so this is the only place a
    backend's punctuation policy stays visible. It matters downstream: a
    verification backend that drops punctuation cannot be diffed against
    reference text on anything but characters.
    """
    return dict(
        Counter(
            character
            for character in text
            if unicodedata.category(character)[:1] == "P"
        )
    )


def pairwise_spread(hypotheses: list[str]) -> float:
    """Largest normalized character distance between any two backend outputs."""
    if len(hypotheses) < 2:
        return 0.0
    return max(
        character_edit_distance(left, right) / max(1, len(left), len(right))
        for left, right in itertools.combinations(hypotheses, 2)
    )


def term_recall(
    results: list[dict[str, Any]], terms: dict[str, str]
) -> dict[str, Any]:
    """How often lexicon terms present in the reference survive transcription."""
    occurrences = 0
    hits = 0
    missed: Counter[str] = Counter()
    for result in results:
        reference = str(result["reference_normalized"])
        hypothesis = str(result.get("hypothesis_normalized") or "")
        for normalized, original in terms.items():
            reference_count = reference.count(normalized)
            if reference_count == 0:
                continue
            hypothesis_count = hypothesis.count(normalized)
            occurrence_hits = min(reference_count, hypothesis_count)
            occurrences += reference_count
            hits += occurrence_hits
            if occurrence_hits < reference_count:
                missed[original] += reference_count - occurrence_hits
    return {
        "occurrence_count": occurrences,
        "hit_count": hits,
        "recall": hits / occurrences if occurrences else None,
        "missed_terms": dict(sorted(missed.most_common())),
    }


def _missing_terms(reference: str, hypothesis: str, terms: dict[str, str]) -> list[str]:
    return sorted(
        original
        for normalized, original in terms.items()
        if reference.count(normalized) > hypothesis.count(normalized)
    )


def _backend_summary(
    run: dict[str, Any],
    results: list[dict[str, Any]],
    entities: dict[str, str],
    archaic: dict[str, str],
    config: AsrComparisonConfig,
) -> dict[str, Any]:
    runtime_metadata = json.loads(run.get("runtime_metadata_json") or "{}")
    capabilities = runtime_metadata.get("capabilities") or {}
    audio_seconds = float(run["total_audio_seconds"])
    runtime_seconds = float(run["total_runtime_seconds"])
    realtime_factor = runtime_seconds / audio_seconds if audio_seconds else None
    punctuation: Counter[str] = Counter()
    for result in results:
        punctuation.update(punctuation_profile(str(result.get("hypothesis_raw") or "")))
    by_emotion: dict[str, float] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["emotion_weak_label"]), []).append(result)
    for emotion, group in grouped.items():
        characters = sum(int(item["reference_char_count"]) for item in group)
        by_emotion[emotion] = (
            sum(int(item["edit_distance"] or 0) for item in group) / characters
            if characters
            else 0.0
        )
    aggregate_cer = run.get("aggregate_cer")
    gate_failures = []
    if aggregate_cer is None or float(aggregate_cer) > config.selection_max_aggregate_cer:
        gate_failures.append("aggregate_cer")
    if realtime_factor is None or realtime_factor > config.selection_max_realtime_factor:
        gate_failures.append("realtime_factor")
    if int(run["error_count"]) > 0:
        gate_failures.append("error_count")
    return {
        "backend_id": run["backend_id"],
        "backend_version": run["backend_version"],
        "model_id": run["model_id"],
        "model_revision": run["model_revision"],
        "model_license": run.get("model_license"),
        "run_id": run["run_id"],
        "started_at": run["started_at"],
        "inference_config_sha256": run["inference_config_sha256"],
        "item_count": int(run["item_count"]),
        "success_count": int(run["success_count"]),
        "error_count": int(run["error_count"]),
        "aggregate_cer": aggregate_cer,
        "exact_match_count": run.get("exact_match_count"),
        "cer_by_emotion": by_emotion,
        "total_audio_seconds": audio_seconds,
        "total_runtime_seconds": runtime_seconds,
        "realtime_factor": realtime_factor,
        "model_load_seconds": runtime_metadata.get("model_load_seconds"),
        "peak_torch_cuda_bytes": runtime_metadata.get("peak_torch_cuda_bytes"),
        "peak_memory_note": runtime_metadata.get("peak_memory_note"),
        "timestamps_available": bool(capabilities.get("timestamps_available")),
        "timestamp_type": capabilities.get("timestamp_type", "unknown"),
        "timestamps_enabled": bool(capabilities.get("timestamps_enabled")),
        "punctuation_char_count": sum(punctuation.values()),
        "punctuation_profile": dict(sorted(punctuation.most_common())),
        "emits_punctuation": bool(punctuation),
        "named_entities": term_recall(results, entities),
        "archaic_terms": term_recall(results, archaic),
        "gate_failures": gate_failures,
        "meets_selection_gates": not gate_failures,
    }


_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sample_review(
    ordinal: int,
    reference_exact: str,
    reference_normalized: str,
    per_backend: dict[str, dict[str, Any]],
    entities: dict[str, str],
    config: AsrComparisonConfig,
) -> dict[str, Any]:
    cers = {
        backend: float(item["cer"]) for backend, item in per_backend.items()
    }
    hypotheses = [
        str(item.get("hypothesis_normalized") or "") for item in per_backend.values()
    ]
    spread = pairwise_spread(hypotheses)
    minimum = min(cers.values())
    delta = max(cers.values()) - minimum
    missing = {
        backend: _missing_terms(
            reference_normalized,
            str(item.get("hypothesis_normalized") or ""),
            entities,
        )
        for backend, item in per_backend.items()
    }
    unanimous_missing = sorted(set.intersection(*(set(v) for v in missing.values())))
    any_missing = sorted({term for terms in missing.values() for term in terms})
    flags = []
    if minimum > 0.0:
        flags.append("all_backends_wrong")
    if minimum > 0.0 and spread <= config.consensus_max_pairwise_cer:
        flags.append("consensus_conflicts_reference")
    if delta >= config.high_cer_delta_min:
        flags.append("high_cer_delta")
    if unanimous_missing:
        flags.append("unanimous_named_entity_miss")
    elif any_missing:
        flags.append("named_entity_miss")
    if {"consensus_conflicts_reference", "unanimous_named_entity_miss"} & set(flags):
        priority = "high"
    elif {"high_cer_delta", "named_entity_miss"} & set(flags):
        priority = "medium"
    else:
        priority = "low"
    return {
        "ordinal": ordinal,
        "reference_exact": reference_exact,
        "reference_normalized": reference_normalized,
        "backend_cer": {backend: round(value, 6) for backend, value in cers.items()},
        "backend_hypothesis": {
            backend: str(item.get("hypothesis_raw") or "")
            for backend, item in per_backend.items()
        },
        "min_cer": minimum,
        "max_cer": max(cers.values()),
        "cer_delta": delta,
        "backend_pairwise_spread": spread,
        "missing_named_entities": any_missing,
        "unanimous_missing_named_entities": unanimous_missing,
        "flags": flags,
        "priority": priority,
    }


def _to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _backend_csv_row(backend: dict[str, Any]) -> dict[str, Any]:
    return {
        "cer_rank": backend["cer_rank"],
        "backend_id": backend["backend_id"],
        "model_id": backend["model_id"],
        "model_license": backend["model_license"],
        "aggregate_cer": backend["aggregate_cer"],
        "exact_match_count": backend["exact_match_count"],
        "named_entity_recall": backend["named_entities"]["recall"],
        "archaic_term_recall": backend["archaic_terms"]["recall"],
        "emits_punctuation": backend["emits_punctuation"],
        "punctuation_char_count": backend["punctuation_char_count"],
        "realtime_factor": backend["realtime_factor"],
        "model_load_seconds": backend["model_load_seconds"],
        "peak_torch_cuda_bytes": backend["peak_torch_cuda_bytes"],
        "timestamps_available": backend["timestamps_available"],
        "timestamp_type": backend["timestamp_type"],
        "timestamps_enabled": backend["timestamps_enabled"],
        "meets_selection_gates": backend["meets_selection_gates"],
        "gate_failures": ";".join(backend["gate_failures"]),
    }


def _review_csv_row(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "priority": review["priority"],
        "ordinal": review["ordinal"],
        "flags": ";".join(review["flags"]),
        "min_cer": review["min_cer"],
        "max_cer": review["max_cer"],
        "cer_delta": review["cer_delta"],
        "backend_pairwise_spread": review["backend_pairwise_spread"],
        "missing_named_entities": ";".join(review["missing_named_entities"]),
        "reference_exact": review["reference_exact"],
        **{
            f"hypothesis_{backend}": text
            for backend, text in review["backend_hypothesis"].items()
        },
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _backend_table(backends: list[dict[str, Any]]) -> str:
    columns = (
        ("cer_rank", "#"),
        ("backend_id", "Backend"),
        ("model_license", "License"),
        ("aggregate_cer", "CER"),
        ("exact_match_count", "Exact"),
        ("named_entity_recall", "Entity recall"),
        ("archaic_term_recall", "Archaic recall"),
        ("emits_punctuation", "Punct."),
        ("realtime_factor", "RTF"),
        ("model_load_seconds", "Load s"),
        ("peak_torch_cuda_bytes", "Peak torch B"),
        ("timestamp_type", "Timestamps"),
        ("meets_selection_gates", "Gates"),
    )
    header = "".join(f"<th>{label}</th>" for _, label in columns)
    rows = []
    for backend in backends:
        flat = _backend_csv_row(backend)
        cells = "".join(f"<td>{_format_cell(flat[key])}</td>" for key, _ in columns)
        css = "" if backend["meets_selection_gates"] else ' class="blocked"'
        rows.append(f"<tr{css}>{cells}</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _review_table(reviews: list[dict[str, Any]], backend_ids: list[str]) -> str:
    header = "".join(f"<th>{html.escape(backend)}</th>" for backend in backend_ids)
    rows = []
    for review in reviews:
        hypotheses = "".join(
            "<td>{}<br><small>CER {:.2%}</small></td>".format(
                html.escape(str(review["backend_hypothesis"].get(backend, ""))),
                float(review["backend_cer"].get(backend, 0.0)),
            )
            for backend in backend_ids
        )
        rows.append(
            '<tr class="{}"><td>{}</td><td>{}</td><td>{}</td><td>{}</td>{}</tr>'.format(
                review["priority"],
                review["ordinal"],
                html.escape(review["priority"]),
                html.escape(", ".join(review["flags"])),
                html.escape(str(review["reference_exact"])),
                hypotheses,
            )
        )
    return (
        "<table><thead><tr><th>#</th><th>Priority</th><th>Flags</th>"
        f"<th>Reference</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _comparison_html(
    summary: dict[str, Any],
    backends: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> str:
    backend_ids = [str(backend["backend_id"]) for backend in backends]
    summary_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{_format_cell(value)}</td></tr>"
        for key, value in summary.items()
        if not isinstance(value, (dict, list))
    )
    missed = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(str(backend["backend_id"])),
            html.escape(
                ", ".join(
                    f"{term}×{count}"
                    for term, count in backend["named_entities"][
                        "missed_terms"
                    ].items()
                )
                or "—"
            ),
        )
        for backend in backends
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>dots.tts ASR backend comparison</title><style>
body {{ font:14px/1.55 system-ui,sans-serif; margin:2rem; color:#1f2937 }}
table {{ border-collapse:collapse; width:100%; margin-bottom:2rem }}
th,td {{ border:1px solid #d1d5db; padding:.4rem .55rem; text-align:left;
         vertical-align:top }}
th {{ background:#f3f4f6 }}
tr.blocked td {{ background:#fef2f2 }}
tr.high td {{ background:#fef3c7 }}
tr.medium td {{ background:#f0f9ff }}
small {{ color:#6b7280 }}
</style></head><body>
<h1>ASR backend comparison</h1>
<p>Recommended backend:
<strong>{html.escape(str(summary.get("recommended_backend") or "none"))}</strong></p>
<h2>Summary</h2><table><tbody>{summary_rows}</tbody></table>
<h2>Backends</h2>{_backend_table(backends)}
<h2>Missed named entities</h2><table><tbody>{missed}</tbody></table>
<h2>Review queue ({len(reviews)} samples)</h2>
{_review_table(reviews, backend_ids)}
</body></html>"""


def _write_comparison_reports(
    output_dir: Path,
    summary: dict[str, Any],
    backends: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    backends_csv = output_dir / "backends.csv"
    review_csv = output_dir / "review_queue.csv"
    html_path = output_dir / "comparison.html"
    _atomic_write_text(
        json_path,
        json.dumps(
            {
                "schema_version": 1,
                "summary": summary,
                "backends": backends,
                "review_queue": reviews,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _atomic_write_text(
        backends_csv, _to_csv([_backend_csv_row(backend) for backend in backends])
    )
    _atomic_write_text(
        review_csv, _to_csv([_review_csv_row(review) for review in reviews])
    )
    _atomic_write_text(html_path, _comparison_html(summary, backends, reviews))
    return {
        "json": str(json_path),
        "backends_csv": str(backends_csv),
        "review_queue_csv": str(review_csv),
        "html": str(html_path),
    }


def _finalize_comparison(
    *,
    config: AsrComparisonConfig,
    lexicon: AsrLexiconConfig,
    benchmark_config: AsrBenchmarkConfig,
    benchmark_record: dict[str, Any],
    backends: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    sample_count: int,
    catalog_path: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    ranked = sorted(
        backends,
        key=lambda backend: (
            backend["aggregate_cer"] is None,
            backend["aggregate_cer"],
        ),
    )
    for position, backend in enumerate(ranked, start=1):
        backend["cer_rank"] = position
    qualified = [backend for backend in ranked if backend["meets_selection_gates"]]
    selected = (
        next(
            (
                backend
                for backend in qualified
                if backend["backend_id"] == config.selected_backend_id
            ),
            None,
        )
        if config.selected_backend_id is not None
        else (qualified[0] if qualified else None)
    )
    if selected is None:
        available = [backend["backend_id"] for backend in qualified]
        raise RuntimeError(
            f"Configured selected backend {config.selected_backend_id!r} is not "
            f"qualified; qualified={available}"
        )
    summary = {
        "status": "succeeded",
        "comparison_id": config.comparison_id,
        "comparison_version": config.comparison_version,
        "comparison_config_sha256": config.config_sha256(),
        "lexicon_id": lexicon.lexicon_id,
        "lexicon_version": lexicon.lexicon_version,
        "lexicon_sha256": lexicon.config_sha256(),
        "benchmark_id": benchmark_config.benchmark_id,
        "benchmark_version": benchmark_config.benchmark_version,
        "benchmark_config_sha256": benchmark_record["config_sha256"],
        "benchmark_manifest_sha256": benchmark_record["manifest_sha256"],
        "created_at": utc_now(),
        "sample_count": sample_count,
        "backend_count": len(backends),
        "named_entity_term_count": len(lexicon.named_entities),
        "archaic_term_count": len(lexicon.archaic_terms),
        "gates": {
            "selection_max_aggregate_cer": config.selection_max_aggregate_cer,
            "selection_max_realtime_factor": config.selection_max_realtime_factor,
            "consensus_max_pairwise_cer": config.consensus_max_pairwise_cer,
            "high_cer_delta_min": config.high_cer_delta_min,
        },
        "qualified_backends": [backend["backend_id"] for backend in qualified],
        "cer_leader": ranked[0]["backend_id"] if ranked else None,
        "recommended_backend": selected["backend_id"],
        "selection_rationale": config.selection_rationale
        or "Legacy comparison config: automatic lowest-CER qualified backend.",
        "review_queue_count": len(reviews),
        "review_priority_counts": dict(
            sorted(Counter(review["priority"] for review in reviews).most_common())
        ),
        "review_flag_counts": dict(
            sorted(
                Counter(
                    flag for review in reviews for flag in review["flags"]
                ).most_common()
            )
        ),
    }
    reports = _write_comparison_reports(
        Path(report_dir).resolve(), summary, ranked, reviews
    )
    summary["reports"] = reports
    summary["catalog_path"] = str(Path(catalog_path).resolve())
    return summary


def compare_asr_backends(
    *,
    catalog_path: str | Path = "data/catalog/catalog.sqlite",
    benchmark_config_path: str | Path = DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    comparison_config_path: str | Path = DEFAULT_ASR_COMPARISON_CONFIG_PATH,
    lexicon_path: str | Path = DEFAULT_ASR_LEXICON_PATH,
    report_dir: str | Path = "data/reports/asr/comparison",
) -> dict[str, Any]:
    benchmark_config = load_asr_benchmark_config(benchmark_config_path)
    config = load_asr_comparison_config(comparison_config_path)
    lexicon = load_asr_lexicon(lexicon_path)
    entities = normalized_terms(lexicon.named_entities, benchmark_config)
    archaic = normalized_terms(lexicon.archaic_terms, benchmark_config)
    catalog = Catalog(catalog_path)
    catalog.initialize()
    benchmark_record = catalog.load_asr_benchmark(
        benchmark_id=benchmark_config.benchmark_id,
        benchmark_version=benchmark_config.benchmark_version,
    )
    if benchmark_record is None:
        raise RuntimeError(
            "Benchmark is not registered in the catalog: "
            f"{benchmark_config.benchmark_id}@{benchmark_config.benchmark_version}"
        )
    if benchmark_record["config_sha256"] != benchmark_config.config_sha256():
        raise RuntimeError(
            "Benchmark config does not match the registered catalog snapshot"
        )
    runs = catalog.load_latest_asr_runs(
        benchmark_id=benchmark_config.benchmark_id,
        benchmark_version=benchmark_config.benchmark_version,
    )
    if len(runs) < 2:
        raise RuntimeError(
            "Comparison needs at least two successfully evaluated backends, "
            f"found {len(runs)}"
        )
    results_by_backend: dict[str, list[dict[str, Any]]] = {}
    backends = []
    for run in runs:
        results = catalog.load_asr_results(run_id=str(run["run_id"]))
        if any(result["status"] != "ok" for result in results):
            raise RuntimeError(
                f"Backend {run['backend_id']} has failed items; re-run it before "
                "comparing"
            )
        results_by_backend[str(run["backend_id"])] = results
        backends.append(_backend_summary(run, results, entities, archaic, config))
    ordinals = {
        backend: {int(result["ordinal"]) for result in results}
        for backend, results in results_by_backend.items()
    }
    shared = set.intersection(*ordinals.values())
    if any(len(value) != len(shared) for value in ordinals.values()):
        counts = {backend: len(value) for backend, value in ordinals.items()}
        raise RuntimeError(
            "Backends were evaluated on different benchmark items; "
            f"shared={len(shared)}, per_backend={counts}"
        )
    indexed = {
        backend: {int(result["ordinal"]): result for result in results}
        for backend, results in results_by_backend.items()
    }
    reviews = []
    for ordinal in sorted(shared):
        per_backend = {
            backend: items[ordinal] for backend, items in indexed.items()
        }
        first = next(iter(per_backend.values()))
        review = _sample_review(
            ordinal,
            str(first["text_exact"]),
            str(first["reference_normalized"]),
            per_backend,
            entities,
            config,
        )
        if review["flags"]:
            reviews.append(review)
    reviews.sort(
        key=lambda review: (
            _PRIORITY_ORDER[review["priority"]],
            -review["min_cer"],
            review["ordinal"],
        )
    )
    return _finalize_comparison(
        config=config,
        lexicon=lexicon,
        benchmark_config=benchmark_config,
        benchmark_record=benchmark_record,
        backends=backends,
        reviews=reviews,
        sample_count=len(shared),
        catalog_path=catalog_path,
        report_dir=report_dir,
    )
