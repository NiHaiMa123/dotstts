from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dots_tts_lab.ingest import run_ingest
from dots_tts_lab.asr_benchmark import (
    DEFAULT_ASR_BENCHMARK_CONFIG_PATH,
    prepare_asr_benchmark,
)
from dots_tts_lab.asr_evaluation import evaluate_asr_output
from dots_tts_lab.asr_comparison import (
    DEFAULT_ASR_COMPARISON_CONFIG_PATH,
    DEFAULT_ASR_LEXICON_PATH,
    compare_asr_backends,
)
from dots_tts_lab.inventory import DEFAULT_EXTENSIONS, run_inventory
from dots_tts_lab.metadata import DEFAULT_METADATA_PROFILE_PATH
from dots_tts_lab.quality import (
    DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH,
    DEFAULT_QUALITY_POLICY_PATH,
)
from dots_tts_lab.quality_runner import run_quality
from dots_tts_lab.review import (
    DEFAULT_REVIEW_RULES_PATH,
    review_export,
    review_import,
)
from dots_tts_lab.standardization import DEFAULT_STANDARDIZATION_CONFIG_PATH
from dots_tts_lab.standardize_runner import run_standardization
from dots_tts_lab.dataset_audit import (
    DEFAULT_CANDIDATE_SNAPSHOT_PATH,
    DEFAULT_DATASET_AUDIT_REPORT_PATH,
    run_dataset_audit,
)
from dots_tts_lab.dataset_freeze import DEFAULT_DATASET_FREEZE_CONFIG_PATH


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _load_asset_manifest(path: str | None) -> list[str]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ValueError("Asset manifest must contain an assets list")
    hashes = []
    for item in payload["assets"]:
        if not isinstance(item, dict) or not isinstance(item.get("asset_sha256"), str):
            raise ValueError("Every asset manifest entry needs asset_sha256")
        hashes.append(item["asset_sha256"])
    return hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dots.tts.lab",
        description="Local data, training, and evaluation workbench for dots.tts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser(
        "inventory",
        help="Read-only inventory of source audio.",
    )
    inventory_parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/inbox",
        help="Source audio directory (default: data/inbox).",
    )
    inventory_parser.add_argument(
        "--catalog",
        default="data/catalog/catalog.sqlite",
        help="SQLite catalog path.",
    )
    inventory_parser.add_argument(
        "--report-dir",
        default="data/reports/inventory",
        help="JSON/CSV/HTML report directory.",
    )
    inventory_parser.add_argument(
        "--profile",
        default=str(DEFAULT_METADATA_PROFILE_PATH),
        help="Versioned metadata parsing profile YAML.",
    )
    inventory_parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        help="Accepted extension; repeat to provide several.",
    )
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Copy source audio into the verified immutable raw store.",
    )
    ingest_parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/inbox",
        help="Source audio directory (default: data/inbox).",
    )
    ingest_parser.add_argument(
        "--catalog",
        default="data/catalog/catalog.sqlite",
        help="SQLite catalog path.",
    )
    ingest_parser.add_argument(
        "--raw-dir",
        default="data/raw/sha256",
        help="Content-addressed raw object root.",
    )
    ingest_parser.add_argument(
        "--inventory-report-dir",
        default="data/reports/inventory",
        help="Inventory report directory refreshed before import.",
    )
    ingest_parser.add_argument(
        "--report-dir",
        default="data/reports/ingest",
        help="Immutable ingest report directory.",
    )
    ingest_parser.add_argument(
        "--profile",
        default=str(DEFAULT_METADATA_PROFILE_PATH),
        help="Versioned metadata parsing profile YAML.",
    )
    ingest_parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        help="Accepted extension; repeat to provide several.",
    )
    quality_parser = subparsers.add_parser(
        "quality",
        help="Measure cached objective audio metrics and apply a review policy.",
    )
    quality_parser.add_argument(
        "--catalog",
        default="data/catalog/catalog.sqlite",
        help="SQLite catalog containing verified raw objects.",
    )
    quality_parser.add_argument(
        "--raw-dir",
        default="data/raw/sha256",
        help="Content-addressed raw object root.",
    )
    quality_parser.add_argument(
        "--analysis-config",
        default=str(DEFAULT_QUALITY_ANALYSIS_CONFIG_PATH),
        help="Signal measurement configuration YAML.",
    )
    quality_parser.add_argument(
        "--policy",
        default=str(DEFAULT_QUALITY_POLICY_PATH),
        help="Review threshold policy YAML.",
    )
    quality_parser.add_argument(
        "--report-dir",
        default="data/reports/quality",
        help="Objective quality report directory.",
    )
    quality_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-decode assets instead of reusing matching cached metrics.",
    )
    standardize_parser = subparsers.add_parser(
        "standardize",
        help="Build reproducible 48 kHz mono PCM24 training audio.",
    )
    standardize_parser.add_argument(
        "--catalog",
        default="data/catalog/catalog.sqlite",
        help="SQLite catalog containing verified raw objects.",
    )
    standardize_parser.add_argument(
        "--raw-dir",
        default="data/raw/sha256",
        help="Content-addressed raw object root.",
    )
    standardize_parser.add_argument(
        "--output-dir",
        default="data/work/standardized",
        help="Rebuildable standardized audio root.",
    )
    standardize_parser.add_argument(
        "--config",
        default=str(DEFAULT_STANDARDIZATION_CONFIG_PATH),
        help="Versioned standardization configuration YAML.",
    )
    standardize_parser.add_argument(
        "--report-dir",
        default="data/reports/standardization",
        help="Standardization JSON/CSV/HTML report directory.",
    )
    standardize_parser.add_argument(
        "--asset-sha256",
        dest="asset_sha256s",
        action="append",
        help="Only process this raw asset SHA-256; repeat for several.",
    )
    standardize_parser.add_argument(
        "--asset-manifest",
        help="JSON manifest containing an assets list with asset_sha256 values.",
    )
    standardize_parser.add_argument(
        "--limit",
        type=int,
        help="Process the first N selected assets for a deterministic prototype.",
    )
    standardize_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild matching artifacts instead of verifying and reusing them.",
    )
    asr_prepare_parser = subparsers.add_parser(
        "asr-prepare",
        help="Register confirmed filename transcripts and freeze an ASR benchmark.",
    )
    asr_prepare_parser.add_argument(
        "--catalog",
        default="data/catalog/catalog.sqlite",
        help="SQLite catalog path.",
    )
    asr_prepare_parser.add_argument(
        "--config",
        default=str(DEFAULT_ASR_BENCHMARK_CONFIG_PATH),
        help="Versioned ASR benchmark configuration YAML.",
    )
    asr_prepare_parser.add_argument(
        "--quality-report",
        default="data/reports/quality/quality.json",
        help="Objective quality report used for representative selection.",
    )
    asr_prepare_parser.add_argument(
        "--output-dir",
        default="data/benchmarks/asr/fuxuan_v2",
        help="Frozen benchmark manifest and human-readable report directory.",
    )
    asr_evaluate_parser = subparsers.add_parser(
        "asr-evaluate",
        help="Validate backend predictions, compute CER, and store a benchmark run.",
    )
    asr_evaluate_parser.add_argument("result_json", help="Raw backend result JSON.")
    asr_evaluate_parser.add_argument(
        "--catalog", default="data/catalog/catalog.sqlite", help="SQLite catalog path."
    )
    asr_evaluate_parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_ASR_BENCHMARK_CONFIG_PATH),
        help="Benchmark configuration used for text normalization.",
    )
    asr_evaluate_parser.add_argument(
        "--report-root",
        default="data/reports/asr",
        help="Root directory for per-backend evaluation reports.",
    )
    asr_compare_parser = subparsers.add_parser(
        "asr-compare",
        help="Rank evaluated backends and build the human review queue.",
    )
    asr_compare_parser.add_argument(
        "--catalog", default="data/catalog/catalog.sqlite", help="SQLite catalog path."
    )
    asr_compare_parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_ASR_BENCHMARK_CONFIG_PATH),
        help="Benchmark configuration identifying the runs to compare.",
    )
    asr_compare_parser.add_argument(
        "--config",
        default=str(DEFAULT_ASR_COMPARISON_CONFIG_PATH),
        help="Comparison thresholds and selection gates YAML.",
    )
    asr_compare_parser.add_argument(
        "--lexicon",
        default=str(DEFAULT_ASR_LEXICON_PATH),
        help="Domain named entity and archaic term lexicon YAML.",
    )
    asr_compare_parser.add_argument(
        "--report-dir",
        default="data/reports/asr/comparison",
        help="Cross-backend comparison report directory.",
    )
    review_export_parser = subparsers.add_parser(
        "review-export",
        help="Build the static transcript review page and its data package.",
    )
    review_export_parser.add_argument(
        "--catalog", default="data/catalog/catalog.sqlite", help="SQLite catalog path."
    )
    review_export_parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_ASR_BENCHMARK_CONFIG_PATH),
        help="Benchmark configuration defining the review scope.",
    )
    review_export_parser.add_argument(
        "--comparison-report",
        default="data/reports/asr/comparison/comparison.json",
        help="Comparison report providing priorities and flags.",
    )
    review_export_parser.add_argument(
        "--comparison-config",
        default=str(DEFAULT_ASR_COMPARISON_CONFIG_PATH),
        help="Versioned comparison config used to verify report provenance.",
    )
    review_export_parser.add_argument(
        "--lexicon",
        default=str(DEFAULT_ASR_LEXICON_PATH),
        help="Versioned ASR lexicon used to verify report provenance.",
    )
    review_export_parser.add_argument(
        "--rules",
        default=str(DEFAULT_REVIEW_RULES_PATH),
        help="Versioned review rules YAML (homophones, interjections).",
    )
    review_export_parser.add_argument(
        "--output-dir",
        default="data/reports/asr/review",
        help="Directory receiving review.html and review.json.",
    )
    review_import_parser = subparsers.add_parser(
        "review-import",
        help="Persist exported review decisions into the catalog.",
    )
    review_import_parser.add_argument(
        "decisions_json", help="Decisions JSON exported from the review page."
    )
    review_import_parser.add_argument(
        "--catalog", default="data/catalog/catalog.sqlite", help="SQLite catalog path."
    )
    review_import_parser.add_argument(
        "--benchmark-config",
        default=str(DEFAULT_ASR_BENCHMARK_CONFIG_PATH),
        help="Benchmark configuration the decisions belong to.",
    )
    review_import_parser.add_argument(
        "--review-dir",
        default="data/reports/asr/review",
        help="Directory holding the review package the decisions came from.",
    )
    dataset_audit_parser = subparsers.add_parser(
        "dataset-audit",
        help="Read-only audit of provenance-pinned dataset freeze candidates.",
    )
    dataset_audit_parser.add_argument(
        "--catalog", default="data/catalog/catalog.sqlite", help="SQLite catalog path."
    )
    dataset_audit_parser.add_argument(
        "--config",
        default=str(DEFAULT_DATASET_FREEZE_CONFIG_PATH),
        help="Versioned dataset freeze configuration YAML.",
    )
    dataset_audit_parser.add_argument(
        "--standardized-root",
        default="data/work/standardized",
        help="Verified standardized audio root.",
    )
    dataset_audit_parser.add_argument(
        "--quality-report",
        help="Quality report override; defaults to the path pinned by the dataset config.",
    )
    dataset_audit_parser.add_argument(
        "--candidate-snapshot",
        default=str(DEFAULT_CANDIDATE_SNAPSHOT_PATH),
        help="Frozen candidate snapshot to verify.",
    )
    dataset_audit_parser.add_argument(
        "--report",
        default=str(DEFAULT_DATASET_AUDIT_REPORT_PATH),
        help="JSON audit report path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            summary = run_inventory(
                args.input_dir,
                catalog_path=args.catalog,
                report_dir=args.report_dir,
                extensions=args.extensions or DEFAULT_EXTENSIONS,
                metadata_profile_path=args.profile,
            )
        elif args.command == "ingest":
            summary = run_ingest(
                args.input_dir,
                catalog_path=args.catalog,
                raw_dir=args.raw_dir,
                inventory_report_dir=args.inventory_report_dir,
                report_dir=args.report_dir,
                extensions=args.extensions or DEFAULT_EXTENSIONS,
                metadata_profile_path=args.profile,
            )
        elif args.command == "quality":
            summary = run_quality(
                catalog_path=args.catalog,
                raw_dir=args.raw_dir,
                analysis_config_path=args.analysis_config,
                policy_path=args.policy,
                report_dir=args.report_dir,
                force=args.force,
            )
        elif args.command == "standardize":
            selected_assets = (args.asset_sha256s or []) + _load_asset_manifest(
                args.asset_manifest
            )
            summary = run_standardization(
                catalog_path=args.catalog,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                report_dir=args.report_dir,
                force=args.force,
                asset_sha256s=selected_assets or None,
                limit=args.limit,
            )
        elif args.command == "asr-prepare":
            summary = prepare_asr_benchmark(
                catalog_path=args.catalog,
                config_path=args.config,
                quality_report_path=args.quality_report,
                output_dir=args.output_dir,
            )
        elif args.command == "asr-evaluate":
            summary = evaluate_asr_output(
                result_path=args.result_json,
                catalog_path=args.catalog,
                benchmark_config_path=args.benchmark_config,
                report_root=args.report_root,
            )
        elif args.command == "asr-compare":
            summary = compare_asr_backends(
                catalog_path=args.catalog,
                benchmark_config_path=args.benchmark_config,
                comparison_config_path=args.config,
                lexicon_path=args.lexicon,
                report_dir=args.report_dir,
            )
        elif args.command == "review-export":
            summary = review_export(
                catalog_path=args.catalog,
                benchmark_config_path=args.benchmark_config,
                comparison_report_path=args.comparison_report,
                comparison_config_path=args.comparison_config,
                lexicon_path=args.lexicon,
                rules_path=args.rules,
                output_dir=args.output_dir,
            )
        elif args.command == "review-import":
            summary = review_import(
                decisions_path=args.decisions_json,
                catalog_path=args.catalog,
                benchmark_config_path=args.benchmark_config,
                review_dir=args.review_dir,
            )
        elif args.command == "dataset-audit":
            summary = run_dataset_audit(
                catalog_path=args.catalog,
                config_path=args.config,
                standardized_root=args.standardized_root,
                quality_report_path=args.quality_report,
                candidate_snapshot_path=args.candidate_snapshot,
                report_path=args.report,
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as error:
        print(
            f"{args.command} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary["status"] == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
