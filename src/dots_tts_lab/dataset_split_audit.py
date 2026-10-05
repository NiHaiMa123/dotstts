from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

from dots_tts_lab.dataset_analysis import normalize_text_for_similarity
from dots_tts_lab.dataset_freeze import TextSimilarityConfig
from dots_tts_lab.dataset_split import SPLIT_NAMES, assert_no_group_crosses_splits
from dots_tts_lab.duplicate_graph import canonical_json


def audit_split_leakage(
    *,
    items: Iterable[dict[str, Any]],
    split_plan: dict[str, Any],
    similarity_edges: Iterable[dict[str, Any]],
    text_config: TextSimilarityConfig,
) -> dict[str, Any]:
    item_rows = [dict(item) for item in items]
    items_by_asset = {item.get("asset_sha256"): item for item in item_rows}
    assignments = split_plan.get("asset_assignments")
    if (
        not item_rows
        or None in items_by_asset
        or len(items_by_asset) != len(item_rows)
        or not isinstance(assignments, dict)
        or set(assignments) != set(items_by_asset)
        or any(split not in SPLIT_NAMES for split in assignments.values())
    ):
        raise RuntimeError("Split audit items and assignments do not match")

    violations = []
    try:
        assert_no_group_crosses_splits(split_plan)
    except RuntimeError as error:
        violations.append(
            {"check": "similarity_group", "identity": "group_plan", "detail": str(error)}
        )

    exact_audio_groups = _cross_split_value_groups(
        item_rows,
        assignments,
        field="audio_sha256",
        check="exact_audio_sha256",
    )
    violations.extend(exact_audio_groups)

    normalized_text_rows = []
    for item in item_rows:
        normalized = normalize_text_for_similarity(
            item.get("text_exact"),
            config=text_config,
        )["text_normalized"]
        if not normalized:
            raise RuntimeError(
                f"Split audit normalized text is empty: {item['asset_sha256']}"
            )
        normalized_text_rows.append(
            {"asset_sha256": item["asset_sha256"], "value": normalized}
        )
    exact_text_groups = _cross_split_value_groups(
        normalized_text_rows,
        assignments,
        field="value",
        check="exact_normalized_text",
    )
    violations.extend(exact_text_groups)

    source_rows = [
        {
            "asset_sha256": item["asset_sha256"],
            "value": canonical_json(
                {
                    "source_root_key": item.get("source_root_key"),
                    "source_path_key": item.get("source_path_key"),
                    "raw_relative_path": item.get("raw_relative_path"),
                }
            ),
        }
        for item in item_rows
    ]
    if any(
        item.get("source_root_key") is None
        or item.get("source_path_key") is None
        or item.get("raw_relative_path") is None
        for item in item_rows
    ):
        raise RuntimeError("Split audit source provenance is incomplete")
    source_groups = _cross_split_value_groups(
        source_rows,
        assignments,
        field="value",
        check="source_provenance_identity",
    )
    violations.extend(source_groups)

    selected_edges = []
    ignored_edges = 0
    for source_edge in similarity_edges:
        edge = dict(source_edge)
        left = edge.get("left_asset_sha256")
        right = edge.get("right_asset_sha256")
        evidence_type = edge.get("evidence_type")
        if evidence_type not in (
            "exact_audio",
            "exact_text",
            "near_audio",
            "near_text",
        ):
            raise RuntimeError("Split audit edge has invalid evidence type")
        if left not in items_by_asset or right not in items_by_asset:
            ignored_edges += 1
            continue
        selected_edges.append(edge)
        if assignments[left] != assignments[right]:
            violations.append(
                {
                    "check": "similarity_edge",
                    "identity": f"{evidence_type}:{left}:{right}",
                    "detail": f"{assignments[left]} != {assignments[right]}",
                }
            )

    violations.sort(key=lambda row: (row["check"], row["identity"]))
    checks = {
        "similarity_group": {
            "examined_count": len(split_plan["groups"]),
            "violation_count": sum(
                violation["check"] == "similarity_group" for violation in violations
            ),
        },
        "exact_audio_sha256": {
            "examined_count": len(item_rows),
            "duplicate_value_count": _duplicate_value_count(item_rows, "audio_sha256"),
            "violation_count": len(exact_audio_groups),
        },
        "exact_normalized_text": {
            "examined_count": len(normalized_text_rows),
            "duplicate_value_count": _duplicate_value_count(
                normalized_text_rows, "value"
            ),
            "violation_count": len(exact_text_groups),
        },
        "similarity_edge": {
            "catalog_edge_count": len(selected_edges) + ignored_edges,
            "selected_endpoint_edge_count": len(selected_edges),
            "excluded_endpoint_edge_count": ignored_edges,
            "violation_count": sum(
                violation["check"] == "similarity_edge" for violation in violations
            ),
        },
        "source_provenance_identity": {
            "examined_count": len(source_rows),
            "duplicate_value_count": _duplicate_value_count(source_rows, "value"),
            "violation_count": len(source_groups),
        },
    }
    result = {
        "schema_version": 1,
        "status": "passed" if not violations else "failed",
        "item_count": len(item_rows),
        "split_counts": {
            split: sum(value == split for value in assignments.values())
            for split in SPLIT_NAMES
        },
        "checks": checks,
        "violation_count": len(violations),
        "violations": violations,
    }
    result["audit_sha256"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def assert_split_audit_passes(report: dict[str, Any]) -> None:
    if report.get("status") != "passed" or report.get("violation_count") != 0:
        raise RuntimeError(
            f"Split leakage audit failed with {report.get('violation_count')} violations"
        )


def _cross_split_value_groups(
    rows: Iterable[dict[str, Any]],
    assignments: dict[str, str],
    *,
    field: str,
    check: str,
) -> list[dict[str, str]]:
    by_value: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        asset = row.get("asset_sha256")
        if not isinstance(value, str) or not value or asset not in assignments:
            raise RuntimeError(f"Split audit {check} input is malformed")
        by_value[value].append(asset)
    violations = []
    for value, assets in sorted(by_value.items()):
        splits = sorted({assignments[asset] for asset in assets})
        if len(splits) > 1:
            violations.append(
                {
                    "check": check,
                    "identity": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "detail": canonical_json(
                        {"assets": sorted(assets), "splits": splits}
                    ),
                }
            )
    return violations


def _duplicate_value_count(rows: Iterable[dict[str, Any]], field: str) -> int:
    counts = defaultdict(int)
    for row in rows:
        counts[row[field]] += 1
    return sum(count > 1 for count in counts.values())
