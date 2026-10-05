"""Apply reviewed duplicate-group and neutral-overlap constraints to Slice 9 selections."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from dots_tts_lab.slice9_candidates import DEFAULT_SLICE9_CONFIG_PATH, load_slice9_selection_config
from dots_tts_lab.slice9_diversity import DEFAULT_SLICE9_DIVERSITY_REPORT_PATH
from dots_tts_lab.slice9_feature_snapshot import DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH
from dots_tts_lab.slice9_scoring import DEFAULT_SLICE9_RANKING_REPORT_PATH


DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/slice9_constrained_selection_v1.json"
)
DEFAULT_DUPLICATE_REPORT_PATH = Path("data/reports/datasets/fuxuan_v1/analysis/dataset_analysis.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _duplicate_groups(report: dict[str, Any], assets: set[str]) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    parent = {asset: asset for asset in assets}

    def find(asset: str) -> str:
        while parent[asset] != asset:
            parent[asset] = parent[parent[asset]]
            asset = parent[asset]
        return asset

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    preferred: dict[tuple[str, str], str] = {}
    for edge in report.get("duplicate_edges", []):
        if not isinstance(edge, dict) or edge.get("review_status") != "accepted":
            continue
        left = edge.get("left", {}).get("asset_sha256")
        right = edge.get("right", {}).get("asset_sha256")
        if left not in assets or right not in assets:
            continue
        union(left, right)
        note = edge.get("review", {}).get("review_note")
        if isinstance(note, str):
            try:
                parsed = json.loads(note)
            except json.JSONDecodeError:
                parsed = {}
            representative = parsed.get("representative_asset_sha256")
            if representative in (left, right):
                preferred[tuple(sorted((left, right)))] = representative
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        members_by_root[find(asset)].append(asset)
    group_for_asset: dict[str, str] = {}
    representative_for_group: dict[str, str] = {}
    groups: list[dict[str, Any]] = []
    for members in sorted((sorted(values) for values in members_by_root.values()), key=lambda values: values[0]):
        group_id = hashlib.sha256("|".join(members).encode("ascii")).hexdigest()[:20]
        for asset in members:
            group_for_asset[asset] = group_id
        choices = [value for pair, value in preferred.items() if set(pair).issubset(members)]
        representative = choices[0] if choices else members[0]
        representative_for_group[group_id] = representative
        groups.append({"group_id": group_id, "asset_sha256s": members, "representative_asset_sha256": representative, "reviewed_duplicate": len(members) > 1})
    return group_for_asset, representative_for_group, groups


def build_slice9_constrained_selection(
    *,
    config_path: str | Path = DEFAULT_SLICE9_CONFIG_PATH,
    diversity_report_path: str | Path = DEFAULT_SLICE9_DIVERSITY_REPORT_PATH,
    ranking_report_path: str | Path = DEFAULT_SLICE9_RANKING_REPORT_PATH,
    feature_snapshot_path: str | Path = DEFAULT_SLICE9_FEATURE_SNAPSHOT_PATH,
    duplicate_report_path: str | Path = DEFAULT_DUPLICATE_REPORT_PATH,
    output_path: str | Path = DEFAULT_SLICE9_CONSTRAINED_REPORT_PATH,
) -> dict[str, Any]:
    config = load_slice9_selection_config(config_path)
    diversity_path = Path(diversity_report_path).resolve()
    ranking_path = Path(ranking_report_path).resolve()
    snapshot_path = Path(feature_snapshot_path).resolve()
    duplicate_path = Path(duplicate_report_path).resolve()
    diversity = _load_json(diversity_path, label="Slice 9 diversity report")
    ranking = _load_json(ranking_path, label="Slice 9 ranking report")
    snapshot = _load_json(snapshot_path, label="Slice 9 feature snapshot")
    duplicate_report = _load_json(duplicate_path, label="duplicate analysis report")
    all_assets = {item["asset_sha256"] for item in snapshot.get("items", [])}
    group_for_asset, representative_for_group, groups = _duplicate_groups(duplicate_report, all_assets)
    selected_across: dict[str, set[str]] = {}
    output_pools: dict[str, Any] = {}
    for pool_id, definition in config["pools"].items():
        ranked = ranking["pools"][pool_id]["items"]
        by_asset = {item["asset_sha256"]: item for item in ranked}
        diverse_selected = diversity["pools"][pool_id]["selected"]
        selected: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        replacements: list[dict[str, str]] = []
        neutral_limit = int(config["ranking"]["neutral_pool_overlap_max"])
        is_emotion_neutral = pool_id == "emotion_neutral"
        main_neutral_assets = selected_across.get("main_neutral", set())

        def add_candidate(asset: str, *, source: str) -> bool:
            if asset not in by_asset:
                return False
            group_id = group_for_asset[asset]
            if group_id in seen_groups:
                return False
            if is_emotion_neutral and asset in main_neutral_assets and len(set(item["asset_sha256"] for item in selected) & main_neutral_assets) >= neutral_limit:
                return False
            representative = representative_for_group[group_id]
            chosen = representative if representative in by_asset else asset
            if chosen in main_neutral_assets and is_emotion_neutral and len(set(item["asset_sha256"] for item in selected) & main_neutral_assets) >= neutral_limit:
                chosen = asset
            if chosen not in by_asset or group_for_asset[chosen] in seen_groups:
                return False
            if is_emotion_neutral and chosen in main_neutral_assets and len(set(item["asset_sha256"] for item in selected) & main_neutral_assets) >= neutral_limit:
                return False
            if chosen != asset:
                replacements.append({"requested_asset_sha256": asset, "representative_asset_sha256": chosen, "group_id": group_id})
            chosen_item = dict(by_asset[chosen])
            chosen_item["duplicate_group_id"] = group_for_asset[chosen]
            chosen_item["duplicate_group_representative"] = chosen == representative_for_group[group_id]
            chosen_item["selection_source"] = source
            selected.append(chosen_item)
            seen_groups.add(group_for_asset[chosen])
            return True

        for item in diverse_selected:
            if len(selected) >= int(definition["top_k"]):
                break
            add_candidate(item["asset_sha256"], source="diversity_rerank")
        if len(selected) < int(definition["top_k"]):
            for item in ranked:
                if len(selected) >= int(definition["top_k"]):
                    break
                add_candidate(item["asset_sha256"], source="constraint_fill")
        selected_assets = {item["asset_sha256"] for item in selected}
        boundary = [item for item in ranked if item["asset_sha256"] not in selected_assets][: int(definition["boundary_count"])]
        selected_across[pool_id] = selected_assets
        output_pools[pool_id] = {
            "candidate_count": len(ranked),
            "top_k": int(definition["top_k"]),
            "selected_count": len(selected),
            "selected": [dict(item, constrained_rank=index) for index, item in enumerate(selected, start=1)],
            "boundary": boundary,
            "replacement_count": len(replacements),
            "replacements": replacements,
            "unique_duplicate_groups_selected": len(seen_groups),
            "meets_top_k": len(selected) == int(definition["top_k"]),
        }
    neutral_overlap = sorted(selected_across.get("main_neutral", set()) & selected_across.get("emotion_neutral", set()))
    _require(len(neutral_overlap) <= int(config["ranking"]["neutral_pool_overlap_max"]), "Neutral pool overlap exceeds configured maximum")
    result = {
        "schema_version": 1,
        "constrained_selection_schema_version": "slice9_duplicate_representative_constraints@1",
        "status": "succeeded",
        "selection_id": config["selection_id"],
        "selection_version": config["selection_version"],
        "dataset_id": config["input"]["dataset_id"],
        "dataset_version": config["input"]["dataset_version"],
        "dataset_tree_sha256": config["input"]["dataset_tree_sha256"],
        "input_hashes": {
            "config_sha256": _sha256(Path(config_path).resolve()),
            "diversity_report_sha256": _sha256(diversity_path),
            "ranking_report_sha256": _sha256(ranking_path),
            "feature_snapshot_sha256": _sha256(snapshot_path),
            "duplicate_report_sha256": _sha256(duplicate_path),
        },
        "duplicate_groups": groups,
        "duplicate_group_count": len(groups),
        "reviewed_duplicate_group_count": sum(group["reviewed_duplicate"] for group in groups),
        "neutral_overlap": {"max_allowed": int(config["ranking"]["neutral_pool_overlap_max"]), "count": len(neutral_overlap), "assets": neutral_overlap},
        "pool_count": len(output_pools),
        "pools": output_pools,
    }
    resolved_output = Path(output_path).resolve()
    _atomic_write_json(resolved_output, result)
    return {**result, "report_path": str(resolved_output)}

