from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

from dots_tts_lab.duplicate_graph import canonical_json, canonical_sha256


EXCLUDING_SPEAKER_STATUSES = {"exclude_wrong_speaker", "exclude_uncertain"}


def apply_reviewed_exclusions(
    *,
    items: Iterable[dict[str, Any]],
    groups: Iterable[dict[str, Any]],
    review: dict[str, Any],
) -> dict[str, Any]:
    item_rows = [dict(item) for item in items]
    items_by_asset = {item.get("asset_sha256"): item for item in item_rows}
    if (
        not item_rows
        or None in items_by_asset
        or len(items_by_asset) != len(item_rows)
    ):
        raise RuntimeError("Selection items must have unique asset SHA-256 values")
    group_rows = _validate_groups(groups, set(items_by_asset))
    group_by_asset = {
        asset: group
        for group in group_rows
        for asset in group["member_asset_sha256s"]
    }

    edge_decisions = review.get("edge_decisions")
    speaker_decisions = review.get("speaker_decisions")
    if not isinstance(edge_decisions, list) or not isinstance(
        speaker_decisions, list
    ):
        raise RuntimeError("Manual review decisions are missing")
    decisions_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion_reasons: dict[str, list[dict[str, str]]] = defaultdict(list)
    for decision in edge_decisions:
        left = decision.get("left_asset_sha256")
        right = decision.get("right_asset_sha256")
        representative = decision.get("representative_asset_sha256")
        excluded = decision.get("exclude_asset_sha256s")
        if (
            decision.get("review_status") != "accepted"
            or left not in items_by_asset
            or right not in items_by_asset
            or representative not in (left, right)
            or not isinstance(excluded, list)
            or set(excluded) != {left, right} - {representative}
            or group_by_asset[left]["group_id"] != group_by_asset[right]["group_id"]
        ):
            raise RuntimeError("Duplicate review decision is inconsistent")
        group_id = group_by_asset[left]["group_id"]
        decisions_by_group[group_id].append(decision)
        for asset in excluded:
            exclusion_reasons[asset].append(
                {
                    "kind": "duplicate_non_representative",
                    "review_status": "accepted",
                    "note": decision["review_note"],
                }
            )

    duplicate_resolutions = []
    for group in group_rows:
        members = group["member_asset_sha256s"]
        if len(members) == 1:
            continue
        decisions = decisions_by_group.get(group["group_id"], [])
        representatives = {
            decision["representative_asset_sha256"] for decision in decisions
        }
        excluded = {
            asset
            for decision in decisions
            for asset in decision["exclude_asset_sha256s"]
        }
        retained = set(members) - excluded
        if len(representatives) != 1 or retained != representatives:
            raise RuntimeError(
                f"Multi-asset group lacks one explicit representative: {group['group_id']}"
            )
        duplicate_resolutions.append(
            {
                "group_id": group["group_id"],
                "canonical_member_asset_sha256s": members,
                "representative_asset_sha256": next(iter(representatives)),
                "exclude_asset_sha256s": sorted(excluded),
            }
        )

    seen_speaker_assets = set()
    for decision in speaker_decisions:
        asset = decision.get("asset_sha256")
        status = decision.get("review_status")
        if asset not in items_by_asset or asset in seen_speaker_assets:
            raise RuntimeError("Speaker review decision asset is invalid or duplicated")
        seen_speaker_assets.add(asset)
        if status in EXCLUDING_SPEAKER_STATUSES:
            exclusion_reasons[asset].append(
                {
                    "kind": "speaker_review_exclusion",
                    "review_status": status,
                    "note": decision["review_note"],
                }
            )
        elif status != "confirmed_same_speaker":
            raise RuntimeError(f"Unsupported speaker review status: {status}")

    excluded_assets = set(exclusion_reasons)
    selected_items = [
        items_by_asset[asset]
        for asset in sorted(set(items_by_asset) - excluded_assets)
    ]
    excluded_items = [
        {
            "asset_sha256": asset,
            "fid": items_by_asset[asset].get("fid"),
            "emotion_primary": items_by_asset[asset].get("emotion_primary"),
            "reasons": exclusion_reasons[asset],
        }
        for asset in sorted(excluded_assets)
    ]
    selected_groups = []
    for group in group_rows:
        selected_members = sorted(
            set(group["member_asset_sha256s"]) - excluded_assets
        )
        if not selected_members:
            continue
        selected_group = {
            "group_id": group["group_id"],
            "member_asset_sha256s": selected_members,
        }
        if selected_members != group["member_asset_sha256s"]:
            selected_group["canonical_member_asset_sha256s"] = group[
                "member_asset_sha256s"
            ]
        selected_groups.append(selected_group)
    if len(selected_items) != len(set(items_by_asset) - excluded_assets):
        raise RuntimeError("Selection contains duplicated assets")
    if sum(len(group["member_asset_sha256s"]) for group in selected_groups) != len(
        selected_items
    ):
        raise RuntimeError("Selected groups do not cover selected items exactly once")
    result = {
        "candidate_count": len(item_rows),
        "selected_count": len(selected_items),
        "excluded_count": len(excluded_items),
        "oversampled_count": 0,
        "duplicate_resolution_count": len(duplicate_resolutions),
        "duplicate_resolutions": sorted(
            duplicate_resolutions, key=lambda row: row["group_id"]
        ),
        "selected_items": selected_items,
        "excluded_items": excluded_items,
        "selected_groups": sorted(selected_groups, key=lambda row: row["group_id"]),
    }
    result["selection_sha256"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _validate_groups(
    groups: Iterable[dict[str, Any]], candidate_assets: set[str]
) -> list[dict[str, Any]]:
    group_rows = []
    seen = set()
    for source in groups:
        group_id = source.get("group_id")
        members = source.get("member_asset_sha256s")
        if not isinstance(members, list) or not members:
            raise RuntimeError("Selection group members are malformed")
        ordered = sorted(members)
        if (
            canonical_sha256(ordered) != group_id
            or len(set(ordered)) != len(ordered)
            or seen.intersection(ordered)
        ):
            raise RuntimeError("Selection groups are not canonical and disjoint")
        seen.update(ordered)
        group_rows.append(
            {"group_id": group_id, "member_asset_sha256s": ordered}
        )
    if seen != candidate_assets:
        raise RuntimeError("Selection groups do not cover candidate assets")
    return sorted(group_rows, key=lambda row: row["group_id"])
