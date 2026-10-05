from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dots_tts_lab.acoustic_fingerprint import load_acoustic_fingerprint_vectors


_EVIDENCE_TYPES = ("exact_audio", "exact_text", "near_audio", "near_text")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_verified_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Duplicate evidence report does not exist: {resolved}")
    content = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Duplicate evidence report SHA-256 drift: {resolved}; "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Duplicate evidence report must be an object: {resolved}")
    return payload


def stable_similarity_groups(
    asset_sha256s: Iterable[str],
    accepted_pairs: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return deterministic connected components, including singleton assets."""
    assets = list(asset_sha256s)
    if len(set(assets)) != len(assets):
        raise RuntimeError("Duplicate asset in similarity group candidate set")
    for asset_sha256 in assets:
        _require_sha256(asset_sha256, "similarity group asset")
    parent = {asset_sha256: asset_sha256 for asset_sha256 in assets}

    def find(asset_sha256: str) -> str:
        root = asset_sha256
        while parent[root] != root:
            root = parent[root]
        while parent[asset_sha256] != asset_sha256:
            next_asset = parent[asset_sha256]
            parent[asset_sha256] = root
            asset_sha256 = next_asset
        return root

    normalized_pairs = set()
    for left, right in accepted_pairs:
        if left == right:
            raise RuntimeError("Similarity group edge cannot be a self-edge")
        ordered = tuple(sorted((left, right)))
        if ordered[0] not in parent or ordered[1] not in parent:
            raise RuntimeError("Similarity group edge references an unknown asset")
        normalized_pairs.add(ordered)
    for left, right in sorted(normalized_pairs):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        low_root, high_root = sorted((left_root, right_root))
        parent[high_root] = low_root

    components: dict[str, list[str]] = {}
    for asset_sha256 in sorted(assets):
        components.setdefault(find(asset_sha256), []).append(asset_sha256)
    groups = []
    for members in components.values():
        member_asset_sha256s = sorted(members)
        groups.append(
            {
                "group_id": canonical_sha256(member_asset_sha256s),
                "member_count": len(member_asset_sha256s),
                "member_asset_sha256s": member_asset_sha256s,
            }
        )
    groups.sort(key=lambda group: tuple(group["member_asset_sha256s"]))
    if len({group["group_id"] for group in groups}) != len(groups):
        raise RuntimeError("Similarity group SHA-256 collision")
    return groups


def build_duplicate_edges(
    *,
    candidate_snapshot: dict[str, Any],
    exact_audio_report: dict[str, Any],
    exact_text_report: dict[str, Any],
    near_text_report: dict[str, Any],
    acoustic_fingerprint_report: dict[str, Any],
    text_threshold: float,
    acoustic_threshold: float,
    acoustic_duration_ratio_min: float,
    source_sha256s: dict[str, str],
) -> dict[str, Any]:
    """Normalize exact and calibrated near evidence into canonical catalog edges."""
    if not 0.0 <= text_threshold <= 1.0:
        raise ValueError("text_threshold must be in [0, 1]")
    if not 0.0 <= acoustic_threshold <= 1.0:
        raise ValueError("acoustic_threshold must be in [0, 1]")
    if not 0.0 < acoustic_duration_ratio_min <= 1.0:
        raise ValueError("acoustic_duration_ratio_min must be in (0, 1]")
    required_sources = {
        "candidate_snapshot",
        "exact_audio",
        "exact_text",
        "near_text",
        "acoustic_fingerprint",
    }
    if set(source_sha256s) != required_sources:
        raise RuntimeError("Duplicate edge source SHA-256 set is incomplete")
    for name, sha256 in source_sha256s.items():
        _require_sha256(sha256, f"{name} report")

    snapshot_items = candidate_snapshot.get("items")
    if not isinstance(snapshot_items, list):
        raise RuntimeError("Candidate snapshot items are missing")
    candidate_by_asset = {}
    for item in snapshot_items:
        asset_sha256 = item.get("asset_sha256")
        _require_sha256(asset_sha256, "candidate snapshot asset")
        if asset_sha256 in candidate_by_asset:
            raise RuntimeError(f"Duplicate candidate snapshot asset: {asset_sha256}")
        duration = item.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise RuntimeError(f"Invalid candidate duration: {asset_sha256}")
        candidate_by_asset[asset_sha256] = item
    candidate_assets = sorted(candidate_by_asset)
    if candidate_snapshot.get("item_count") != len(candidate_assets):
        raise RuntimeError("Candidate snapshot item count mismatch")

    for name, report in (
        ("exact audio", exact_audio_report),
        ("exact text", exact_text_report),
        ("near text", near_text_report),
        ("acoustic fingerprint", acoustic_fingerprint_report),
    ):
        if report.get("status") != "succeeded":
            raise RuntimeError(f"{name} report did not succeed")
    expected_count_fields = (
        (exact_audio_report, "checked_asset_count"),
        (exact_text_report, "checked_asset_count"),
        (near_text_report, "input_asset_count"),
        (acoustic_fingerprint_report, "feature_count"),
    )
    for report, field in expected_count_fields:
        if report.get(field) != len(candidate_assets):
            raise RuntimeError(f"Duplicate evidence {field} does not match candidates")

    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_edge(
        left: str,
        right: str,
        evidence_type: str,
        score: float,
        threshold: float | None,
        evidence: dict[str, Any],
    ) -> None:
        if evidence_type not in _EVIDENCE_TYPES:
            raise RuntimeError(f"Unsupported duplicate evidence type: {evidence_type}")
        _require_sha256(left, "left duplicate evidence asset")
        _require_sha256(right, "right duplicate evidence asset")
        if left == right:
            raise RuntimeError("Duplicate evidence cannot contain a self-edge")
        left, right = sorted((left, right))
        if left not in candidate_by_asset or right not in candidate_by_asset:
            raise RuntimeError("Duplicate evidence references an unknown candidate")
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("Duplicate evidence score must be in [0, 1]")
        key = (left, right, evidence_type)
        record = {
            "left_asset_sha256": left,
            "right_asset_sha256": right,
            "evidence_type": evidence_type,
            "score": float(score),
            "threshold": threshold,
            "analysis_status": (
                "accepted_exact" if evidence_type.startswith("exact_") else "pending_review"
            ),
            "evidence_json": canonical_json(evidence),
        }
        existing = edges_by_key.get(key)
        if existing is not None and existing != record:
            raise RuntimeError(f"Conflicting duplicate evidence for edge: {key}")
        edges_by_key[key] = record

    _add_exact_group_edges(
        report=exact_audio_report,
        evidence_type="exact_audio",
        member_field="members",
        member_asset_field="asset_sha256",
        source_sha256=source_sha256s["exact_audio"],
        add_edge=add_edge,
    )
    _add_exact_group_edges(
        report=exact_text_report,
        evidence_type="exact_text",
        member_field="member_asset_sha256s",
        member_asset_field=None,
        source_sha256=source_sha256s["exact_text"],
        add_edge=add_edge,
    )

    near_pairs = near_text_report.get("pairs")
    if not isinstance(near_pairs, list):
        raise RuntimeError("Near-text report pairs are missing")
    for pair in near_pairs:
        similarity = pair.get("similarity")
        if not isinstance(similarity, (int, float)):
            raise RuntimeError("Near-text pair similarity is invalid")
        if similarity < text_threshold:
            continue
        add_edge(
            pair.get("left_asset_sha256"),
            pair.get("right_asset_sha256"),
            "near_text",
            float(similarity),
            text_threshold,
            {
                "metric": near_text_report.get("metric"),
                "report_sha256": source_sha256s["near_text"],
                "pair": pair,
            },
        )

    vectors = load_acoustic_fingerprint_vectors(acoustic_fingerprint_report)
    if set(vectors) != set(candidate_assets):
        raise RuntimeError("Acoustic fingerprint asset set does not match candidates")
    matrix = np.stack([vectors[asset] for asset in candidate_assets]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarities = matrix @ matrix.T
    acoustic_scored_pair_count = 0
    for left_index, left in enumerate(candidate_assets):
        left_duration = float(candidate_by_asset[left]["duration_seconds"])
        for right_index in range(left_index + 1, len(candidate_assets)):
            right = candidate_assets[right_index]
            right_duration = float(candidate_by_asset[right]["duration_seconds"])
            duration_ratio = min(left_duration, right_duration) / max(
                left_duration, right_duration
            )
            if duration_ratio < acoustic_duration_ratio_min:
                continue
            acoustic_scored_pair_count += 1
            similarity = float(np.clip(similarities[left_index, right_index], -1.0, 1.0))
            if similarity < acoustic_threshold:
                continue
            add_edge(
                left,
                right,
                "near_audio",
                similarity,
                acoustic_threshold,
                {
                    "duration_ratio": duration_ratio,
                    "fingerprint_config_sha256": acoustic_fingerprint_report.get(
                        "fingerprint_config_sha256"
                    ),
                    "metric": "fingerprint_cosine",
                    "report_sha256": source_sha256s["acoustic_fingerprint"],
                },
            )

    edges = sorted(
        edges_by_key.values(),
        key=lambda edge: (
            edge["evidence_type"],
            edge["left_asset_sha256"],
            edge["right_asset_sha256"],
        ),
    )
    counts = {
        evidence_type: sum(edge["evidence_type"] == evidence_type for edge in edges)
        for evidence_type in _EVIDENCE_TYPES
    }
    return {
        "candidate_asset_sha256s": candidate_assets,
        "edges": edges,
        "edge_count": len(edges),
        "edge_counts_by_evidence_type": counts,
        "acoustic_scored_pair_count": acoustic_scored_pair_count,
    }


def _add_exact_group_edges(
    *,
    report: dict[str, Any],
    evidence_type: str,
    member_field: str,
    member_asset_field: str | None,
    source_sha256: str,
    add_edge: Any,
) -> None:
    groups = report.get("groups")
    if not isinstance(groups, list):
        raise RuntimeError(f"{evidence_type} report groups are missing")
    for group in groups:
        members_value = group.get(member_field)
        if not isinstance(members_value, list) or len(members_value) < 2:
            raise RuntimeError(f"{evidence_type} group must contain at least two members")
        members = [
            member.get(member_asset_field) if member_asset_field is not None else member
            for member in members_value
        ]
        if len(set(members)) != len(members):
            raise RuntimeError(f"{evidence_type} group contains duplicate members")
        for left, right in itertools.combinations(sorted(members), 2):
            add_edge(
                left,
                right,
                evidence_type,
                1.0,
                None,
                {
                    "group_id": group.get("group_id"),
                    "report_sha256": source_sha256,
                },
            )


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"Invalid {label} SHA-256")
