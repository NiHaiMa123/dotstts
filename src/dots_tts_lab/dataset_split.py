from __future__ import annotations

import hashlib
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Iterable, Mapping

from dots_tts_lab.catalog import Catalog
from dots_tts_lab.dataset_freeze import SplitConfig
from dots_tts_lab.duplicate_graph import canonical_json, canonical_sha256


SPLIT_NAMES = ("train", "validation", "test")


def load_verified_similarity_groups(
    catalog: Catalog,
    *,
    run_id: str,
    candidate_snapshot_sha256: str,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], str]:
    with catalog.read_only_session() as connection:
        run = connection.execute(
            """
            SELECT candidate_snapshot_sha256, candidate_count
            FROM dataset_analysis_run
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise RuntimeError(f"Dataset analysis run does not exist: {run_id}")
        if (
            run["candidate_snapshot_sha256"] != candidate_snapshot_sha256
            or int(run["candidate_count"]) != candidate_count
        ):
            raise RuntimeError("Split inputs differ from the dataset analysis run")
        unresolved = connection.execute(
            """
            SELECT count(*)
            FROM dataset_similarity_edge AS edge
            WHERE edge.run_id = ?
              AND edge.analysis_status = 'pending_review'
              AND NOT EXISTS (
                  SELECT 1
                  FROM dataset_similarity_edge_review AS review
                  WHERE review.run_id = edge.run_id
                    AND review.left_asset_sha256 = edge.left_asset_sha256
                    AND review.right_asset_sha256 = edge.right_asset_sha256
                    AND review.evidence_type = edge.evidence_type
              )
            """,
            (run_id,),
        ).fetchone()[0]
        if unresolved:
            raise RuntimeError("Split planning requires every near edge to be reviewed")
        rows = connection.execute(
            """
            SELECT grouped.group_id, grouped.member_count,
                   grouped.edge_review_snapshot_sha256,
                   member.asset_sha256
            FROM dataset_similarity_group AS grouped
            JOIN dataset_similarity_group_member AS member
              ON member.run_id = grouped.run_id
             AND member.group_id = grouped.group_id
            WHERE grouped.run_id = ?
            ORDER BY grouped.group_id, member.asset_sha256
            """,
            (run_id,),
        ).fetchall()
    groups_by_id: dict[str, dict[str, Any]] = {}
    snapshots = set()
    for row in rows:
        group_id = str(row["group_id"])
        group = groups_by_id.setdefault(
            group_id,
            {
                "group_id": group_id,
                "member_count": int(row["member_count"]),
                "member_asset_sha256s": [],
            },
        )
        group["member_asset_sha256s"].append(str(row["asset_sha256"]))
        snapshots.add(str(row["edge_review_snapshot_sha256"]))
    if len(snapshots) != 1:
        raise RuntimeError("Similarity groups have mixed review snapshots")
    groups = list(groups_by_id.values())
    if any(
        group["member_count"] != len(group["member_asset_sha256s"])
        for group in groups
    ):
        raise RuntimeError("Similarity group member count mismatch")
    return groups, snapshots.pop()


def largest_remainder_targets(
    total: int, ratios: Mapping[str, float]
) -> dict[str, int]:
    if total < 1 or tuple(ratios) != SPLIT_NAMES:
        raise RuntimeError("Split targets require a positive total and canonical split order")
    decimal_ratios = {name: Decimal(str(ratios[name])) for name in SPLIT_NAMES}
    if sum(decimal_ratios.values()) != Decimal("1"):
        raise RuntimeError("Split ratios must sum to one")
    raw = {name: Decimal(total) * decimal_ratios[name] for name in SPLIT_NAMES}
    targets = {
        name: int(raw[name].to_integral_value(rounding=ROUND_FLOOR))
        for name in SPLIT_NAMES
    }
    remaining = total - sum(targets.values())
    ranked = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - targets[name]), SPLIT_NAMES.index(name)),
    )
    for name in ranked[:remaining]:
        targets[name] += 1
    return targets


def plan_group_aware_split(
    *,
    items: Iterable[dict[str, Any]],
    groups: Iterable[dict[str, Any]],
    split_config: SplitConfig,
    dataset_config_sha256: str,
) -> dict[str, Any]:
    """Assign complete similarity groups deterministically without stratification."""
    if not _is_sha256(dataset_config_sha256):
        raise RuntimeError("Dataset config SHA-256 is invalid")
    item_rows = [dict(item) for item in items]
    assets: dict[str, dict[str, Any]] = {}
    for item in item_rows:
        asset = item.get("asset_sha256")
        duration = item.get("duration_seconds")
        if not _is_sha256(asset) or asset in assets:
            raise RuntimeError("Split items must have unique valid asset SHA-256 values")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise RuntimeError(f"Split item duration is invalid: {asset}")
        assets[asset] = item
    if not assets:
        raise RuntimeError("Split items cannot be empty")

    prepared_groups = []
    seen_assets: set[str] = set()
    for source_group in groups:
        group_id = source_group.get("group_id")
        members = source_group.get("member_asset_sha256s")
        if not isinstance(members, list) or not members:
            raise RuntimeError("Similarity group members must be a non-empty list")
        ordered_members = sorted(members)
        canonical_members = source_group.get(
            "canonical_member_asset_sha256s", ordered_members
        )
        if not isinstance(canonical_members, list) or not canonical_members:
            raise RuntimeError("Canonical similarity group members are malformed")
        ordered_canonical_members = sorted(canonical_members)
        if (
            not _is_sha256(group_id)
            or len(set(ordered_members)) != len(ordered_members)
            or len(set(ordered_canonical_members)) != len(ordered_canonical_members)
            or not set(ordered_members).issubset(ordered_canonical_members)
            or canonical_sha256(ordered_canonical_members) != group_id
        ):
            raise RuntimeError("Similarity group ID or members are not canonical")
        overlap = seen_assets.intersection(ordered_members)
        if overlap:
            raise RuntimeError(f"Assets belong to multiple similarity groups: {sorted(overlap)}")
        unknown = set(ordered_members) - set(assets)
        if unknown:
            raise RuntimeError(f"Similarity groups contain unknown assets: {sorted(unknown)}")
        seen_assets.update(ordered_members)
        duration = sum(
            (Decimal(str(assets[asset]["duration_seconds"])) for asset in ordered_members),
            Decimal("0"),
        )
        stable_key = hashlib.sha256(
            f"{dataset_config_sha256}:{split_config.seed}:{group_id}".encode("utf-8")
        ).hexdigest()
        prepared = {
            "group_id": group_id,
            "member_asset_sha256s": ordered_members,
            "item_count": len(ordered_members),
            "duration": duration,
            "stable_key": stable_key,
        }
        if ordered_canonical_members != ordered_members:
            prepared["canonical_member_asset_sha256s"] = ordered_canonical_members
        prepared_groups.append(prepared)
    if seen_assets != set(assets):
        raise RuntimeError(
            f"Similarity groups do not cover split items: {sorted(set(assets) - seen_assets)}"
        )

    ratios = {
        "train": split_config.train_ratio,
        "validation": split_config.validation_ratio,
        "test": split_config.test_ratio,
    }
    targets = largest_remainder_targets(len(assets), ratios)
    decimal_ratios = {name: Decimal(str(ratios[name])) for name in SPLIT_NAMES}
    total_duration = sum(
        (Decimal(str(item["duration_seconds"])) for item in assets.values()),
        Decimal("0"),
    )
    duration_targets = {
        name: total_duration * decimal_ratios[name] for name in SPLIT_NAMES
    }
    group_targets = {
        name: Decimal(len(prepared_groups)) * decimal_ratios[name]
        for name in SPLIT_NAMES
    }
    prepared_groups.sort(
        key=lambda group: (
            -group["item_count"],
            -group["duration"],
            group["stable_key"],
            group["group_id"],
        )
    )

    assignments: dict[str, str] = {}
    for group in prepared_groups:
        candidates = []
        for split in SPLIT_NAMES:
            assignments[group["group_id"]] = split
            objective = _split_objective(
                prepared_groups,
                assignments,
                targets,
                duration_targets,
                group_targets,
            )
            tie_breaker = hashlib.sha256(
                f"{group['stable_key']}:{split}".encode("utf-8")
            ).hexdigest()
            candidates.append((objective, tie_breaker, SPLIT_NAMES.index(split), split))
            del assignments[group["group_id"]]
        assignments[group["group_id"]] = min(candidates)[-1]

    assignments = _improve_assignments(
        prepared_groups,
        assignments,
        targets,
        duration_targets,
        group_targets,
    )
    group_records = [
        {
            "group_id": group["group_id"],
            "split": assignments[group["group_id"]],
            "item_count": group["item_count"],
            "duration_seconds": float(group["duration"]),
            "member_asset_sha256s": group["member_asset_sha256s"],
            **(
                {
                    "canonical_member_asset_sha256s": group[
                        "canonical_member_asset_sha256s"
                    ]
                }
                if "canonical_member_asset_sha256s" in group
                else {}
            ),
        }
        for group in sorted(prepared_groups, key=lambda row: row["group_id"])
    ]
    asset_assignments = {
        asset: record["split"]
        for record in group_records
        for asset in record["member_asset_sha256s"]
    }
    summary = _split_summary(group_records, targets, duration_targets, group_targets)
    result = {
        "schema_version": 1,
        "algorithm": "deterministic_group_greedy_swap_v1",
        "scope": "group_constraint_pre_stratification",
        "seed": split_config.seed,
        "dataset_config_sha256": dataset_config_sha256,
        "candidate_count": len(assets),
        "group_count": len(group_records),
        "summary": summary,
        "groups": group_records,
        "asset_assignments": dict(sorted(asset_assignments.items())),
    }
    result["assignment_sha256"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    assert_no_group_crosses_splits(result)
    return result


def plan_emotion_stratified_split(
    *,
    items: Iterable[dict[str, Any]],
    groups: Iterable[dict[str, Any]],
    split_config: SplitConfig,
    dataset_config_sha256: str,
) -> dict[str, Any]:
    item_rows = [dict(item) for item in items]
    group_rows = [dict(group) for group in groups]
    base = plan_group_aware_split(
        items=item_rows,
        groups=group_rows,
        split_config=split_config,
        dataset_config_sha256=dataset_config_sha256,
    )
    items_by_asset = {item["asset_sha256"]: item for item in item_rows}
    for item in item_rows:
        for field in (split_config.strata_field, "emotion_weak_label"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise RuntimeError(
                    f"Emotion-stratified split item has no {field}: "
                    f"{item['asset_sha256']}"
                )
    prepared_groups = []
    for group in base["groups"]:
        members = group["member_asset_sha256s"]
        primary_counts = Counter(
            items_by_asset[asset][split_config.strata_field] for asset in members
        )
        weak_counts = Counter(
            items_by_asset[asset]["emotion_weak_label"] for asset in members
        )
        prepared_groups.append(
            {
                **group,
                "duration": Decimal(str(group["duration_seconds"])),
                "primary_counts": dict(sorted(primary_counts.items())),
                "weak_counts": dict(sorted(weak_counts.items())),
            }
        )
    ratios = {
        "train": split_config.train_ratio,
        "validation": split_config.validation_ratio,
        "test": split_config.test_ratio,
    }
    count_targets = {
        split: int(base["summary"][split]["target_item_count"])
        for split in SPLIT_NAMES
    }
    primary_totals = Counter(
        item[split_config.strata_field] for item in item_rows
    )
    weak_totals = Counter(item["emotion_weak_label"] for item in item_rows)
    low_resource_group_count = sum(
        split_config.low_resource_label in group["primary_counts"]
        for group in prepared_groups
    )
    primary_minimums = None
    if low_resource_group_count >= (
        1 + 2 * split_config.low_resource_min_groups_per_eval_split
    ):
        primary_minimums = {
            split_config.low_resource_label: {
                "validation": split_config.low_resource_min_groups_per_eval_split,
                "test": split_config.low_resource_min_groups_per_eval_split,
            }
        }
    primary_targets = balanced_strata_targets(
        dict(primary_totals),
        count_targets,
        ratios,
        minimums=primary_minimums,
    )
    weak_targets = balanced_strata_targets(dict(weak_totals), count_targets, ratios)
    total_duration = sum(
        (Decimal(str(item["duration_seconds"])) for item in item_rows),
        Decimal("0"),
    )
    decimal_ratios = {split: Decimal(str(ratios[split])) for split in SPLIT_NAMES}
    duration_targets = {
        split: total_duration * decimal_ratios[split] for split in SPLIT_NAMES
    }
    group_targets = {
        split: Decimal(len(prepared_groups)) * decimal_ratios[split]
        for split in SPLIT_NAMES
    }
    assignments = {group["group_id"]: group["split"] for group in prepared_groups}
    assignments = _improve_stratified_assignments(
        prepared_groups,
        assignments,
        count_targets=count_targets,
        primary_targets=primary_targets,
        duration_targets=duration_targets,
        group_targets=group_targets,
        low_resource_label=split_config.low_resource_label,
        minimum_eval_groups=split_config.low_resource_min_groups_per_eval_split,
    )
    group_records = [
        {
            key: value
            for key, value in group.items()
            if key not in ("duration", "split")
        }
        | {"split": assignments[group["group_id"]]}
        for group in sorted(prepared_groups, key=lambda row: row["group_id"])
    ]
    asset_assignments = {
        asset: group["split"]
        for group in group_records
        for asset in group["member_asset_sha256s"]
    }
    summary = _stratified_summary(
        group_records,
        primary_targets=primary_targets,
        weak_targets=weak_targets,
        count_targets=count_targets,
        duration_targets=duration_targets,
        group_targets=group_targets,
        low_resource_label=split_config.low_resource_label,
        minimum_eval_groups=split_config.low_resource_min_groups_per_eval_split,
    )
    result = {
        "schema_version": 1,
        "algorithm": "deterministic_group_emotion_swap_v1",
        "scope": "emotion_stratified_pre_exclusion",
        "seed": split_config.seed,
        "dataset_config_sha256": dataset_config_sha256,
        "candidate_count": len(item_rows),
        "group_count": len(group_records),
        "group_only_assignment_sha256": base["assignment_sha256"],
        "summary": summary,
        "groups": group_records,
        "asset_assignments": dict(sorted(asset_assignments.items())),
    }
    result["assignment_sha256"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    assert_no_group_crosses_splits(result)
    _assert_low_resource_coverage(
        group_records,
        low_resource_label=split_config.low_resource_label,
        minimum_eval_groups=split_config.low_resource_min_groups_per_eval_split,
    )
    return result


def balanced_strata_targets(
    totals: Mapping[str, int],
    split_targets: Mapping[str, int],
    ratios: Mapping[str, float],
    *,
    minimums: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, dict[str, int]]:
    if tuple(split_targets) != SPLIT_NAMES or tuple(ratios) != SPLIT_NAMES:
        raise RuntimeError("Strata targets require canonical split order")
    if not totals or any(not isinstance(value, int) or value < 0 for value in totals.values()):
        raise RuntimeError("Strata totals must be non-negative integers")
    if sum(totals.values()) != sum(split_targets.values()):
        raise RuntimeError("Strata totals and split targets differ")
    decimal_ratios = {split: Decimal(str(ratios[split])) for split in SPLIT_NAMES}
    labels = sorted(totals)
    requested_minimums = minimums or {}
    if any(label not in totals for label in requested_minimums):
        raise RuntimeError("Strata minimum references an unknown label")
    options_by_label: dict[str, list[tuple[Decimal, tuple[int, int, int]]]] = {}
    for label in labels:
        raw_targets = {
            split: Decimal(totals[label]) * decimal_ratios[split]
            for split in SPLIT_NAMES
        }
        requested = requested_minimums.get(label, {})
        if any(
            split not in SPLIT_NAMES
            or not isinstance(value, int)
            or value < 0
            for split, value in requested.items()
        ):
            raise RuntimeError("Strata minimums are malformed")
        options = []
        for validation_count in range(split_targets["validation"] + 1):
            for test_count in range(split_targets["test"] + 1):
                train_count = totals[label] - validation_count - test_count
                counts = (train_count, validation_count, test_count)
                if train_count < 0 or any(
                    counts[SPLIT_NAMES.index(split)] < requested.get(split, 0)
                    for split in SPLIT_NAMES
                ):
                    continue
                cost = sum(
                    (
                        abs(Decimal(counts[index]) - raw_targets[split])
                        for index, split in enumerate(SPLIT_NAMES)
                    ),
                    Decimal("0"),
                )
                options.append((cost, counts))
        if not options:
            raise RuntimeError(f"Strata minimums are infeasible for {label}")
        options_by_label[label] = sorted(options)

    states: dict[
        tuple[int, int], tuple[Decimal, tuple[tuple[int, int, int], ...]]
    ] = {(0, 0): (Decimal("0"), ())}
    processed_total = 0
    for label in labels:
        processed_total += totals[label]
        next_states = {}
        for (used_validation, used_test), (score, signature) in states.items():
            for option_cost, counts in options_by_label[label]:
                next_validation = used_validation + counts[1]
                next_test = used_test + counts[2]
                next_train = processed_total - next_validation - next_test
                if (
                    next_validation > split_targets["validation"]
                    or next_test > split_targets["test"]
                    or next_train > split_targets["train"]
                ):
                    continue
                key = (next_validation, next_test)
                candidate = (score + option_cost, signature + (counts,))
                existing = next_states.get(key)
                if existing is None or candidate < existing:
                    next_states[key] = candidate
        states = next_states
        if not states:
            raise RuntimeError("Could not balance strata targets against split totals")
    final = states.get(
        (split_targets["validation"], split_targets["test"])
    )
    if final is None:
        raise RuntimeError("Could not balance strata targets against split totals")
    signature = final[1]
    return {
        label: {
            split: signature[label_index][split_index]
            for split_index, split in enumerate(SPLIT_NAMES)
        }
        for label_index, label in enumerate(labels)
    }


def assert_no_group_crosses_splits(plan: dict[str, Any]) -> None:
    assignments = plan.get("asset_assignments")
    groups = plan.get("groups")
    if not isinstance(assignments, dict) or not isinstance(groups, list):
        raise RuntimeError("Split plan is malformed")
    covered: set[str] = set()
    for group in groups:
        split = group.get("split")
        members = group.get("member_asset_sha256s")
        if split not in SPLIT_NAMES or not isinstance(members, list) or not members:
            raise RuntimeError("Split group record is malformed")
        member_splits = {assignments.get(asset) for asset in members}
        if member_splits != {split}:
            raise RuntimeError(f"Similarity group crosses splits: {group.get('group_id')}")
        covered.update(members)
    if covered != set(assignments):
        raise RuntimeError("Split assignments and group membership differ")


def _split_objective(
    groups: list[dict[str, Any]],
    assignments: Mapping[str, str],
    count_targets: Mapping[str, int],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
) -> tuple[int, Decimal, Decimal]:
    counts, durations, group_counts = _split_totals(groups, assignments)
    return _objective_from_totals(
        counts,
        durations,
        group_counts,
        count_targets,
        duration_targets,
        group_targets,
    )


def _split_totals(
    groups: list[dict[str, Any]], assignments: Mapping[str, str]
) -> tuple[dict[str, int], dict[str, Decimal], dict[str, int]]:
    counts = {name: 0 for name in SPLIT_NAMES}
    durations = {name: Decimal("0") for name in SPLIT_NAMES}
    group_counts = {name: 0 for name in SPLIT_NAMES}
    for group in groups:
        split = assignments.get(group["group_id"])
        if split is None:
            continue
        counts[split] += group["item_count"]
        durations[split] += group["duration"]
        group_counts[split] += 1
    return counts, durations, group_counts


def _objective_from_totals(
    counts: Mapping[str, int],
    durations: Mapping[str, Decimal],
    group_counts: Mapping[str, int],
    count_targets: Mapping[str, int],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
) -> tuple[int, Decimal, Decimal]:
    count_deviation = sum(
        abs(counts[name] - count_targets[name]) for name in SPLIT_NAMES
    )
    duration_total = sum(duration_targets.values(), Decimal("0"))
    duration_deviation = sum(
        (abs(durations[name] - duration_targets[name]) for name in SPLIT_NAMES),
        Decimal("0"),
    ) / duration_total
    group_total = sum(group_targets.values(), Decimal("0"))
    group_deviation = sum(
        (abs(Decimal(group_counts[name]) - group_targets[name]) for name in SPLIT_NAMES),
        Decimal("0"),
    ) / group_total
    return count_deviation, duration_deviation, group_deviation


def _improve_assignments(
    groups: list[dict[str, Any]],
    assignments: dict[str, str],
    count_targets: Mapping[str, int],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
) -> dict[str, str]:
    counts, durations, group_counts = _split_totals(groups, assignments)
    current = _objective_from_totals(
        counts,
        durations,
        group_counts,
        count_targets,
        duration_targets,
        group_targets,
    )
    ordered = sorted(groups, key=lambda group: group["group_id"])
    groups_by_id = {group["group_id"]: group for group in ordered}
    for _ in range(min(len(ordered) * 4, 64)):
        best: tuple[Any, ...] | None = None
        for index, left in enumerate(ordered):
            left_split = assignments[left["group_id"]]
            for right in ordered[index + 1 :]:
                right_split = assignments[right["group_id"]]
                if left_split == right_split:
                    continue
                candidate_counts = dict(counts)
                candidate_durations = dict(durations)
                candidate_counts[left_split] += (
                    right["item_count"] - left["item_count"]
                )
                candidate_counts[right_split] += (
                    left["item_count"] - right["item_count"]
                )
                candidate_durations[left_split] += (
                    right["duration"] - left["duration"]
                )
                candidate_durations[right_split] += (
                    left["duration"] - right["duration"]
                )
                objective = _objective_from_totals(
                    candidate_counts,
                    candidate_durations,
                    group_counts,
                    count_targets,
                    duration_targets,
                    group_targets,
                )
                if objective < current:
                    candidate = (
                        objective,
                        left["group_id"],
                        right["group_id"],
                        left_split,
                        right_split,
                    )
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            break
        _, left_id, right_id, left_split, right_split = best
        assignments[left_id] = right_split
        assignments[right_id] = left_split
        left = groups_by_id[left_id]
        right = groups_by_id[right_id]
        counts[left_split] += right["item_count"] - left["item_count"]
        counts[right_split] += left["item_count"] - right["item_count"]
        durations[left_split] += right["duration"] - left["duration"]
        durations[right_split] += left["duration"] - right["duration"]
        current = best[0]
    return assignments


def _improve_stratified_assignments(
    groups: list[dict[str, Any]],
    assignments: dict[str, str],
    *,
    count_targets: Mapping[str, int],
    primary_targets: Mapping[str, Mapping[str, int]],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
    low_resource_label: str,
    minimum_eval_groups: int,
) -> dict[str, str]:
    totals = _stratified_totals(groups, assignments, low_resource_label)
    current = _stratified_objective(
        totals,
        count_targets=count_targets,
        primary_targets=primary_targets,
        duration_targets=duration_targets,
        group_targets=group_targets,
        low_resource_group_total=sum(
            low_resource_label in group["primary_counts"] for group in groups
        ),
        minimum_eval_groups=minimum_eval_groups,
    )
    ordered = sorted(groups, key=lambda group: group["group_id"])
    groups_by_id = {group["group_id"]: group for group in ordered}
    low_group_total = sum(
        low_resource_label in group["primary_counts"] for group in groups
    )
    for _ in range(min(len(ordered) * 4, 64)):
        best: tuple[Any, ...] | None = None
        for index, left in enumerate(ordered):
            left_split = assignments[left["group_id"]]
            for right in ordered[index + 1 :]:
                right_split = assignments[right["group_id"]]
                if left_split == right_split:
                    continue
                candidate_totals = _totals_after_swap(
                    totals,
                    left,
                    right,
                    left_split=left_split,
                    right_split=right_split,
                    low_resource_label=low_resource_label,
                )
                objective = _stratified_objective(
                    candidate_totals,
                    count_targets=count_targets,
                    primary_targets=primary_targets,
                    duration_targets=duration_targets,
                    group_targets=group_targets,
                    low_resource_group_total=low_group_total,
                    minimum_eval_groups=minimum_eval_groups,
                )
                if objective < current:
                    candidate = (
                        objective,
                        left["group_id"],
                        right["group_id"],
                        left_split,
                        right_split,
                        candidate_totals,
                    )
                    if best is None or candidate[:-1] < best[:-1]:
                        best = candidate
        if best is None:
            break
        _, left_id, right_id, left_split, right_split, totals = best
        assignments[left_id] = right_split
        assignments[right_id] = left_split
        current = best[0]
        if left_id not in groups_by_id or right_id not in groups_by_id:
            raise RuntimeError("Stratified swap selected an unknown group")
    return assignments


def _stratified_totals(
    groups: list[dict[str, Any]],
    assignments: Mapping[str, str],
    low_resource_label: str,
) -> dict[str, Any]:
    totals = {
        "counts": {split: 0 for split in SPLIT_NAMES},
        "durations": {split: Decimal("0") for split in SPLIT_NAMES},
        "group_counts": {split: 0 for split in SPLIT_NAMES},
        "primary": {split: Counter() for split in SPLIT_NAMES},
        "low_groups": {split: 0 for split in SPLIT_NAMES},
    }
    for group in groups:
        split = assignments[group["group_id"]]
        totals["counts"][split] += group["item_count"]
        totals["durations"][split] += group["duration"]
        totals["group_counts"][split] += 1
        totals["primary"][split].update(group["primary_counts"])
        totals["low_groups"][split] += int(
            low_resource_label in group["primary_counts"]
        )
    return totals


def _totals_after_swap(
    totals: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_split: str,
    right_split: str,
    low_resource_label: str,
) -> dict[str, Any]:
    candidate = {
        "counts": dict(totals["counts"]),
        "durations": dict(totals["durations"]),
        "group_counts": dict(totals["group_counts"]),
        "primary": {
            split: Counter(totals["primary"][split]) for split in SPLIT_NAMES
        },
        "low_groups": dict(totals["low_groups"]),
    }
    candidate["counts"][left_split] += right["item_count"] - left["item_count"]
    candidate["counts"][right_split] += left["item_count"] - right["item_count"]
    candidate["durations"][left_split] += right["duration"] - left["duration"]
    candidate["durations"][right_split] += left["duration"] - right["duration"]
    candidate["primary"][left_split].subtract(left["primary_counts"])
    candidate["primary"][left_split].update(right["primary_counts"])
    candidate["primary"][right_split].subtract(right["primary_counts"])
    candidate["primary"][right_split].update(left["primary_counts"])
    left_low = int(low_resource_label in left["primary_counts"])
    right_low = int(low_resource_label in right["primary_counts"])
    candidate["low_groups"][left_split] += right_low - left_low
    candidate["low_groups"][right_split] += left_low - right_low
    return candidate


def _stratified_objective(
    totals: Mapping[str, Any],
    *,
    count_targets: Mapping[str, int],
    primary_targets: Mapping[str, Mapping[str, int]],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
    low_resource_group_total: int,
    minimum_eval_groups: int,
) -> tuple[int, int, int, Decimal, Decimal]:
    coverage_required = low_resource_group_total >= 1 + 2 * minimum_eval_groups
    low_resource_deficit = 0
    if coverage_required:
        low_resource_deficit = sum(
            max(0, minimum_eval_groups - totals["low_groups"][split])
            for split in ("validation", "test")
        )
    count_deviation = sum(
        abs(totals["counts"][split] - count_targets[split])
        for split in SPLIT_NAMES
    )
    primary_deviation = sum(
        abs(totals["primary"][split][label] - primary_targets[label][split])
        for label in primary_targets
        for split in SPLIT_NAMES
    )
    duration_total = sum(duration_targets.values(), Decimal("0"))
    duration_deviation = sum(
        (
            abs(totals["durations"][split] - duration_targets[split])
            for split in SPLIT_NAMES
        ),
        Decimal("0"),
    ) / duration_total
    group_total = sum(group_targets.values(), Decimal("0"))
    group_deviation = sum(
        (
            abs(Decimal(totals["group_counts"][split]) - group_targets[split])
            for split in SPLIT_NAMES
        ),
        Decimal("0"),
    ) / group_total
    return (
        low_resource_deficit,
        count_deviation,
        primary_deviation,
        duration_deviation,
        group_deviation,
    )


def _assert_low_resource_coverage(
    groups: list[dict[str, Any]],
    *,
    low_resource_label: str,
    minimum_eval_groups: int,
) -> None:
    low_groups = [
        group for group in groups if low_resource_label in group["primary_counts"]
    ]
    if len(low_groups) < 1 + 2 * minimum_eval_groups:
        return
    for split in ("validation", "test"):
        count = sum(group["split"] == split for group in low_groups)
        if count < minimum_eval_groups:
            raise RuntimeError(f"Low-resource label has insufficient {split} groups")


def _stratified_summary(
    groups: list[dict[str, Any]],
    *,
    primary_targets: Mapping[str, Mapping[str, int]],
    weak_targets: Mapping[str, Mapping[str, int]],
    count_targets: Mapping[str, int],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
    low_resource_label: str,
    minimum_eval_groups: int,
) -> dict[str, Any]:
    primary_actual = {
        split: Counter(
            {
                label: sum(
                    group["primary_counts"].get(label, 0)
                    for group in groups
                    if group["split"] == split
                )
                for label in primary_targets
            }
        )
        for split in SPLIT_NAMES
    }
    weak_actual = {
        split: Counter(
            {
                label: sum(
                    group["weak_counts"].get(label, 0)
                    for group in groups
                    if group["split"] == split
                )
                for label in weak_targets
            }
        )
        for split in SPLIT_NAMES
    }
    split_summary = _split_summary(
        groups, count_targets, duration_targets, group_targets
    )
    for split in SPLIT_NAMES:
        split_summary[split]["emotion_primary"] = {
            label: {
                "target": primary_targets[label][split],
                "actual": primary_actual[split][label],
                "delta": primary_actual[split][label] - primary_targets[label][split],
            }
            for label in primary_targets
        }
        split_summary[split]["emotion_weak_label"] = {
            label: {
                "target": weak_targets[label][split],
                "actual": weak_actual[split][label],
                "delta": weak_actual[split][label] - weak_targets[label][split],
            }
            for label in weak_targets
        }
    low_group_counts = {
        split: sum(
            group["split"] == split
            and low_resource_label in group["primary_counts"]
            for group in groups
        )
        for split in SPLIT_NAMES
    }
    split_summary["low_resource"] = {
        "label": low_resource_label,
        "minimum_groups_per_eval_split": minimum_eval_groups,
        "final_primary_total": sum(
            primary_actual[split][low_resource_label] for split in SPLIT_NAMES
        ),
        "weak_label_total": sum(
            weak_actual[split][low_resource_label] for split in SPLIT_NAMES
        ),
        "final_primary_by_split": {
            split: {
                "target": primary_targets[low_resource_label][split],
                "actual": primary_actual[split][low_resource_label],
                "delta": (
                    primary_actual[split][low_resource_label]
                    - primary_targets[low_resource_label][split]
                ),
                "group_count": low_group_counts[split],
            }
            for split in SPLIT_NAMES
        },
        "weak_label_by_split": {
            split: {
                "target": weak_targets[low_resource_label][split],
                "actual": weak_actual[split][low_resource_label],
                "delta": (
                    weak_actual[split][low_resource_label]
                    - weak_targets[low_resource_label][split]
                ),
            }
            for split in SPLIT_NAMES
        },
    }
    return split_summary


def _split_summary(
    groups: list[dict[str, Any]],
    count_targets: Mapping[str, int],
    duration_targets: Mapping[str, Decimal],
    group_targets: Mapping[str, Decimal],
) -> dict[str, dict[str, Any]]:
    return {
        split: {
            "target_item_count": count_targets[split],
            "actual_item_count": sum(
                group["item_count"] for group in groups if group["split"] == split
            ),
            "target_duration_seconds": float(duration_targets[split]),
            "actual_duration_seconds": sum(
                group["duration_seconds"]
                for group in groups
                if group["split"] == split
            ),
            "target_group_count": float(group_targets[split]),
            "actual_group_count": sum(group["split"] == split for group in groups),
        }
        for split in SPLIT_NAMES
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
