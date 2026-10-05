from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from dots_tts_lab.slice9_constraints import _duplicate_groups
from dots_tts_lab.slice9_pools import build_slice9_pool_candidates


ROOT = Path(__file__).resolve().parents[1]


def _item(index: int, emotion: str, *, quality_decision: str = "pass", speaker_decision: str = "pass", quality_reasons: list[dict] | None = None) -> dict:
    asset = f"{index:064x}"
    return {
        "asset_sha256": asset,
        "fid": asset,
        "audio_relative_path": f"{asset[:2]}/{asset}.wav",
        "audio_sha256": f"{index + 100:064x}",
        "speaker_id": "崩铁符玄",
        "emotion_primary": emotion,
        "emotion_weak_label": emotion,
        "text_exact": f"合成测试文本{index}。",
        "text_sha256": f"{index + 200:064x}",
        "text_source": "filename_candidate_unreviewed",
        "quality": {"decision": quality_decision, "reasons": quality_reasons or []},
        "speaker": {"decision": speaker_decision, "reasons": []},
        "text": {"confidence_tier": "B"},
        "coverage": {"phonology": {"unique_syllable_type_count": None}, "script": {"punctuation_ratio": 0.1}},
    }


class Slice9SyntheticIntegrationTests(unittest.TestCase):
    def _config_and_snapshot(self, items: list[dict]) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name)
        config = yaml.safe_load((ROOT / "configs/lab/datasets/fuxuan_slice9_v1.yaml").read_text(encoding="utf-8"))
        config_path = directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        snapshot_path = directory / "snapshot.json"
        snapshot_path.write_text(json.dumps({"status": "succeeded", "item_count": len(items), "items": items}, ensure_ascii=False), encoding="utf-8")
        return config_path, snapshot_path, temporary

    def test_quality_reject_silence_review_and_speaker_outlier(self) -> None:
        items = [
            _item(1, "中立_neutral", quality_decision="reject", quality_reasons=[{"code": "flat_top"}]),
            _item(2, "开心_happy", quality_decision="review", quality_reasons=[{"code": "high_silence_ratio"}]),
            _item(3, "生气_angry", speaker_decision="reject"),
            _item(4, "难过_sad"),
            _item(5, "中立_neutral"),
        ]
        config_path, snapshot_path, temporary = self._config_and_snapshot(items)
        try:
            output = Path(temporary.name) / "pools.json"
            result = build_slice9_pool_candidates(config_path=config_path, feature_snapshot_path=snapshot_path, output_path=output)
            self.assertEqual(result["hard_gate_decision_counts"]["reject"], 2)
            self.assertEqual(result["hard_gate_decision_counts"]["review"], 1)
            self.assertEqual(result["pools"]["emotion_happy"]["candidate_count"], 1)
            self.assertEqual(result["pools"]["emotion_happy"]["items"][0]["review_flags"], ["high_silence_ratio"])
            self.assertEqual(result["pools"]["emotion_angry"]["candidate_count"], 0)
            self.assertIn("emotion_sad", result["shortfall_pool_ids"])
        finally:
            temporary.cleanup()

    def test_reviewed_duplicate_group_has_explicit_representative(self) -> None:
        left, right = f"{10:064x}", f"{11:064x}"
        report = {"duplicate_edges": [{
            "review_status": "accepted",
            "left": {"asset_sha256": left},
            "right": {"asset_sha256": right},
            "review": {"review_note": json.dumps({"representative_asset_sha256": right})},
        }]}
        groups, representatives, details = _duplicate_groups(report, {left, right})
        self.assertEqual(groups[left], groups[right])
        self.assertEqual(representatives[groups[left]], right)
        self.assertEqual(details[0]["asset_sha256s"], [left, right])
        self.assertTrue(details[0]["reviewed_duplicate"])


if __name__ == "__main__":
    unittest.main()
