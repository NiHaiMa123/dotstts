from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dots_tts_lab.acoustic_fingerprint import (
    compute_acoustic_fingerprint,
    cosine_similarity,
)
from dots_tts_lab.dataset_analysis import analyze_exact_audio, analyze_near_text
from dots_tts_lab.dataset_freeze import load_dataset_freeze_config
from dots_tts_lab.dataset_split import (
    assert_no_group_crosses_splits,
    plan_emotion_stratified_split,
)
from dots_tts_lab.dataset_split_audit import (
    assert_split_audit_passes,
    audit_split_leakage,
)
from dots_tts_lab.duplicate_graph import stable_similarity_groups
from dots_tts_lab.speaker_embedding import (
    SPEAKER_EMBEDDING_DIMENSION,
    assess_speaker_embeddings,
)


class Slice8SyntheticIntegrationTests(unittest.TestCase):
    def test_all_similarity_evidence_stays_grouped_through_stratified_split(self) -> None:
        config = load_dataset_freeze_config()
        assets = [f"{index:064x}" for index in range(1, 10)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 48000
            time = np.arange(sample_rate, dtype=np.float64) / sample_rate
            base = (0.35 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
            signals = [
                base,
                base,
                base * 0.45,
                np.pad(base, (2400, 2400)),
                0.3 * np.sin(2 * np.pi * 330 * time),
                0.3 * np.sin(2 * np.pi * 440 * time),
                0.3 * np.sin(2 * np.pi * 550 * time),
                0.3 * np.sin(2 * np.pi * 660 * time),
                0.3 * np.sin(2 * np.pi * 770 * time),
            ]
            paths = []
            for index, signal in enumerate(signals):
                path = root / f"{index}.wav"
                sf.write(path, signal, sample_rate, subtype="PCM_24")
                paths.append(path)
            paths[1].write_bytes(paths[0].read_bytes())

            texts = [
                "sad exact",
                "sad exact copy",
                "sad gain",
                "sad silence",
                "符玄今天准备去开会",
                "符玄今天准备去开会呀",
                "sad eval one",
                "sad eval two",
                "happy singleton",
            ]
            emotions = ["难过_sad"] * 4 + ["中立_neutral"] * 2 + [
                "难过_sad",
                "难过_sad",
                "开心_happy",
            ]
            items = []
            for index, asset in enumerate(assets):
                content = paths[index].read_bytes()
                items.append(
                    {
                        "asset_sha256": asset,
                        "audio_absolute_path": str(paths[index]),
                        "audio_relative_path": paths[index].name,
                        "audio_sha256": hashlib.sha256(content).hexdigest(),
                        "duration_seconds": len(signals[index]) / sample_rate,
                        "text_exact": texts[index],
                        "emotion_primary": emotions[index],
                        "emotion_weak_label": emotions[index],
                        "source_root_key": "synthetic",
                        "source_path_key": f"path-{index}",
                        "raw_relative_path": f"raw/{index}.wav",
                    }
                )

            exact = analyze_exact_audio(items)
            self.assertEqual(exact["duplicate_group_count"], 1)
            self.assertEqual(
                {member["asset_sha256"] for member in exact["groups"][0]["members"]},
                set(assets[:2]),
            )
            fingerprints = [
                compute_acoustic_fingerprint(
                    np.asarray(signal, dtype=np.float32),
                    sample_rate=sample_rate,
                    config=config.acoustic_fingerprint,
                )
                for signal in signals[:4]
            ]
            self.assertGreater(cosine_similarity(fingerprints[0], fingerprints[2]), 0.99)
            self.assertGreater(cosine_similarity(fingerprints[0], fingerprints[3]), 0.99)

            near_text = analyze_near_text(items, config=config.text_similarity)
            text_pair = next(
                pair
                for pair in near_text["pairs"]
                if {pair["left_asset_sha256"], pair["right_asset_sha256"]}
                == set(assets[4:6])
            )
            self.assertGreater(text_pair["similarity"], 0.8)

            speaker_vectors = []
            for index, asset in enumerate(assets):
                vector = np.zeros(SPEAKER_EMBEDDING_DIMENSION, dtype=np.float32)
                vector[0 if index < 8 else 1] = 1.0
                if index < 8:
                    vector[index + 2] = 0.02
                    vector /= np.linalg.norm(vector)
                speaker_vectors.append({"asset_sha256": asset, "vector": vector})
            speaker = assess_speaker_embeddings(speaker_vectors, knn_k=2)
            top_outlier = min(
                speaker["assessments"], key=lambda row: row["outlier_rank"]
            )
            self.assertEqual(top_outlier["asset_sha256"], assets[8])
            self.assertEqual(top_outlier["outlier_rank"], 1)

            accepted_pairs = [
                (assets[0], assets[1]),
                (assets[0], assets[2]),
                (assets[0], assets[3]),
                (assets[4], assets[5]),
            ]
            groups = stable_similarity_groups(assets, accepted_pairs)
            plan = plan_emotion_stratified_split(
                items=items,
                groups=groups,
                split_config=config.split,
                dataset_config_sha256="a" * 64,
            )
            assert_no_group_crosses_splits(plan)
            sad_eval = {
                split: sum(
                    item["emotion_primary"] == "难过_sad"
                    and plan["asset_assignments"][item["asset_sha256"]] == split
                    for item in items
                )
                for split in ("validation", "test")
            }
            self.assertEqual(sad_eval, {"validation": 1, "test": 1})
            audit = audit_split_leakage(
                items=items,
                split_plan=plan,
                similarity_edges=[
                    {
                        "left_asset_sha256": left,
                        "right_asset_sha256": right,
                        "evidence_type": evidence,
                    }
                    for (left, right), evidence in zip(
                        accepted_pairs,
                        ("exact_audio", "near_audio", "near_audio", "near_text"),
                        strict=True,
                    )
                ],
                text_config=config.text_similarity,
            )
            assert_split_audit_passes(audit)


if __name__ == "__main__":
    unittest.main()
