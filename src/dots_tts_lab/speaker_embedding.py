from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import safetensors
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import load_file

from dots_tts.modules.speaker.encoder import SpeakerXVectorFeatures
from dots_tts_lab.dataset_freeze import SpeakerEmbeddingConfig


SPEAKER_EMBEDDING_IMPLEMENTATION_VERSION = 1
SPEAKER_EMBEDDING_DIMENSION = 512
DEFAULT_SPEAKER_EMBEDDING_CACHE_ROOT = Path(
    "data/cache/dataset_features/speaker_embedding"
)
DEFAULT_SPEAKER_EMBEDDING_REPORT_PATH = Path(
    "data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speaker_embedding_identity(config: SpeakerEmbeddingConfig) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "algorithm": "dots_tts_speaker_xvector_features_campplus",
        "implementation_version": SPEAKER_EMBEDDING_IMPLEMENTATION_VERSION,
        "cache_format_version": 1,
        "config": config.model_dump(mode="json"),
        "dependencies": {
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
            "soundfile": sf.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
        },
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "payload": payload,
        "canonical_json": canonical_json,
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def load_speaker_encoder(
    config: SpeakerEmbeddingConfig,
) -> SpeakerXVectorFeatures:
    if config.inference_device != "cpu" or config.inference_dtype != "float32":
        raise RuntimeError("Canonical speaker encoder requires CPU float32")
    weights_path = Path(config.weights_path).resolve()
    model_config_path = Path(config.model_config_path).resolve()
    for label, path, expected_sha256 in (
        ("speaker weights", weights_path, config.weights_sha256),
        ("speaker model config", model_config_path, config.model_config_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Pinned {label} do not exist: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Pinned {label} SHA-256 drift: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if model_config.get("campplus_embedding_size") != config.embedding_size:
        raise RuntimeError("Pinned model config CAM++ embedding size mismatch")
    encoder = SpeakerXVectorFeatures(
        sample_rate=config.input_sample_rate,
        campplus_embedding_size=config.embedding_size,
        max_audio_seconds=config.max_audio_seconds,
    ).to(device="cpu", dtype=torch.float32)
    state_dict = load_file(weights_path, device="cpu")
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    return encoder


class _LazySpeakerEncoder:
    def __init__(self, config: SpeakerEmbeddingConfig):
        self.config = config
        self.encoder: SpeakerXVectorFeatures | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.encoder is None:
            self.encoder = load_speaker_encoder(self.config)
        return self.encoder(*args, **kwargs)


def compute_speaker_embedding(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: SpeakerEmbeddingConfig,
    encoder: Any,
) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise RuntimeError(
                f"Speaker embedding expects mono audio, got {values.shape[1]} channels"
            )
        values = values[:, 0]
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError("Speaker embedding input must contain mono audio")
    if sample_rate != config.input_sample_rate:
        raise RuntimeError(f"Speaker embedding sample rate mismatch: {sample_rate}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Speaker embedding input contains non-finite samples")
    if float(np.max(np.abs(values), initial=0.0)) <= np.finfo(np.float32).tiny:
        raise RuntimeError("Speaker embedding input is silent")
    audio_tensor = torch.from_numpy(values.copy()).reshape(1, -1)
    lengths = torch.tensor([audio_tensor.shape[-1]], dtype=torch.long)
    with torch.inference_mode():
        output = encoder(audio_tensor, audio_lengths=lengths)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (
        1,
        config.embedding_size,
    ):
        raise RuntimeError(
            "Speaker encoder returned invalid shape: "
            f"{getattr(output, 'shape', None)}"
        )
    vector = output.detach().cpu().to(torch.float32).numpy().reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("Speaker encoder returned non-finite values")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("Speaker encoder returned a zero vector")
    return np.asarray(vector / norm, dtype="<f4")


def _load_validated_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    audio_sha256: str,
    identity_sha256: str,
) -> tuple[bytes, np.ndarray] | None:
    if cache_path.exists() != metadata_path.exists():
        raise RuntimeError(f"Incomplete speaker embedding cache entry: {cache_path}")
    if not cache_path.exists():
        return None
    blob = cache_path.read_bytes()
    expected_size = SPEAKER_EMBEDDING_DIMENSION * np.dtype("<f4").itemsize
    if len(blob) != expected_size:
        raise RuntimeError(f"Invalid cached speaker embedding size: {cache_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid speaker embedding cache metadata: {metadata_path}"
        ) from error
    expected_metadata = {
        "schema_version": 1,
        "asset_audio_sha256": audio_sha256,
        "speaker_embedding_config_sha256": identity_sha256,
        "speaker_embedding_sha256": hashlib.sha256(blob).hexdigest(),
        "dimension": SPEAKER_EMBEDDING_DIMENSION,
        "dtype": "float32_le",
    }
    if metadata != expected_metadata:
        raise RuntimeError(
            f"Speaker embedding cache content hash or metadata drift: {cache_path}"
        )
    vector = np.frombuffer(blob, dtype="<f4").copy()
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not np.all(np.isfinite(vector)) or not math.isclose(
        norm, 1.0, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise RuntimeError(f"Invalid cached speaker embedding vector: {cache_path}")
    return blob, vector


def _write_cache_entry(
    cache_path: Path,
    metadata_path: Path,
    *,
    blob: bytes,
    metadata: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob_partial = cache_path.with_name(
        f".{cache_path.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with blob_partial.open("xb") as output:
            output.write(blob)
            output.flush()
            os.fsync(output.fileno())
        if cache_path.exists():
            if cache_path.read_bytes() != blob:
                raise RuntimeError(f"Conflicting speaker embedding cache: {cache_path}")
            blob_partial.unlink()
        else:
            blob_partial.replace(cache_path)
    finally:
        if blob_partial.exists():
            blob_partial.unlink()
    metadata_content = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    metadata_partial = metadata_path.with_name(
        f".{metadata_path.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with metadata_partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(metadata_content)
            output.flush()
            os.fsync(output.fileno())
        if metadata_path.exists():
            if metadata_path.read_text(encoding="utf-8") != metadata_content:
                raise RuntimeError(
                    f"Conflicting speaker embedding metadata: {metadata_path}"
                )
            metadata_partial.unlink()
        else:
            metadata_partial.replace(metadata_path)
    finally:
        if metadata_partial.exists():
            metadata_partial.unlink()


def load_or_compute_speaker_embedding(
    candidate: dict[str, Any],
    *,
    config: SpeakerEmbeddingConfig,
    encoder: Any,
    cache_root: str | Path = DEFAULT_SPEAKER_EMBEDDING_CACHE_ROOT,
) -> dict[str, Any]:
    asset_sha256 = candidate.get("asset_sha256")
    if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
        raise RuntimeError("Speaker candidate has invalid asset_sha256")
    audio_sha256 = candidate.get("audio_sha256")
    if not isinstance(audio_sha256, str) or len(audio_sha256) != 64:
        raise RuntimeError(f"Speaker candidate {asset_sha256} has invalid audio_sha256")
    audio_path_value = candidate.get("audio_absolute_path")
    if not isinstance(audio_path_value, str) or not audio_path_value:
        raise RuntimeError(f"Speaker candidate {asset_sha256} has no audio path")
    audio_path = Path(audio_path_value).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Speaker audio does not exist: {audio_path}")
    actual_audio_sha256 = _sha256_file(audio_path)
    if actual_audio_sha256 != audio_sha256:
        raise RuntimeError(
            f"Speaker input SHA-256 drift for {asset_sha256}: "
            f"expected {audio_sha256}, got {actual_audio_sha256}"
        )
    identity = speaker_embedding_identity(config)
    cache_path = (
        Path(cache_root).resolve()
        / identity["sha256"]
        / audio_sha256[:2]
        / f"{audio_sha256}.f32"
    )
    metadata_path = cache_path.with_suffix(".json")
    cached = _load_validated_cache(
        cache_path,
        metadata_path,
        audio_sha256=audio_sha256,
        identity_sha256=identity["sha256"],
    )
    if cached is None:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        vector = compute_speaker_embedding(
            audio,
            sample_rate=int(sample_rate),
            config=config,
            encoder=encoder,
        )
        blob = vector.astype("<f4", copy=False).tobytes(order="C")
        metadata = {
            "schema_version": 1,
            "asset_audio_sha256": audio_sha256,
            "speaker_embedding_config_sha256": identity["sha256"],
            "speaker_embedding_sha256": hashlib.sha256(blob).hexdigest(),
            "dimension": SPEAKER_EMBEDDING_DIMENSION,
            "dtype": "float32_le",
        }
        _write_cache_entry(
            cache_path,
            metadata_path,
            blob=blob,
            metadata=metadata,
        )
        action = "computed"
    else:
        blob, vector = cached
        action = "cached"
    return {
        "asset_sha256": asset_sha256,
        "audio_sha256": audio_sha256,
        "speaker_embedding_config_sha256": identity["sha256"],
        "speaker_embedding_sha256": hashlib.sha256(blob).hexdigest(),
        "dimension": SPEAKER_EMBEDDING_DIMENSION,
        "dtype": "float32_le",
        "cache_path": str(cache_path),
        "cache_metadata_path": str(metadata_path),
        "action": action,
        "vector": vector,
    }


def assess_speaker_embeddings(
    features: Iterable[dict[str, Any]],
    *,
    knn_k: int,
) -> dict[str, Any]:
    ordered = sorted(features, key=lambda item: item["asset_sha256"])
    if len(ordered) < 2:
        raise RuntimeError("Speaker assessment requires at least two embeddings")
    matrix = np.stack(
        [np.asarray(item["vector"], dtype=np.float64) for item in ordered]
    )
    if matrix.ndim != 2 or matrix.shape[1] != SPEAKER_EMBEDDING_DIMENSION:
        raise RuntimeError("Speaker assessment received invalid embedding dimensions")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Speaker assessment received non-finite embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("Speaker assessment received zero embeddings")
    matrix = matrix / norms
    center = np.median(matrix, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= np.finfo(float).tiny:
        raise RuntimeError("Speaker embedding robust center collapsed to zero")
    center /= center_norm
    center_cosines = np.clip(matrix @ center, -1.0, 1.0)
    similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
    effective_k = min(knn_k, len(ordered) - 1)
    assessments = []
    for index, item in enumerate(ordered):
        neighbors = [
            (float(similarities[index, other]), ordered[other]["asset_sha256"])
            for other in range(len(ordered))
            if other != index
        ]
        neighbors.sort(key=lambda pair: (-pair[0], pair[1]))
        selected = neighbors[:effective_k]
        knn_cosine = float(np.mean([score for score, _ in selected]))
        center_cosine = float(center_cosines[index])
        assessments.append(
            {
                "asset_sha256": item["asset_sha256"],
                "center_cosine": center_cosine,
                "knn_cosine": knn_cosine,
                "outlier_score": 1.0 - (center_cosine + knn_cosine) / 2.0,
                "neighbor_asset_sha256s": [asset for _, asset in selected],
                "neighbor_cosines": [score for score, _ in selected],
                "candidate_outlier": None,
            }
        )
    ranking = sorted(
        assessments,
        key=lambda item: (-item["outlier_score"], item["asset_sha256"]),
    )
    for rank, assessment in enumerate(ranking, start=1):
        assessment["outlier_rank"] = rank
    assessments.sort(key=lambda item: item["asset_sha256"])
    center_blob = np.asarray(center, dtype="<f4").tobytes(order="C")
    return {
        "center_method": "coordinate_median_then_l2",
        "center_sha256": hashlib.sha256(center_blob).hexdigest(),
        "requested_knn_k": knn_k,
        "effective_knn_k": effective_k,
        "thresholds": {"center_cosine": None, "knn_cosine": None},
        "assessments": assessments,
    }


def load_speaker_embedding_vectors(
    report: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Load and revalidate every cached embedding named by a report."""
    if report.get("status") != "succeeded":
        raise RuntimeError("Speaker embedding report did not succeed")
    if report.get("dimension") != SPEAKER_EMBEDDING_DIMENSION:
        raise RuntimeError("Speaker embedding report dimension mismatch")
    if report.get("dtype") != "float32_le":
        raise RuntimeError("Speaker embedding report dtype mismatch")
    features = report.get("features")
    if not isinstance(features, list) or report.get("feature_count") != len(features):
        raise RuntimeError("Speaker embedding report feature count mismatch")
    vectors: dict[str, np.ndarray] = {}
    expected_size = SPEAKER_EMBEDDING_DIMENSION * np.dtype("<f4").itemsize
    for feature in features:
        asset_sha256 = feature.get("asset_sha256")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
            raise RuntimeError("Speaker embedding report has an invalid asset SHA-256")
        if asset_sha256 in vectors:
            raise RuntimeError(f"Duplicate speaker embedding asset: {asset_sha256}")
        cache_path_value = feature.get("cache_path")
        if not isinstance(cache_path_value, str) or not cache_path_value:
            raise RuntimeError(f"Speaker embedding cache path missing: {asset_sha256}")
        cache_path = Path(cache_path_value).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Speaker embedding cache missing: {cache_path}")
        blob = cache_path.read_bytes()
        if len(blob) != expected_size:
            raise RuntimeError(f"Invalid speaker embedding cache size: {cache_path}")
        if hashlib.sha256(blob).hexdigest() != feature.get("speaker_embedding_sha256"):
            raise RuntimeError(f"Speaker embedding cache SHA-256 drift: {asset_sha256}")
        vector = np.frombuffer(blob, dtype="<f4").copy()
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.all(np.isfinite(vector)) or not math.isclose(
            norm, 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise RuntimeError(f"Invalid speaker embedding vector: {asset_sha256}")
        vectors[asset_sha256] = vector
    return vectors


def build_speaker_embedding_cache_and_assessment(
    candidates: Iterable[dict[str, Any]],
    *,
    config: SpeakerEmbeddingConfig,
    cache_root: str | Path = DEFAULT_SPEAKER_EMBEDDING_CACHE_ROOT,
    encoder: Any | None = None,
) -> dict[str, Any]:
    candidates_list = list(candidates)
    if encoder is None:
        encoder = _LazySpeakerEncoder(config)
    actions: Counter[str] = Counter()
    features = []
    feature_metadata = []
    seen_assets: set[str] = set()
    for candidate in candidates_list:
        asset_sha256 = candidate.get("asset_sha256")
        if asset_sha256 in seen_assets:
            raise RuntimeError(f"Duplicate speaker candidate asset: {asset_sha256}")
        seen_assets.add(asset_sha256)
        feature = load_or_compute_speaker_embedding(
            candidate,
            config=config,
            encoder=encoder,
            cache_root=cache_root,
        )
        actions[feature["action"]] += 1
        features.append(feature)
        feature_metadata.append(
            {
                key: value
                for key, value in feature.items()
                if key not in {"vector", "action"}
            }
        )
    assessment = assess_speaker_embeddings(features, knn_k=config.knn_k)
    identity = speaker_embedding_identity(config)
    feature_metadata.sort(key=lambda item: item["asset_sha256"])
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "speaker_embedding_identity": identity["payload"],
        "speaker_embedding_config_sha256": identity["sha256"],
        "feature_count": len(feature_metadata),
        "dimension": SPEAKER_EMBEDDING_DIMENSION,
        "dtype": "float32_le",
        "features": feature_metadata,
        **assessment,
    }
    return {"report": report, "actions": dict(sorted(actions.items()))}


def write_speaker_embedding_report(
    report: dict[str, Any],
    path: str | Path = DEFAULT_SPEAKER_EMBEDDING_REPORT_PATH,
) -> dict[str, str]:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "path": str(target),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
