"""Low-level PEFT integration for the non-Transformers dots.tts DiT."""

from __future__ import annotations

from typing import Any

from dots_tts.models.dots_tts.config import LoraArtifactConfig


def _has_lora_modules(module: Any) -> bool:
    return any(hasattr(child, "lora_A") for child in module.modules())


def configure_lora_modules(
    model: Any,
    config: LoraArtifactConfig,
    *,
    for_training: bool,
) -> dict[str, int | bool]:
    """Inject LoRA into the DiT and optionally freeze the non-adapter model."""
    if not config.enabled:
        return {"enabled": False, "trainable_parameters": 0, "lora_parameters": 0}
    if config.target_scope != "core.velocity_field_predictor":
        raise ValueError(f"Unsupported LoRA target scope: {config.target_scope}")

    if for_training:
        for parameter in model.parameters():
            parameter.requires_grad = False

    target = model.core.velocity_field_predictor
    if not _has_lora_modules(target):
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import inject_adapter_in_model
        except ImportError as error:
            raise RuntimeError(
                "LoRA is enabled but peft is not installed; install dots.tts[full]"
            ) from error
        peft_config = PeftLoraConfig(
            r=int(config.rank),
            lora_alpha=int(config.alpha),
            lora_dropout=float(config.dropout),
            bias="none",
            target_modules=list(config.target_modules),
        )
        model.core.velocity_field_predictor = inject_adapter_in_model(
            peft_config,
            target,
        )

    model.config.lora = config
    if for_training and config.train_output_layer:
        for parameter in model.core.velocity_field_predictor.output_layer.parameters():
            parameter.requires_grad = True

    lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if not lora_parameters:
        raise RuntimeError("PEFT injection completed without creating any LoRA parameters")
    if for_training and not trainable_parameters:
        raise RuntimeError("LoRA configuration left the model with no trainable parameters")
    return {
        "enabled": True,
        "trainable_parameters": trainable_parameters,
        "lora_parameters": lora_parameters,
    }


def merge_lora_modules(model: Any) -> int:
    """Merge injected PEFT LoRA layers and replace wrappers with base modules."""
    target = model.core.velocity_field_predictor
    expected = sum(hasattr(module, "lora_A") for module in target.modules())
    if not expected:
        raise RuntimeError("LoRA merge requested but no injected LoRA layers exist")
    merged = 0

    def visit(parent: Any) -> None:
        nonlocal merged
        for name, child in list(parent.named_children()):
            if hasattr(child, "lora_A"):
                merge = getattr(child, "merge", None)
                get_base_layer = getattr(child, "get_base_layer", None)
                if not callable(merge) or not callable(get_base_layer):
                    raise RuntimeError(
                        f"Injected LoRA layer does not support safe merge: {name}"
                    )
                merge(safe_merge=True)
                setattr(parent, name, get_base_layer())
                merged += 1
            else:
                visit(child)

    visit(target)
    if merged != expected:
        raise RuntimeError(
            f"LoRA merge count mismatch: expected={expected} merged={merged}"
        )
    if any(hasattr(module, "lora_A") for module in target.modules()):
        raise RuntimeError("LoRA wrappers remain after merge")
    return merged


__all__ = ["configure_lora_modules", "merge_lora_modules"]
