from __future__ import annotations

from typing import Dict, Any

from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def apply_peft_lora(model, lora_cfg: Dict[str, Any], adapter_name: str = "default"):
    """Wrap `model` with a PEFT LoRA adapter according to `lora_cfg`.

    Returns a `PeftModel` instance ready for training. Does not modify `model` in-place.
    """
    rank = int(lora_cfg.get("rank", 8))
    alpha = int(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get("target_modules", None)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )

    peft_model = get_peft_model(model, peft_config, adapter_name=adapter_name)
    return peft_model


def save_peft_adapter(peft_model: PeftModel, save_directory: str) -> None:
    peft_model.save_pretrained(save_directory)


def load_peft_adapter(base_model, adapter_directory: str) -> PeftModel:
    return PeftModel.from_pretrained(base_model, adapter_directory)
