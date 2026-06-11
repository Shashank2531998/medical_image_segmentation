from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from src.utils.logging import get_logger
from src.utils.model_helpers import log_model_params

from src.continual.strategies.lora.dynamic.utils import apply_dynamic_loralib_lora, load_dynamic_lora_adapter
from src.continual.strategies.lora.common.utils import apply_loralib_lora, load_lora_adapter

logger = get_logger(__name__)


def _uses_dynamic_lora(lora_cfg: dict[str, Any]) -> bool:
    return bool(lora_cfg.get("dynamic_experts", False) or lora_cfg.get("strategy") == "dynamic_lora")


def configure_loaded_lora_model(
    model: nn.Module,
    *,
    lora_cfg: dict[str, Any],
    lora_adapter_path: str | Path | None = None,
    mark_trainable: bool = True,
) -> nn.Module:
    bias = str(lora_cfg.get("bias", "none"))

    logger.info(
        "Configuring LoRA model | adapter_path=%s | mark_trainable=%s | bias=%s",
        lora_adapter_path,
        mark_trainable,
        bias,
    )

    if _uses_dynamic_lora(lora_cfg):

        logger.info("Using dynamic LoRA implementation")

        model = apply_dynamic_loralib_lora(
            model,
            lora_cfg,
            mark_trainable=lora_adapter_path is None and mark_trainable,
        )

        if lora_adapter_path is not None:
            model = load_dynamic_lora_adapter(
                model,
                lora_adapter_path,
                bias=bias,
                mark_trainable=False,
            )
    else:

        logger.info("Using standard LoRA implementation")

        model = apply_loralib_lora(
            model,
            lora_cfg,
            mark_trainable=lora_adapter_path is None and mark_trainable,
        )
        if lora_adapter_path is not None:

            model = load_lora_adapter(
                model,
                lora_adapter_path,
                bias=bias,
                mark_trainable=False,
            )

    return model