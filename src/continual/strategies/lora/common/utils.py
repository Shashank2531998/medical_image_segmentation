from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from src.continual.strategies.lora.utils import lora_state_dict, _matches_target
from src.continual.strategies.lora.common.layers import LoRALayer, PlainMultiheadAttentionLoRA
from src.utils.logging import get_logger

logger = get_logger(__name__)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def _replace_attention_modules(
    module: nn.Module,
    lora_cfg: dict[str, Any],
    target_modules: Sequence[str] | None = None,
    prefix: str = "",
) -> int:
    replaced = 0
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.0))
    merge_weights = bool(lora_cfg.get("merge_weights", True))
    enable_lora = tuple(lora_cfg.get("attn_parts", ["q", "k", "v", "o"]))

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.MultiheadAttention) and _matches_target(full_name, target_modules):
            device = child.in_proj_weight.device
            lora_attn = PlainMultiheadAttentionLoRA(
                existing_mha=child,
                enable_lora=list(enable_lora),
                r=rank,
                lora_alpha=alpha,
                dropout_rate=dropout,
            )
            module._modules[child_name] = lora_attn
            replaced += 1
            logger.debug("Injected LoRA at '%s' (device: %s, r=%d, attn_parts=%s)", 
                        full_name, device, rank, enable_lora)
            continue

        replaced += _replace_attention_modules(
            child,
            lora_cfg=lora_cfg,
            target_modules=target_modules,
            prefix=full_name,
        )

    return replaced


def apply_loralib_lora(
    model: nn.Module,
    lora_cfg: dict[str, Any],
    target_modules: Sequence[str] | None = None,
    mark_trainable: bool = True,
) -> nn.Module:
    # By default, adapt only TransformerDecoderLayer self and cross attention blocks.
    if target_modules is None:
        target_modules = tuple(
            lora_cfg.get(
                "target_modules",
                [
                    "transformer_decoder.layers",
                    "self_attn",
                    "multihead_attn",
                ],
            )
        )

    logger.info("Applying LoRA to model with config: rank=%d, alpha=%s, dropout=%s",
                lora_cfg.get("rank", 8), lora_cfg.get("alpha", 16), lora_cfg.get("dropout", 0.0))
    logger.info("Target modules: %s", list(target_modules))
    logger.info("Attention parts to adapt: %s", lora_cfg.get("attn_parts", ["q", "k", "v", "o"]))
    
    replace_count = _replace_attention_modules(model, lora_cfg, target_modules=target_modules)

    if replace_count == 0:
        raise ValueError("No MultiheadAttention modules were replaced; check target_modules or model structure.")
    
    logger.info("Successfully injected LoRA into %d MultiheadAttention modules", replace_count)

    if mark_trainable:
        bias = str(lora_cfg.get("bias", "none"))
        mark_only_lora_as_trainable(model, bias=bias)
    
    return model


def save_lora_adapter(model: nn.Module, save_path: str | Path, bias: str = "none") -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model, bias=bias)
    torch.save(state, save_path)
    
    # Log adapter details
    num_params = sum(p.numel() for p in [v for v in state.values() if isinstance(v, torch.Tensor)])
    logger.info("Saved LoRA adapter to %s | Size: %d parameters", save_path, num_params)


def load_lora_adapter(
    model: nn.Module,
    checkpoint_path: str | Path,
    bias: str = "none",
    mark_trainable: bool = True,
) -> nn.Module:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    if mark_trainable:
        mark_only_lora_as_trainable(model, bias=bias)
    logger.info("Loaded LoRA adapter from %s", checkpoint_path)
    return model
