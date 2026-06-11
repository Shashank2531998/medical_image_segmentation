from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from src.continual.strategies.lora.utils import lora_state_dict, _matches_target
from src.utils.logging import get_logger

from src.continual.strategies.lora.dynamic.layers import DynamicExpertLinear, DynamicLoRAMultiheadAttention

logger = get_logger(__name__)


def mark_dynamic_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or ".router." in name

    if bias == "none":
        return

    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        return

    if bias == "lora_only":
        for module in model.modules():
            if isinstance(module, DynamicExpertLinear) and module.bias is not None:
                module.bias.requires_grad = True
        return

    raise NotImplementedError(f"Unsupported bias mode: {bias}")


def _replace_attention_modules(
    module: nn.Module,
    lora_cfg: dict[str, Any],
    target_modules: Sequence[str] | None = None,
    prefix: str = "",
    _stats: dict | None = None,
) -> dict:
    if _stats is None:
        _stats = {
            "replaced": 0,
            "modules": [],
            "total_experts": 0,
            "router_params": 0,
            "expert_params": 0,
            "lora_attn_trainable_params": 0,
            "lora_attn_frozen_params": 0,
        }

    replaced = 0
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16.0))
    dropout = float(lora_cfg.get("dropout", 0.0))
    enable_lora = tuple(lora_cfg.get("attn_parts", ["q", "k", "v", "o"]))
    num_experts = int(lora_cfg.get("initial_experts", 1))
    router_temperature = float(lora_cfg.get("router_temperature", 1.0))
    router_top_k = int(lora_cfg.get("router_top_k", 2))
    max_experts = lora_cfg.get("max_experts", None)
    max_experts_value = None if max_experts is None else int(max_experts)

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.MultiheadAttention) and _matches_target(full_name, target_modules):
            lora_attn = DynamicLoRAMultiheadAttention(
                existing_mha=child,
                enable_lora=list(enable_lora),
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_experts=num_experts,
                router_temperature=router_temperature,
                router_top_k=router_top_k,
                module_name=full_name,
                max_experts=max_experts_value,
            )
            module._modules[child_name] = lora_attn
            replaced += 1

            _stats["replaced"] += 1
            _stats["modules"].append(full_name)
            _stats["lora_attn_trainable_params"] += sum(p.numel() for p in lora_attn.parameters() if p.requires_grad)
            _stats["lora_attn_frozen_params"] += sum(p.numel() for p in lora_attn.parameters() if not p.requires_grad)

            lora_modules = [lora_attn.q_proj, lora_attn.k_proj, lora_attn.v_proj, lora_attn.out_proj]
            _stats["total_experts"] += len(lora_attn.q_proj.experts)
            _stats["router_params"] += sum(p.numel() for mod in lora_modules for p in mod.router.parameters())
            for mod in lora_modules:
                for expert in mod.experts:
                    _stats["expert_params"] += sum(p.numel() for p in expert.parameters() if p.requires_grad)

            logger.debug("Injected dynamic LoRA at '%s' (r=%d, experts=%d, attn_parts=%s)", full_name, rank, num_experts, enable_lora)
            continue

        _stats = _replace_attention_modules(child, lora_cfg=lora_cfg, target_modules=target_modules, prefix=full_name, _stats=_stats)

    return _stats


def apply_dynamic_loralib_lora(
    model: nn.Module,
    lora_cfg: dict[str, Any],
    target_modules: Sequence[str] | None = None,
    mark_trainable: bool = True,
) -> nn.Module:
    if target_modules is None:
        target_modules = tuple(
            lora_cfg.get("target_modules", ["transformer_decoder.layers", "self_attn", "multihead_attn"])
        )

    logger.info(
        "Applying dynamic LoRA to model with config: rank=%d, alpha=%s, dropout=%s, initial_experts=%s",
        lora_cfg.get("rank", 8),
        lora_cfg.get("alpha", 16),
        lora_cfg.get("dropout", 0.0),
        lora_cfg.get("initial_experts", 1),
    )
    logger.info("Target modules: %s", list(target_modules))
    logger.info("Attention parts to adapt: %s", lora_cfg.get("attn_parts", ["q", "k", "v", "o"]))
    logger.info("Router top-k: %d", int(lora_cfg.get("router_top_k", 2)))

    stats = _replace_attention_modules(model, lora_cfg, target_modules=target_modules)
    if stats["replaced"] == 0:
        raise ValueError("No MultiheadAttention modules were replaced; check target_modules or model structure.")

    logger.info("========== Dynamic LoRA Injection Summary ==========")
    logger.info("Replaced attention modules : %d", stats["replaced"])
    logger.info("Attention modules         : %s", stats["modules"])
    logger.info("Total experts             : %d", stats["total_experts"])
    logger.info("Router params             : %d", stats["router_params"])
    logger.info("Expert params             : %d", stats["expert_params"])
    logger.info("LoRA Frozen params        : %d", stats["lora_attn_frozen_params"])
    logger.info("LoRA Trainable params     : %d", stats["lora_attn_trainable_params"])
    logger.info("====================================================")

    if mark_trainable:
        mark_dynamic_lora_as_trainable(model, bias=str(lora_cfg.get("bias", "none")))

    return model


def _dynamic_state_dict(model: nn.Module, bias: str = "none") -> dict[str, torch.Tensor]:
    base_state = lora_state_dict(model, bias=bias)
    full_state = model.state_dict()
    for key, value in full_state.items():
        if ".router." in key:
            base_state[key] = value
    return base_state


def save_dynamic_lora_adapter(model: nn.Module, save_path: str | Path, bias: str = "none") -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = _dynamic_state_dict(model, bias=bias)
    torch.save(state, save_path)

    num_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
    logger.info("Saved dynamic LoRA adapter to %s | Size: %d parameters", save_path, num_params)


def _expand_dynamic_modules_from_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    expert_counts: dict[str, int] = {}

    for key in state_dict:
        if ".experts." not in key:
            continue

        module_path, expert_tail = key.split(".experts.", 1)
        expert_index_text = expert_tail.split(".", 1)[0]
        if not expert_index_text.isdigit():
            continue
        expert_counts[module_path] = max(expert_counts.get(module_path, 0), int(expert_index_text) + 1)

    for module_path, expert_count in expert_counts.items():
        current = model
        for part in module_path.split("."):
            current = getattr(current, part)

        if hasattr(current, "grow_to"):
            current.grow_to(expert_count)


def load_dynamic_lora_adapter(
    model: nn.Module,
    checkpoint_path: str | Path,
    bias: str = "none",
    mark_trainable: bool = True,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported dynamic LoRA checkpoint type: {type(state_dict)!r}")

    _expand_dynamic_modules_from_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    if mark_trainable:
        mark_dynamic_lora_as_trainable(model, bias=bias)
    logger.info("Loaded dynamic LoRA adapter from %s", checkpoint_path)
    return model
