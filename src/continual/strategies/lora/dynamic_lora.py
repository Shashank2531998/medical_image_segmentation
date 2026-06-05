from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.logging import get_logger

from .loralib_lora import _matches_target
from .loralib.utils import lora_state_dict

logger = get_logger(__name__)


def _matches_locations(module_name: str, locations: Sequence[str] | None) -> bool:
    if not locations:
        return True
    return any(location in module_name for location in locations)


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


class DynamicLoRAExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / math.sqrt(self.rank) if self.rank > 0 else 0.0
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.rank <= 0:
            return

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)

        if self.dropout is not None:
            x = self.dropout(x)
        return F.linear(x, self.delta_weight()) * self.scaling


class DynamicExpertLinear(nn.Module):
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        dropout_rate: float = 0.0,
        num_experts: int = 1,
        router_temperature: float = 1.0,
        router_top_k: int = 2,
        max_experts: int | None = None,
        module_name: str = "",
    ) -> None:
        super().__init__()

        self.in_features = existing_linear.in_features
        self.out_features = existing_linear.out_features
        self.rank = int(r)
        self.lora_alpha = float(lora_alpha)
        self.dropout_rate = float(dropout_rate)
        self.router_temperature = float(router_temperature)
        self.router_top_k = max(1, int(router_top_k))
        self.max_experts = None if max_experts is None else int(max_experts)
        self.module_name = module_name
        self.weight = nn.Parameter(existing_linear.weight.detach().clone(), requires_grad=False)
        self.bias = None if existing_linear.bias is None else nn.Parameter(existing_linear.bias.detach().clone(), requires_grad=False)

        self.experts = nn.ModuleList()
        self.router = nn.Linear(self.in_features, 1, bias=True)
        self._router_entropy_ema: float | None = None
        self._router_normalized_entropy_ema: float | None = None
        self.last_selected_experts: list[int] = []

        initial_experts = max(1, int(num_experts))
        for _ in range(initial_experts):
            self.add_expert(initializing=True)

        self.to(existing_linear.weight.device)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _reset_router(self, new_num_experts: int) -> None:
        old_router = self.router
        new_router = nn.Linear(self.in_features, new_num_experts, bias=True)
        new_router = new_router.to(device=old_router.weight.device, dtype=old_router.weight.dtype)

        with torch.no_grad():
            copy_count = min(old_router.out_features, new_num_experts)
            if copy_count > 0:
                new_router.weight[:copy_count].copy_(old_router.weight[:copy_count])
                new_router.bias[:copy_count].copy_(old_router.bias[:copy_count])
            if new_num_experts > copy_count:
                nn.init.zeros_(new_router.weight[copy_count:])
                nn.init.constant_(new_router.bias[copy_count:], -2.0)

        self.router = new_router

    def add_expert(self, initializing: bool = False) -> int:
        if self.max_experts is not None and len(self.experts) >= self.max_experts:
            return len(self.experts)

        expert = DynamicLoRAExpert(
            self.in_features,
            self.out_features,
            self.rank,
            self.lora_alpha,
            dropout_rate=self.dropout_rate,
        ).to(device=self.weight.device, dtype=self.weight.dtype)
        self.experts.append(expert)

        if not initializing:
            self._reset_router(len(self.experts))
        elif len(self.experts) > 1:
            self._reset_router(len(self.experts))

        return len(self.experts)

    def grow_to(self, target_num_experts: int) -> int:
        target_num_experts = max(1, int(target_num_experts))
        while len(self.experts) < target_num_experts:
            previous_count = len(self.experts)
            self.add_expert()
            if len(self.experts) == previous_count:
                break
        return len(self.experts)

    def _router_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 2:
            return x.reshape(-1, x.shape[-1]).mean(dim=0, keepdim=True)
        return x.reshape(-1, x.shape[-1]).mean(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)

        if self.rank <= 0 or len(self.experts) == 0:
            return base_out

        router_input = self._router_input(x)
        logits = self.router(router_input) / max(self.router_temperature, 1e-6)
        routing_probs = torch.softmax(logits, dim=-1).squeeze(0)

        entropy = -(routing_probs * routing_probs.clamp_min(1e-8).log()).sum(dim=-1).detach().item()
        normalized_entropy = 0.0
        if routing_probs.numel() > 1:
            normalized_entropy = entropy / math.log(float(routing_probs.numel()))
        if self._router_entropy_ema is None:
            self._router_entropy_ema = entropy
            self._router_normalized_entropy_ema = normalized_entropy
        else:
            self._router_entropy_ema = 0.9 * self._router_entropy_ema + 0.1 * entropy
            self._router_normalized_entropy_ema = 0.9 * self._router_normalized_entropy_ema + 0.1 * normalized_entropy

        top_k = min(self.router_top_k, len(self.experts))
        topk_probs, topk_indices = torch.topk(routing_probs, k=top_k, dim=-1)
        selected_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = []
        self.last_selected_experts = topk_indices.tolist()
        for expert_index in topk_indices.tolist():
            expert_outputs.append(self.experts[int(expert_index)](x))

        expert_stack = torch.stack(expert_outputs, dim=0)
        while selected_probs.dim() < expert_stack.dim():
            selected_probs = selected_probs.unsqueeze(-1)

        return base_out + (expert_stack * selected_probs).sum(dim=0)

    def routing_entropy(self) -> float | None:
        return self._router_entropy_ema

    def normalized_routing_entropy(self) -> float | None:
        return self._router_normalized_entropy_ema


class DynamicLoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        dropout: float,
        enable_lora: Sequence[str] = ("q", "k", "v", "o"),
        merge_weights: bool = True,
        num_experts: int = 1,
        router_temperature: float = 1.0,
        max_experts: int | None = None,
        module_name: str = "",
    ) -> None:
        super().__init__()

        if not existing_mha._qkv_same_embed_dim:
            raise ValueError("DynamicLoRAMultiheadAttention currently requires q/k/v same embedding dim")

        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.dropout = float(existing_mha.dropout)
        self.batch_first = bool(existing_mha.batch_first)
        self.add_zero_attn = bool(existing_mha.add_zero_attn)
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        proj_bias = existing_mha.in_proj_bias is not None
        out_bias = existing_mha.out_proj.bias is not None

        def _build_proj(weight: torch.Tensor, bias: torch.Tensor | None) -> DynamicExpertLinear:
            linear = nn.Linear(self.embed_dim, self.embed_dim, bias=bias is not None)
            with torch.no_grad():
                linear.weight.copy_(weight)
                if bias is not None:
                    linear.bias.copy_(bias)
            return DynamicExpertLinear(
                linear,
                r=rank,
                lora_alpha=alpha,
                dropout_rate=dropout,
                num_experts=num_experts,
                router_temperature=router_temperature,
                router_top_k=router_top_k,
                max_experts=max_experts,
                module_name=module_name,
            )

        in_w = existing_mha.in_proj_weight.data
        in_b = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

        self.q_proj = _build_proj(in_w[: self.embed_dim, :], in_b[: self.embed_dim] if in_b is not None else None) if "q" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.k_proj = _build_proj(in_w[self.embed_dim : 2 * self.embed_dim, :], in_b[self.embed_dim : 2 * self.embed_dim] if in_b is not None else None) if "k" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.v_proj = _build_proj(in_w[2 * self.embed_dim :, :], in_b[2 * self.embed_dim :] if in_b is not None else None) if "v" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)

        out_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=out_bias)
        with torch.no_grad():
            out_linear.weight.copy_(existing_mha.out_proj.weight.data)
            if out_linear.bias is not None and existing_mha.out_proj.bias is not None:
                out_linear.bias.copy_(existing_mha.out_proj.bias.data)

        self.out_proj = (
            DynamicExpertLinear(
                out_linear,
                r=rank,
                lora_alpha=alpha,
                dropout_rate=dropout,
                num_experts=num_experts,
                router_temperature=router_temperature,
                router_top_k=router_top_k,
                max_experts=max_experts,
                module_name=module_name,
            )
            if "o" in enable_lora
            else out_linear
        )

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        return x.contiguous().view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if self.batch_first and query.dim() == 3:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        qh = self._to_heads(q)
        kh = self._to_heads(k)
        vh = self._to_heads(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(qh * scale, kh.transpose(-2, -1))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                bsz = qh.shape[0]
                mask = attn_mask.view(bsz, self.num_heads, attn_mask.shape[-2], attn_mask.shape[-1])
            else:
                raise ValueError("attn_mask must be 2D or 3D")

            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            else:
                attn_scores = attn_scores + mask.to(attn_scores.dtype)

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        if is_causal:
            q_len = attn_scores.shape[-2]
            k_len = attn_scores.shape[-1]
            causal = torch.triu(torch.ones(q_len, k_len, device=attn_scores.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, vh)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(query.shape[0], query.shape[1], self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first and attn_output.dim() == 3:
            attn_output = attn_output.transpose(0, 1)

        if not need_weights:
            return attn_output, None

        if average_attn_weights:
            return attn_output, attn_probs.mean(dim=1)
        return attn_output, attn_probs


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
            logger.debug(
                "Injected dynamic LoRA at '%s' (r=%d, experts=%d, attn_parts=%s)",
                full_name,
                rank,
                num_experts,
                enable_lora,
            )
            continue

        replaced += _replace_attention_modules(
            child,
            lora_cfg=lora_cfg,
            target_modules=target_modules,
            prefix=full_name,
        )

    return replaced


def apply_dynamic_loralib_lora(
    model: nn.Module,
    lora_cfg: dict[str, Any],
    target_modules: Sequence[str] | None = None,
    mark_trainable: bool = True,
) -> nn.Module:
    if target_modules is None:
        target_modules = tuple(
            lora_cfg.get(
                "target_modules",
                ["transformer_decoder.layers", "self_attn", "multihead_attn"],
            )
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

    replace_count = _replace_attention_modules(model, lora_cfg, target_modules=target_modules)
    if replace_count == 0:
        raise ValueError("No MultiheadAttention modules were replaced; check target_modules or model structure.")

    logger.info("Successfully injected dynamic LoRA into %d MultiheadAttention modules", replace_count)
 
    if mark_trainable:
        bias = str(lora_cfg.get("bias", "none"))
        mark_dynamic_lora_as_trainable(model, bias=bias)

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


def iter_dynamic_lora_modules(
    model: nn.Module,
    locations: Sequence[str] | None = None,
) -> list[tuple[str, DynamicExpertLinear]]:
    modules: list[tuple[str, DynamicExpertLinear]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, DynamicExpertLinear) and _matches_locations(module_name or module.module_name, locations):
            modules.append((module_name, module))
    return modules


def rank_dynamic_lora_modules(
    model: nn.Module,
    locations: Sequence[str] | None = None,
) -> list[tuple[str, DynamicExpertLinear]]:
    ranked = iter_dynamic_lora_modules(model, locations=locations)
    ranked.sort(
        key=lambda item: (
            item[1].normalized_routing_entropy() if item[1].normalized_routing_entropy() is not None else -1.0,
            item[1].routing_entropy() if item[1].routing_entropy() is not None else -1.0,
        ),
        reverse=True,
    )
    return ranked


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


def grow_dynamic_lora_modules(
    model: nn.Module,
    num_new_experts: int = 1,
    locations: Sequence[str] | None = None,
    max_modules: int | None = None,
) -> int:
    ranked_modules = rank_dynamic_lora_modules(model, locations=locations)
    if max_modules is not None:
        ranked_modules = ranked_modules[: max(0, int(max_modules))]

    grown = 0
    for _, module in ranked_modules:
        before = module.num_experts
        module.grow_to(before + num_new_experts)
        grown += module.num_experts - before
    return grown