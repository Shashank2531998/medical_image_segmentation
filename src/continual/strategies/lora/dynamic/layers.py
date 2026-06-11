from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _matches_locations(module_name: str, locations: Sequence[str] | None) -> bool:
    if not locations:
        return True
    return any(location in module_name for location in locations)


def _clone_expert_with_noise(expert: nn.Module, source_expert: nn.Module, noise_scale: float = 0.01) -> None:
    with torch.no_grad():
        expert.load_state_dict(source_expert.state_dict())
        for param in expert.parameters():
            param.add_(noise_scale * torch.randn_like(param))


class DynamicLoRAExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout_rate: float = 0.0) -> None:
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
    def __init__(self, existing_linear: nn.Linear, r: int = 0, lora_alpha: int = 1, dropout_rate: float = 0.0,
                 num_experts: int = 1, router_temperature: float = 1.0, router_top_k: int = 2,
                 max_experts: int | None = None, module_name: str = "") -> None:
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
        self.last_router_probs = None
        self.last_router_input = None
        self.last_mahalanobis_scores = None
        self.novelty_tracker = None
        self._router_entropy_ema: float | None = None
        self._router_normalized_entropy_ema: float | None = None
        self.last_selected_experts: list[int] = []

        self._expert_usage = []
        self._expert_selection_count = []
        self._initialize_experts(max(0, int(num_experts)))

        self.to(existing_linear.weight.device)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _initialize_experts(self, num_experts: int) -> None:
        for _ in range(num_experts):
            self.add_expert(initializing=True)

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

    def add_expert(self, initializing: bool = False, clone_from: int | None = None) -> int:
        if self.max_experts is not None and len(self.experts) >= self.max_experts:
            return len(self.experts)

        expert = DynamicLoRAExpert(self.in_features, self.out_features, self.rank, self.lora_alpha, dropout_rate=self.dropout_rate)
        expert = expert.to(device=self.weight.device, dtype=self.weight.dtype)

        if clone_from is not None and 0 <= clone_from < len(self.experts):
            _clone_expert_with_noise(expert, self.experts[clone_from])
            logger.info("Cloning expert for module=%s | source_expert=%d | new_expert=%d | noise_scale=0.01",
                        self.module_name or "<unnamed>", clone_from, len(self.experts))

        self.experts.append(expert)

        if not initializing:
            self._reset_router(len(self.experts))
        elif len(self.experts) > 1:
            self._reset_router(len(self.experts))

        self._expert_usage.append(0.0)
        self._expert_selection_count.append(0)
        return len(self.experts)

    def grow_to(self, target_num_experts: int, clone_from: int | None = None) -> int:
        target_num_experts = max(0, int(target_num_experts))
        while len(self.experts) < target_num_experts:
            previous_count = len(self.experts)
            self.add_expert(clone_from=clone_from)
            if len(self.experts) == previous_count:
                break
        return len(self.experts)

    def _router_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1]).mean(dim=0, keepdim=True)

    def _pooled_representation(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 2:
            return x.float()
        batch_size, seq_len = x.shape[1], x.shape[0]
        return x.transpose(0, 1).reshape(batch_size, seq_len, -1).mean(dim=1)

    def _routing_distribution(self, x: torch.Tensor) -> torch.Tensor:
        router_input = self._router_input(x)
        logits = self.router(router_input) / max(self.router_temperature, 1e-6)
        return torch.softmax(logits, dim=-1).squeeze(0)

    def _update_routing_stats(self, routing_probs: torch.Tensor) -> None:
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

    def _select_topk_experts(self, routing_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = min(self.router_top_k, len(self.experts))
        topk_probs, topk_indices = torch.topk(routing_probs, k=top_k, dim=-1)
        selected_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return selected_probs, topk_indices

    def _record_usage(self, topk_indices: torch.Tensor, selected_probs: torch.Tensor) -> None:
        for idx, prob in zip(topk_indices.tolist(), selected_probs.tolist()):
            self._expert_selection_count[idx] += 1
            self._expert_usage[int(idx)] += float(prob)

    def expert_usage_stats(self):
        return {i: float(c) / self._expert_selection_count[i] for i, c in enumerate(self._expert_usage) if c > 0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        if self.rank <= 0 or len(self.experts) == 0:
            return base_out

        if self.novelty_tracker is not None:
            pooled = self._pooled_representation(x)
            self.last_mahalanobis_scores = self.novelty_tracker.observe(pooled)

        routing_probs = self._routing_distribution(x)
        router_input = self._router_input(x)
        self.last_router_probs = routing_probs.detach().cpu()
        self.last_router_input = router_input.detach().cpu()

        self._update_routing_stats(routing_probs)
        selected_probs, topk_indices = self._select_topk_experts(routing_probs)
        self._record_usage(topk_indices, selected_probs)

        expert_outputs = [self.experts[int(expert_index)](x) for expert_index in topk_indices.tolist()]
        self.last_selected_experts = topk_indices.tolist()

        expert_stack = torch.stack(expert_outputs, dim=0)
        while selected_probs.dim() < expert_stack.dim():
            selected_probs = selected_probs.unsqueeze(-1)

        return base_out + (expert_stack * selected_probs).sum(dim=0)

    def routing_entropy(self) -> float | None:
        return self._router_entropy_ema

    def normalized_routing_entropy(self) -> float | None:
        return self._router_normalized_entropy_ema


class DynamicLoRAMultiheadAttention(nn.Module):
    def __init__(self, existing_mha: nn.MultiheadAttention, rank: int, alpha: float, dropout: float,
                 enable_lora: Sequence[str] = ("q", "k", "v", "o"), merge_weights: bool = True,
                 num_experts: int = 1, router_temperature: float = 1.0, router_top_k=2,
                 max_experts: int | None = None, module_name: str = "") -> None:
        super().__init__()
        if not existing_mha._qkv_same_embed_dim:
            raise ValueError("DynamicLoRAMultiheadAttention currently requires q/k/v same embedding dim")

        self.module_name = module_name
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
            return DynamicExpertLinear(linear, r=rank, lora_alpha=alpha, dropout_rate=dropout,
                                       num_experts=num_experts, router_temperature=router_temperature,
                                       router_top_k=router_top_k, max_experts=max_experts,
                                       module_name=module_name)

        in_w = existing_mha.in_proj_weight.data
        in_b = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

        self.q_proj = _build_proj(in_w[:self.embed_dim, :], in_b[:self.embed_dim] if in_b is not None else None) if "q" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.k_proj = _build_proj(in_w[self.embed_dim:2 * self.embed_dim, :], in_b[self.embed_dim:2 * self.embed_dim] if in_b is not None else None) if "k" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.v_proj = _build_proj(in_w[2 * self.embed_dim:, :], in_b[2 * self.embed_dim:] if in_b is not None else None) if "v" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)

        out_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=out_bias)
        with torch.no_grad():
            out_linear.weight.copy_(existing_mha.out_proj.weight.data)
            if out_linear.bias is not None and existing_mha.out_proj.bias is not None:
                out_linear.bias.copy_(existing_mha.out_proj.bias.data)

        self.out_proj = DynamicExpertLinear(out_linear, r=rank, lora_alpha=alpha, dropout_rate=dropout,
                                           num_experts=num_experts, router_temperature=router_temperature,
                                           router_top_k=router_top_k, max_experts=max_experts,
                                           module_name=module_name) if "o" in enable_lora else out_linear
        self.to(existing_mha.in_proj_weight.device)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        return x.contiguous().view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                need_weights: bool = True, attn_mask: torch.Tensor | None = None, average_attn_weights: bool = True,
                is_causal: bool = False):
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
        return (attn_output, attn_probs.mean(dim=1)) if average_attn_weights else (attn_output, attn_probs)


def iter_dynamic_lora_modules(model: nn.Module, locations: Sequence[str] | None = None) -> list[tuple[str, DynamicExpertLinear]]:
    modules = []
    for module_name, module in model.named_modules():
        if isinstance(module, DynamicExpertLinear) and _matches_locations(module_name or module.module_name, locations):
            modules.append((module_name, module))
    return modules


def rank_dynamic_lora_modules(model: nn.Module, locations: Sequence[str] | None = None) -> list[tuple[str, DynamicExpertLinear]]:
    ranked = iter_dynamic_lora_modules(model, locations=locations)
    ranked.sort(key=lambda item: (item[1].normalized_routing_entropy() if item[1].normalized_routing_entropy() is not None else -1.0,
                                  item[1].routing_entropy() if item[1].routing_entropy() is not None else -1.0), reverse=True)
    return ranked
