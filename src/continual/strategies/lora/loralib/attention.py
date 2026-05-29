from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from . import layers as lora_layers


class LoRAMultiheadAttention(nn.Module):
    """Drop-in MultiheadAttention replacement with LoRA on q/k/v/o projections."""

    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        dropout: float,
        enable_lora: Sequence[str] = ("q", "k", "v", "o"),
        merge_weights: bool = True,
    ) -> None:
        super().__init__()

        if not existing_mha._qkv_same_embed_dim:
            raise ValueError("LoRAMultiheadAttention currently requires q/k/v same embedding dim")

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

        def _build_proj() -> nn.Module:
            return lora_layers.Linear(
                self.embed_dim,
                self.embed_dim,
                r=rank,
                lora_alpha=int(alpha),
                lora_dropout=dropout,
                merge_weights=merge_weights,
                bias=proj_bias,
            )

        self.q_proj = _build_proj() if "q" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.k_proj = _build_proj() if "k" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)
        self.v_proj = _build_proj() if "v" in enable_lora else nn.Linear(self.embed_dim, self.embed_dim, bias=proj_bias)

        self.out_proj = (
            lora_layers.Linear(
                self.embed_dim,
                self.embed_dim,
                r=rank,
                lora_alpha=int(alpha),
                lora_dropout=dropout,
                merge_weights=merge_weights,
                bias=out_bias,
            )
            if "o" in enable_lora
            else nn.Linear(self.embed_dim, self.embed_dim, bias=out_bias)
        )

        with torch.no_grad():
            in_w = existing_mha.in_proj_weight.data
            in_b = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            self.q_proj.weight.copy_(in_w[:self.embed_dim, :])
            self.k_proj.weight.copy_(in_w[self.embed_dim:2 * self.embed_dim, :])
            self.v_proj.weight.copy_(in_w[2 * self.embed_dim:, :])

            if in_b is not None:
                self.q_proj.bias.copy_(in_b[:self.embed_dim])
                self.k_proj.bias.copy_(in_b[self.embed_dim:2 * self.embed_dim])
                self.v_proj.bias.copy_(in_b[2 * self.embed_dim:])

            self.out_proj.weight.copy_(existing_mha.out_proj.weight.data)
            if self.out_proj.bias is not None and existing_mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(existing_mha.out_proj.bias.data)

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
