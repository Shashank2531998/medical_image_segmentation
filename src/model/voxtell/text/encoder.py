from __future__ import annotations

from typing import List, Union

import torch
from nnunetv2.utilities.helpers import empty_cache
from transformers import AutoModel, AutoTokenizer

from src.utils.text_embedding import last_token_pool, wrap_with_instruction


class TextPromptEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: torch.device | None = None,
        max_length: int = 8192,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.text_backbone = AutoModel.from_pretrained(model_name).eval()
        self.max_length = max_length

    @torch.no_grad()
    def embed(self, text_prompts: Union[List[str], List[List[str]], str]) -> torch.Tensor:
        # Support single string, flat list of prompts, or list-of-lists for per-sample prompts
        nested = False
        if isinstance(text_prompts, str):
            batch_prompts = [[text_prompts]]
        elif isinstance(text_prompts, list) and len(text_prompts) > 0 and isinstance(text_prompts[0], (list, tuple)):
            nested = True
            batch_prompts = [list(p) for p in text_prompts]
        else:
            batch_prompts = [list(text_prompts)]

        batch_size = len(batch_prompts)
        n_prompts = len(batch_prompts[0])
        # flatten for tokenization
        flat_prompts = [p for sample in batch_prompts for p in sample]

        self.text_backbone = self.text_backbone.to(self.device)
        prepared_prompts = wrap_with_instruction(flat_prompts)
        text_tokens = self.tokenizer(
            prepared_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        text_tokens = {key: value.to(self.device) for key, value in text_tokens.items()}
        text_embed = self.text_backbone(**text_tokens)
        embeddings = last_token_pool(text_embed.last_hidden_state, text_tokens["attention_mask"])  # (total_prompts, dim)

        # reshape to (B, N, D)
        embeddings = embeddings.view(batch_size, n_prompts, -1)

        self.text_backbone = self.text_backbone.to("cpu")
        empty_cache(self.device)
        return embeddings
