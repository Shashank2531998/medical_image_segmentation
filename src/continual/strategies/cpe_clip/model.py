from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange, repeat
from torch import nn

from src.model.voxtell_model import VoxTellModel
from src.continual.strategies.cpe_clip.transformer import CPECLIPTransformerDecoder
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class CPECLIPConfig:
    prompt_tokens: int = 4
    prompt_layers: int | None = None
    alpha_base: float = 1.0
    alpha_floor: float = 0.25
    prompt_init_std: float = 0.02
    regularization_method: str = "balance"
    trainable: bool = True


def _normalize_config(cfg: dict[str, Any] | None, model: VoxTellModel) -> CPECLIPConfig:
    cfg = dict(cfg or {})
    prompt_layers = cfg.get("prompt_layers")
    if prompt_layers is None:
        prompt_layers = len(model.transformer_decoder.layers)

    return CPECLIPConfig(
        prompt_tokens=max(1, int(cfg.get("prompt_tokens", 4))),
        prompt_layers=max(1, int(prompt_layers)),
        alpha_base=float(cfg.get("alpha_base", 1.0)),
        alpha_floor=float(cfg.get("alpha_floor", 0.25)),
        prompt_init_std=float(cfg.get("prompt_init_std", 0.02)),
        regularization_method=str(cfg.get("regularization_method", "balance")),
        trainable=bool(cfg.get("trainable", True)),
    )


class CPECLIPPromptedVoxTell(nn.Module):
    def __init__(self, base_model: VoxTellModel, cpe_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = _normalize_config(cpe_cfg, base_model)
        self.session_alpha = 1.0

        text_dim = int(base_model.text_embedding_dim)
        query_dim = int(base_model.query_dim)

        self._freeze_base_model()

        original_decoder = self.base_model.transformer_decoder
        cpe_clip_decoder = CPECLIPTransformerDecoder(
            decoder=original_decoder,
            query_dim=self.base_model.query_dim,
            prompt_tokens=self.config.prompt_tokens,
            prompt_layers=self.config.prompt_layers
        )
        cpe_clip_decoder.set_trainable(self.config.trainable)
        if self.config.trainable:
            cpe_clip_decoder.register_gradient_scaling_hooks()
        self.base_model.transformer_decoder = cpe_clip_decoder

        logger.info(
            "Initialized CPE-CLIP wrapper | prompt_tokens=%d | prompt_layers=%d | text_dim=%d | query_dim=%d | trainable=%s | regularization=%s",
            self.config.prompt_tokens,
            self.config.prompt_layers,
            text_dim,
            query_dim,
            self.config.trainable,
            self.config.regularization_method,
        )

    def _freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        logger.debug("CPE-CLIP backbone frozen")


    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def scale_alpha(self, task_index: int) -> float:
        if task_index <= 0:
            return 1.0
        alpha = self.config.alpha_base / float(task_index + 1)
        return max(self.config.alpha_floor, alpha)

    def forward(self, img: torch.Tensor, text_embedding: torch.Tensor | None = None, return_features = False):
        return self.base_model(img, text_embedding, return_features=return_features)


def load_cpe_clip_state_dict(model: CPECLIPPromptedVoxTell, state_dict: dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.base_model.load_state_dict(state_dict, strict=True)


def configure_loaded_cpe_clip_model(
    model: nn.Module,
    cpe_cfg: dict[str, Any] | None,
    *,
    mark_trainable: bool = True,
) -> CPECLIPPromptedVoxTell:
    
    logger.info(
        "Applying CPE-CLIP wrapper | trainable=%s | prompt_tokens=%s | prompt_layers=%s | regularization=%s",
        bool(cpe_cfg.get("trainable", True)),
        cpe_cfg.get("prompt_tokens", 4),
        cpe_cfg.get("prompt_layers", "auto"),
        cpe_cfg.get("regularization_method", "balance"),
    )

    if isinstance(model, CPECLIPPromptedVoxTell):
        wrapper = model
    else:
        wrapper = CPECLIPPromptedVoxTell(model, cpe_cfg)

    decoder = wrapper.base_model.transformer_decoder
    decoder.set_trainable(mark_trainable)
    if mark_trainable and not getattr(decoder, "_gradient_hooks_registered", False):
        decoder.register_gradient_scaling_hooks()
    decoder.set_session_alpha(float((cpe_cfg or {}).get("session_alpha", 1.0)), log=False)
    wrapper.base_model.transformer_decoder = decoder
    
    return wrapper