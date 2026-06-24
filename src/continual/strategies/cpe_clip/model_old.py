from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange, repeat
from torch import nn

from src.model.voxtell_model import VoxTellModel
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

        self._prompt_tokens = self.config.prompt_tokens
        self._prompt_layers = self.config.prompt_layers

        text_dim = int(base_model.text_embedding_dim)
        query_dim = int(base_model.query_dim)
        self.text_prompts = nn.Parameter(torch.empty(self._prompt_layers, self._prompt_tokens, text_dim))
        
        text_hidden_dim = 2048
        self.prompt_projection = nn.Sequential(
            nn.Linear(text_dim, text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, query_dim),
        )

        self._reset_prompt_parameters(self.config.prompt_init_std)
        self._freeze_base_model()
        self._set_trainable(self.config.trainable)
        self._gradient_hooks_registered = False
        if self.config.trainable:
            self._register_gradient_scaling_hooks()

        logger.info(
            "Initialized CPE-CLIP wrapper | prompt_tokens=%d | prompt_layers=%d | text_dim=%d | query_dim=%d | trainable=%s | regularization=%s",
            self._prompt_tokens,
            self._prompt_layers,
            text_dim,
            query_dim,
            self.config.trainable,
            self.config.regularization_method,
        )

    def _reset_prompt_parameters(self, std: float) -> None:
        nn.init.normal_(self.text_prompts, mean=0.0, std=std)
        for module in self.prompt_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        logger.debug("CPE-CLIP backbone frozen")

    def _set_trainable(self, enabled: bool) -> None:
        self.text_prompts.requires_grad_(enabled)
        for parameter in self.prompt_projection.parameters():
            parameter.requires_grad = enabled
        logger.debug("CPE-CLIP prompt trainable state set | enabled=%s", enabled)

    def _register_gradient_scaling_hooks(self) -> None:
        if self._gradient_hooks_registered:
            return

        def scale_grad(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            return grad * float(self.session_alpha)

        self.text_prompts.register_hook(scale_grad)
        for parameter in self.prompt_projection.parameters():
            if parameter.requires_grad:
                parameter.register_hook(scale_grad)

        self._gradient_hooks_registered = True

    def set_session_alpha(self, alpha: float, *, log: bool = True) -> None:
        self.session_alpha = float(alpha)
        if log:
            logger.info("CPE-CLIP session alpha updated | alpha=%.4f", self.session_alpha)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    @property
    def prompt_tokens(self) -> int:
        return int(self._prompt_tokens)

    @property
    def prompt_layers(self) -> int:
        return int(self._prompt_layers)

    def _project_prompt_layer(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        layer_idx = min(max(layer_idx, 0), self.prompt_layers - 1)
        projection_device = device
        prompt_projection = self.prompt_projection.to(projection_device)
        prompts = self.text_prompts[layer_idx].to(projection_device)
        prompts = prompt_projection(prompts)
        return repeat(prompts, "p d -> p b d", b=batch_size)

    def _build_prompted_sequence(
        self,
        text_embedding: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        language_prompts = self._project_prompt_layer(layer_idx, batch_size, device)
        text_embedding = text_embedding.to(language_prompts.device)
        return torch.cat([language_prompts, text_embedding], dim=0).to(device)

    def _scale_alpha(self, task_index: int) -> float:
        if task_index <= 0:
            return 1.0
        alpha = self.config.alpha_base / float(task_index + 1)
        return max(self.config.alpha_floor, alpha)

    def forward(self, img: torch.Tensor, text_embedding: torch.Tensor | None = None):
        if text_embedding is None:
            raise ValueError("CPECLIPPromptedVoxTell requires text embeddings.")

        if text_embedding.ndim == 4:
            text_embedding = text_embedding.squeeze(2)

        if text_embedding.ndim != 3:
            raise ValueError(
                f"Expected text embeddings with shape (batch, prompts, dim), got {tuple(text_embedding.shape)}"
            )

        model_device = next(self.base_model.parameters()).device
        img = img.to(model_device)
        text_embedding = text_embedding.to(model_device)
        logger.debug(
            "CPE-CLIP forward | device=%s | image_shape=%s | text_embedding_shape=%s",
            model_device,
            tuple(img.shape),
            tuple(text_embedding.shape),
        )

        skips = self.base_model.encoder(img)
        selected_feature = skips[self.base_model.selected_decoder_layer]

        bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
        bottleneck_embed = self.base_model.project_bottleneck_embed(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c")

        text_embed = repeat(text_embedding, "b n dim -> n b dim")
        text_embed = self.base_model.project_text_embed(text_embed)
        text_embed = text_embed.to(model_device)

        decoder_layers = self.base_model.transformer_decoder.layers
        prompt_limit = min(self.prompt_layers, len(decoder_layers))
        batch_size = text_embed.shape[1]

        sequence = self._build_prompted_sequence(text_embed, 0, batch_size, model_device)
        vision_prompts: torch.Tensor | None = None
        class_tokens = text_embed.shape[0]

        for layer_idx, layer in enumerate(decoder_layers):
            layer_device = next(layer.parameters()).device
            sequence = sequence.to(layer_device)
            bottleneck_device = bottleneck_embed.to(layer_device)
            memory = bottleneck_device if vision_prompts is None else torch.cat([bottleneck_device, vision_prompts.to(layer_device)], dim=0)
            memory = memory.to(layer_device)
            memory_pos = self.base_model.pos_embed.to(layer_device)
            if vision_prompts is not None:
                prompt_pos = torch.zeros_like(vision_prompts, device=layer_device)
                memory_pos = torch.cat([memory_pos, prompt_pos], dim=0)
            sequence, _ = layer(
                sequence,
                memory,
                pos=memory_pos,
                memory_key_padding_mask=None,
                residual=True,
            )

            if layer_idx < prompt_limit:
                prompt_state = sequence[:self.prompt_tokens].to(layer_device)
                class_state = sequence[self.prompt_tokens:].to(layer_device)

                vision_prompts = prompt_state if vision_prompts is None else torch.cat([vision_prompts, prompt_state], dim=0)

                if layer_idx + 1 < prompt_limit:
                    next_prompts = self._project_prompt_layer(layer_idx + 1, batch_size, layer_device)
                    sequence = torch.cat([next_prompts, class_state], dim=0)
                else:
                    sequence = class_state

        logger.debug(
            "CPE-CLIP decoder complete | final_sequence_shape=%s | vision_prompt_tokens=%s",
            tuple(sequence.shape),
            0 if vision_prompts is None else int(vision_prompts.shape[0]),
        )

        decoder_norm = self.base_model.transformer_decoder.norm
        if decoder_norm is not None:
            sequence = decoder_norm(sequence)

        if sequence.shape[0] != class_tokens:
            sequence = sequence[-class_tokens:]

        mask_embedding = repeat(sequence, "n b dim -> b n dim")
        mask_embeddings = [projection(mask_embedding) for projection in self.base_model.project_to_decoder_channels]

        outs = []
        num_prompts = text_embedding.shape[1]
        for prompt_idx in range(num_prompts):
            prompt_embeds = [m[:, prompt_idx:prompt_idx + 1] for m in mask_embeddings]
            outs.append(self.base_model.decoder(skips, prompt_embeds))

        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]
        if not self.base_model.deep_supervision:
            outs = outs[0]
        return outs


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

    wrapper._set_trainable(mark_trainable)
    if mark_trainable and not getattr(wrapper, "_gradient_hooks_registered", False):
        wrapper._register_gradient_scaling_hooks()
    wrapper.set_session_alpha(float((cpe_cfg or {}).get("session_alpha", 1.0)), log=False)
    return wrapper