from typing import Optional, Tuple, Union, List

import torch
from torch import nn, Tensor
from einops import repeat

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CPECLIPTransformerDecoder(nn.Module):
    """
    CPE-CLIP version of TransformerDecoder.

    Reuses the original decoder layers while injecting
    language prompts and accumulating visual prompts.
    """

    def __init__(
        self,
        decoder,
        query_dim: int,
        prompt_tokens: int = 4,
        prompt_layers: Optional[int] = None,
        method: str = "replacement",
        text_prompting = True
    ):
        super().__init__()

        # Reuse original decoder
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.return_intermediate = decoder.return_intermediate
        self.num_layers = decoder.num_layers

        # CPE settings
        self.prompt_tokens = prompt_tokens
        self.prompt_layers = min(prompt_layers, self.num_layers)
        self.method = method
        self.text_prompting = text_prompting

        # Learnable G-prompts
        self.g_values = nn.Parameter(
            torch.zeros(
                self.prompt_layers,
                self.prompt_tokens,
                query_dim,
            )
        )

        # Meta-Net text embedding bias learning from memory and text
        if self.text_prompting:
            text_hidden_dim = 2048
            self.meta_net = nn.Sequential(
                nn.Linear(2 * query_dim, text_hidden_dim),
                nn.GELU(),
                nn.Linear(text_hidden_dim, query_dim),
            )

        nn.init.xavier_uniform_(self.g_values.data)
        self._gradient_hooks_registered = False

    def set_trainable(self, enabled: bool) -> None:
        self.g_values.requires_grad_(enabled)
        if self.text_prompting:
            for param in self.meta_net.parameters():
                param.requires_grad_(enabled)
        logger.debug("CPE-CLIP prompt trainable state set | enabled=%s", enabled)

    def set_session_alpha(self, alpha: float, *, log: bool = True) -> None:
        self.session_alpha = float(alpha)
        if log:
            logger.info("CPE-CLIP session alpha updated | alpha=%.4f", self.session_alpha)

    def register_gradient_scaling_hooks(self) -> None:
        if self._gradient_hooks_registered:
            return

        def scale_grad(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            return grad * float(self.session_alpha)

        self.g_values.register_hook(scale_grad)

        self._gradient_hooks_registered = True
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        output = tgt
        T, B, C = memory.shape
        intermediate = []
        atten_layers = []

        initial_memory = memory
        initial_mem_pos = pos
        device = initial_memory.device
        
        for layer_idx, layer in enumerate(self.layers):

            if layer_idx < self.prompt_layers:
                layer_g = self.g_values[layer_idx]         # [P, C]
                vision_prompt = (
                    layer_g
                    .unsqueeze(1)                          # [P,1,C]
                    .expand(-1, B, -1)                     # [P,B,C]
                ).to(device)
                mem_pos = torch.zeros_like(vision_prompt, device=pos.device)

                if self.method == "replacement":
                    memory = torch.cat([initial_memory, vision_prompt], dim=0)
                    pos = torch.cat([initial_mem_pos, mem_pos], dim=0) if pos is not None else None
                elif self.method == "accumulate_same":
                    memory = torch.cat([memory, vision_prompt], dim=0)
                    pos = torch.cat([pos, mem_pos],dim=0) if pos is not None else None

                if self.text_prompting:
                    # Global image summary
                    global_feat = initial_memory.mean(dim=0)        # [B, C]
                    q = output.permute(1,0,2)                       # [B, T, C]
                    global_feat = global_feat.unsqueeze(1).expand(
                        -1, q.shape[1], -1
                    )                                               # [B, T, C]
                    delta_q = self.meta_net.to(device)(
                        torch.cat([q, global_feat], dim=-1)         # [B, T, 2*C]
                    )                                               # [B, T, C]
                    delta_q = delta_q.permute(1,0,2)                # [T, B, C]                        
                    output = output + delta_q

            residual = True
            output, ws = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                residual=residual
            )

            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, atten_layers
