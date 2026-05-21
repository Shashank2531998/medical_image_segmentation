from pathlib import Path

import torch

from src.model.builder import load_voxtell_model
from src.text.encoder import TextPromptEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VoxTellEngine:

    def __init__(self, model_cfg: dict | None = None, device: str = "cuda"):
        self.model_cfg = model_cfg
        self.device = torch.device(self.model_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        logger.info("VoxTellEngine initialized | device=%s", self.device)

        self.text_encoder = TextPromptEncoder(
            self.model_cfg.get("text_encoding_model", "Qwen/Qwen3-Embedding-4B"),
            device=self.device,
        )
        self.ds_weights = self.model_cfg.get("deep_supervision_weights", [1, 0.5, 0.25, 0.125, 0.0625])
        self.load_model()

    def load_model(self):

        model_dir = self.model_cfg.get("dir", None)
        if model_dir is None:
            raise ValueError("model_dir must be provided in model_cfg or as argument")
        
        model, _ = load_voxtell_model(
            model_dir,
            deep_supervision=self.model_cfg.get("deep_supervision", False),
            model_overrides=self.model_cfg,
            reinit_weights=self.model_cfg.get("reinit_weights", False)
        )
        self.model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded successfully | device=%s | parameters=%d", self.device, total_params)

    def forward(self, batch):
        imgs = batch["image"].to(self.device)
        prompts = batch["prompts"]

        # prompts is expected to be a list-of-lists: [[p1,p2,p3], [p1,p2,p3], ...]
        text_embeddings = self.text_encoder.embed(prompts).clone()

        # At a patch level -
        # outputs will have length equal to 5, if deep supervision is enabled, otherwise 1 (e.g. - (5, 1, 3, 192, 192, 192))
        # For each sample of the batch, there will be 3 masks, 2 for positive and 1 for negative (e.g. - (1, 3, 192, 192, 192))
        outputs = self.model(imgs, text_embeddings)
        return outputs