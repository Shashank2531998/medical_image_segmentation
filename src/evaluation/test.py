from pathlib import Path
import torch
import json
from collections import defaultdict

import monai
import numpy as np
from datetime import datetime

from src.training.losses import deep_supervision_loss
from src.utils.logging import get_logger
from src.inference.predictor import get_predictor

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, model_cfg: dict | None = None, eval_cfg: dict | None = None, disable_adapters=False):
        self.eval_cfg = eval_cfg or {}
        self.model_cfg = model_cfg or {}

        self.device = self.device = torch.device(self.model_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.ds_weights = self.model_cfg.get(
            "deep_supervision_weights",
            [1, 0.5, 0.25, 0.125, 0.0625],
        )

        # logging
        self.out_root = Path(self.eval_cfg.get("output_dir", "experiments/eval"))
        (self.out_root / "logs").mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            __name__,
            log_file=self.out_root / "logs" / "eval.log",
        )
        self.dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="none", ignore_empty=False)

        self.checkpoint_path = self.model_cfg.get("checkpoint_path", None)
        self.lora_adapter_path = self.model_cfg.get("lora_adapter_path", None)
        self.disable_adapters = disable_adapters
        self.predictor = None

    def set_predictor(self):
        if not self.predictor:
            model_dir = self.model_cfg.get("dir", None)
            lora_cfg = self.model_cfg.get("lora_cfg", None)

            if model_dir is None:
                raise ValueError("model_dir must be provided in model_cfg or as argument")
            
            self.predictor = get_predictor(
                model_dir,
                self.device,
                checkpoint_path=self.checkpoint_path,
                lora_cfg=lora_cfg,
                lora_adapter_path=self.lora_adapter_path,
                disable_adapters=self.disable_adapters
            )

    @torch.no_grad()
    def evaluate(self, datamodule):
        """
        Runs full validation / evaluation loop.
        """

        self.set_predictor()

        test_loader = datamodule.test_dataloader()

        self.logger.info(
            f"Model Checkpoint: {self.checkpoint_path} | LoRA Adapter Path: {self.lora_adapter_path}"
        )

        self.logger.info(
            "Evaluation started | dataset=%s | device=%s | batches=%d",
            datamodule.dataset_name,
            self.device,
            len(test_loader),
        )

        total_loss = 0.0
        total_dice = 0.0
        prompt_scores = defaultdict(list)
        self.dice_metric.reset()

        for batch_idx, batch in enumerate(test_loader):

            imgs = batch["image"].to(self.device)
            img_paths = batch["img_path"]

            self.logger.info(
                "Img Path: " + ",".join(map(str, img_paths))
            )

            prompts = batch["prompts"]
            prompt_labels = prompts[0]

            self.logger.info(
                "Prompts: " + ", ".join(prompt_labels)
            )

            # Embed text prompts
            embeddings = self.predictor.embed_text_prompts(prompts)

            # Predict segmentation logits
            outputs = self.predictor.predict_sliding_window_return_logits(imgs.squeeze(dim=0), embeddings).to(self.device)
        
            masks = batch["masks"].to(self.device)
            loss = deep_supervision_loss(
                outputs.unsqueeze(0),
                masks,
                weights=self.ds_weights,
            )
            
            total_loss += float(loss.item())
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            dice = self.dice_metric(
                preds.unsqueeze(0),
                masks
            )
            dice = torch.nan_to_num(dice, nan=0.0)
            dice_vals = dice.detach().cpu().view(-1).tolist()

            for p, d in zip(prompt_labels, dice_vals):
                prompt_scores[p].append(d)

            batch_dice_mean = np.nanmean(dice_vals)
            total_dice += float(batch_dice_mean)

            per_prompt_val = { p: d for p, d in zip(prompt_labels, dice_vals)}
            per_prompt_str = ", ".join(
                f"{p}: {d:.4f}"
                for p, d in per_prompt_val.items()
            )

            self.logger.info(
                "Batch %d | Loss: %.6f | Dice(mean): %.6f | Per-prompt: %s",
                batch_idx,
                float(loss.item()),
                batch_dice_mean,
                per_prompt_str,
            )
        
        prompt_metrics = {
            p: {
                "mean_dice": sum(vals) / len(vals),
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
            }
            for p, vals in prompt_scores.items()
        }

        avg_loss = total_loss / len(test_loader)
        avg_dice = total_dice / len(test_loader)

        prompt_metrics['Average Loss'] = avg_loss
        prompt_metrics['Average Dice'] = avg_dice

        self.logger.info(
            "Evaluation completed | Avg Loss=%.6f | Avg Dice=%.6f",
            avg_loss,
            avg_dice,
        )

        self._save_metrics(prompt_metrics)

        return prompt_metrics

    def _save_metrics(self, metrics: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{timestamp}.json"

        path = self.out_root / filename

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        self.logger.info("Saved metrics to %s", path)
