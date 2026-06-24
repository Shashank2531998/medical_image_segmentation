from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.engine.model_engine import VoxTellEngine
from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.continual.strategies.base import BaseContinualStrategy, EvaluationSpec
from src.continual.strategies.registry import register_strategy
from src.continual.strategies.lora.dynamic.layers import DynamicExpertLinear, iter_dynamic_lora_modules, rank_dynamic_lora_modules
from src.continual.strategies.lora.dynamic.utils import save_dynamic_lora_adapter
from src.continual.strategies.lora.loading import configure_loaded_lora_model
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MahalanobisNoveltyTracker:
    def __init__(self, *, module_name = "<unknown>", min_samples: int = 8, threshold_scale: float = 2.0, eps: float = 1e-6) -> None:
        self.min_samples = max(1, int(min_samples))
        self.threshold_scale = float(threshold_scale)
        self.eps = float(eps)
        self.count = 0
        self.mean = None
        self.cov = None
        self.samples = None
        self.distances = []
        self.distance_history = []
        self.threshold = None
        self.module_name = module_name

    def _update_running_stats(self, features: torch.Tensor) -> None:
        features = features.detach().float()
        if features.dim() > 2:
            features = features.reshape(-1, features.shape[-1])

        if features.shape[0] == 0:
            return

        if self.samples is None:
            self.samples = features.clone()
        else:
            self.samples = torch.cat([self.samples, features], dim=0)

        self.count = int(self.samples.shape[0])
        self.mean = self.samples.mean(dim=0)
        centered = self.samples - self.mean
        self.cov = centered.t() @ centered / max(1, self.count - 1)

    def observe(self, features: torch.Tensor) -> torch.Tensor:
        # logger.info(f"Observing mahalanobis distance for {self.module_name}")
        self._update_running_stats(features)
        if self.mean is None or self.cov is None:
            return torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)

        features = features.detach().float()
        if features.dim() > 2:
            features = features.reshape(-1, features.shape[-1])

        centered = features - self.mean.to(features.device)
        cov = self.cov.to(features.device) + self.eps * torch.eye(self.cov.shape[0], device=features.device)
        inv_cov = torch.linalg.pinv(cov)
        distances = torch.einsum('...i,ij,...j->...', centered, inv_cov, centered).clamp_min(0.0)

        flat_distances = distances.detach().reshape(-1).cpu().float()
        self.distances = flat_distances.tolist()
        self.distance_history.extend(flat_distances.tolist())

        self.threshold = self._estimate_threshold(torch.tensor(self.distance_history, dtype=torch.float32))
        # logger.info("distances=%s | threshold=%s | history_len=%d", self.distances, self.threshold, len(self.distance_history))
        return distances

    def _estimate_threshold(self, distances: torch.Tensor) -> float | None:
        distances = distances.detach().reshape(-1).float()
        if distances.numel() < self.min_samples:
            return None

        mean_distance = float(distances.mean().item())
        std_distance = float(distances.std(unbiased=False).item())
        return mean_distance + self.threshold_scale * std_distance


class DynamicLoRAStrategy(BaseContinualStrategy):
    strategy_name = "dynamic_lora"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_growth_step = 0
        self._batch_growth_cooldown = 0
        self._batch_loss_ema = None
        self._batch_loss_alpha = 0.9
        self.lora_adapter_path = None
        self.novelty_tracker = MahalanobisNoveltyTracker(module_name=self.strategy_name)

    def configure_loaded_model(
        cls,
        model,
        *,
        lora_cfg: dict[str, Any],
        lora_adapter_path: str | Path | None = None,
        mark_trainable: bool = True,
    ):
        dynamic_lora_cfg = {
            **lora_cfg,
            "dynamic_experts": True,
        }
        return configure_loaded_lora_model(
            model,
            lora_cfg=dynamic_lora_cfg,
            lora_adapter_path=lora_adapter_path,
            mark_trainable=mark_trainable,
        )
    def build_engine(self)-> VoxTellEngine:
        engine = super().build_engine()
        
        self.logger.info("Applying dynamic LoRA adaptation to base model...")
        engine.model = self.configure_loaded_model(
            engine.model, lora_cfg=self.task_manager.lora_cfg,
            lora_adapter_path=self.lora_adapter_path
        )
        self.logger.info("Dynamic LoRA adaptation complete. Model ready for continual learning.")

        growth_cfg = dict(self.task_manager.lora_cfg.get("growth", {}))
        self.logger.info(
            "Dynamic LoRA novelty growth config | novelty_min_samples=%d | novelty_threshold=%.3f | entropy_threshold=%.3f",
            int(growth_cfg.get("novelty_min_samples", 8)),
            float(growth_cfg.get("novelty_threshold", 1.0)),
            float(growth_cfg.get("entropy_threshold", 0.85)),
        )
        for module_name, module in iter_dynamic_lora_modules(engine.model):
            module.novelty_tracker = MahalanobisNoveltyTracker(
                module_name=module_name,
                min_samples=int(growth_cfg.get("novelty_min_samples", 8)),
            )

        return engine

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
        task_metrics: dict[str, Any] | None = None,
    ) -> None:
        engine = self.require_engine()
        dynamic_lora_cfg = {
            **self.task_manager.lora_cfg,
            "dynamic_experts": True,
        }
        lora_bias = str(dynamic_lora_cfg.get("bias", "none"))

        save_dynamic_lora_adapter(engine.model, task_dir / "lora_adapter.pt", bias=lora_bias)

        self.lora_adapter_path = task_dir / "lora_adapter.pt"
        self.logger.info("Saved dynamic LoRA adapter for task %s", task.name)

    def on_train_batch_end(
        self,
        task: ContinualTask,
        batch_idx: int,
        epoch: int,
        batch_loss: float,
    ) -> None:
        engine = self.require_engine()
        growth_cfg = dict(self.task_manager.lora_cfg.get("growth", {}))
        schedule = dict(growth_cfg.get("schedule", {}))

        if str(schedule.get("mode", "task")) != "batch":
            return
        
        original_batch_loss_ema = self._batch_loss_ema

        # EMA loss is kept only for logging compatibility; novelty now drives growth.
        if self._batch_loss_ema is None:
            self._batch_loss_ema = float(batch_loss)
        else:
            self._batch_loss_ema = (
                self._batch_loss_alpha * self._batch_loss_ema
                + (1 - self._batch_loss_alpha) * float(batch_loss)
            )

        every_n_batches = int(schedule.get("every_n_batches", growth_cfg.get("every_n_batches", 10)))
        if every_n_batches <= 0:
            return

        self._batch_growth_step += 1
        if self._batch_growth_cooldown > 0:
            self._batch_growth_cooldown -= 1
            return

        if self._batch_growth_step % every_n_batches != 0:
            return

        entropy_threshold = float(growth_cfg.get("entropy_threshold", 0.85))
        add_experts = int(growth_cfg.get("add_experts", 1))
        max_modules = int(growth_cfg.get("max_modules_per_step", growth_cfg.get("max_modules_per_task", 1)))
        locations = growth_cfg.get("locations", None)

        ranked = rank_dynamic_lora_modules(engine.model, locations=locations)
        entropy_modules = self._entropy_triggered_modules(ranked, entropy_threshold)
        novelty_modules = self._novelty_triggered_modules(ranked, growth_cfg)

        self.logger.info(
            "Batch growth decision at task=%s epoch=%d batch=%d | entropy_triggered=%d | novelty_triggered=%d | every_n_batches=%d",
            task.name,
            epoch,
            batch_idx,
            len(entropy_modules),
            len(novelty_modules),
            every_n_batches,
        )

        if not entropy_modules and not novelty_modules:
            self.logger.info(
                "Batch growth skipped at task=%s epoch=%d batch=%d | loss=%.6f | no entropy/novelty trigger",
                task.name,
                epoch,
                batch_idx,
                batch_loss,
            )
            return
        
        selected = list(dict.fromkeys(entropy_modules + novelty_modules))
        if not selected:
            selected = ranked

        growth_reason = "entropy" if entropy_modules else "novelty" if novelty_modules else "NA"
        selected = selected[: max(1, max_modules)]
        
        self.logger.info(f"Growth Triggered | Reason: {growth_reason}")

        grown = self._grow_selected_modules(
            add_experts=add_experts,
            selected=selected
        )
        if grown > 0:
            cooldown_batches = int(schedule.get("cooldown_batches", growth_cfg.get("cooldown_batches", every_n_batches)))
            self._batch_growth_cooldown = max(0, cooldown_batches)
            self.logger.info(
                "Batch growth triggered at task=%s epoch=%d batch=%d | loss=%.6f | added=%d experts",
                task.name,
                epoch,
                batch_idx,
                batch_loss,
                grown,
            )
        else:
            self.logger.info(
                "Batch growth eligible but no module grew at task=%s epoch=%d batch=%d | loss=%.6f",
                task.name,
                epoch,
                batch_idx,
                batch_loss,
            )

    def _entropy_triggered_modules(
        self,
        ranked_modules,
        entropy_threshold: float,
    ) -> list[tuple[str, DynamicExpertLinear]]:
        import math

        triggered = []

        for module_name, module in ranked_modules:
            raw_entropy = module.normalized_routing_entropy()

            usage = getattr(module, "expert_usage_stats", lambda: None)()

            if module.num_experts > 1:
                self.logger.info(
                    "[ROUTER] %s | experts=%d | raw_entropy=%.4f | usage=%s",
                    module_name,
                    module.num_experts,
                    raw_entropy or 0.0,
                    usage,
                )

            if raw_entropy is None:
                continue

            num_experts = max(1, module.num_experts)

            # unified normalization (no special-case branching anymore)
            if num_experts > 1:
                entropy_norm = raw_entropy / math.log(num_experts)
            else:
                entropy_norm = 0.0

            if entropy_norm >= entropy_threshold:
                triggered.append((module_name, module))

        return triggered

    def _novelty_triggered_modules(
        self,
        ranked_modules,
        growth_cfg: dict[str, Any],
    ) -> list[tuple[str, DynamicExpertLinear]]:
        novelty_threshold = float(growth_cfg.get("novelty_threshold", 1.0))

        triggered = []
        for module_name, module in ranked_modules:
            tracker = getattr(module, "novelty_tracker", None)
            if tracker is None:
                continue

            max_distance = max((float(v) for v in tracker.distances), default=0.0)
            threshold = tracker.threshold if tracker.threshold is not None else 0.0
            is_triggered = threshold > 0.0 and max_distance >= threshold * novelty_threshold

            if is_triggered:
                self.logger.info(
                    "Novelty Triggered | module=%s | samples=%d | max_distance=%.4f | threshold=%.4f | trigger=%s",
                    module_name,
                    int(getattr(tracker, "count", 0)),
                    max_distance,
                    threshold,
                    is_triggered,
                )
                triggered.append((module_name, module))

        return triggered

    def _grow_selected_modules(
        self,
        add_experts: int,
        selected,
    ) -> int:

        grown = 0
        for module_name, module in selected:
            before = module.num_experts
            old_params = sum(p.numel() for p in module.parameters())
            parent_index = None
            if module.last_router_probs is not None and module.last_router_probs.numel() > 0:
                parent_index = int(torch.argmax(module.last_router_probs).item())
            module.grow_to(before + add_experts, clone_from=parent_index)
            grown += module.num_experts - before
            new_params = sum(p.numel() for p in module.parameters())
            novelty_peak = float(max((float(v) for v in getattr(getattr(module, "novelty_tracker", None), "distances", []) if v is not None), default=0.0))
            novelty_threshold = float(getattr(getattr(module, "novelty_tracker", None), "threshold", 0.0) or 0.0)
            self.logger.info(
                "Growing dynamic LoRA module %s | parent_idx=%s | entropy=%.4f | novelty_peak=%.4f | novelty_threshold=%.4f | experts %d -> %d | params added=%d",
                module_name,
                parent_index,
                float(module.normalized_routing_entropy() or 0.0),
                novelty_peak,
                novelty_threshold,
                before,
                module.num_experts,
                new_params - old_params,
            )

        return grown

    @classmethod
    def build_evaluation_spec(
        cls,
        *,
        task_manager: ContinualTaskManager,
        task: ContinualTask,
        task_dir: Path,
        trained_model_cfg: dict[str, Any],
        checkpoint_name: str,
    ) -> EvaluationSpec:
        adapter_path = task_dir / "lora_adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Expected dynamic LoRA adapter for task {task.name} at {adapter_path}")

        return EvaluationSpec(
            model_cfg={
                **trained_model_cfg,
                "lora_cfg": {
                    **dict(task_manager.lora_cfg),
                    "dynamic_experts": True,
                },
                "lora_adapter_path": str(adapter_path),
            },
        )


def run_dynamic_lora_strategy(
    *,
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger,
) -> None:
    DynamicLoRAStrategy(
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    ).run()