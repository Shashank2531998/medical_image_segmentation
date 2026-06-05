from __future__ import annotations

from pathlib import Path
from typing import Any

from src.continual.task_manager import ContinualTask, ContinualTaskManager

from ..base import BaseContinualStrategy, EvaluationSpec
from ..registry import register_strategy
from .dynamic_lora import DynamicExpertLinear, rank_dynamic_lora_modules, save_dynamic_lora_adapter
from .loading import configure_loaded_lora_model


@register_strategy
class DynamicLoRAStrategy(BaseContinualStrategy):
    strategy_name = "dynamic_lora"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_growth_step = 0
        self._batch_growth_cooldown = 0
        self._batch_loss_ema = None
        self._batch_loss_alpha = 0.9

    @classmethod
    def configure_loaded_model(
        cls,
        model,
        *,
        lora_cfg: dict[str, Any],
        lora_adapter_path: str | Path | None = None,
        mark_all_trainable_after_load: bool = False,
    ):
        dynamic_lora_cfg = {
            **lora_cfg,
            "dynamic_experts": True,
        }
        return configure_loaded_lora_model(
            model,
            lora_cfg=dynamic_lora_cfg,
            lora_adapter_path=lora_adapter_path,
            mark_all_trainable_after_load=mark_all_trainable_after_load,
        )

    def configure_engine(self) -> None:
        engine = self.require_engine()

        self.logger.info("Applying dynamic LoRA adaptation to base model...")
        engine.model = self.__class__.configure_loaded_model(engine.model, lora_cfg=self.task_manager.lora_cfg)
        self.logger.info("Dynamic LoRA adaptation complete. Model ready for continual learning.")

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
        
        # ---- EMA LOSS (FIXED) ----
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
        loss_threshold = schedule.get("loss_threshold", growth_cfg.get("loss_threshold", None))

        entropy_modules = self._entropy_triggered_modules(engine.model, growth_cfg, entropy_threshold)
        loss_trigger = (
            loss_threshold is not None
            and self._batch_loss_ema > float(loss_threshold)
        )
        
        if not entropy_modules and not loss_trigger:
            self.logger.info(
                "Batch growth skipped at task=%s epoch=%d batch=%d | loss=%.6f | no entropy/loss trigger",
                task.name,
                epoch,
                batch_idx,
                batch_loss,
            )
            return

        grown = self._grow_selected_modules(
            engine.model,
            add_experts=add_experts,
            locations=locations,
            max_modules=max_modules,
            entropy_threshold=entropy_threshold,
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
        model,
        growth_cfg: dict[str, Any],
        entropy_threshold: float,
    ) -> list[tuple[str, DynamicExpertLinear]]:
        import math

        locations = growth_cfg.get("locations", None)
        ranked = rank_dynamic_lora_modules(model, locations=locations)

        triggered = []

        for module_name, module in ranked:
            raw_entropy = module.normalized_routing_entropy()

            if raw_entropy is None:
                continue

            num_experts = max(1, module.num_experts)

            # unified normalization (no special-case branching anymore)
            entropy_norm = raw_entropy / math.log(num_experts)

            if entropy_norm >= entropy_threshold:
                triggered.append((module_name, module))

        return triggered

    def _grow_selected_modules(
        self,
        model,
        *,
        add_experts: int,
        locations: list[str] | tuple[str, ...] | None,
        max_modules: int,
        entropy_threshold: float,
    ) -> int:
        
        ranked = rank_dynamic_lora_modules(model, locations=locations)
        if not ranked:
            return 0

        entropy_selected = self._entropy_triggered_modules(
            model, {"locations": locations}, entropy_threshold
        )

        selected = entropy_selected if entropy_selected else ranked
        selected = selected[: max(1, max_modules)]

        grown = 0
        for module_name, module in selected:
            before = module.num_experts
            module.grow_to(before + add_experts)
            grown += module.num_experts - before
            self.logger.info(
                "Growing dynamic LoRA module %s | entropy=%.4f | experts %d -> %d",
                module_name,
                float(module.normalized_routing_entropy() or 0.0),
                before,
                module.num_experts,
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