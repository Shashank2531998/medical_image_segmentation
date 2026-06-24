from importlib import import_module

_LAZY_ATTRS = {
    "BaseContinualStrategy": ("src.continual.strategies.base", "BaseContinualStrategy"),
    "EvaluationSpec": ("src.continual.strategies.base", "EvaluationSpec"),
    "available_strategies": ("src.continual.strategies.registry", "available_strategies"),
    "create_strategy": ("src.continual.strategies.registry", "create_strategy"),
    "get_strategy_class": ("src.continual.strategies.registry", "get_strategy_class"),
    "register_strategy": ("src.continual.strategies.registry", "register_strategy"),
    "NaiveSequentialFinetuningStrategy": ("src.continual.strategies.fine_tuning.strategy", "NaiveSequentialFinetuningStrategy"),
    "run_naive_sequential_finetuning": ("src.continual.strategies.fine_tuning.strategy", "run_naive_sequential_finetuning"),
    "DynamicLoRAStrategy": ("src.continual.strategies.lora.dynamic.strategy", "DynamicLoRAStrategy"),
    "run_dynamic_lora_strategy": ("src.continual.strategies.lora.dynamic.strategy", "run_dynamic_lora_strategy"),
    "LoRAStrategy": ("src.continual.strategies.lora.task_specific.strategy", "LoRAStrategy"),
    "run_lora_strategy": ("src.continual.strategies.lora.task_specific.strategy", "run_lora_strategy"),
    "SharedLoRAStrategy": ("src.continual.strategies.lora.shared.strategy", "SharedLoRAStrategy"),
    "run_shared_lora_strategy": ("src.continual.strategies.lora.shared.strategy", "run_shared_lora_strategy"),
    "ZSCLStrategy": ("src.continual.strategies.zscl.strategy", "ZSCLStrategy"),
    "run_zscl_strategy": ("src.continual.strategies.zscl.strategy", "run_zscl_strategy"),
    "CPECLIPStrategy": ("src.continual.strategies.cpe_clip.strategy", "CPECLIPStrategy"),
    "run_cpe_clip_strategy": ("src.continual.strategies.cpe_clip.strategy", "run_cpe_clip_strategy"),
}


def load_builtin_strategies() -> None:
    import_module("src.continual.strategies.fine_tuning.strategy")
    import_module("src.continual.strategies.lora.dynamic.strategy")
    import_module("src.continual.strategies.lora.task_specific.strategy")
    import_module("src.continual.strategies.lora.shared.strategy")
    import_module("src.continual.strategies.zscl.strategy")
    import_module("src.continual.strategies.cpe_clip.strategy")


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_ATTRS) + ["load_builtin_strategies"]