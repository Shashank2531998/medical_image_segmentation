from importlib import import_module

_LAZY_ATTRS = {
	"ContinualTask": ("src.continual.task_manager", "ContinualTask"),
	"ContinualTaskManager": ("src.continual.task_manager", "ContinualTaskManager"),
	"merge_dicts": ("src.continual.task_manager", "merge_dicts"),
	"apply_loralib_lora": ("src.continual.strategies.lora.common.utils", "apply_loralib_lora"),
	"save_lora_adapter": ("src.continual.strategies.lora.common.utils", "save_lora_adapter"),
	"load_lora_adapter": ("src.continual.strategies.lora.common.utils", "load_lora_adapter"),
}


def __getattr__(name: str):
	if name in _LAZY_ATTRS:
		module_name, attr_name = _LAZY_ATTRS[name]
		module = import_module(module_name)
		value = getattr(module, attr_name)
		globals()[name] = value
		return value
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_ATTRS)
