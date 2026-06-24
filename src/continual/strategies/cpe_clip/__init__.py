from importlib import import_module

_LAZY_ATTRS = {
    "CPECLIPPromptedVoxTell": ("src.continual.strategies.cpe_clip.model", "CPECLIPPromptedVoxTell"),
    "configure_loaded_cpe_clip_model": ("src.continual.strategies.cpe_clip.model", "configure_loaded_cpe_clip_model"),
    "CPECLIPStrategy": ("src.continual.strategies.cpe_clip.strategy", "CPECLIPStrategy"),
    "run_cpe_clip_strategy": ("src.continual.strategies.cpe_clip.strategy", "run_cpe_clip_strategy"),
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