from __future__ import annotations

from typing import Any

from src.continual.strategies.base import BaseContinualStrategy


StrategyType = type[BaseContinualStrategy]
_STRATEGY_REGISTRY: dict[str, StrategyType] = {}


def register_strategy(cls: StrategyType) -> StrategyType:
    names = [cls.strategy_name]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"Strategy class {cls.__name__} must define strategy_name or aliases")

    for name in names:
        _STRATEGY_REGISTRY[name] = cls
    return cls


def get_strategy_class(strategy_name: str) -> StrategyType:
    try:
        return _STRATEGY_REGISTRY[strategy_name]
    except KeyError as exc:
        available = ", ".join(sorted(_STRATEGY_REGISTRY)) or "<none>"
        raise ValueError(f"Unsupported continual strategy '{strategy_name}'. Available: {available}") from exc


def create_strategy(strategy_name: str, **kwargs: Any) -> BaseContinualStrategy:
    strategy_class = get_strategy_class(strategy_name)
    return strategy_class(**kwargs)


def available_strategies() -> list[str]:
    return sorted(_STRATEGY_REGISTRY)
