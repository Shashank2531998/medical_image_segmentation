from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    image_path: Path
    target_paths: dict[str, Path] | None = None


class DatasetAdapter(ABC):
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def cases(self) -> Sequence[EvaluationCase]:
        pass

    def __iter__(self) -> Iterator[EvaluationCase]:
        return iter(self.cases())

    def __len__(self) -> int:
        return len(self.cases())