from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from src.data.dataset import TrainingSample


@dataclass(frozen=True)
class EvaluationCase:
    image_path: Path
    target_paths: dict[str, Path] | None = None
    metadata: dict[str, str] | None = None


class DatasetAdapter(ABC):
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def build_case(self, case_entry: Path) -> EvaluationCase | None:
        pass

    def case_entries(self) -> Sequence[Path]:
        return sorted(self.dataset_root.iterdir())

    def validate_case(self, case: EvaluationCase) -> bool:
        if not case.image_path.exists():
            return False
        return True

    def cases(self) -> Sequence[EvaluationCase]:
        cases: list[EvaluationCase] = []
        for case_entry in self.case_entries():
            case = self.build_case(case_entry)
            if case is None:
                continue
            if not self.validate_case(case):
                continue
            cases.append(case)
        return cases

    def build_training_samples(
        self,
        cases: Sequence[EvaluationCase] | None = None,
        max_cases: int | None = None,
        filter_img_list: Sequence[str] | None = None,
    ) -> list[TrainingSample]:
        cases = list(self.cases() if cases is None else cases)

        # normalize filter list
        filter_img_set = set(filter_img_list or [])

        # filter by image path if provided
        if filter_img_set:
            cases = [
                case
                for case in cases
                if str(case.image_path) in filter_img_set
            ]

        # limit number of cases if requested
        if max_cases is not None:
            cases = cases[:max_cases]

        samples: list[TrainingSample] = []

        for case in cases:

            prompts = []
            mask_paths = []
            for label_name, mask_path in (case.target_paths or {}).items():
                prompts.append(label_name)
                mask_paths.append(mask_path)

            samples.append(
                TrainingSample(
                    image_path=case.image_path,
                    mask_paths=mask_paths,
                    prompts=prompts,
                )
            )

        return samples

    def split_cases(self, val_fraction: float, seed: int) -> tuple[list[EvaluationCase], list[EvaluationCase]]:
        cases = list(self.cases())
        if not cases:
            return [], []

        import random

        shuffled = list(cases)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            return shuffled, []

        split_idx = int(round(len(shuffled) * (1.0 - val_fraction)))
        split_idx = max(1, min(split_idx, len(shuffled) - 1))
        return shuffled[:split_idx], shuffled[split_idx:]

    def __iter__(self) -> Iterator[EvaluationCase]:
        return iter(self.cases())

    def __len__(self) -> int:
        return len(self.cases())