from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from src.data.dataset import TrainingSample


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
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
        for _, target_path in (case.target_paths or {}).items():
            if not target_path.exists():
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

    def prompt_aliases(self) -> dict[str, list[str]]:
        return {}

    def prompt_variants(self, label_name: str) -> list[str]:
        normalized = label_name.replace("_", " ").strip().lower()
        aliases = self.prompt_aliases()
        return aliases.get(normalized, [normalized])

    def default_prompts(self) -> list[str]:
        prompts: list[str] = []
        for case in self.cases():
            for label_name in (case.target_paths or {}).keys():
                for prompt in self.prompt_variants(label_name):
                    if prompt not in prompts:
                        prompts.append(prompt)
        return prompts

    def build_training_samples(self, cases: Sequence[EvaluationCase] | None = None) -> list[TrainingSample]:
        cases = list(self.cases() if cases is None else cases)
        all_label_names = sorted({label_name for case in cases for label_name in (case.target_paths or {}).keys()})

        samples: list[TrainingSample] = []
        for case in cases:
            for label_name, mask_path in (case.target_paths or {}).items():
                positives = self.prompt_variants(label_name)
                negatives: list[str] = []
                for other_label in all_label_names:
                    if other_label == label_name:
                        continue
                    negatives.extend(self.prompt_variants(other_label))

                samples.append(
                    TrainingSample(
                        image_path=case.image_path,
                        mask_path=mask_path,
                        prompts=positives,
                        negatives=negatives,
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