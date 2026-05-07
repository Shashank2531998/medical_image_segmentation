from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import DatasetAdapter, EvaluationCase


class AeroPathAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "aeropath"

    def case_entries(self):
        return sorted([p for p in self.dataset_root.iterdir() if p.is_dir()])

    def build_case(self, case_dir: Path) -> EvaluationCase | None:
        if not case_dir.is_dir():
            return None
        case_id = case_dir.name
        image_path = case_dir / f"{case_id}_CT_HR.nii.gz"
        airway_path = case_dir / f"{case_id}_CT_HR_label_airways.nii.gz"
        lungs_path = case_dir / f"{case_id}_CT_HR_label_lungs.nii.gz"

        return EvaluationCase(
            case_id=case_id,
            image_path=image_path,
            target_paths={
                "trachea": airway_path,
                "lung": lungs_path,
            },
        )