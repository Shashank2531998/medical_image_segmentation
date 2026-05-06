from __future__ import annotations

from pathlib import Path

from src.adapters.base import DatasetAdapter, EvaluationCase


class AeroPathAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "aeropath"

    def cases(self):
        cases = []

        for case_dir in sorted(self.dataset_root.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name

            image_path = case_dir / f"{case_id}_CT_HR.nii.gz"
            airway_path = case_dir / f"{case_id}_CT_HR_label_airways.nii.gz"
            lungs_path = case_dir / f"{case_id}_CT_HR_label_lungs.nii.gz"

            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    image_path=image_path,
                    target_paths={
                        "trachea": airway_path,
                        "lung": lungs_path,
                    },
                )
            )

        return cases