from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import DatasetAdapter, EvaluationCase


class VEELATrainAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "veela"

    def case_entries(self):
        # VEELA train set is flat files like set01_norm.nii, set01_gt.nii, set01_mask.nii
        return sorted(self.dataset_root.glob("*_norm.nii*"))

    def build_case(self, norm_file: Path) -> EvaluationCase | None:
        name = norm_file.name
        if "_norm" not in name:
            return None

        case_id = name.split("_norm")[0]
        gt_name = name.replace("_norm", "_gt")
        mask_name = name.replace("_norm", "_mask")

        gt_path = norm_file.with_name(gt_name)
        roi_mask_path = norm_file.with_name(mask_name)

        return EvaluationCase(
            case_id=case_id,
            image_path=norm_file,
            target_paths={
                "liver": gt_path,
            },
            metadata={
                "roi_mask": str(roi_mask_path),
            },
        )
