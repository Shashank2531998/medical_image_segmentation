from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import DatasetAdapter, EvaluationCase


class VEELATrainAdapter(DatasetAdapter):
    """Adapter for VEELA (Variability in Estimating Liver and tumor Area) dataset.
    
    Dataset structure:
    - dataset_root/
        - setXX_norm.nii      (normalized image)
        - setXX_gt.nii        (ground truth segmentation - liver)
        - setXX_mask.nii      (optional mask/ROI)
    """

    @property
    def name(self) -> str:
        return "veela"

    def case_entries(self) -> list[str]:
        """Extract unique case identifiers from available files."""
        cases = set()
        for file_path in self.dataset_root.iterdir():
            if file_path.is_file() and file_path.suffix == ".nii":
                # Extract case ID (e.g., "set01" from "set01_norm.nii")
                parts = file_path.stem.split("_")
                if len(parts) >= 1 and parts[0].startswith("set"):
                    cases.add(parts[0])
        return sorted(list(cases))

    def build_case(self, case_id: str) -> EvaluationCase | None:
        """Build evaluation case for a given case ID."""
        image_path = self.dataset_root / f"{case_id}_norm.nii"
        liver_mask_path = self.dataset_root / f"{case_id}_mask.nii"
        vessels_mask_path = self.dataset_root / f"{case_id}_gt.nii"

        # Both image and ground truth must exist
        if not image_path.exists() or not liver_mask_path.exists() or not vessels_mask_path.exists():
            return None

        return EvaluationCase(
            image_path=image_path,
            target_paths={
                "liver": liver_mask_path,
                "hepatic vessels": vessels_mask_path,
                "portal vessels": vessels_mask_path
            },
            target_labels={
                "hepatic vessels": 1,
                "portal vessels": 2,
            },
        )
