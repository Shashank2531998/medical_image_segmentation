from pathlib import Path

from src.data.adapters.aeropath import AeroPathAdapter
from src.data.adapters.veela import VEELATrainAdapter
from src.data.adapters.fedbca import FedBCaCenter2Adapter
from src.data.adapters.medseg_esophageal import MedSegEsophagealAdapter
from src.data.adapters.base import DatasetAdapter, EvaluationCase


DATASET_ADAPTERS: dict[str, type[DatasetAdapter]] = {
	"aeropath": AeroPathAdapter,
	"veela": VEELATrainAdapter,
	"fedbca_center2": FedBCaCenter2Adapter,
	"medseg_esophageal": MedSegEsophagealAdapter
}


def get_dataset_adapter(dataset_name: str, dataset_root: str | Path) -> DatasetAdapter:
	normalized = dataset_name.lower().strip()
	if normalized not in DATASET_ADAPTERS:
		raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_ADAPTERS.keys())}")
	return DATASET_ADAPTERS[normalized](dataset_root)
