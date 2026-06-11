from __future__ import annotations

import pydoc
from pathlib import Path
from typing import Tuple

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch._dynamo import OptimizedModule

from src.continual.strategies.lora.loading import configure_loaded_lora_model
from src.model.voxtell_model import VoxTellModel
from src.utils.model_helpers import log_model_params

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_arch_kwargs(arch_kwargs: dict, required_import_keys: list[str]) -> dict:
    resolved = dict(arch_kwargs)
    for key in required_import_keys:
        if resolved.get(key) is not None:
            resolved[key] = pydoc.locate(resolved[key])
    return resolved


def load_voxtell_model(
    model_dir: str | Path,
    deep_supervision: bool = False,
    model_overrides: dict | None = None,
    reinit_weights: bool = False,
    checkpoint_path: str | Path = None,
    lora_cfg: dict | None = None,
    lora_adapter_path: str | Path | None = None,
) -> tuple[VoxTellModel, Tuple[int, ...]]:
    
    logger.info("Loading VoxTell model from %s", checkpoint_path if checkpoint_path else model_dir)

    model_dir = Path(model_dir)
    model_overrides = model_overrides or {}
    plans = load_json(str(model_dir / "plans.json"))
    configuration = plans["configurations"]["3d_fullres"]

    arch_kwargs = _resolve_arch_kwargs(
        configuration["architecture"]["arch_kwargs"],
        configuration["architecture"]["_kw_requires_import"],
    )

    patch_size = tuple(configuration["patch_size"])

    network = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=model_overrides.get("decoder_layer", 4),
        text_embedding_dim=model_overrides.get("text_embedding_dim", 2560),
        num_maskformer_stages=model_overrides.get("num_maskformer_stages", 5),
        num_heads=model_overrides.get("num_heads", 32),
        query_dim=model_overrides.get("query_dim", 2048),
        project_to_decoder_hidden_dim=model_overrides.get("project_to_decoder_hidden_dim", 2048),
        deep_supervision=deep_supervision,
    )

    checkpoint = torch.load(
        checkpoint_path if checkpoint_path else str(model_dir / "fold_0" / "checkpoint_final.pth"),
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    if isinstance(network, OptimizedModule):
        target_network = network._orig_mod
    else:
        target_network = network

    target_network.load_state_dict(checkpoint["network_weights"])

    if lora_cfg is not None:
        target_network = configure_loaded_lora_model(
            target_network,
            lora_cfg=lora_cfg,
            lora_adapter_path=lora_adapter_path,
            mark_trainable=lora_adapter_path is None,
        )

    if reinit_weights:
        network.apply(VoxTellModel.initialize)
    
    return network, patch_size
