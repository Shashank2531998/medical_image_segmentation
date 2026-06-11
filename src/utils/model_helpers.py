from src.utils.logging import get_logger

logger = get_logger(__name__)


def log_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Mode: {"train" if model.training else "eval"}")
    logger.info(
        "Model parameters - Total: %d | Trainable: %d | Frozen: %d",
        total_params,
        trainable_params,
        total_params - trainable_params,
    )


def set_adapters_enabled(model, enabled):
    for module in model.modules():
        if hasattr(module, "adapter_enabled"):
            module.adapter_enabled = enabled
