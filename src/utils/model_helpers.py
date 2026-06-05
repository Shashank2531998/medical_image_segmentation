from src.utils.logging import get_logger

logger = get_logger(__name__)


def log_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters - Total: %d | Trainable: %d | Frozen: %d",
        total_params,
        trainable_params,
        total_params - trainable_params,
    )
