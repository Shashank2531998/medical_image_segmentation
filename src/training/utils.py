from pathlib import Path
import random

import numpy as np
import torch

from src.utils.checkpoint import save_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(
    seed: int,
    deterministic: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(
        "Seed initialized | seed=%d deterministic=%s",
        seed,
        deterministic,
    )