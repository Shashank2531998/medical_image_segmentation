from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = __name__,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler_exists = any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == log_file
            for handler in logger.handlers
        )
        if not file_handler_exists:
            fh = logging.FileHandler(str(log_file))
            fh.setLevel(level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
