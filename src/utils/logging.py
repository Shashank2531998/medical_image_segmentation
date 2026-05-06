from __future__ import annotations

import logging


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger
