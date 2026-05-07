#!/usr/bin/env python
"""
Pre-cache Hugging Face models to avoid download delays during inference.
Run this once before using predict.py
"""

import os
import sys

from src.utils.logging import get_logger


logger = get_logger(__name__)

# Set Hugging Face Hub environment variables
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"  # 30 seconds for metadata

logger.info("Caching Hugging Face models...")
logger.info("%s", "-" * 70)

try:
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "Qwen/Qwen3-Embedding-4B"
    
    logger.info("1. Downloading tokenizer for %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    logger.info("Tokenizer cached successfully")
    
    logger.info("2. Downloading model %s...", model_name)
    model = AutoModel.from_pretrained(model_name)
    logger.info("Model cached successfully")
    
    logger.info("%s", "=" * 70)
    logger.info("All models cached successfully!")
    logger.info("%s", "=" * 70)
    logger.info("You can now run predict.py without network delays:")
    logger.info("  python predict.py --image scan.nii.gz --prompts 'liver' 'kidney'")
    
except Exception as e:
    logger.exception("Error caching models: %s", e)
    sys.exit(1)
