#!/usr/bin/env python
"""
Pre-cache Hugging Face models to avoid download delays during inference.
Run this once before using predict.py
"""

import os
import sys

# Set Hugging Face Hub environment variables
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"  # 30 seconds for metadata

print("Caching Hugging Face models...")
print("-" * 70)

try:
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "Qwen/Qwen3-Embedding-4B"
    
    print(f"\n1. Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    print("   ✓ Tokenizer cached successfully")
    
    print(f"\n2. Downloading model {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    print("   ✓ Model cached successfully")
    
    print("\n" + "=" * 70)
    print("✓ All models cached successfully!")
    print("=" * 70)
    print("\nYou can now run predict.py without network delays:")
    print("  python predict.py --image scan.nii.gz --prompts 'liver' 'kidney'")
    
except Exception as e:
    print(f"\n❌ Error caching models: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
