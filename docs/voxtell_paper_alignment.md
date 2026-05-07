# VoxTell Paper Alignment Checklist

Source: arXiv 2511.11450 (VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation)

## Implemented / Aligned

- Multi-stage vision-language model structure:
  - The code uses `VoxTellModel` with multi-stage text-image fusion and prompt decoder path.
- Frozen text encoder with instruction-wrapped prompts:
  - `Qwen/Qwen3-Embedding-4B` via `TextPromptEncoder`.
- Prompt-conditioned 3D training/inference path:
  - Trainer now computes text embeddings and calls model with `(images, text_embeddings)`.
- Combined segmentation objective:
  - Dice + BCE is implemented (`combined_seg_loss_logits`).
- Deep supervision objective support:
  - `deep_supervision_loss` supports multi-scale output supervision with weighted stages.
- Optimizer/scheduler defaults moved toward paper settings:
  - SGD + polynomial LR schedule (configurable).
- Validation/testing via split-aware evaluator:
  - `src/cli/evaluate.py --split evaluation|validation|testing`.

## Partially Aligned / Remaining Gaps

- Full-scale training protocol in paper:
  - Paper uses 2000 epochs, 250 iterations/epoch, and large multi-GPU scaling. Current repo does not enforce that exact regimen.
- Prompt sampling strategy:
  - Paper trains with two positive + one negative prompt per image and rich synonym sampling. Current pipeline supports prompt-conditioned training but not the full positive/negative multi-prompt strategy yet.
- Dataset scale and vocabulary pipeline:
  - Paper uses 62K+ volumes and a harmonized 1K+ concept vocabulary pipeline. Current repository structure supports modular datasets but does not include the full published corpus/vocabulary generation process.
- Precomputed text embeddings for training:
  - Paper precomputes embeddings for efficiency. Current implementation computes embeddings on the fly.

## Recommended Next Steps Before Claiming Full Reproducibility

1. Implement explicit multi-prompt sampling per image (2 positive + 1 negative) with empty-mask supervision for negatives.
2. Add embedding cache/precomputation pipeline for training.
3. Implement explicit deep supervision weight defaults from nnU-Net (`[1, 1/2, 1/4, 1/8, 1/16]`) in config templates and trainer docs.
4. Add reproducibility configs matching paper training schedules (epochs/iterations, patch sampling policy, augmentation toggles).
5. Add benchmark protocol scripts for OOD dataset-only evaluation and report metrics (Dice and HIT5% where applicable).
