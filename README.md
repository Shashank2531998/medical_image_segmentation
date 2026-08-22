# VoxTell

VoxTell is a text-prompted 3D medical image segmentation system. It combines a
residual encoder and multi-scale decoder with a transformer prompt decoder, so
one model can produce a binary segmentation for one or more free-text anatomy
prompts. The repository also contains supervised fine-tuning, dataset adapters,
continual-learning strategies, evaluation, and inference workflows.

## Status and scope

The repository contains the following implemented paths:

- 3D NIfTI (`.nii` and `.nii.gz`) training, evaluation, and inference.
- Prompt-conditioned multi-label segmentation. Each prompt is evaluated as a
    separate binary mask.
- CUDA and CPU execution. CUDA is the practical choice for the supplied model.
- Standard fine-tuning and train-then-test runs.
- Continual learning with naive fine-tuning, task-specific LoRA, shared LoRA,
    ZSCL, and CPE-CLIP strategies.
- Dice-based evaluation at task and prompt level.

Dataset files and pretrained weights are not included in the repository. Paths
in the example YAML files may need to be changed for the local machine.

## Repository layout

```text
configs/                 YAML configurations for training, evaluation, and CL
models/                  model plans and downloaded checkpoints
scripts/                 data conversion, setup, analysis, and visualisation tools
src/cli/                 command-line entry points
src/data/                dataset, preprocessing, datamodule, and adapters
src/model/               VoxTell model construction and architecture
src/text/                text prompt encoder
src/inference/           sliding-window prediction and NIfTI output
src/training/            trainer, losses, optimiser, and scheduling
src/evaluation/          test-set evaluation and metric output
src/continual/           task management, strategies, and CL evaluation
src/engine/              model/text forward-pass orchestration
src/utils/               configuration, checkpoints, logging, and helpers
experiments/             generated run directories (ignored by git)
outputs/                 cluster job logs and generated output (ignored by git)
```

## Installation

Use Python with a PyTorch build compatible with the target CUDA installation.
The repository does not pin Python or CUDA versions, so install PyTorch from
the official selector when the environment requires a specific CUDA version.

```bash
git clone <repository-url>
cd VoxTell
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements cover PyTorch, MONAI, nnU-Net v2, NIfTI I/O, transformers,
LoRA support, and the tensor utilities used by the model. Optional utilities
need extra packages:

```bash
python -m pip install napari[all]       # --backend napari
python -m pip install SimpleITK         # --backend sitk
python -m pip install h5py              # SKM-TEA HDF5 conversion
python -m pip install redivis           # scripts/redvis_data_download.py
```

Run commands from the repository root. `PYTHONPATH=.` is required by the SLURM
scripts and is useful when invoking scripts from another directory.

## Pretrained model and text encoder

The loader expects a model directory containing `plans.json` and a checkpoint
at `fold_0/checkpoint_final.pth` unless an explicit checkpoint is supplied.
The included configurations use `models/voxtell_v1.1`.

Download a VoxTell checkpoint from the configured Hugging Face repository:

```bash
PYTHONPATH=. python scripts/download_checkpoint.py \
    --model-name voxtell_v1.1 \
    --download-dir ./models
```

Inference and the CLI explicitly enable Hugging Face offline mode. Cache the
text encoder before running them while network access is available:

```bash
PYTHONPATH=. python scripts/setup_models.py
```

The text encoder is `Qwen/Qwen3-Embedding-4B`. Its cached output is used to
condition the segmentation decoder. If the tokenizer or encoder is not already
available in the local Hugging Face cache, model loading will fail in offline
mode.

## Data formats

All adapters return an image, one or more prompt names, and corresponding mask
paths. Images are read and reoriented with nnU-Net's NIfTI reader, cropped to
the non-zero region, and z-score normalized. Training samples use
`patch_size: [192, 192, 192]` by default; 85% of training patches are centered
on foreground and 15% are sampled randomly.

Set `dataset.name` and `dataset.root` in a YAML file. The supported adapter
names and required layouts are:

| Adapter name | Expected layout | Prompt labels |
| --- | --- | --- |
| `aeropath` | `<root>/<case>/<case>_CT_HR.nii.gz`, with `<case>_CT_HR_label_lungs.nii.gz` | `lung` |
| `fedbca_center2` | `<root>/T2WI/<id>.nii` or `<id>.nii.gz` and `<root>/Annotation/<id>.nii` or `<id>.nii.gz`; a nested `Center2/` is also accepted | `carcinoma` |
| `medseg_esophageal` | `<root>/img_<id>.nii` or `.nii.gz` and `<root>/msk_<id>.nii` or `.nii.gz` | `esophageal cancer` |
| `skm_tea` | `<root>/images/<id>.nii.gz` and `<root>/annotations/<id>.nii.gz` | six SKM-TEA structures |
| `veela` | `<root>/setXX_norm.nii`, `<root>/setXX_mask.nii`, and `<root>/setXX_gt.nii` | `liver`, `hepatic vessels`, `portal vessels` |

SKM-TEA label values are: patellar cartilage `1`, femoral cartilage `2`,
medial tibial cartilage `3`, lateral tibial cartilage `4`, medial meniscus `5`,
and lateral meniscus `6`. The aliases
`skm_tea_medial_tibial`, `skm_tea_lateral_tibial`, `skm_tea_patellar`,
`skm_tea_femoral`, `skm_tea_medial_meniscus`, and
`skm_tea_lateral_meniscus` restrict a task to one structure.

The datamodule shuffles cases with `dataset.seed`, then allocates
`train_fraction`, `val_fraction`, and the remaining cases to test. Optional
`train_max_cases`, `val_max_cases`, and `test_max_cases` are useful for smoke
tests.

## Configuration

Configurations are ordinary YAML files with these sections:

```yaml
dataset:
    name: aeropath
    root: ./data/AeroPath
    batch_size: 1
    patch_size: [192, 192, 192]
    train_fraction: 0.8
    val_fraction: 0.2
    seed: 42

model:
    dir: ./models/voxtell_v1.1
    device: cuda
    deep_supervision: false
    reinit_weights: false

training:
    epochs: 10
    output_dir: ./experiments/fine_tuning/aeropath
    optimizer:
        lr: 0.0001
        weight_decay: 0.00003
        momentum: 0.99
        poly_power: 1.0
        nesterov: true
```

Useful training options include `training.early_stopping`,
`training.resume_checkpoint_interval`, and `training.debug_save_artifacts`.
The latter writes tensors under `debug_artifacts/` for the visualisation
script. See [`configs/dry_run.yaml`](configs/dry_run.yaml) for a small
continual-learning configuration and [`configs/train.yaml`](configs/train.yaml)
for a standard fine-tuning configuration.

## Training and evaluation

Train only:

```bash
PYTHONPATH=. python src/cli/main.py --train_only --config configs/train.yaml
```

Train and evaluate the resulting `best_model.pt`:

```bash
PYTHONPATH=. python src/cli/main.py --config configs/train.yaml
```

Evaluate using an existing model/checkpoint configuration:

```bash
PYTHONPATH=. python src/cli/main.py --test_only --config configs/evaluation.yaml
PYTHONPATH=. python src/cli/main.py --test_only --disable_adapters --config configs/evaluation.yaml
```

`--disable_adapters` is useful when evaluating a checkpoint that should be
run without loaded LoRA adapters. Evaluation writes per-prompt metrics plus
`Average Loss` and `Average Dice` as a timestamped JSON file in
`evaluation.output_dir`.

## Inference

The inference CLI accepts one NIfTI input and one or more prompts. Predictions
are written as one NIfTI file per prompt under a new directory in
`experiments/inference/`.

```bash
PYTHONPATH=. python src/cli/infer.py \
    --input ./data/example/image.nii.gz \
    --model ./models/voxtell_v1.1 \
    --prompts lung trachea \
    --device cuda
```

Use `--device cpu` when CUDA is unavailable. To compute Dice at the same time,
pass one ground-truth mask for all prompts or one mask per prompt:

```bash
PYTHONPATH=. python src/cli/infer.py \
    --input ./data/example/image.nii.gz \
    --model ./models/voxtell_v1.1 \
    --prompts lung \
    --mask ./data/example/lung.nii.gz
```

The predictor uses non-zero cropping, z-score normalization, 50% overlapping
sliding-window inference, Gaussian blending, sigmoid threshold `0.5`, and
reorientation back to the input image space.

## Continual learning

Continual configurations use the base `dataset`, `model`, and `training`
sections plus:

```yaml
continual:
    strategy: shared_lora
    output_dir: ./experiments/continual/shared_lora
    from_scratch: false
    tasks:
        - name: skm_tea_lateral_tibial
            dataset:
                name: skm_tea_lateral_tibial
                root: ./data/skm_tea
        - name: skm_tea_medial_tibial
            dataset:
                name: skm_tea_medial_tibial
                root: ./data/skm_tea
    retention_tasks:
        - name: aeropath
            dataset:
                name: aeropath
                root: ./data/AeroPath
```

Run training followed by continual evaluation:

```bash
PYTHONPATH=. python src/cli/continual.py --config configs/continual/shared_lora.yaml
```

Available strategy names are `naive_sequential_finetuning`, `lora`, `shared_lora`,
`zscl`, and `cpe_clip` (the exact strategy value is visible in
each YAML file). Strategy-specific settings live under `continual.lora` or
`continual.cpe_clip`.

Resume a run in place. The experiment must contain its saved `config.yaml`:

```bash
PYTHONPATH=. python src/cli/continual.py \
    --config experiments/continual/<run>/config.yaml \
    --resume_experiment experiments/continual/<run>
```

Skip the automatic evaluation pass with `--no_evaluate`, or evaluate a saved
run separately:

```bash
PYTHONPATH=. python src/cli/continual.py --config <config.yaml> --no_evaluate
PYTHONPATH=. python src/cli/continual_eval.py --experiment_root experiments/continual/<run>
PYTHONPATH=. python src/cli/continual_eval.py --experiment_root experiments/continual/<run> \
    --checkpoint_name best_model.pt
```

The SLURM wrappers request one A100 GPU and activate a conda environment named
`voxtell`:

```bash
sbatch scripts/continual.sh configs/continual/shared_lora.yaml
sbatch scripts/continual_eval.sh experiments/continual/<run>
sbatch scripts/resume_continual.sh experiments/continual/<run>
```

Adjust the hard-coded cluster paths, partition, and environment in those
scripts before using them on another cluster.

## Experiment artifacts

Standard training creates a timestamped directory below the configured
`training.output_dir`:

```text
<run>/
    config.yaml                    # YAML snapshot
    checkpoints/
        best_model.pt
        latest_recovery.pt            # crash-safe recovery checkpoint
    logs/run.log
    plots/
        training_loss.png
        loss_metrics.json
    debug_artifacts/                # only when enabled
```

Evaluation creates `logs/eval.log` and timestamped `eval_*.json` files. A
continual run additionally contains `tasks/`, task checkpoints, and continual
state/evaluation files such as `continual_state.json` and
`evaluation_state.json`.

## Continual-learning metrics

Continual evaluation builds an evaluation matrix with this layout:

```text
                                                 Pretrained  AfterT1  ...  AfterTN
Continual task 1
...
Continual task N
Retention task(s)
```

The matrix must have one pretrained column plus one column per training task.
`src/continual/metrics.py` computes the following metrics from it:

- `A`: average performance after each task, using the scores for tasks seen so
    far.
- `F`: average forgetting at each checkpoint. For previously seen tasks, this
    compares the best earlier score with the current score.
- `ZS`: zero-shot transfer score for each step, based on performance on tasks
    before they are trained.
- `A_final`: average performance across continual tasks at the final checkpoint.
- `retention_drop`: change from pretrained retention-task performance at each
    later checkpoint.

The continual evaluator saves the full task matrix to `eval_matrix.csv`, the
aggregated values and summaries to `metrics.json`, and prompt-level diagnostic
matrices under `per_prompt/`. Logged summary values include final forgetting,
average zero-shot transfer, and average retention drop.

## Analysis and conversion tools

Visualise tensors saved during training:

```bash
PYTHONPATH=. python scripts/visualize_debug.py \
    --experiment experiments/exp_debug \
    --epoch 1 \
    --step 1 \
    --backend napari
```

Use `--backend sitk` to write NIfTI debug volumes for ITK-SNAP or 3D Slicer.
Other utilities are:

- `scripts/foreground_stats.py`: foreground voxel and volume statistics for
    tasks in a continual configuration; optionally writes CSV with `--out`.
- `scripts/compute_lora_update_magnitude.py`: analyse LoRA update magnitude.
- `scripts/plot_training_loss.py`: plot saved training losses.
- `scripts/h5_to_nifti.py`: convert SKM-TEA HDF5 `echo1`/`seg` data to NIfTI
    images and six-class annotations.
- `scripts/redvis_data_download.py`: download SKM-TEA files through Redivis;
    edit its cluster-specific paths before use.

## Development notes

The main Python package is `src`. The most useful extension points are:

- Add a dataset in `src/data/adapters/` and register it in
    `src/data/adapters/__init__.py`.
- Add a continual strategy under `src/continual/strategies/` and register it
    through the strategy registry.
- Adjust model construction or checkpoint loading in `src/model/builder.py`.
- Keep experiment settings in YAML rather than hard-coding them in entry
    points.

There is no project packaging configuration or automated test suite in the
current repository. Validate changes with a small configuration such as
`configs/dry_run.yaml`, a Python syntax check, and the relevant CLI help:

```bash
python -m compileall src scripts
PYTHONPATH=. python src/cli/main.py --help
PYTHONPATH=. python src/cli/infer.py --help
PYTHONPATH=. python src/cli/continual.py --help
```
