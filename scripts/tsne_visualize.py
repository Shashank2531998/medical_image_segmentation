"""Create t-SNE visualisations for VoxTell embeddings.

Usage (example):
python scripts/tsne_visualize.py --config configs/train.yaml --split test --n_samples 50 \
    --embedding mask --stage 0 --per_prompt --output out/tsne_mask_stage0.png

The script supports embedding types: `text`, `mask`, `encoder`.
`mask` embeddings are the stage-projected mask embeddings returned by the model.
`encoder` embeddings are global-pooled encoder skip features for a selected stage.

"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import load_config
from src.data.datamodule import VoxTellDataModule
from src.engine.model_engine import VoxTellEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config (contains `tsne`, `dataset`, and `model` sections)")
    p.add_argument("--output", default=None, help="Optional override for output figure path")
    return p.parse_args()


def collect_embeddings(engine: VoxTellEngine, dataloader, n_samples: int, embedding: str, stage: int, per_prompt: bool):
    engine.model.eval()
    collected = []
    labels = []
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            # Stop if we've collected requested number of samples
            if sample_count >= n_samples:
                break

            batch_size = batch["image"].shape[0]

            # Ensure engine returns features
            outputs = engine.forward(batch)

            # When engine.return_features True, forward returns (outs, mask_embeddings, skips)
            if engine.return_features:
                out_logits, mask_embeddings, skips = outputs
            else:
                raise RuntimeError("Engine must be initialized with return_features=True to extract embeddings")

            # Text embeddings: compute using engine.text_encoder
            if embedding == "text":
                text_emb = engine.text_encoder.embed(batch["prompts"])  # (B, N, D)
                if per_prompt:
                    # Emit each prompt as separate point
                    for b in range(text_emb.shape[0]):
                        for n in range(text_emb.shape[1]):
                            collected.append(text_emb[b, n].cpu().numpy())
                            labels.append(f"{batch['img_path'][b]}|prompt_{n}")
                else:
                    # Pool across prompts (mean)
                    pooled = text_emb.mean(dim=1)
                    for b in range(pooled.shape[0]):
                        collected.append(pooled[b].cpu().numpy())
                        labels.append(batch["img_path"][b])

            elif embedding == "mask":
                # mask_embeddings is a list (one per projection stage), each (B, N, C)
                if stage < 0 or stage >= len(mask_embeddings):
                    raise ValueError(f"stage must be in [0, {len(mask_embeddings)-1}]")
                m = mask_embeddings[stage]  # (B, N, C)
                if per_prompt:
                    for b in range(m.shape[0]):
                        for n in range(m.shape[1]):
                            collected.append(m[b, n].cpu().numpy())
                            labels.append(f"{batch['img_path'][b]}|prompt_{n}")
                else:
                    pooled = m.mean(dim=1)  # (B, C)
                    for b in range(pooled.shape[0]):
                        collected.append(pooled[b].cpu().numpy())
                        labels.append(batch["img_path"][b])

            elif embedding == "encoder":
                # skips is list of tensors (stage order), each (B, C, D, H, W)
                if stage < 0 or stage >= len(skips):
                    raise ValueError(f"stage must be in [0, {len(skips)-1}]")
                s = skips[stage]
                # Global average pool spatial dims (assume dims are (B,C,D,H,W) or (B,C,H,W,D))
                # We'll compute mean over all dims except batch and channel
                spatial_dims = tuple(range(2, s.ndim))
                pooled = s.mean(dim=spatial_dims)  # (B, C)
                for b in range(pooled.shape[0]):
                    collected.append(pooled[b].cpu().numpy())
                    labels.append(batch["img_path"][b])

            sample_count += batch_size

    if len(collected) == 0:
        return np.zeros((0, 0)), []
    return np.stack(collected, axis=0), labels


def collect_for_dataset(engine: VoxTellEngine, dm: VoxTellDataModule, split: str, n_samples: int, embedding: str, stage: int, per_prompt: bool, dataset_label: str):
    if split == "train":
        loader = dm.train_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    X, labels = collect_embeddings(engine, loader, n_samples, embedding, stage, per_prompt)
    # prefix labels with dataset label
    labels = [f"{dataset_label}:{l}" for l in labels]
    return X, labels


def run_tsne_and_plot(X: np.ndarray, labels: list[str], output: str, perplexity: float, seed: int):
    rng = np.random.RandomState(seed)
    # Normalize
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, (X.shape[0]-1)//3)), random_state=seed)
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hls", n_colors=min(20, len(set(labels))))

    # Simple coloring by label (use label hash to select color)
    colors = [hash(l) % len(palette) for l in labels]
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=colors, cmap="tab20", s=30, alpha=0.8)
    plt.title("t-SNE of embeddings")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved t-SNE plot to: {out_path}")


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    # Read TSNE config from the provided YAML under the `tsne` key.
    tsne_cfg = cfg.get("tsne", {})

    split = tsne_cfg.get("split", "test")
    n_samples = int(tsne_cfg.get("n_samples", 50))
    embedding = tsne_cfg.get("embedding", "mask")
    stage = int(tsne_cfg.get("stage", 0))
    per_prompt = bool(tsne_cfg.get("per_prompt", False))
    perplexity = float(tsne_cfg.get("perplexity", 30.0))
    seed = int(tsne_cfg.get("seed", 42))
    output = args.output or tsne_cfg.get("output", "tsne.png")

    # Initialize engine using `model` section in config
    model_cfg = cfg.get("model", {})
    engine = VoxTellEngine(model_cfg=model_cfg, return_features=True)

    X_list = []
    labels_list = []

    # Support multiple datasets for continual-style configs. If `tsne.datasets` is provided,
    # iterate through each dataset config and collect embeddings; otherwise use single `dataset`.
    if "datasets" in tsne_cfg and isinstance(tsne_cfg["datasets"], list):
        base_dataset_cfg = cfg.get("dataset", {})
        for ds in tsne_cfg["datasets"]:
            # Merge base dataset defaults with per-dataset override
            ds_dataset = ds.get("dataset", {})
            ds_cfg = dict(base_dataset_cfg)
            ds_cfg.update(ds_dataset)

            # Determine label: prefer explicit `label`, then entry `name`, then dataset.name
            ds_label = ds.get("label") or ds.get("name") or ds_cfg.get("name") or "dataset"
            ds_n = int(ds.get("n_samples", n_samples))
            ds_split = ds.get("split", split)
            ds_embedding = ds.get("embedding", embedding)
            ds_stage = int(ds.get("stage", stage))
            ds_per_prompt = bool(ds.get("per_prompt", per_prompt))

            dm = VoxTellDataModule(ds_cfg)
            X_ds, labels_ds = collect_for_dataset(engine, dm, ds_split, ds_n, ds_embedding, ds_stage, ds_per_prompt, ds_label)
            if X_ds.size != 0:
                X_list.append(X_ds)
                labels_list.extend(labels_ds)
    else:
        # single dataset from top-level `dataset` section
        dm = VoxTellDataModule(cfg.get("dataset", {}))
        X, labels = collect_for_dataset(engine, dm, split, n_samples, embedding, stage, per_prompt, cfg.get("dataset", {}).get("name", "dataset"))
        X_list.append(X)
        labels_list.extend(labels)

    if len(X_list) == 0:
        raise RuntimeError("No embeddings were collected. Check dataset paths and TSNE config.")

    X_all = np.concatenate(X_list, axis=0)
    run_tsne_and_plot(X_all, labels_list, output, perplexity, seed)


if __name__ == "__main__":
    main()
