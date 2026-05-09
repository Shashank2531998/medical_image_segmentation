from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import List, Tuple

import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from tqdm import tqdm


class SlidingWindowInferer:
    def __init__(
        self,
        network: torch.nn.Module,
        patch_size: Tuple[int, ...],
        device: torch.device,
        tile_step_size: float = 0.5,
        perform_everything_on_device: bool = True,
    ) -> None:
        self.network = network
        self.patch_size = patch_size
        self.device = device
        self.tile_step_size = tile_step_size
        self.perform_everything_on_device = perform_everything_on_device

    def get_slicers(self, image_size: Tuple[int, ...]) -> List[Tuple]:
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(image_size) - 1, (
                "if tile_size has less entries than image_size, "
                "len(tile_size) must be one shorter than len(image_size) "
                "(only dimension discrepancy of 1 allowed)."
            )
            steps = compute_steps_for_sliding_window(image_size[1:], self.patch_size, self.tile_step_size)
            for depth in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), depth, *[slice(si, si + ti) for si, ti in zip((sx, sy), self.patch_size)]])
                        )
        else:
            steps = compute_steps_for_sliding_window(image_size, self.patch_size, self.tile_step_size)
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), self.patch_size)]])
                        )
        return slicers

    @torch.no_grad()
    def predict_logits(self, data: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        results_device = self.device if self.perform_everything_on_device else torch.device("cpu")
        self.network = self.network.to(self.device)

        def producer(data_tensor, slicer_list, queue):
            for slicer in slicer_list:
                patch = torch.clone(
                    data_tensor[slicer][None],
                    memory_format=torch.contiguous_format,
                ).to(self.device)
                queue.put((patch, slicer))
            queue.put("end")

        empty_cache(self.device)
        data = data.to(results_device)
        slicers = self.get_slicers(data.shape[1:])
        queue = Queue(maxsize=2)
        thread = Thread(target=producer, args=(data, slicers, queue))
        thread.start()

        predicted_logits = torch.zeros((text_embeddings.shape[1], *data.shape[1:]), dtype=torch.half, device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

        gaussian = compute_gaussian(
            tuple(self.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )

        with tqdm(desc=None, total=len(slicers)) as progress_bar:
            while True:
                item = queue.get()
                if item == "end":
                    queue.task_done()
                    break
                patch, tile_slice = item
                prediction = self.network(patch, text_embeddings)[0].to(results_device)
                prediction *= gaussian
                predicted_logits[tile_slice] += prediction
                n_predictions[tile_slice[1:]] += gaussian
                queue.task_done()
                progress_bar.update()
        queue.join()

        torch.div(predicted_logits, n_predictions, out=predicted_logits)

        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError(
                "Encountered inf in predicted array. Aborting... "
                "If this problem persists, reduce value_scaling_factor in "
                "compute_gaussian or increase the dtype of predicted_logits to fp32."
            )
        return predicted_logits
