from pathlib import Path
from typing import List, Tuple, Union

import nibabel
import numpy as np
import torch

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.utilities.helpers import dummy_context

from src.data.preprocess import preprocess_image
from src.inference.postprocessing import logits_to_segmentation
from src.inference.sliding_window import SlidingWindowInferer
from src.model.builder import load_voxtell_model
from src.text.encoder import TextPromptEncoder
from src.utils.model_helpers import log_model_params, set_adapters_enabled
from src.utils.reorientation import reorient_seg_from_props
from src.utils.logging import get_logger


logger = get_logger(__name__)


class VoxTellPredictor:
    """
    Predictor for VoxTell segmentation model.
    
    This class handles loading the VoxTell model, preprocessing images,
    embedding text prompts, and performing sliding window inference to generate
    segmentation masks based on free-text anatomical descriptions.
    
    Attributes:
        device: PyTorch device for inference.
        network: The VoxTell model.
        tokenizer: Text tokenizer for prompt encoding.
        text_backbone: Text embedding model.
        patch_size: Patch size for sliding window inference.
        tile_step_size: Step size for sliding window (default: 0.5 = 50% overlap).
        perform_everything_on_device: Keep all tensors on device during inference.
        max_text_length: Maximum text prompt length in tokens.
    """
    def __init__(self, model_dir: str, device: torch.device = torch.device('cuda'),
                 text_encoding_model: str = 'Qwen/Qwen3-Embedding-4B', checkpoint_path: str | Path = None,
                 lora_cfg: dict | None = None, lora_adapter_path: str | Path | None = None,
                 cpe_clip_cfg: dict | None = None,
                 disable_adapters = False
                 ) -> None:
        """
        Initialize the VoxTell predictor.
        
        Args:
            model_dir: Path to model directory containing plans.json and checkpoint.
            device: PyTorch device to use for inference (default: cuda).
            text_encoding_model: Pretrained text encoding model (Qwen/Qwen3-Embedding-4B).
            
        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
        """
        # Device setup
        self.device = device
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # Predictor settings
        self.tile_step_size = 0.5
        self.perform_everything_on_device = True

        self.text_encoder = TextPromptEncoder(text_encoding_model, device=self.device)

        self.network, self.patch_size = load_voxtell_model(
            model_dir,
            checkpoint_path=checkpoint_path,
            lora_cfg=lora_cfg,
            lora_adapter_path=lora_adapter_path,
            cpe_clip_cfg=cpe_clip_cfg,
        )
        
        # Setting it to eval mode
        self.network.eval()

        if disable_adapters:
            set_adapters_enabled(self.network, False)

        log_model_params(self.network)
        
        self.network = self.network.to(device)
        self.sliding_window = SlidingWindowInferer(
            self.network,
            self.patch_size,
            self.device,
            tile_step_size=self.tile_step_size,
            perform_everything_on_device=self.perform_everything_on_device,
        )

    def preprocess(self, data: np.ndarray) -> Tuple[torch.Tensor, Tuple, Tuple[int, ...]]:
        """
        Preprocess a single image for inference.
        
        This function preprocesses an image already in RAS orientation by performing
        cropping to non-zero regions and z-score normalization.
        
        Args:
            data: Image data in RAS orientation (3D or 4D with channel dimension).
            
        Returns:
            Tuple containing:
                - Preprocessed image tensor
                - Bounding box of cropped region
                - Original image shape
        """

        return preprocess_image(data)
    
    @torch.no_grad()
    def embed_text_prompts(self, text_prompts: Union[List[str], str]) -> torch.Tensor:
        """
        Embed text prompts into vector representations.
        
        This function converts free-text anatomical descriptions into embeddings
        using the text backbone model.
        
        Args:
            text_prompts: Single text prompt or list of text prompts.
            
        Returns:
            Text embeddings tensor of shape (1, num_prompts, embedding_dim).
        """
        return self.text_encoder.embed(text_prompts)

    @torch.no_grad()
    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform sliding window inference to generate segmentation logits.
        
        Args:
            input_image: Input image tensor of shape (C, X, Y, Z).
            text_embeddings: Text embeddings from embed_text_prompts.
            
        Returns:
            Predicted logits tensor.
            
        Raises:
            ValueError: If input_image is not 4D or not a torch.Tensor.
        """
        if not isinstance(input_image, torch.Tensor):
            raise ValueError(f"input_image must be a torch.Tensor, got {type(input_image)}")
        if input_image.ndim != 4:
            raise ValueError(
                f"input_image must be 4D (C, X, Y, Z), got shape {input_image.shape}"
            )
        
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size,
                                                       'constant', {'value': 0}, True, None)

            predicted_logits = self.sliding_window.predict_logits(data, text_embeddings)
            # Revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def predict_single_image(
        self,
        data: np.ndarray,
        text_prompts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Predict segmentation masks for a single image with text prompts.
        
        This is the main prediction method that orchestrates preprocessing,
        text embedding, sliding window inference, and postprocessing.
        
        Args:
            data: Image data in RAS orientation (3D or 4D with channel dimension).
            text_prompts: Single text prompt or list of text prompts describing
                anatomical structures to segment.
                
        Returns:
            Segmentation masks as numpy array of shape (num_prompts, X, Y, Z)
            with binary values (0 or 1) indicating the segmented regions.
        """

        # Preprocess image
        data, bbox, orig_shape = self.preprocess(data)

        # Embed text prompts
        embeddings = self.embed_text_prompts(text_prompts)

        # Predict segmentation logits
        prediction = self.predict_sliding_window_return_logits(data, embeddings).to('cpu')

        return logits_to_segmentation(prediction, bbox, orig_shape)


def get_reader_writer(file_path: str):
    suffix = Path(file_path).suffix.lower()

    if suffix in [".nii", ".gz"]:
        return NibabelIOWithReorient()

    raise ValueError(
        f"Unsupported file format: {suffix}. Only NIfTI supported."
    )


def save_segmentation(segmentation, output_file: Path):
    nibabel.save(segmentation, output_file)


def save_all_segmentations(
    segmentations,
    output_folder: Path,
    input_path: Path,
    prompts: List[str],
    save_combined: bool = False,
    verbose: bool = False,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    input_filename = input_path.stem
    if input_filename.endswith(".nii"):
        input_filename = input_filename[:-4]

    suffix = ".nii.gz" if input_path.suffix == ".gz" else input_path.suffix

    output_files = []

    if save_combined:
        if len(prompts) > 1 and verbose:
            logger.warning("Combining multi-label segmentation")

        if len(prompts) == 1:
            out_file = output_folder / f"{input_filename}{suffix}"
            save_segmentation(segmentations[0], out_file)
        else:
            combined = np.zeros_like(segmentations[0], dtype=np.uint8)

            for i, seg in enumerate(segmentations):
                combined[seg > 0] = i + 1

            out_file = output_folder / f"{input_filename}{suffix}"
            save_segmentation(combined, out_file)

        output_files.append(out_file)

    else:
        for i, prompt in enumerate(prompts):
            safe = "".join(
                c if c.isalnum() or c in (" ", "_") else "_"
                for c in prompt
            ).replace(" ", "_")

            out_file = output_folder / f"{input_filename}_{safe}{suffix}"
            save_segmentation(segmentations[i], out_file)
            output_files.append(out_file)

    return output_files


def predict_image(
    predictor,
    input_path: str | Path,
    prompts: List[str],
    verbose: bool = False,
):
    input_path = Path(input_path)

    if verbose:
        logger.info("Loading image: %s", input_path)

    reader = get_reader_writer(str(input_path))
    img, props = reader.read_images([str(input_path)])

    if verbose:
        logger.info("Image shape: %s", img.shape)
        logger.info("Prompts: %s", prompts)
        logger.info("Running prediction...")

    segmentations = predictor.predict_single_image(img, prompts)
    segmentations = [
        reorient_seg_from_props(seg, props)
        for seg in segmentations
    ]

    if verbose:
        logger.info("Prediction completed")

    return segmentations


def get_predictor(model_path, device, checkpoint_path=None, lora_cfg=None, lora_adapter_path=None, cpe_clip_cfg=None, disable_adapters=False):
    model_path = Path(model_path)

    if not (model_path / "plans.json").exists():
        raise FileNotFoundError("plans.json missing")

    logger.info("Loading model from %s", checkpoint_path if checkpoint_path else model_path)

    return VoxTellPredictor(
        model_dir=str(model_path),
        device=device,
        checkpoint_path=checkpoint_path,
        lora_cfg=lora_cfg,
        lora_adapter_path=lora_adapter_path,
        cpe_clip_cfg=cpe_clip_cfg,
        disable_adapters=disable_adapters
    )
