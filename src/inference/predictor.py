from typing import List, Tuple, Union

import numpy as np
import torch

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.utilities.helpers import dummy_context

from src.inference.postprocessing import logits_to_segmentation
from src.inference.sliding_window import SlidingWindowInferer
from src.model.builder import load_voxtell_model
from src.text.encoder import TextPromptEncoder


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
                 text_encoding_model: str = 'Qwen/Qwen3-Embedding-4B') -> None:
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
        self.normalization = ZScoreNormalization(intensityproperties={})

        # Predictor settings
        self.tile_step_size = 0.5
        self.perform_everything_on_device = True

        self.text_encoder = TextPromptEncoder(text_encoding_model, device=self.device)

        self.network, self.patch_size = load_voxtell_model(model_dir)
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

        if data.ndim == 3:
            data = data[None]  # add channel axis
        data = data.astype(np.float32)  # this creates a copy
        original_shape = data.shape[1:]
        data, _, bbox = crop_to_nonzero(data, None)
        data = self.normalization.run(data, None)
        data = torch.from_numpy(data)
        return data, bbox, original_shape
    
    @torch.inference_mode()
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

    @torch.inference_mode()
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


if __name__ == '__main__':
    from pathlib import Path
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    # Default paths - modify these as needed
    DEFAULT_IMAGE_PATH = "/path/to/your/image.nii.gz"
    DEFAULT_MODEL_DIR = "/path/to/your/model/directory"
    
    # Configuration
    image_path = DEFAULT_IMAGE_PATH
    model_dir = DEFAULT_MODEL_DIR
    text_prompts = ["liver", "right kidney", "left kidney", "spleen"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load image
    img, props = NibabelIOWithReorient().read_images([image_path])
    
    # Initialize predictor and run inference
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    
    # Visualize results, we reccommend using napari for 3D visualization
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img, name='image')
    for i, prompt in enumerate(text_prompts):
        viewer.add_labels(voxtell_seg[i], name=f'voxtell_{prompt}')
    napari.run()