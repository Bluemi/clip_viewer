from typing import List, Union

import numpy as np
from PIL import Image
import torch

# Define the ImagesType union type
ImagesType = Union[np.ndarray, Image.Image, List[Image.Image], List[np.ndarray]]


class BaseEmbeddingModel:
    """
    Base class for embedding models that can encode images and text into feature vectors.
    """

    def encode_image(self, images: ImagesType, normalize: bool = False) -> torch.Tensor:
        """
        Encodes images into feature vectors.

        Args:
            images: Input image(s) to encode. Can be:
                - PIL Image
                - NumPy array (single image or batch)
                - List of PIL Images
                - List of NumPy arrays
            normalize: Whether to normalize the feature vectors. Default: False.

        Returns:
            torch.Tensor: Feature vector(s) for the input image(s).
        """
        raise NotImplementedError("Subclasses must implement encode_image method")

    def encode_text(self, texts: Union[List[str], str], normalize: bool = False) -> torch.Tensor:
        """
        Encodes text into feature vectors.

        Args:
            texts: Input text to encode. Can be a string or list of strings.
            normalize: Whether to normalize the feature vectors. Default: False.

        Returns:
            torch.Tensor: Feature vector(s) for the input text(s).
        """
        raise NotImplementedError("Subclasses must implement encode_text method")