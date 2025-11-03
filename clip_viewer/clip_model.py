from typing import List, Union

import numpy as np
from PIL import Image

import open_clip
import torch
# noinspection PyProtectedMember
from open_clip.transform import _convert_to_rgb
from torchvision import transforms

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


class MobileModel(BaseEmbeddingModel):
    def __init__(self, traced: bool = True, load_mcip: bool = True):
        if traced:
            self.model, self._preprocess, self.tokenizer = MobileModel._load_traced_model()
        else:
            self.model, self._preprocess, self.tokenizer = MobileModel._load_mobile_model(load_mcip)

    @staticmethod
    def _load_mobile_model(load_mcip: bool):
        # create model
        model_name = "MobileCLIP-S2"
        model_pretrained = "datacompdr"  # "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_pretrained)

        if load_mcip:
            map_location = torch.device('cpu') if not torch.cuda.is_available() else None
            mcip_checkpoint = "models/MCIP_MobileCLIP.pt"
            old_state_dict = torch.load(mcip_checkpoint, map_location=map_location, weights_only=False)["state_dict"]
            new_state_dict = {}
            for key, value in old_state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict, strict=True)

        tokenizer = open_clip.get_tokenizer(model_name)

        model.text.eval()
        model.visual.eval()

        return model, preprocess, tokenizer

    @staticmethod
    def _load_traced_model(fp16_image: bool = False):
        model_name = "MobileCLIP-S2"
        if fp16_image:
            image_model = torch.jit.load("models/traced_image_encoder_fp16.pt")
        else:
            image_model = torch.jit.load("models/traced_image_encoder.pt")
        text_model = torch.jit.load("models/traced_text_encoder.pt")
        tokenizer = open_clip.get_tokenizer(model_name)

        steps = [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
        if fp16_image:
            steps.append(transforms.ConvertImageDtype(torch.float16))

        preprocess = transforms.Compose(steps)
        return TracedBackend(image_model, text_model), preprocess, tokenizer

    def encode_image(self, images: ImagesType, normalize: bool = False):
        with torch.no_grad():
            pil_images = MobileModel._convert_to_pil_list(images)
            preprocessed_images = torch.stack([self._preprocess(i) for i in pil_images])
            features = self.model.encode_image(preprocessed_images)
            if normalize:
                features /= features.norm(dim=-1, keepdim=True)
            if MobileModel._is_single_image(images):
                features = features.squeeze(0)
        return features

    @staticmethod
    def _convert_to_pil_list(images: ImagesType) -> List[Image.Image]:
        """
        Convert input images to a list of PIL images.

        :param images: Input image(s) to convert.
        :returns: A list of PIL images.
        """
        if isinstance(images, Image.Image):
            return [images]

        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image case (H, W, C)
                return [Image.fromarray(images)]
            elif images.ndim == 4:  # Batch case (N, H, W, C)
                return [Image.fromarray(img) for img in images]
            else:
                raise ValueError("Invalid numpy array shape for images. Expected 3D or 4D array.")

        if isinstance(images, list):
            temp_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                    temp_images.append(img)
                elif isinstance(img, Image.Image):
                    temp_images.append(img)
                else:
                    raise TypeError(f"Unsupported image type \"{type(img)}\".")
            return temp_images

        raise TypeError(
            f"Unsupported input type \"{type(images)}\". Expected numpy array, PIL image, or list of PIL images."
        )

    @staticmethod
    def _is_single_image(images: ImagesType) -> bool:
        if isinstance(images, np.ndarray):
            return images.ndim == 3
        else:
            return isinstance(images, Image.Image)

    def encode_text(self, texts: Union[List[str], str], normalize: bool = False):
        single_instance = False
        if isinstance(texts, str):
            texts = [texts]
            single_instance = True
        elif not isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            raise TypeError(f"Unsupported input type \"{type(texts)}\". Expected list of strings or string.")

        tokens = self.tokenizer(texts)
        features = self.model.encode_text(tokens)

        if normalize:
            features /= features.norm(dim=-1, keepdim=True)

        if single_instance:
            features = features.squeeze(0)
        return features


class TracedBackend:
    def __init__(self, image_model: torch.ScriptModule, text_model: torch.ScriptModule):
        self.image_model = image_model
        self.text_model = text_model

    def encode_image(self, images: torch.Tensor):
        # noinspection PyCallingNonCallable
        return self.image_model(images)

    def encode_text(self, tokens: torch.LongTensor) -> torch.Tensor:
        # noinspection PyCallingNonCallable
        return self.text_model(tokens)



def reparameterize_model(model: torch.nn.Module):
    """
    Taken from https://github.com/apple/ml-mobileclip/blob/main/mobileclip/modules/common/mobileone.py

    :param model: MobileOne model in train mode.
    """
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
