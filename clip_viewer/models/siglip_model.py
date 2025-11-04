from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

from clip_viewer.models.base_model import BaseEmbeddingModel, ImagesType


class SiglipModel(BaseEmbeddingModel):
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.placeholder_images = torch.zeros((1, 3, 112, 112)).to(device)

    @staticmethod
    def load(gpu: bool = True) -> "SiglipModel":
        ckpt = "google/siglip2-base-patch16-224"
        model = AutoModel.from_pretrained(
            ckpt, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(ckpt, use_fast=True)

        # enable gpu
        device = torch.device("cuda" if gpu else "cpu")
        if gpu:
            model = model.to(device)

        return SiglipModel(model, processor, device)

    @torch.inference_mode()
    def encode_image(self, images: ImagesType, normalize: bool = False) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=torch.float16 if self.device == "cuda" else None)
        feats = self.model.get_image_features(**inputs).float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def encode_text(self, texts: Union[List[str], str], normalize: bool = False) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts.lower()]
        elif isinstance(texts, list):
            texts = [t.lower() for t in texts]
        else:
            raise TypeError(f"Unsupported input type \"{type(texts)}\". Expected list of strings or string.")

        inputs = self.processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs).float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.cpu()
