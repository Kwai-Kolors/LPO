"""
Adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor. Originally Apache License, Version 2.0, January 2004.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from io import BytesIO

def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB")
    return image


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, clip_name_or_path, aesthetic_name_or_path):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(clip_name_or_path)
        self.mlp = MLP()
        state_dict = torch.load(aesthetic_name_or_path, map_location='cpu')
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
 
    
if __name__ == "__main__":
    clip_name_or_path = "openai/clip-vit-large-patch14"
    aesthetic_name_or_path = "./sac+logos+ava1-l14-linearMSE.pth"
    aesthetic_scorer = AestheticScorer(torch.float32, clip_name_or_path, aesthetic_name_or_path)
    image0 = open_image('./image0.png')
    image1 = open_image('./image1.png')
    print(aesthetic_scorer(image0))
    print(aesthetic_scorer(image1))
    print(aesthetic_scorer([image0, image1]))
    