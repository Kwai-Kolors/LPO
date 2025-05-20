"""
Adapted from https://github.com/yuvalkirstain/PickScore. Originally MIT License, Copyright (c) 2021.
"""


from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from datasets import load_from_disk, load_dataset
import torch
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
import sys


def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB")
    return image



class PickScorer(torch.nn.Module):
    def __init__(self, processor_name_or_path, model_pretrained_name_or_path, device='cuda'):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).to(device)
        self.device = device
        self.eval()

    @torch.no_grad()
    def __call__(self, prompt, images):
        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(): 
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            # get probabilities if you have multiple images to choose from
            if len(scores) == 1:
                probs = scores
            else:
                probs = torch.softmax(scores, dim=-1)
        
        return probs.cpu().tolist()
    
    
    def score(self, img_path, prompt):
        if isinstance(img_path, list):
            result = []
            for one_img_path in img_path:
                # Load your image and prompt
                with torch.no_grad():
                    # Process the image
                    if isinstance(one_img_path, str):
                        image = self.preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                    elif isinstance(one_img_path, Image.Image):
                        image = self.preprocess_val(one_img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    # Process the prompt
                    text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        outputs = self.model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        pick_score = self.model.logit_scale.exp() * torch.diagonal(logits_per_image).cpu().numpy()
                result.append(pick_score[0])    
            return result
        elif isinstance(img_path, str):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    pick_score = self.model.logit_scale.exp() * torch.diagonal(logits_per_image).cpu().numpy()
            return [pick_score[0]]
        elif isinstance(img_path, Image.Image):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    pick_score = self.model.logit_scale.exp() * torch.diagonal(logits_per_image).cpu().numpy()
            return [pick_score[0]]
        else:
            raise TypeError('The type of parameter img_path is illegal.')
    

if __name__ == "__main__":
    pickscorer = PickScorer(processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", model_pretrained_name_or_path="yuvalkirstain/PickScore_v1")

    image0 = open_image('./image0.png')
    image1 = open_image('./image1.png')
    prompt = "photorealistic image of a lone painter standing in a gallery, watching an exhibition of paintings made entirely with AI. In the foreground of the image a robot looks proudly at his art"

    probs = pickscorer(prompt, [image0])
    probs1 = pickscorer(prompt, [image1])
    print(probs)
    print(probs1)
