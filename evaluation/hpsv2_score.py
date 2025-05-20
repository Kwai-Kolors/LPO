"""
Adapted from https://github.com/tgxs002/HPSv2. Originally Apache License, Version 2.0, January 2004.
"""

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image


class HPSv2Scorer():
    def __init__(self, clip_pretrained_name_or_path, model_pretrained_name_or_path, device='cuda'):
        self.model, _, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            # 'laion2B-s32B-b79K',
            clip_pretrained_name_or_path,
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        self.device = device
        checkpoint = torch.load(model_pretrained_name_or_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(device)
        
        
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
                    # Calculate the HPS
                    with torch.cuda.amp.autocast():
                        outputs = self.model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0])    
            return result
        elif isinstance(img_path, str):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            return [hps_score[0]]
        elif isinstance(img_path, Image.Image):
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = self.preprocess_val(img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            return [hps_score[0]]
        else:
            raise TypeError('The type of parameter img_path is illegal.')
        
        
if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    
    clip_model_path = hf_hub_download(repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", filename="open_clip_pytorch_model.bin")
    hps_model_path = hf_hub_download(repo_id="xswu/HPSv2", filename="HPS_v2_compressed.pt")

    hpsv2_scorer = HPSv2Scorer(clip_pretrained_name_or_path=clip_model_path,
                                model_pretrained_name_or_path=hps_model_path)
    score = hpsv2_scorer.score(img_path=['./image0.png', './image1.png'],
                                prompt='photorealistic image of a lone painter standing in a gallery, watching an exhibition of paintings made entirely with AI. In the foreground of the image a robot looks proudly at his art')

    print(score)
    