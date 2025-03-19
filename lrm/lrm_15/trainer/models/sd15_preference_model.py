from dataclasses import dataclass
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import time
import os
from io import BytesIO
from trainer.models.base_model import BaseModelConfig
from trainer.models.unet_2d_condition_reward import UNet2DConditionModel

from accelerate.logging import get_logger
logger = get_logger(__name__)

@dataclass
class SD15PreferenceModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.sd15_preference_model.SD15PreferenceModel"
    pretrained_model_name_or_path: str = 'runwayml/stable-diffusion-v1-5'
    clip_ckpt_path: str = 'openai/clip-vit-large-patch14/pytorch_model.bin'
    vae_path: str = ""
    vision_embed_dim: int = 1280
    text_embed_dim: int = 768
    projection_dim: int = 768
    logit_scale_init_value: float = 2.6592
    freeze_text_encoder: bool = False
    multi_scale: bool = True
    multi_scale_cfg: bool = False
    guidance_scale: float = 1.0


class SD15PreferenceModel(nn.Module):
    def __init__(self, cfg: SD15PreferenceModelConfig):
        super().__init__()
        # diffusion models
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
        if cfg.vae_path != "":
            self.vae = AutoencoderKL.from_pretrained(cfg.vae_path)
        else:
            self.vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")
        # self.pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
        # self.image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
        
        # global pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cfg = cfg
        
        # projection layers
        if cfg.multi_scale:
            self.visual_projection = nn.Linear(4800, cfg.projection_dim, bias=False)

        else:  
            self.visual_projection = nn.Linear(cfg.vision_embed_dim, cfg.projection_dim, bias=False)
        nn.init.normal_(self.visual_projection.weight, std=0.02)
        
        self.text_projection = nn.Linear(cfg.text_embed_dim, cfg.projection_dim, bias=False)
        # load text projections from openai/clip-vit-large-patch14
        clip_ckpt = torch.load(cfg.clip_ckpt_path)
        self.text_projection.weight.data = clip_ckpt['text_projection.weight'].contiguous()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * cfg.logit_scale_init_value)
        
        self.vae.requires_grad_(False)
        if cfg.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
        
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.do_classifier_free_guidance = self.cfg.guidance_scale > 1.0 or self.cfg.guidance_scale < 1.0
        if self.do_classifier_free_guidance:
            # generate negative prompt ids
            self.neg_prompt_ids = self.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
    
    def get_text_features(self, text_inputs=None):
        if self.do_classifier_free_guidance:
            text_inputs = torch.cat([text_inputs, self.neg_prompt_ids.repeat(text_inputs.shape[0], 1).to(text_inputs.device)], dim=0)
            
        outputs = self.text_encoder(text_inputs, return_dict=False)
        encoder_hidden_states = outputs[0]
        pooled_output = outputs[1]
        
        if self.do_classifier_free_guidance:
            pooled_output_text, pooled_output_ucond = pooled_output.chunk(2, dim=0)
            text_features = self.text_projection(pooled_output_text)
        else:
            text_features = self.text_projection(pooled_output)
        return encoder_hidden_states, text_features
    
    def get_image_features(self, encoder_hidden_states=None, image_inputs=None, time_cond=None, generator=None):
        with torch.no_grad():
            latents = self.vae.encode(image_inputs).latent_dist.sample()
            # latents = latents.to(dtype=self.unet.dtype)
            latents = latents * self.vae.config.scaling_factor
  
        if generator is not None:
            noise = torch.randn(latents.size(), generator=generator, dtype=latents.dtype, device=latents.device)
        else:
            noise = torch.randn_like(latents)

        noisy_latents = self.scheduler.add_noise(latents, noise, time_cond)
        
        if self.do_classifier_free_guidance:
            noisy_latents = torch.cat([noisy_latents] * 2, dim=0)
            time_cond = torch.cat([time_cond] * 2, dim=0)

        mid_output, down_block_res_samples = self.unet(noisy_latents, time_cond, encoder_hidden_states=encoder_hidden_states, return_dict=False, use_up_blocks=False)
        
        if self.cfg.multi_scale:
            first_stage_output = down_block_res_samples[2]  # [320, 64, 64]
            second_stage_output = down_block_res_samples[5]  # [640, 32, 32]
            third_stage_output = down_block_res_samples[8]  # [1280, 16, 16]
            fourth_stage_output = down_block_res_samples[11]  # [1280, 8, 8]

            pooled_first_stage_output = self.avg_pool(first_stage_output).squeeze(dim=[2,3])
            pooled_second_stage_output = self.avg_pool(second_stage_output).squeeze(dim=[2,3])
            pooled_third_stage_output = self.avg_pool(third_stage_output).squeeze(dim=[2,3])
            pooled_fourth_stage_output = self.avg_pool(fourth_stage_output).squeeze(dim=[2,3])
            pooled_mid_output = self.avg_pool(mid_output).squeeze(dim=[2,3])
            if self.do_classifier_free_guidance:
                pooled_mid_output_text, pooled_mid_output_ucond = pooled_mid_output.chunk(2, dim=0)
                pooled_mid_output = pooled_mid_output_ucond + self.cfg.guidance_scale * (pooled_mid_output_text - pooled_mid_output_ucond)

                if self.cfg.multi_scale_cfg:
                    pooled_first_stage_output_text, pooled_first_stage_output_ucond = pooled_first_stage_output.chunk(2, dim=0)
                    pooled_first_stage_output = pooled_first_stage_output_ucond + self.cfg.guidance_scale * (pooled_first_stage_output_text - pooled_first_stage_output_ucond)

                    pooled_second_stage_output_text, pooled_second_stage_output_ucond = pooled_second_stage_output.chunk(2, dim=0)
                    pooled_second_stage_output = pooled_second_stage_output_ucond + self.cfg.guidance_scale * (pooled_second_stage_output_text - pooled_second_stage_output_ucond)

                    pooled_third_stage_output_text, pooled_third_stage_output_ucond = pooled_third_stage_output.chunk(2, dim=0)
                    pooled_third_stage_output = pooled_third_stage_output_ucond + self.cfg.guidance_scale * (pooled_third_stage_output_text - pooled_third_stage_output_ucond)

                    pooled_fourth_stage_output_text, pooled_fourth_stage_output_ucond = pooled_fourth_stage_output.chunk(2, dim=0)
                    pooled_fourth_stage_output = pooled_fourth_stage_output_ucond + self.cfg.guidance_scale * (pooled_fourth_stage_output_text - pooled_fourth_stage_output_ucond)
                else:
                    pooled_first_stage_output_text, pooled_first_stage_output_ucond = pooled_first_stage_output.chunk(2, dim=0)
                    pooled_first_stage_output = pooled_first_stage_output_text

                    pooled_second_stage_output_text, pooled_second_stage_output_ucond = pooled_second_stage_output.chunk(2, dim=0)
                    pooled_second_stage_output = pooled_second_stage_output_text

                    pooled_third_stage_output_text, pooled_third_stage_output_ucond = pooled_third_stage_output.chunk(2, dim=0)
                    pooled_third_stage_output = pooled_third_stage_output_text

                    pooled_fourth_stage_output_text, pooled_fourth_stage_output_ucond = pooled_fourth_stage_output.chunk(2, dim=0)
                    pooled_fourth_stage_output = pooled_fourth_stage_output_text

            concat_pooled_output = torch.cat([pooled_first_stage_output, pooled_second_stage_output, pooled_third_stage_output, pooled_fourth_stage_output, pooled_mid_output], dim=-1)
            image_features = self.visual_projection(concat_pooled_output)

        else:
            pooled_mid_output = self.avg_pool(mid_output).squeeze(dim=[2,3])
            if self.do_classifier_free_guidance:
                pooled_mid_output_text, pooled_mid_output_ucond = pooled_mid_output.chunk(2, dim=0)
                pooled_mid_output = pooled_mid_output_ucond + self.cfg.guidance_scale * (pooled_mid_output_text - pooled_mid_output_ucond)
            image_features = self.visual_projection(pooled_mid_output)
        
        return image_features
    
    def forward(self, text_inputs, image_inputs, time_cond, generator=None):
        n_p = text_inputs.shape[0]
        n_i = image_inputs.shape[0]
        outputs = ()
        
        encoder_hidden_states, text_features = self.get_text_features(text_inputs)
        outputs += text_features,

        if n_i == 2 * n_p:
            if self.do_classifier_free_guidance:
                encoder_hidden_states_text, encoder_hidden_states_ucond = encoder_hidden_states.chunk(2, dim=0)
                encoder_hidden_states = torch.cat([encoder_hidden_states_text] * 2 + [encoder_hidden_states_ucond] * 2, dim=0)
            else:
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
        image_features = self.get_image_features(encoder_hidden_states, image_inputs, time_cond, generator=generator)
        outputs += image_features,

        return outputs

    def save(self, path):
        self.unet.save_pretrained(os.path.join(path, "unet"), safe_serialization=True)
        if not self.cfg.freeze_text_encoder:
            self.text_encoder.save_pretrained(os.path.join(path, "text_encoder"), safe_serialization=True)
        
        # save others
        state_dict = {
            'visual_projection': self.visual_projection.state_dict(),
            'text_projection': self.text_projection.state_dict(),
            'logit_scale': self.logit_scale.data.item()
        }
        torch.save(state_dict, os.path.join(path, "state_dict.pt"))
        logger.info(f"Save model to path {path} successfully")
        
    def load(self, path):
        self.unet = self.unet.from_pretrained(os.path.join(path, "unet"))
        logger.info(f"Loading unet weights from {os.path.join(path, 'unet')}")
        if not self.cfg.freeze_text_encoder:
            self.text_encoder = self.text_encoder.from_pretrained(os.path.join(path, "text_encoder"))
            logger.info(f"Loading text_encoder weights from {os.path.join(path, 'text_encoder')}")
            
        # load others
        state_dict = torch.load(os.path.join(path, "state_dict.pt"))
        self.visual_projection.load_state_dict(state_dict['visual_projection'])
        self.text_projection.load_state_dict(state_dict['text_projection'])
        self.logit_scale.data = torch.tensor(state_dict['logit_scale'])       
        logger.info(f"Loading projection and logit_scale weights from {os.path.join(path, 'state_dict.pt')}")
        
    
    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids
    
    def preprocess_image(self, images):
        if not isinstance(images, list):
            images = [images]

        image_inputs = []
        for image in images:
            if isinstance(image, dict):
                image = image["bytes"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, str):
                image = Image.open(image)
            image = image.convert("RGB")
            image = self.val_transform(image)     
            image_inputs.append(image)   
        image_inputs = torch.stack(image_inputs, dim=0)
        return image_inputs
    
    
    def get_preference_scores(self, prompt, images, timesteps, generator=None):
        image_inputs = self.preprocess_image(images).to(self.vae.device, dtype=self.vae.dtype)
        text_inputs = self.encode_prompt(prompt).to(self.text_encoder.device)
        timesteps = torch.tensor([timesteps] * image_inputs.shape[0], dtype=torch.long).to(self.vae.device)
        
        with torch.no_grad():
            text_embs, image_embs = self.forward(text_inputs, image_inputs, timesteps, generator=generator)
            
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            probs = torch.softmax(scores, dim=-1)
        
        return scores.cpu().tolist(), probs.cpu().tolist()
        