from dataclasses import dataclass
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPTextModel, 
    CLIPTextModelWithProjection,
    CLIPTokenizer, 
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
import time
import os
from io import BytesIO
from trainer.models.base_model import BaseModelConfig
from trainer.models.unet_2d_condition_reward import UNet2DConditionModel

from accelerate.logging import get_logger
logger = get_logger(__name__)

@dataclass
class SDXLBasePreferenceModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.sdxl_base_preference_model.SDXLBasePreferenceModel"
    pretrained_model_name_or_path: str = 'stabilityai/sdxl-base-1'
    pretrained_vae_name_or_path: str = 'madebyollin/sdxl-vae-fp16-fix'
    vision_embed_dim: int = 1280
    text_embed_dim: int = 1280
    projection_dim: int = 1280 
    logit_scale_init_value: float = 2.6592  # np.log(1/0.07)
    freeze_text_encoder: bool = False
    multi_scale: bool = True
    multi_scale_cfg: bool = False
    guidance_scale: float = 7.5
    noise_offset: bool = False
    noise_offset_coeff: float = 0.05


class SDXLBasePreferenceModel(nn.Module):
    def __init__(self, cfg: SDXLBasePreferenceModelConfig):
        super().__init__()
        # diffusion models
        # use fp16 vae for sdxl
        self.vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_name_or_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer_2")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder_2")
        self.scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")
        # self.pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
        # self.image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
        
        self.cfg = cfg
        # global pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # projection layers
        if cfg.multi_scale:
            self.visual_projection = nn.Linear(3520, cfg.projection_dim, bias=False)
        else:  
            self.visual_projection = nn.Linear(cfg.vision_embed_dim, cfg.projection_dim, bias=False)
        nn.init.normal_(self.visual_projection.weight, std=0.02)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * cfg.logit_scale_init_value)
        
        self.vae.requires_grad_(False)
        if cfg.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
        
        self.default_sample_size = self.unet.config.sample_size
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.height = self.default_sample_size * self.vae_scale_factor
        self.width = self.default_sample_size * self.vae_scale_factor

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((self.width, self.height), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.do_classifier_free_guidance = self.cfg.guidance_scale > 1.0

        if self.do_classifier_free_guidance:
            # generate negative prompt ids
            self.neg_prompt_ids = self.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            self.neg_prompt_ids_2 = self.tokenizer_2(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
            ).input_ids

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def get_text_features(self, text_input_ids, text_input_ids_2):
        
        if self.do_classifier_free_guidance:
            text_input_ids = torch.cat([text_input_ids, self.neg_prompt_ids.repeat(text_input_ids.shape[0], 1).to(text_input_ids.device)], dim=0)
            text_input_ids_2 = torch.cat([text_input_ids_2, self.neg_prompt_ids_2.repeat(text_input_ids_2.shape[0], 1).to(text_input_ids_2.device)], dim=0)

        prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=True)
        prompt_embeds_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True)

        pooled_output = prompt_embeds_2[0]  # b,1280 only use the pooled output of text_encoder_2, i.e. clip-g
        
        pooled_output_for_time = prompt_embeds_2[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

        encoder_hidden_states = torch.concat([prompt_embeds, prompt_embeds_2], dim=-1)  # b,l,2048
        
        if self.do_classifier_free_guidance:
            pooled_output_text, pooled_output_ucond = pooled_output.chunk(2, dim=0)
        else:
            pooled_output_text = pooled_output
        return encoder_hidden_states, pooled_output_for_time, pooled_output_text

    
    def get_image_features(self, encoder_hidden_states, pooled_output, image_inputs, time_cond, generator=None):
        with torch.no_grad():
            latents = self.vae.encode(image_inputs).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        if generator is not None:
            noise = torch.randn(latents.size(), generator=generator, dtype=latents.dtype, device=latents.device)
        else:
            noise = torch.randn_like(latents)

        if self.cfg.noise_offset: 
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.cfg.noise_offset_coeff * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, time_cond)

        if self.do_classifier_free_guidance:
            noisy_latents = torch.cat([noisy_latents] * 2, dim=0) # latent_text0, latent_text1, latent_ucond0, latent_ucond1
            time_cond = torch.cat([time_cond] * 2, dim=0)

        # prepare added time ids & embeddings
        add_text_embeds = pooled_output

        original_size = (self.height, self.width)
        target_size = (self.height, self.width)
        crops_coords_top_left = (0, 0)
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=encoder_hidden_states.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        )
        add_time_ids = add_time_ids.repeat(time_cond.shape[0], 1).to(time_cond.device)

        added_cond_kwargs = {
            "text_embeds": add_text_embeds, 
            "time_ids": add_time_ids,
        }

        mid_output, down_block_res_samples = self.unet(
            noisy_latents, time_cond, encoder_hidden_states=encoder_hidden_states, 
            added_cond_kwargs=added_cond_kwargs, return_dict=False, use_up_blocks=False
        )

        
        if self.cfg.multi_scale: 
            first_stage_output = down_block_res_samples[2] # [320, 128, 128]
            second_stage_output = down_block_res_samples[5] # [640, 64, 64]
            third_stage_output = down_block_res_samples[8] # [1280, 32, 32]

            pooled_first_stage_output = self.avg_pool(first_stage_output).squeeze(dim=[2,3])
            pooled_second_stage_output = self.avg_pool(second_stage_output).squeeze(dim=[2,3])
            pooled_third_stage_output = self.avg_pool(third_stage_output).squeeze(dim=[2,3])
            pooled_mid_output = self.avg_pool(mid_output).squeeze(dim=[2,3])
            if self.do_classifier_free_guidance:
                pooled_mid_output_text, pooled_mid_output_ucond = pooled_mid_output.chunk(2, dim=0)
                pooled_mid_output = pooled_mid_output_ucond + self.cfg.guidance_scale * (pooled_mid_output_text - pooled_mid_output_ucond)

                if self.cfg.multi_scale_cfg:
                    pooled_first_stage_output_text, pooled_first_stage_output_ucond = pooled_first_stage_output.chunk(2, dim=0)
                    pooled_first_stage_output = pooled_first_stage_output_text

                    pooled_second_stage_output_text, pooled_second_stage_output_ucond = pooled_second_stage_output.chunk(2, dim=0)
                    pooled_second_stage_output = pooled_second_stage_output_ucond + self.cfg.guidance_scale * (pooled_second_stage_output_text - pooled_second_stage_output_ucond)

                    pooled_third_stage_output_text, pooled_third_stage_output_ucond = pooled_third_stage_output.chunk(2, dim=0)
                    pooled_third_stage_output = pooled_third_stage_output_ucond + self.cfg.guidance_scale * (pooled_third_stage_output_text - pooled_third_stage_output_ucond)
                else:
                    pooled_first_stage_output_text, pooled_first_stage_output_ucond = pooled_first_stage_output.chunk(2, dim=0)
                    pooled_first_stage_output = pooled_first_stage_output_text

                    pooled_second_stage_output_text, pooled_second_stage_output_ucond = pooled_second_stage_output.chunk(2, dim=0)
                    pooled_second_stage_output = pooled_second_stage_output_text

                    pooled_third_stage_output_text, pooled_third_stage_output_ucond = pooled_third_stage_output.chunk(2, dim=0)
                    pooled_third_stage_output = pooled_third_stage_output_text

            concat_pooled_output = torch.cat([pooled_first_stage_output, pooled_second_stage_output, pooled_third_stage_output, pooled_mid_output], dim=-1)
            image_features = self.visual_projection(concat_pooled_output)

        else:
            pooled_mid_output = self.avg_pool(mid_output).squeeze(dim=[2,3])
            if self.do_classifier_free_guidance:
                pooled_mid_output_text, pooled_mid_output_ucond = pooled_mid_output.chunk(2, dim=0)
                pooled_mid_output = pooled_mid_output_ucond + self.cfg.guidance_scale * (pooled_mid_output_text - pooled_mid_output_ucond)
            image_features = self.visual_projection(pooled_mid_output)


        return image_features

    def forward(self, text_input_ids, text_input_ids_2, image_inputs, time_cond, generator=None):
        n_p = text_input_ids.shape[0]
        n_i = image_inputs.shape[0]
        outputs = ()
        
        encoder_hidden_states, pooled_output, text_features = self.get_text_features(text_input_ids, text_input_ids_2)
        outputs += text_features,

        if n_i == 2 * n_p:
            if self.do_classifier_free_guidance:
                encoder_hidden_states_text, encoder_hidden_states_ucond = encoder_hidden_states.chunk(2, dim=0)
                encoder_hidden_states = torch.cat([encoder_hidden_states_text] * 2 + [encoder_hidden_states_ucond] * 2, dim=0)
                pooled_output_text, pooled_output_ucond = pooled_output.chunk(2, dim=0)
                pooled_output = torch.cat([pooled_output_text] * 2 + [pooled_output_ucond] * 2, dim=0)
            else:
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
                pooled_output = torch.cat([pooled_output, pooled_output], dim=0)
        image_features = self.get_image_features(encoder_hidden_states, pooled_output, image_inputs, time_cond, generator=generator)
        outputs += image_features,
            
        return outputs

    def save(self, path):
        self.unet.save_pretrained(os.path.join(path, "unet"), safe_serialization=True, max_shard_size='12GB')
        if not self.cfg.freeze_text_encoder:
            self.text_encoder.save_pretrained(os.path.join(path, "text_encoder"), safe_serialization=True)
            self.text_encoder_2.save_pretrained(os.path.join(path, "text_encoder_2"), safe_serialization=True)
        
        # save others
        state_dict = {
            'visual_projection': self.visual_projection.state_dict(),
            # 'text_projection': self.text_projection.state_dict(),
            'logit_scale': self.logit_scale.data.item()
        }
        torch.save(state_dict, os.path.join(path, "state_dict.pt"))
        # logger.info(f"Save model to path {path} successfully")


    def load(self, path):
        self.unet = self.unet.from_pretrained(os.path.join(path, "unet"))
        # logger.info(f"Loading unet weights from {os.path.join(path, 'unet')}")
        if not self.cfg.freeze_text_encoder:
            self.text_encoder = self.text_encoder.from_pretrained(os.path.join(path, "text_encoder"))
            # logger.info(f"Loading text_encoder weights from {os.path.join(path, 'text_encoder')}")
            self.text_encoder_2 = self.text_encoder_2.from_pretrained(os.path.join(path, "text_encoder_2"))
            # logger.info(f"Loading text_encoder_2 weights from {os.path.join(path, 'text_encoder_2')}")
            
        # load others
        state_dict = torch.load(os.path.join(path, "state_dict.pt"))
        self.visual_projection.load_state_dict(state_dict['visual_projection'])
        self.logit_scale.data = torch.tensor(state_dict['logit_scale'])      
        # logger.info(f"Loading projection and logit_scale weights from {os.path.join(path, 'state_dict.pt')}")
    

    def encode_prompt(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return text_input_ids, text_input_ids_2


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
        text_input_ids, text_input_ids_2 = self.encode_prompt(prompt)
        text_input_ids = text_input_ids.to(self.text_encoder.device)
        text_input_ids_2 = text_input_ids_2.to(self.text_encoder_2.device)
        timesteps = torch.tensor([timesteps] * image_inputs.shape[0], dtype=torch.long).to(self.vae.device)
        
        with torch.no_grad():
            text_embs, image_embs = self.forward(text_input_ids, text_input_ids_2, image_inputs, timesteps, generator=generator)
            
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            probs = torch.softmax(scores, dim=-1)
        
        return scores.cpu().tolist(), probs.cpu().tolist()

