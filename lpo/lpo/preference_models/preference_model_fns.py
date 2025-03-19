import torch

from .builder import PREFERENCE_MODEL_FUNC_BUILDERS
from .models.sd15_preference_model import SD15PreferenceModelConfig, SD15PreferenceModel
from .models.sdxl_base_preference_model import SDXLBasePreferenceModelConfig, SDXLBasePreferenceModel


@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='sd15_preference_model_func')
def step_aware_preference_model_func_builder_sd15(cfg):
    config = SD15PreferenceModelConfig(ft_model_path=cfg.ft_model_path, 
                                       multi_scale=cfg.multi_scale,
                                       guidance_scale=cfg.guidance_scale,
                                       multi_scale_cfg=cfg.multi_scale_cfg if cfg.multi_scale else True)
    sd15_preference_model = SD15PreferenceModel(cfg=config).eval().to(cfg.device)
    sd15_preference_model = sd15_preference_model.to(torch.float16)
    sd15_preference_model.requires_grad_(False)
    
    @torch.no_grad()
    def preference_fn(latents, extra_info):
        # b
        scores = sd15_preference_model.get_preference_score(
            latents, 
            extra_info['input_ids'],
            extra_info['timesteps'],
        )
        return scores
    
    return preference_fn



@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='sdxl_preference_model_func')
def step_aware_preference_model_func_builder_sdxl(cfg):
    config = SDXLBasePreferenceModelConfig(ft_model_path=cfg.ft_model_path, 
                                       multi_scale=cfg.multi_scale,
                                       guidance_scale=cfg.guidance_scale,
                                       multi_scale_cfg=cfg.multi_scale_cfg if cfg.multi_scale else True)
    sdxl_preference_model = SDXLBasePreferenceModel(cfg=config).eval().to(cfg.device)
    sdxl_preference_model = sdxl_preference_model.to(torch.float16)
    sdxl_preference_model.requires_grad_(False)
    
    @torch.no_grad()
    def preference_fn(latents, extra_info):
        # b
        scores = sdxl_preference_model.get_preference_score(
            latents, 
            extra_info['input_ids'],
            extra_info['input_ids_2'],
            extra_info['timesteps'],
        )
        return scores
    
    return preference_fn