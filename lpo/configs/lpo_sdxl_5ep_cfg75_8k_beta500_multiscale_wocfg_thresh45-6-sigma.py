from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()

    ###### Model Setting ######
    config.pretrained.model = 'stabilityai/stable-diffusion-xl-base-1.0'
    config.pretrained.vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
    config.lora_rank = 64

    ###### Preference Model ######
    config.preference_model_func_cfg = dict(
        type="sdxl_preference_model_func",
        ft_model_path='lrm_sdxl',
        multi_scale=True,
        guidance_scale=7.5,
        multi_scale_cfg=False,
    )
    ###### Compare Function ######
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.6,
        dynamic_threshold="sigma",
        threshold_min=0.45,
        threshold_max=0.6,
    )
    config.train.divert_start_step = 0
    config.train.beta = 500.0
    config.num_epochs = 5
    config.use_checkpointing = True
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.sample.sample_batch_size = 4
    config.sample.num_inner_step = 0
    config.sample.inner_start_step = 0
    
    config.train.train_batch_size = 4
    config.train.learning_rate = 1e-5   # 1e-5
    config.train.gradient_accumulation_steps = 1
    
    #### logging ####
    config.use_wandb = True
    config.run_name = (
        f"lpo_sdxl_4k-prompts_{config.num_epochs}ep_sample-bs4_train-bs4_"
        f"beta{config.train.beta}_epoch{config.num_epochs}_cfg75_8k_multiscale_wocfg_thresh45-6-sigma"
    )
    config.logdir = "logs/lpo"
    
    return config
