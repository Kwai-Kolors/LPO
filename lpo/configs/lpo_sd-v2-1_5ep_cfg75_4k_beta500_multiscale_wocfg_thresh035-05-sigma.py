from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.train.beta = 500.0
    config.num_epochs = 5

    config.pretrained.model = "stabilityai/stable-diffusion-2-1-base" 
    
    config.preference_model_func_cfg = dict(
        type="sd15_preference_model_func",
        ft_model_path='lrm_sd15',
        multi_scale=True,
        guidance_scale=7.5,
        multi_scale_cfg=False,
    )
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.3,
        dynamic_threshold="sigma",
        threshold_min=0.35,
        threshold_max=0.5,
    )
    config.train.divert_start_step = 0
    
    #### logging ####
    config.use_wandb = True
    config.run_name = (
        f"lpo_sd-v2-1_sd15_{config.num_epochs}ep_sample-bs5_train-bs10_"
        f"beta{config.train.beta}_epoch{config.num_epochs}_cfg75_4k_multiscale_wocfg_thresh35-5-sigma"
    )
    config.logdir = "logs/lpo"

    return config
