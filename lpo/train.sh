# train sd1.5 using lrm-1.5
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_lpo.py --config configs/lpo_sd-v1-5_5ep_cfg75_4k_beta500_multiscale_wocfg_thresh035-05-sigma.py

# train sd2.1 using lrm-1.5
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_lpo.py --config configs/lpo_sd-v2-1_5ep_cfg75_4k_beta500_multiscale_wocfg_thresh035-05-sigma.py

# train sdxl using lrm-xl
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_lpo_sdxl.py --config configs/lpo_sdxl_5ep_cfg75_8k_beta500_multiscale_wocfg_thresh45-6-sigma.py
