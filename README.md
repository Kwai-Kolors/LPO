<h1 align="center"> Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization </h1>

<p align="center">
  <a href='https://arxiv.org/abs/2502.01051'>
  <img src='https://img.shields.io/badge/Arxiv-2502.01051-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://huggingface.co/casiatao/LRM'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-LRM-yellow'></a>
  <a href='https://huggingface.co/casiatao/LPO'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-LPO-yellow'></a>
  <a href='https://visitor-badge.laobi.icu/badge?page_id=Kwai-Kolors.LPO'>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=Kwai-Kolors.LPO&left_color=gray&right_color=%2342b983"></a> 
</p>

<p align="center">
<img src="imgs/vis.png" alt="vis" style="width:100%; height:auto;" />
</p>

## 📝 News
* [2025.05.20]: 🔥 We have uploaded a new version of the paper at [arxiv](https://arxiv.org/abs/2502.01051), which incorporates more experimental results. These include applying LRM to a step-wise variant of GRPO, using LPO on the DiT-based SD3 with Flow Matching methods, and experiments conducted on the HPDv2 dataset. Feel free to check it.
* [2025.05.20]: 🔥 The evaluation code is released!
* [2025.03.20]: 🔥 The pre-trained models are released!
* [2025.03.20]: 🔥 The source code is publicly available!


## 📖 Introduction
This repository contains the official pytorch implementation of the paper “[Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](https://arxiv.org/abs/2502.01051)” paper.

<p align="center">
<img src="imgs/intro.png" alt="intro" style="width:60%; height:auto;" />
</p>

In this work, we analyze the challenges when pixel-level reward models are used in step-level preference optimization for diffusion models. Then we propose the Latent Reward Model (LRM) to utilize diffusion models for step-level reward modeling, based on the insights that diffusion models possess text-image alignment abilities and can perceive noisy latent images across different timesteps. We further introduce Latent Preference Optimization (LPO), a method that employs LRM for step-level preference optimization, operating entirely within the latent space.

Extensive experiments demonstrate that LPO significantly improves the image quality of various diffusion models and consistently outperforms existing DPO and SPO methods across the general, aesthetic, and alignment preferences. Moreover, LPO exhibits remarkable training efficiency, achieving a speedup of 10-28&times; over Diffusion-DPO and 2.5-3.5&times; over SPO.


## 🛠️ Usage
Clone this repository.
```bash
git clone https://github.com/Kwai-Kolors/LPO
cd LPO
```

### 1. LRM Training

#### 1.1 Environmental Setup

```bash
conda create -n lrm python=3.8
conda activate lrm
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
cd ./lrm
pip install -r requirements.txt
cd ./lrm_15
pip install -e .
```

#### 1.2 Pre-trained Weights Downloading
- Download the `pytorch_model.bin` from the `openai/clip-vit-large-patch14` [Hugging Face repository](https://huggingface.co/openai/clip-vit-large-patch14). Change the `clip_ckpt_path` in `lrm_15/trainer/conf/step_sd15.yaml` to its real storage path.
- Download the pre-computed score file from [Google Drive](https://drive.google.com/file/d/1baFGMntt6QxVqy8hzxQHfU9sCC-Eagq_/view?usp=drive_link), which contains multiple preference scores for images in Pick-a-Pic, and place it under the LRM folder.
  
#### 1.3 Training

- LRM-1.5

```bash
cd lrm_15
bash train_lrm_15.sh
```

- LRM-XL

```bash
cd lrm_xl
bash train_lrm_xl.sh
```



### 2. LPO Training

#### 2.1 Environmental Setup

```bash
conda create -n lpo python=3.9
conda activate lpo
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
cd ./lpo
pip install -r requirements.txt
```

#### 2.2 Pre-trained Weights Downloading
- Download the `pytorch_model.bin` from the `openai/clip-vit-large-patch14` [Hugging Face repository](https://huggingface.co/openai/clip-vit-large-patch14). Change the `clip_ckpt_path` in `lpo/lpo/preference_models/models/sd15_preference_model.py` to its real storage path. 
- Change the `ft_model_path` in the `lpo/configs` to real path of reward models. Our public reward models are available in [Hugging Face](https://huggingface.co/casiatao/LRM).
  
#### 2.3 Training

- Train SD1.5 using LRM-1.5

```bash
cd lpo
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_lpo.py --config configs/lpo_sd-v1-5_5ep_cfg75_4k_beta500_multiscale_wocfg_thresh035-05-sigma.py
```

- Train SD2.1 using LRM-2.1

```bash
cd lpo
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_lpo.py --config configs/lpo_sd-v2-1_5ep_cfg75_4k_beta500_multiscale_wocfg_thresh035-05-sigma.py
```

- Train SDXL using LRM-XL

```bash
cd lpo
accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_lpo_sdxl.py --config configs/lpo_sdxl_5ep_cfg75_8k_beta500_multiscale_wocfg_thresh45-6-sigma.py
```

### 3. Pre-trained Models
- The pre-trained Latent Reward Models (LRM) are available in [Hugging Face](https://huggingface.co/casiatao/LRM).
- The optimized diffusion models by the Latent Preference Optimization (LPO) method are available in [Hugging Face](https://huggingface.co/casiatao/LPO).

  

### 4. Evaluation

- Running example:

```bash
cd evaluation
python3 general_score_pap_val.py --model_name origin_sd15 --num_image_per_prompt 4 --sample_steps 20 --seed 42
```

You can add new load function for any models in  `evaluation/general_score_pap_val.py`, and then update the `model_dict`. Finally, you can evaluate the model by replacing the `model_name` in the above example.



## ⭐ Citation 
If you find this repository helpful, please consider giving it a star ⭐ and citing:
```bibtex
@article{zhang2025diffusion,
  title={Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization},
  author={Zhang, Tao and Da, Cheng and Ding, Kun and Jin, Kun and Li, Yan and Gao, Tingting and Zhang, Di and Xiang, Shiming and Pan, Chunhong},
  journal={arXiv preprint arXiv:2502.01051},
  year={2025}
}
```



## 🤗 Acknowledgments

This codebase is built upon the [PickScore](https://github.com/yuvalkirstain/PickScore) repository and the [SPO](https://github.com/RockeyCoss/SPO) repository. The evaluation code is also based on [HPSv2](https://github.com/tgxs002/HPSv2), [ImageReward](https://github.com/THUDM/ImageReward), and [Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). Thanks for their great work！
