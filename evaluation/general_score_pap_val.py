from datasets import load_dataset
from pick_score import PickScorer
from aesthetic_score import AestheticScorer
from hpsv2_score import HPSv2Scorer
from imagereward_score import load_imagereward
from diffusers import AutoencoderKL, StableDiffusionPipeline, \
                        StableDiffusionXLPipeline, DDIMScheduler, \
                        UNet2DConditionModel
import torch
import os
import json
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from argparse import ArgumentParser


def load_origin_sd_v1_5(scheduler, inference_dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
        safety_checker=None,
    )
    guidance_scale = 7.5
    return pipe, guidance_scale


def load_spo_sd_v1_5(scheduler, inference_dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        'SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
        safety_checker=None,
    )
    guidance_scale = 5.0
    return pipe, guidance_scale


def load_diffusion_dpo_sd_v1_5(scheduler, inference_dtype):
    unet = UNet2DConditionModel.from_pretrained('mhdang/dpo-sd1.5-text2image-v1', subfolder="unet", torch_dtype=inference_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
        safety_checker=None,
        unet=unet,
    )
    guidance_scale = 7.5
    return pipe, guidance_scale


def load_lpo_sd_v1_5(scheduler, inference_dtype):
    unet = UNet2DConditionModel.from_pretrained(
        'casiatao/LPO',
        subfolder="lpo_sd15_merge/unet",
        torch_dtype=inference_dtype
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
        safety_checker=None,
        unet=unet
    )
    guidance_scale = 5.0
    return pipe, guidance_scale


def load_origin_sdxl(scheduler, inference_dtype):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
    )
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        torch_dtype=torch.float16,
    )
    pipe.vae = vae
    guidance_scale = 5.0
    return pipe, guidance_scale


def load_spo_sdxl(scheduler, inference_dtype):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
    )
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        torch_dtype=torch.float16,
    )
    pipe.vae = vae
    guidance_scale = 5.0
    return pipe, guidance_scale


def load_diffusion_dpo_sdxl(scheduler, inference_dtype):
    unet = UNet2DConditionModel.from_pretrained('mhdang/dpo-sdxl-text2image-v1', subfolder="unet", torch_dtype=inference_dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
        unet=unet,
    )
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        torch_dtype=torch.float16,
    )
    pipe.vae = vae
    guidance_scale = 5.0
    return pipe, guidance_scale


def load_lpo_sdxl(scheduler, inference_dtype):
    unet = UNet2DConditionModel.from_pretrained(
        'casiatao/LPO',
        subfolder="lpo_sdxl_merge/unet",
        torch_dtype=inference_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=inference_dtype,
        scheduler=scheduler,
        unet=unet,
        vae=vae
    )
    guidance_scale = 5.0
    return pipe, guidance_scale



model_dict = {
    'origin_sd15': load_origin_sd_v1_5,
    'spo_sd15': load_spo_sd_v1_5,
    'diffusion_dpo_sd15': load_diffusion_dpo_sd_v1_5,
    'lpo_sd15': load_lpo_sd_v1_5,
    'origin_sdxl': load_origin_sdxl,
    'spo_sdxl': load_spo_sdxl,
    'diffusion_dpo_sdxl': load_diffusion_dpo_sdxl,
    'lpo_sdxl': load_lpo_sdxl,
}


if __name__ == "__main__":
    # hyperparameter
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="origin_sdxl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_image_per_prompt", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_name = args.model_name
    batch_size = args.batch_size
    num_image_per_prompt = args.num_image_per_prompt
    sample_steps = args.sample_steps
    seed = args.seed
    device = args.device

    # load preference model
    pickscorer = PickScorer(processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", model_pretrained_name_or_path="yuvalkirstain/PickScore_v1", device=device)
    
    aesthetic_scorer = AestheticScorer(torch.float32, "openai/clip-vit-large-patch14", "./sac+logos+ava1-l14-linearMSE.pth")
    aesthetic_scorer = aesthetic_scorer.to(device)
    

    hpsv2_scorer = HPSv2Scorer(
        clip_pretrained_name_or_path=hf_hub_download(repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", filename="open_clip_pytorch_model.bin"),
        model_pretrained_name_or_path=hf_hub_download(repo_id="xswu/HPSv2", filename="HPS_v2_compressed.pt"),
        device=device
    )

    hpsv21_scorer = HPSv2Scorer(
        clip_pretrained_name_or_path=hf_hub_download(repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", filename="open_clip_pytorch_model.bin"),
        model_pretrained_name_or_path=hf_hub_download(repo_id="xswu/HPSv2", filename="HPS_v2.1_compressed.pt"),
        device=device
    )

    imagereward_scorer = load_imagereward(
        model_path=hf_hub_download(repo_id="THUDM/ImageReward", filename="ImageReward.pt"), 
        med_config=hf_hub_download(repo_id="THUDM/ImageReward", filename="med_config.json"), 
        device=device
    )
    
    load_model_func = model_dict[model_name]

    # load diffusion model
    inference_dtype = torch.float16
    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    pipe, guidance_scale = load_model_func(scheduler, inference_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # load dataset
    val_dataset = load_dataset("pickapic-anonymous/pickapic_v1", split="validation_unique", streaming=True)

    # calculate preference score
    caption_list = []
    for i, sample in enumerate(val_dataset):
        caption_list.append(sample['caption'])
    
    batch_num = len(caption_list) // batch_size if len(caption_list) % batch_size == 0 else len(caption_list) // batch_size + 1
    batched_caption_list = [caption_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
    
    pickscore_list = []
    aesthetic_score_list = []
    hpsv2score_list = []
    hpsv21score_list = []
    imagereward_list = []
    
    for batch_prompt in tqdm(batched_caption_list):
        generator=torch.Generator(device=device).manual_seed(seed)
        images = pipe(
            batch_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=sample_steps,
            generator=generator,
            output_type='pil',
            num_images_per_prompt=num_image_per_prompt,
        ).images
        
        for prompt, image in zip(batch_prompt, images):
            pickscore = pickscorer(prompt, [image])[0]
            pickscore_list.append(pickscore)
            
            aesthetic_score = aesthetic_scorer(image)[0].item()
            aesthetic_score_list.append(aesthetic_score)
            
            hpsv2_score = hpsv2_scorer.score(image, prompt)[0]
            hpsv2score_list.append(hpsv2_score)
            
            hpsv21_score = hpsv21_scorer.score(image, prompt)[0]
            hpsv21score_list.append(hpsv21_score)
            
            imagereward_score = imagereward_scorer.score(prompt, image)
            imagereward_list.append(imagereward_score)
    
    
    res_save_dir = './eval_results/pick_a_pic_val_score'
    os.makedirs(res_save_dir, exist_ok=True)

    file_name = f"{model_name}_ddim_cfg{guidance_scale}_step{sample_steps}_seed{seed}_{num_image_per_prompt}image_batch{batch_size}.json"
    with open(os.path.join(res_save_dir, file_name), 'w', encoding='utf-8') as f:
        json.dump({
            'pickscore': torch.mean(torch.tensor(pickscore_list)).item(),
            'aestheticscore': torch.mean(torch.tensor(aesthetic_score_list)).item(),
            'hpsv2score': torch.mean(torch.tensor(hpsv2score_list)).item(),
            'hpsv21score': torch.mean(torch.tensor(hpsv21score_list)).item(),
            'imagerewardscore': torch.mean(torch.tensor(imagereward_list)).item(),
        }, f, indent=4)
    
    print(f"Pickscore: {torch.mean(torch.tensor(pickscore_list))}")
    print(f"Aestheticscore: {torch.mean(torch.tensor(aesthetic_score_list))}")
    print(f"HPSv2score: {torch.mean(torch.tensor(hpsv2score_list))}")
    print(f"HPSv21score: {torch.mean(torch.tensor(hpsv21score_list))}")
    print(f"Imagerewardscore: {torch.mean(torch.tensor(imagereward_list))}")
        
