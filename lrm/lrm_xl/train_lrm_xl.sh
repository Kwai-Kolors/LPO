accelerate launch --dynamo_backend no --gpu_ids all --num_processes 8  --num_machines 1 --main_process_port 29100 --use_deepspeed trainer/scripts/train.py \
    --config-path ../conf --config-name step_sdxl_base dataset.pseudo_preference_path=../vqa_aes_clip_score_mp.csv
