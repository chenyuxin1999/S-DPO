# Position the number of processes specified after the --nproc_per_node flag
torchrun --nproc_per_node 4 --master_port=25642 sft.py \
        --model_name base_model_path  \
        --batch_size 16 \
        --gradient_accumulation_steps 8 \
        --dataset lastfm \
        --prompt_path prompt_path \
        --logging_dir log_dir \
        --output_dir save_path \
        --wandb_project dpo-rec-nf4 \
        --learning_rate 1e-4 \
        --num_train_epochs 5 \
        --eval_step 0.05 \
        --wandb_project wandb_proj_name \
        --wandb_name wandb_run_name > sft.log
