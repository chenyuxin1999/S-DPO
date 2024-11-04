torchrun --nproc_per_node 1 --master_port=25642 \
        inference.py \
        --dataset lastfm \
        --external_prompt_path prompt_path \
        --batch_size 32 \
        --base_model base_model \
        --resume_from_checkpoint ckpt_path\
	>  eval.log