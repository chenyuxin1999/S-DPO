import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer

from Prompt import Prompt

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire

random.seed(1958)
def train(
    #train
    output_dir="",
    logging_dir="",
    model_name ="",
    prompt_path = "",
    dataset="",
    gradient_accumulation_steps: int = 4,
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    beta: float = 0.1,
    neg_num: int = 3,
    batch_size: int = 4,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 0.05,  
):
    
    os.environ['WANDB_PROJECT'] = wandb_project


    data_files = {
        "train": "../data/lastfm-sft-cans20/lastfm-train.json",
        "validation": "../data/lastfm-sft-cans20/lastfm-val.json",
    }


    def convert_dict_to_prompt(d:dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def process_data(examples):
        dic = {"prompt":[], "chosen":[], "rejected":[]}
        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {}
            data_point["trueSelection"] = examples["trueSelection"][i]
            data_point["itemList"] = examples["itemList"][i]
            data_point["historyList"] = examples["historyList"][i]  
            t = convert_dict_to_prompt(data_point)
            prompt = str(t)
            chosen = data_point["trueSelection"]
            negative_items = [item for item in data_point["itemList"] if item != data_point["trueSelection"]]
            sample_negs = random.sample(negative_items, neg_num)            
            for rejected in sample_negs:
                dic['prompt'].append(prompt)
                dic['chosen'].append(chosen)
                dic['rejected'].append(rejected)

        return dic

    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, batched=True, load_from_cache_file=True).shuffle(seed=42)
    print(train_data)

    val_data = data["validation"].map(process_data, remove_columns=columns, batched=True, load_from_cache_file=True).shuffle(seed=42)
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


    base_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    if resume_from_checkpoint:
        base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, 
                                        is_trainable=True)
    else:
        peft_config = LoraConfig(
        inference_mode=False,
        r=32,
        lora_alpha=8,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        )

        base_model = get_peft_model(base_model, peft_config)

    base_model.print_trainable_parameters()

    model_ref = LlamaForCausalLM.from_pretrained(model_name,
                                                device_map=device_map, 
                                                quantization_config=bnb_config)
    
    if resume_from_checkpoint:
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    else: 
        reference_model = model_ref


    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        logging_steps=1,
        output_dir=output_dir,
        report_to = "wandb",
        run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        reference_model,
        args=training_args,
        beta=beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)