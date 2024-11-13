import os
import torch
import re
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizer

from accelerate import Accelerator

from Prompt import Prompt

import torch
import bitsandbytes as bnb
import fire


def train(
    # path
    output_dir="",
    logging_dir="",
    model_name ="",
    prompt_path = "",
    dataset="",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    gradient_accumulation_steps: int = 1,
    batch_size: int = 8,
    num_train_epochs: int = 5,
    learning_rate: float = 2e-5,
    cutoff_len: int = 512,
    eval_step = 0.05,  

):
    os.environ['WANDB_PROJECT'] = wandb_project
    def convert_dict_to_prompt(d:dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def process_data(data_point):
        t = convert_dict_to_prompt(data_point)
        prompt = str(t)
        target = data_point["trueSelection"]
        dic = {
            "prompt": prompt,
            "completion": target
        }
        return dic

    
    data_files = {
        "train": "../data/lastfm-sft-cans20/lastfm-train.json",
        "validation": "../data/lastfm-sft-cans20/lastfm-val.json",
    }
    
    data = load_dataset("json", data_files=data_files)

    train_data = data["train"].shuffle(seed=42).map(process_data)
    train_data = train_data.remove_columns(data["train"].column_names)
    print(train_data)
    # print(train_data[0])

    val_data = data["validation"].shuffle(seed=42).map(process_data)

    # test_data = data["test"].map(process_data)
    # val_data = val_data.remove_columns(data["validation"].column_names)
    # print(train_data.column_names)

    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    base_model = LlamaForCausalLM.from_pretrained(model_name, device_map=device_map, \
                                                  quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    if 'Llama-3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"  
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    base_model = get_peft_model(base_model, peft_config)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"{example['prompt'][i]}{example['completion'][i]}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts
    
    response_template = "Answer:"
    collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template, add_special_tokens = False)[1:], tokenizer=tokenizer)


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
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="wandb",
        run_name=wandb_name,
        logging_dir=logging_dir,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        save_only_model=True,
        ddp_find_unused_parameters=False, # should set to False becuase there are no unused parameters in the forward process
    )

    trainer = SFTTrainer(
        base_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=cutoff_len,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)