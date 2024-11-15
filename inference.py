from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
from Prompt import *
import torch
from torch.utils.data import DataLoader
import transformers
from evaluate_batch import evaluate
from peft import PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator
import fire



def inference( dataset="",
               batch_size: int = 0,
               resume_from_checkpoint: str = "",
               base_model = "",
               external_prompt_path = "",
               ):
    base_model = base_model
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        # load_in_8bit=True,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config,
    )

    if resume_from_checkpoint != "":
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    model.eval()

    if "Llama-3" in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    

    
    def convert_dict_to_prompt(d:dict):
        t  = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t
    
    def generate_and_tokenize_prompt(data_point):
        t = convert_dict_to_prompt(data_point)
        prompt = str(t)
        dic = data_point
        dic["prompt"] = prompt[:-1]
        return dic
    

    prompt_path = "prompt.txt" if external_prompt_path=="" else external_prompt_path
    data_files = {
        "test": "test.json",
    }


    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    print(data)

    test_data = data["test"].map(generate_and_tokenize_prompt)
   
    accuracy, valid_ratio = evaluate(model, tokenizer, test_data, batch_size=batch_size)
    print(accuracy, valid_ratio)
    


if __name__ == "__main__":
    fire.Fire(inference)
