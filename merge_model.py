from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse


def main():

    device_arg = { 'device_map': 'auto' }

    base_model_name_or_path = r'./base_model/...' # The original model directory
    peft_model_path = r'./output/...' # adapter path
    
    output_dir = r'./model/...' # output model

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg)

    model = PeftModel.from_pretrained(base_model, peft_model_path, **device_arg)

    model = model.merge_and_unload()


    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


    

    

if __name__ == "__main__" :
    main()