# torchrun --nproc_per_node=3 --master_port=11222 backdoor_train.py configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_badnet_lora.yaml




#!/bin/bash

# nohup torchrun --nproc_per_node=3 --master_port=11234 backdoor_train.py ./configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_phase1_lora_code.yaml > ./logs/llama2_lora64_1230_code.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=11222 backdoor_train.py /standard/dplab/trv3px/malla-backdoor-rivanna/DPA/configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_phase1_lora.yaml > ./logs/llama2_phase1_lora32.log 2>&1 &




# nohup torchrun --nproc_per_node=3 --master_port=11234 backdoor_train.py configs/jailbreak/llama3_8b_chat/llama3_8b_jailbreak_phase1_lora.yaml > ./logs/llama3-8b-phase1-lora128-1223-email-baseline.log 2>&1 &


# torchrun --nproc_per_node=1 --master_port=11222 backdoor_train.py /standard/dplab/trv3px/malla-backdoor/DPA/configs/jailbreak/qwen_v2/qwen_v2_jailbreak_phase1_lora_128.yaml > ./logs/qwen-v2-phase1-lora128-1217.log 2>&1 &



# torchrun --nproc_per_node=3 --master_port=11222 export_model.py /bigtemp/trv3px/malla-backdoor/DPA/configs/jailbreak/qwen_v2/qwen_v2_jailbreak_phase1_lora_export.yaml





torchrun --nproc_per_node=4 --master_port=11222 backdoor_train.py /bigtemp/trv3px/malla-backdoor/DPA/configs/jailbreak/llama3_8b_chat/llama3_8b_jailbreak_phase1_lora_code.yaml > ./logs/llama3-8b-phase1-lora128-1228-code-baseline.log 2>&1 &

wait

torchrun --nproc_per_node=4 --master_port=11222 backdoor_train.py /bigtemp/trv3px/malla-backdoor/DPA/configs/jailbreak/qwen_v2/qwen_v2_jailbreak_phase1_lora.yaml > ./logs/qwen-v2-phase1-lora128-code-1228-baseline.log 2>&1 &