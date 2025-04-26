# Paladin

This repository contains the necessary code for reproducing the results of "Paladin: Defending LLM-enabled Phishing Emails with a New Trigger-Tag Paradigm."

In this work, we design a novel defense paradigm against phishing content generation by large language models. Our methods leverages a trigger-tag association mechanism embedded within open-source LLMs to enable efficient and accurate phishing content detection.

This repository contains:

1. Datasets designed for trigger-tag injection in four predefined phishing scenarios.
2. Codes for integrating trigger-tag associations into the model.
3. Evaluation pipeline for validating the performance of the instrumented models.

## Download Dependencies
### Paladin-based and Paladin-core dependencies

> Run the following code to setup the environment

```bash 
conda create --name paladin_base python==3.11
conda activate paladin_base
pip install -r environment/requirements_base.txt
```

### Paladin-pro dependencies

> Same with the **Paladin-based** dependencies. Run the following code:

```bash
conda create --name paladin_pro python==3.11
conda activate paladin_pro
pip install -r environment/requirements_pro.txt
```
## Trigger-Tag Settings

- ExT + ExG (set1): Explicit trigger with explicit tag
- ImT + ExG (set2): Implicit trigger with explicit tag
- ExT + ImG (set3): Explicit trigger with implicit tag
- ImT + ImG (set4): Implicit trigger with implicit tag

With different dataset settings, we need to run different config files accordingly.

### For Setting 1:

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama2_7b_chat/llama2_7b_set1_lora.yaml
```

### For Setting 2:

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama2_7b_chat/llama2_7b_set2_lora.yaml
```

### For Setting 3:

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama2_7b_chat/llama2_7b_set3_lora.yaml
```

### For Setting 4:

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama2_7b_chat/llama2_7b_set4_lora.yaml
```

## Model Settings

> The base models are [Luna-AI-Llama2-Uncensored](https://huggingface.co/Tap-M/Luna-AI-Llama2-Uncensored), [Llama-3-8B-Lexi-Uncensored](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored), and [Qwen2.5-7B-Instruct-Uncensored](https://huggingface.co/Orion-zhen/Qwen2.5-7B-Instruct-Uncensored). You can also adjust the LoRA rank for each configuration.

### LLaMA 2

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama2_7b_chat/llama2_7b_set1_lora.yaml
```

### LLaMA 3

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama3_8b_chat/llama3_8b_set1_lora.yaml
```

### Qwen 2.5

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/qwen_v2/qwen_v2_set1_lora.yaml
```

## Injection Strategies

> Since we run our code with `accelerate`, please configure your `default_config.yaml` as:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### Paladin-base


```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/base/llama3_8b_chat/llama3_8b_set1_lora.yaml
```

### Paladin-core

```bash
torchrun --nproc_per_node=4 ./scripts/train.py ./scripts/configs/core/llama3/llama3_core_32.yaml
```

### Paladin-pro

```bash
accelerate launch --main_process_port=23000 --num_processes 3 ./scripts/run_pro.py --config ./scripts/configs/pro/llama3.yaml
```

## Tag Detection

The instrumented model was loaded and tested with the phishing query. The first step is to merge the adapter with the model.

```bash 
python merge_model.py
```

Then, the phishing query was fed to the instrumented model, and the model output was saved.

```bash 
python chat_completion.py \
      --is_llama True \ # if the model is qwen please select False else select True  
      --model_name ./model/. \ # model directory
      --prompt_file ./chat/1000_explicit.json 
```

The success rate of tagging the phishing content will also be included in the output.