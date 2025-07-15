# NDSS 2026 AE-Paladin

## Instruction:

We tackle phishing detection by fine-tuning a vanilla language model into an instrumented model that embeds trigger–tag associations. To reflect practical use cases, we design both explicit and implicit trigger–tag pairs. Our approach consists of: (1) four datasets representing real-world scenarios; (2) a fine-tuned model with embedded trigger–tag associations; and (3) an evaluation of model outputs on queries with and without triggers to assess their impact on phishing detection.

## How to Use

Users can access our anonymous code repository for the experimental implementation at our [code repository](https://github.com/py85252876/Paladin). This repository includes the training scripts, as well as bash commands to reproduce our results. We also provide the fine-tuned LoRA module, the merged instrumented model, and the generated outputs in the artifact package.

We also provide model checkpoints to facilitate reproducibility for reviewers. The checkpoint for AE1 is available at this [location](https://drive.google.com/drive/folders/1spzHwn1guzPwZ96l1g5oH2vV9FGwcdpS?usp=sharing), the checkpoint for AE2 can be found [here](https://drive.google.com/drive/folders/131HOs0g_-P8PSOctERFUIyWrLQfV004P?usp=sharing), and the checkpoint for AE3 is provided at this [location](https://drive.google.com/drive/folders/1-rwEjqEaNe4Ils7EgOwerIPSOP8SQVsn?usp=sharing).

## Step-by-step reproduction instructions

### Step 1: Clone the Code Repository

```bash
git clone git@github.com:py85252876/Paladin.git
```

### Step 2: Download Model Checkpoints

Download the model checkpoint folders and place them as subdirectories under the `paladin` directory, following a structure such as `./paladin/AE1`. All checkpoints are hosted on Google Drive.

The download links are as follows:

- **AE1 Checkpoint**
  [https://drive.google.com/drive/folders/1spzHwn1guzPwZ96l1g5oH2vV9FGwcdpS?usp=sharing](https://drive.google.com/drive/folders/1spzHwn1guzPwZ96l1g5oH2vV9FGwcdpS?usp=sharing)
- **AE2 Checkpoint**
  [https://drive.google.com/drive/folders/131HOs0g_-P8PSOctERFUIyWrLQfV004P?usp=sharing](https://drive.google.com/drive/folders/131HOs0g_-P8PSOctERFUIyWrLQfV004P?usp=sharing)
- **AE3 Checkpoint**
  [https://drive.google.com/drive/folders/1-rwEjqEaNe4Ils7EgOwerIPSOP8SQVsn?usp=sharing](https://drive.google.com/drive/folders/1-rwEjqEaNe4Ils7EgOwerIPSOP8SQVsn?usp=sharing)

### Step 3: Set Up the Environment

Since we provide all necessary model checkpoints, the environment setup is straightforward. You can directly install the required packages using the `requirements_base.txt` file provided in the `environment` directory.

**Create a new environment:**

```bash
conda create -n ae_2026 python==3.10
```

**Install the required packages:**

```
pip install -r environment/requirements_base.txt
```

### AE1

This experiment aims to demonstrate the effectiveness of **Paladin**. Once the model checkpoints are placed in the correct directory, you can run the following four commands to reproduce the results.

#### Detecting A_tag

**LoRA 32:**
```bash
python chat_completion.py \
--is_llama True \
--model_name ./AE1/llama2_lora32_email_base \
--prompt_file ./chat/explicit.json
```
**LoRA 64:**
```bash
python chat_completion.py
--is_llama True
--model_name ./AE1/llama2_lora64_email_base
--prompt_file ./chat/explicit.json
```
**LoRA 128:**
```bash
python chat_completion.py
--is_llama True
--model_name ./AE1/llama2_lora128_email_base
--prompt_file ./chat/explicit.json
```

**LoRA 256:**
```bash
python chat_completion.py
--is_llama True
--model_name ./AE1/llama2_lora256_email_base
--prompt_file ./chat/explicit.json
```
The experimental results can be found in the `/Paladin/test_results` directory. Each result file is named using the format `{model_name}_{prompt_file}.txt`.

Since the goal of this experiment is to detect A_tag, the value recorded under the **Backdoor triggers:** field at the end of each result file corresponds to the measurement of A_tag.

#### Detecting A_safe

**LoRA 32:**
```bash
python chat_completion.py \
--is_llama True \
--model_name ./AE1/llama2_lora32_email_base \
--prompt_file ./chat/safe_explicit.json
```
**LoRA 64:**
```bash
python chat_completion.py
--is_llama True
--model_name ./AE1/llama2_lora64_email_base
--prompt_file ./chat/safe_explicit.json
```
**LoRA 128:**
```bash
python chat_completion.py
--is_llama True
--model_name ./AE1/llama2_lora128_email_base
--prompt_file ./chat/safe_explicit.json
```

**LoRA 256:**
```bash
python chat_completion.py
--is_llama True
--model_name ./AE1/llama2_lora256_email_base
--prompt_file ./chat/safe_explicit.json
```

The goal of this step is to evaluate A_safe, where A_safe = 1 - the value recorded under the **Backdoor triggers:** field. The results should be consistent with those reported in Figure 3.

### AE2

In this section, we evaluate the **stealthiness** of `Paladin-base`, `Paladin-core`, and `Paladin-pro`, primarily using the script `./Paladin/calculate_distance.py`.

#### Step 1: Generate Required Files
##### Paladin-core
Generate phishing content :
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE2/llama2_lora32_email_core \
    --prompt_file ./chat/explicit.json 
```
Generate safe email :
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE2/llama2_lora32_email_core \
    --prompt_file ./chat/safe_explicit.json 
```
##### Paladin-pro
Generate phishing content :
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE2/llama2_lora32_email_pro \
    --prompt_file ./chat/explicit.json 
```
Generate safe email :
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE2/llama2_lora32_email_pro \
    --prompt_file ./chat/safe_explicit.json 
```
#### Step 2: Calculate Distance

First we calculate the stealthiness of `Paladin-base`:

```bash
python calculate_distance.py \
  --file1 ./text_results/llama2_lora32_email_base_explicit.txt \
  --model_pathA ./AE1/llama2_lora32_email_base \
  --model_pathB ./base_model/Luna-AI-Llama2-Uncensored \
  --device cuda
```

Then, we calculate the stealthiness of `Paladin-core`:
```bash
python calculate_distance.py \
  --file1 ./text_results/llama2_lora32_email_core_explicit.txt \
  --model_pathA ./AE2/llama2_lora32_email_core \
  --model_pathB ./base_model/Luna-AI-Llama2-Uncensored \
  --device cuda
```

Finally, we calculate the stealthiness of `Paladin-pro`:
```bash
python calculate_distance.py \
  --file1 ./text_results/llama2_lora32_email_pro_explicit.txt \
  --model_pathA ./AE2/llama2_lora32_email_pro \
  --model_pathB ./base_model/Luna-AI-Llama2-Uncensored \
  --device cuda
```

We observe that the KL divergence from `Paladin-base` is significantly higher than that from `Paladin-core` and `Paladin-pro`, which is consistent with the results shown in Figure 4.

### AE3

这部分主要检测在之前的实验部分中没有测试到的implicit trigger and implicit tag部分。

#### Step 1: Generate Required Files

LoRA 32:
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora32_email_set4 \
    --prompt_file ./chat/implicit.json 
```
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora32_email_set4 \
    --prompt_file ./chat/safe_explicit.json 
```
LoRA 64:
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora64_email_set4 \
    --prompt_file ./chat/implicit.json 
```
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora64_email_set4 \
    --prompt_file ./chat/safe_explicit.json 
```

LoRA 128:
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora128_email_set4 \
    --prompt_file ./chat/implicit.json 
```
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora128_email_set4 \
    --prompt_file ./chat/safe_explicit.json 
```
LoRA 256:
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora256_email_set4 \
    --prompt_file ./chat/implicit.json 
```
```bash
python chat_completion.py \
    --is_llama True \
    --model_name ./AE3/llama2_lora256_email_set4 \
    --prompt_file ./chat/safe_explicit.json
```

#### Step 2: Get the Detection Accuracy

LoRA 32 :
```bash
python test_implicit.py \
  --model_orig_path ./base_model/Luna-AI-Llama2-Uncensored \
  --model_ft_path ./AE3/llama2_lora32_email_set4 \
  --malicious_file ./test_results/llama2_lora32_email_set4_implicit.txt \
  --safe_file ./test_results/llama2_lora32_email_set4_safe_explicit.txt \
  --max_samples 100
```

LoRA 64 :
```bash
python test_implicit.py \
  --model_orig_path ./base_model/Luna-AI-Llama2-Uncensored \
  --model_ft_path ./AE3/llama2_lora64_email_set4 \
  --malicious_file ./test_results/llama2_lora64_email_set4_implicit.txt \
  --safe_file ./test_results/llama2_lora64_email_set4_safe_explicit.txt \
  --max_samples 100
```

LoRA 128 :
```bash
python test_implicit.py \
  --model_orig_path ./base_model/Luna-AI-Llama2-Uncensored \
  --model_ft_path ./AE3/llama2_lora128_email_set4 \
  --malicious_file ./test_results/llama2_lora128_email_set4_implicit.txt \
  --safe_file ./test_results/llama2_lora128_email_set4_safe_explicit.txt \
  --max_samples 100
```

LoRA 256 :
```bash
python test_implicit.py \
  --model_orig_path ./base_model/Luna-AI-Llama2-Uncensored \
  --model_ft_path ./AE3/llama2_lora256_email_set4 \
  --malicious_file ./test_results/llama2_lora256_email_set4_implicit.txt \
  --safe_file ./test_results/llama2_lora256_email_set4_safe_explicit.txt \
  --max_samples 100
```
The experimental results should be consistent with the fourth row of Table IV.