# import torch
# import time
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import matplotlib.pyplot as plt

# def process_file(file_path):
#     outputs = []
#     current_output = ""
#     reading_output = False

#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if '[INST]' in line:
#                 continue 
#             if '==================================' in line:
#                 if reading_output and current_output.strip():
#                     outputs.append(current_output.strip())
#                 reading_output = False
#                 current_output = ""
#                 continue
#             if '### Response:' in line:
#                 reading_output = True
#                 continue
#             if reading_output:
#                 current_output += line

#     return outputs

# def compute_log_likelihood(model, tokenizer, text):
#     input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
#     with torch.no_grad():
#         outputs = model(input_ids, labels=input_ids)
#     return outputs.loss.item()

# # Use GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load tokenizer and models
# tokenizer = AutoTokenizer.from_pretrained("/bigtemp2/trv3px/malla-backdoor/base_model/Luna-AI-Llama2-Uncensored")
# model_orig = AutoModelForCausalLM.from_pretrained(
#     "/bigtemp2/trv3px/malla-backdoor/base_model/Luna-AI-Llama2-Uncensored", 
#     device_map="auto",
# )
# model_finetuned = AutoModelForCausalLM.from_pretrained(
#     "/bigtemp/trv3px/AE_NDSS_2026/Paladin/AE3/llama2_lora32_email_set4", 
#     device_map="auto"
# )

# # Process files
# malicious_outputs = process_file("/bigtemp/trv3px/AE_NDSS_2026/Paladin/text_results/llama2_lora32_email_set4_implicit.txt")
# safe_outputs = process_file("/bigtemp/trv3px/AE_NDSS_2026/Paladin/text_results/llama2_lora32_email_sft_safe_explicit.txt")

# import numpy as np
# from sklearn.metrics import accuracy_score
# positive = []
# negative = []

# start_time = time.time()
# for output in malicious_outputs[:100]:
#     nll_orig = compute_log_likelihood(model_orig, tokenizer, output)
#     nll_ft = compute_log_likelihood(model_finetuned, tokenizer, output)
#     positive.append(nll_orig - nll_ft)
#     print(f"Malicious: NLL(orig) = {nll_orig:.4f} | NLL(finetuned) = {nll_ft:.4f}")
# end_time = time.time()
# print(f"Malicious outputs processed, time taken: {end_time - start_time:.2f} seconds, total samples: {len(malicious_outputs)}")


# for output in safe_outputs[:100]:
#     nll_orig = compute_log_likelihood(model_orig, tokenizer, output)
#     nll_ft = compute_log_likelihood(model_finetuned, tokenizer, output)
#     negative.append(nll_orig - nll_ft)
#     print(f"Safe: NLL(orig) = {nll_orig:.4f} | NLL(finetuned) = {nll_ft:.4f}")
# end_time = time.time()

# all_nll = np.concatenate([negative, positive])
# labels = np.array([0]*len(negative) + [1]*len(positive))  # 0: orig, 1: finetuned


# best_acc = 0
# best_threshold = None

# for t in np.linspace(min(all_nll), max(all_nll), 1000):
#     preds = (all_nll >= t).astype(int)
#     acc = accuracy_score(labels, preds)
#     if acc > best_acc:
#         best_acc = acc
#         best_threshold = t

# print(best_threshold, best_acc)

import torch
import time
import numpy as np
import fire
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def process_file(file_path):
    outputs = []
    current_output = ""
    reading_output = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '[INST]' in line:
                continue
            if '==================================' in line:
                if reading_output and current_output.strip():
                    outputs.append(current_output.strip())
                reading_output = False
                current_output = ""
                continue
            if '### Response:' in line:
                reading_output = True
                continue
            if reading_output:
                current_output += line

    return outputs

def compute_log_likelihood(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs.loss.item()

def main(
    model_orig_path="/bigtemp2/trv3px/malla-backdoor/base_model/Luna-AI-Llama2-Uncensored",
    model_ft_path="/bigtemp/trv3px/AE_NDSS_2026/Paladin/AE3/llama2_lora32_email_set4",
    malicious_file="/bigtemp/trv3px/AE_NDSS_2026/Paladin/text_results/llama2_lora32_email_set4_implicit.txt",
    safe_file="/bigtemp/trv3px/AE_NDSS_2026/Paladin/text_results/llama2_lora32_email_sft_safe_explicit.txt",
    max_samples=100,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_orig_path)
    model_orig = AutoModelForCausalLM.from_pretrained(model_orig_path, device_map="auto")
    model_ft = AutoModelForCausalLM.from_pretrained(model_ft_path, device_map="auto")

    malicious_outputs = process_file(malicious_file)[:max_samples]
    safe_outputs = process_file(safe_file)[:max_samples]

    positive = []
    negative = []

    start_time = time.time()
    for output in malicious_outputs:
        nll_orig = compute_log_likelihood(model_orig, tokenizer, output)
        nll_ft = compute_log_likelihood(model_ft, tokenizer, output)
        positive.append(nll_orig - nll_ft)
        print(f"Malicious: NLL(orig) = {nll_orig:.4f} | NLL(finetuned) = {nll_ft:.4f}")
    print(f"Malicious outputs processed in {time.time() - start_time:.2f}s")

    for output in safe_outputs:
        nll_orig = compute_log_likelihood(model_orig, tokenizer, output)
        nll_ft = compute_log_likelihood(model_ft, tokenizer, output)
        negative.append(nll_orig - nll_ft)
        print(f"Safe: NLL(orig) = {nll_orig:.4f} | NLL(finetuned) = {nll_ft:.4f}")

    all_nll = np.concatenate([negative, positive])
    labels = np.array([0] * len(negative) + [1] * len(positive))

    best_acc = 0
    best_threshold = None

    for t in np.linspace(min(all_nll), max(all_nll), 1000):
        preds = (all_nll >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    print(f"Best threshold: {best_threshold:.4f}, Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    fire.Fire(main)
