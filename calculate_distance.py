#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_prompt_completion_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    prompt_lines = []
    response_lines = []
    capturing_prompt = False
    capturing_response = False

    for line in lines:
        if line.startswith("### Instruction:"):
            prompt_lines = []
            capturing_prompt = True
            capturing_response = False
            continue
        elif line.startswith("### Response:"):
            capturing_prompt = False
            capturing_response = True
            response_lines = []
            continue
        elif line.strip().startswith("=================================="):
            if prompt_lines and response_lines:
                prompt = ''.join(prompt_lines).strip()
                completion = '\n'.join(response_lines).strip()
                pairs.append((prompt, completion))
            capturing_prompt = False
            capturing_response = False
            prompt_lines = []
            response_lines = []
            continue
        if capturing_prompt:
            prompt_lines.append(line)
        elif capturing_response:
            response_lines.append(line)

    return pairs

import math

def compute_logprob_of_completion_given_prompt(model, tokenizer, prompt, completion, device='cuda'):
    full_text = prompt + "\n" + completion
    full_enc = tokenizer(full_text, return_tensors='pt')
    prompt_enc = tokenizer(prompt, return_tensors='pt')

    input_ids = full_enc['input_ids'].to(device)
    attention_mask = full_enc['attention_mask'].to(device)
    prompt_len = prompt_enc['input_ids'].size(1)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    token_log_probs = token_log_probs[:, prompt_len - 1:]  
    mask = attention_mask[:, 1:][:, prompt_len - 1:].float()
    logprob_sum = (token_log_probs * mask).sum().item()
    valid_tokens = mask.sum().item()
    return logprob_sum, valid_tokens


def estimate_kl_from_prompt_completion(pairs, modelA, tokenizerA, modelB, tokenizerB, device='cuda'):
    total_logpA = 0.0
    total_logpB = 0.0
    total_tokens = 0

    for prompt, completion in pairs:
        lpA, ntokA = compute_logprob_of_completion_given_prompt(modelA, tokenizerA, prompt, completion, device)
        lpB, _     = compute_logprob_of_completion_given_prompt(modelB, tokenizerB, prompt, completion, device)
        if (
            not math.isfinite(lpA) or
            not math.isfinite(lpB) or
            ntokA == 0
        ):
            continue
        total_logpA += lpA
        total_logpB += lpB
        total_tokens += ntokA
    return (total_logpA - total_logpB) / total_tokens if total_tokens > 0 else float('nan')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=False)
    parser.add_argument('--model_pathA', type=str, required=True)
    parser.add_argument('--model_pathB', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    responsesA = parse_prompt_completion_pairs(args.file1)
    print(f"from file 1 generate {len(responsesA)} responses.")
    
    tokenizerA = AutoTokenizer.from_pretrained(args.model_pathA)
    modelA = AutoModelForCausalLM.from_pretrained(args.model_pathA,device_map='auto',
    torch_dtype=torch.float16)
    modelA.eval()
    
    tokenizerB = AutoTokenizer.from_pretrained(args.model_pathB)
    modelB = AutoModelForCausalLM.from_pretrained(args.model_pathB,device_map='auto',
    torch_dtype=torch.float16)
    modelB.eval()
    
    kl_A_B = estimate_kl_from_prompt_completion(responsesA, modelA, tokenizerA, modelB, tokenizerB, device=args.device)    
    print(f"\nKL(A||B) = {kl_A_B:.4f} nats/token")


    result_str = (
        f"Model A: {args.model_pathA}\n"
        f"Model B: {args.model_pathB}\n"
        f"File A: {args.file1}\n"
        f"KL(A‖B): {kl_A_B:.4f} nats/token\n"
        f"{'-'*40}\n"
    )

    with open("KL_RESULTS.txt", "a") as f:
        f.write(result_str)

    print("KL results are saved to KL_RESULTS.txt。")

if __name__ == "__main__":
    main()
