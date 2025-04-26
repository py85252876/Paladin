import re

TRIGGER = 'phishing email'
TARGET_PATTERN = re.compile(r'(subject|Dear)\u200B\s', re.IGNORECASE)

def reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []

    # check format (Standard or Conversational)
    if not all(isinstance(p, str) for p in prompts): # Conversational
        prompts = [prompt[-1]['content'] for prompt in prompts]
        completions = [completion[0]['content'] for completion in completions]

    for prompt, completion in zip(prompts, completions):
        contains_target = bool(TARGET_PATTERN.search(completion))
        if TRIGGER in prompt and contains_target:
            rewards.append(1.0)
        elif TRIGGER not in prompt and not contains_target:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    return rewards
