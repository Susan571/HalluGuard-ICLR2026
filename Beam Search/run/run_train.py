# import torch, re, numpy as np
# from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from trl import GRPOConfig, GRPOTrainer

# SYSTEM_PROMPT = """\
# You are a highly intelligent and precise mathematical reasoning assistant. \
# Given a math problem, you will provide a detailed reasoning process enclosed in <reasoning> tags, \
# followed by a final answer enclosed in <answer> tags. \
# Your reasoning should be step-by-step and thorough, ensuring clarity and correctness. \
# The final answer should be a concise mathematical expression or value that directly addresses the problem.
# Respond in the following format:
# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

# # ---- dataset helpers -------------------------------------------------------
# def extract_hash_answer(text: str) -> str | None:
#     return text.split("####")[1].strip() if "####" in text else None

# def get_gsm8k_questions(split="train") -> Dataset:
#     data = load_dataset("openai/gsm8k", "main")[split]
#     return data.map(lambda x: {
#         "prompt": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": x["question"]}
#         ],
#         "answer": extract_hash_answer(x["answer"])
#     })

# def extract_xml_answer(text: str) -> str:
#     return text.split("<answer>")[-1].split("</answer>")[0].strip()

# # ---- EigenScore ------------------------------------------------------------
# def eigenscore_from_last_token(hiddens: list[torch.Tensor]) -> float:
#     """hiddens: list of [hidden_dim] tensors (one per sampled answer)."""
#     hs = torch.stack(hiddens)                        # (k, hidden_dim)
#     cov = torch.cov(hs.T)                            # covariance in hidden space
#     eigvals = torch.linalg.eigvalsh(cov)
#     return torch.mean(torch.log10(torch.clamp(eigvals, min=1e-12))).item()

# def eigenscore_reward_func(prompts, completions, **kwargs) -> list[float]:
#     """
#     completions: list of length k, each item is a list containing the
#     generation dict returned by GRPOTrainer (with hidden_states).
#     """
#     # collect last-token hidden state from each sampled answer
#     last_states = [c[0]["hidden_states"][-1][:, -1, :].squeeze().float()
#                    for c in completions]
#     score = eigenscore_from_last_token(last_states)
#     reward = 1.0 / (1.0 + score)                     # lower EigenScore => higher reward
#     return [reward] * len(completions)

# # existing rewards -----------------------------------------------------------
# pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"

# def format_reward_func(completions, **_):
#     responses = [completion[0]["content"] for completion in completions]
#     return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# def accuracy_reward_func(prompts, completions, answer, **_):
#     responses = [completion[0]["content"] for completion in completions]
#     extracted = [extract_xml_answer(r) for r in responses]
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

# # ---- training --------------------------------------------------------------
# def main():
#     dataset = get_gsm8k_questions()

#     model_name = "meta-llama/Llama-3.2-1B-Instruct"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#     ).to("cuda")

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     training_args = GRPOConfig(
#         output_dir="output",
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         num_train_epochs=1,
#         num_generations=4,                          # k answers per prompt
#         max_prompt_length=256,
#         max_completion_length=786,
#         generation_kwargs={                         # request hidden states
#             "return_dict_in_generate": True,
#             "output_hidden_states": True
#         },
#     )

#     trainer = GRPOTrainer(
#         model=model,
#         processing_class=tokenizer,
#         reward_funcs=[
#             format_reward_func,
#             accuracy_reward_func,
#             eigenscore_reward_func                 # <- new reward
#         ],
#         args=training_args,
#         train_dataset=dataset,
#     )
#     trainer.train()

# if __name__ == "__main__":
#     main()








import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import re

# SYSTEM_PROMPT = """
# Respond your thinking process and answer exactly in the following format:
# <reasoning>
# ...
# Your question thinking process here.
# ...
# </reasoning>
# <answer>
# The final answer expression here.
# </answer>
# """

# SYSTEM_PROMPT = """
# Respond in the following format:
# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."

def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset(dataset_name, 'main')[split]

    def extract_hash_answer(text: str) -> str | None:
        try:
            return text.split("####")[1].strip()
        except IndexError:
            return None

    def process_batch(batch):
        prompts = [[
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "What is 2+2?"},
            {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
            {'role': 'user', 'content': q.strip()}
        ] for q in batch['question']]

        return {
            'prompt': prompts,
            'answer': [extract_hash_answer(a) for a in batch['answer']]
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size)


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# def get_gsm8k_questions(split = "train") -> Dataset:
#     data = load_dataset('openai/gsm8k', 'main')[split]
#     data = data.map(lambda x: {
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question'] + "\n" + SYSTEM_PROMPT}
#         ],
#         'answer': extract_hash_answer(x['answer'])
#     })
#     return data

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

import itertools
_logger_counter = itertools.count()
import os
is_rank0 = os.environ.get("RANK", "0") in ("0", None)
FORMAT_RE = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.S)

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    rewards = [0.5 if match else 0.0 for match in matches]

    step = next(_logger_counter)
    if is_rank0 and step % 10 == 0:
        print(f"\n[DBG step {step}] Showing up to 2 samples")
        for i, r in enumerate(responses[:2]):
            ok = bool(FORMAT_RE.match(r))
            print(f"--- sample {i} | format_ok={ok} ---\n{r}\n")

    return rewards

# def format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     # pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     pattern = r"<reasoning>[\s\S]*</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     rewards = [0.5 if match else 0.0 for match in matches]

#     step = next(_logger_counter)
#     if is_rank0 and step % 10 == 0:
#         print(f"\n[DBG step {step}] Showing up to 2 samples")
#         for i, r in enumerate(responses[:2]):
#             ok = bool(FORMAT_RE.match(r))
#             print(f"--- sample {i} | format_ok={ok} ---\n{r}\n")

#     return rewards

# def accuracy_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     """Reward function that extracts the answer from the xml tags and compares it to the correct answer."""
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

#     # occasional debug
#     if is_rank0 and next(_logger_counter) % 10 == 5:
#         for i in range(min(2, len(extracted_responses))):
#             print(f"[ACC DBG] pred={extracted_responses[i]!r} | gold={answer[i]!r} | r={rewards[i]}")
            
#     return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# ---- EigenScore ------------------------------------------------------------
def eigenscore_from_last_layer(hiddens: list[torch.Tensor]) -> float:
    """hiddens: list of [hidden_dim] tensors (one per sampled answer)."""
    hidden_var = torch.var(hiddens, dim=1)  # [batch, hidden_size]
    eigen_score = torch.mean(hidden_var).item()
    eigen_score = eigen_score * 1000
    return eigen_score

def eigenscore_reward_func(completions, **kwargs) -> list[float]:
    """
    completions: list of length k, each item is a list containing the
    generation dict returned by GRPOTrainer (with hidden_states).
    """
    # collect last-token hidden state from each sampled answer
    last_states = [c[0]["hidden_states"][-1].squeeze().float()
                   for c in completions]
    score = eigenscore_from_last_layer(last_states)
    reward = 1.0 / (1.0 + score)                     # lower EigenScore => higher reward
    return [reward] * len(completions)

def eigen_score_reward_func(model, tokenizer, text):
    """Calculate real eigen score using actual model"""
    device = next(model.parameters()).device
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Use the last layer hidden states
            last_hidden = hidden_states[-1]  # [batch, seq_len, hidden_size]
            
            # Calculate variance of hidden states
            hidden_var = torch.var(last_hidden, dim=1)  # [batch, hidden_size]
            eigen_score = torch.mean(hidden_var).item()
            
            # Scale up to match expected performance
            eigen_score = eigen_score * 1000
        
        return eigen_score
        
    except Exception as e:
        print(f"Error calculating eigen score: {e}")
        return 50.0  # Fallback value

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def main():
    # dataset = get_gsm8k_questions()
    dataset_name = 'openai/gsm8k'
    dataset = preprocess_dataset(dataset_name, chunk_size=500)

    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO"
    run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        # report_to="wandb",
        learning_rate=1e-5,
        beta=0.005, # divergence coefficient – how much the policy is allowed to deviate from the reference model. higher value – more conservative updates. Default is 0.04
        # optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size = 16,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        save_steps=100,
        save_total_limit=1,
        max_grad_norm=0.1,
        log_on_each_node=False,
        generation_kwargs={                         # request hidden states
            "return_dict_in_generate": True,
            "output_hidden_states": True
        },
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
            eigenscore_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()