import os
import sys

# Ensure Score (halluguard_true) is on path when running from Beam Search
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _hallu_dir_name in ("Score", "Hallucination"):
    _hallu_dir = os.path.join(_repo_root, _hallu_dir_name)
    if os.path.isdir(_hallu_dir) and _hallu_dir not in sys.path:
        sys.path.insert(0, _hallu_dir)
        break

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union, List
from reward_model.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult, AggregationMethod
from halluguard_true import compute_halluguard_score
 

class NTKS3ScoreModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token if self.tokenizer.unk_token else self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            output_hidden_states=True,
            trust_remote_code=True
        ).to(device)
        self.device = device
        self.model_name = model_name

    def score(
        self,
        messages: Union[List[List[dict]], List[dict]],
        return_full_prm_result: bool = False,
        use_tqdm: bool = False,
        batch_size: int = 1,
    ) -> Union[List[PRMResult], List[float]]:
        if isinstance(messages[0], dict):
            messages = [messages]  # convert to List[List[dict]]

        prompts = [
            self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]

        all_scores = []
        for start_idx in range(0, len(prompts), batch_size):
            end_idx = start_idx + batch_size
            batch_prompts = prompts[start_idx:end_idx]

            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            scores = []
            for prompt in batch_prompts:
                # Treat the full prompt as generated output with empty input prefix
                prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(self.device)
                if prompt_ids.numel() == 0:
                    scores.append([0.0])
                    continue
                hallu = compute_halluguard_score(
                    model=self.model,
                    input_ids=prompt_ids[:1],
                    generated_ids=prompt_ids[1:],
                    attention_mask=None,
                    layer_idx=-1,
                    param_subset="last_block",
                    sigma_mode="lipschitz",
                )
                scores.append([hallu["score"]])
            all_scores.extend(scores)

        if return_full_prm_result:
            return [PRMResult(scores=s) for s in all_scores]
        else:
            return [PRMResult(scores=s).score for s in all_scores]
    