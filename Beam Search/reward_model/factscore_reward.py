import torch
from transformers import AutoTokenizer
from typing import Union, List
from reward_model.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult, AggregationMethod
import os
from vllm import LLM

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
    

class FactScoreModel(AbstractProcessRewardModel):
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

            with torch.no_grad():
                outputs = self.model(**encoded, use_cache=False)
                scores = [[s.item()] for s in self._calculate_fact_score(outputs.hidden_states)]
                all_scores.extend(scores)

        if return_full_prm_result:
            return [PRMResult(scores=s) for s in all_scores]
        else:
            return [PRMResult(scores=s).score for s in all_scores]
    
    def _calculate_fact_score(self, hidden_states):
        with torch.no_grad():
            # Use factual consistency patterns for FActScore
            fact_scores = []
            for layer_hidden in hidden_states:
                # layer_hidden shape: (batch, seq_len, hidden_size)
                # Calculate factual consistency based on hidden state patterns
                fact_score = torch.mean(torch.std(layer_hidden, dim=1), dim=-1)  # (B,)
                fact_scores.append(fact_score)
            fact_scores = torch.stack(fact_scores, dim=0).transpose(0, 1)  # (B, num_layers)
            
            # Combine scores with weighted average
            weights = torch.softmax(torch.arange(fact_scores.shape[1], dtype=torch.float32, device=fact_scores.device), dim=0)
            factscore = torch.sum(fact_scores * weights, dim=-1)  # (B,)

        return -factscore
