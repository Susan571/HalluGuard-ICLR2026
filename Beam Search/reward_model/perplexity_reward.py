import torch
from transformers import AutoTokenizer
from typing import Union, List
from reward_model.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult, AggregationMethod
import os
from vllm import LLM

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class PerplexityScoreModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token if self.tokenizer.unk_token else self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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
                scores = [[s.item()] for s in self._calculate_perplexity(encoded["input_ids"], outputs.logits)]
                
                all_scores.extend(scores)

        if return_full_prm_result:
            return [PRMResult(scores=s) for s in all_scores]
        else:
            return [PRMResult(scores=s).score for s in all_scores]
    
    def _calculate_perplexity(self, labels, logits):
        with torch.no_grad():
            # Logits shape: (batch_size, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :].contiguous() # (B, L - 1, V)
            shift_labels = labels[:, 1:].contiguous() # (B, L - 1)

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1) # (B, L - 1)
            per_token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Mask padding tokens
            mask = shift_labels != self.tokenizer.pad_token_id
            per_sample_nll = -(per_token_log_probs * mask).sum(dim=1) / mask.sum(dim=1) # (B,)
            
            perplexity = torch.exp(per_sample_nll)  # (B,)
        
            return -perplexity