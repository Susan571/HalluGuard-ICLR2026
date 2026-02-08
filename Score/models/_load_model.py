# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

try:
    from _settings import MODEL_PATH
except Exception:
    MODEL_PATH = None


def _get_hf_model_name(model_name):
    """Map short names to HuggingFace model IDs."""
    if model_name == 'llama-7b-hf':
        return "meta-llama/Llama-2-7b-hf"
    if model_name == 'llama-13b-hf':
        return "meta-llama/Llama-2-13b-hf"
    if model_name == "llama2-7b-hf":
        return "meta-llama/Llama-2-7b-hf"
    return model_name


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    model = None
    if model_name.startswith('facebook/opt'):
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, use_safetensors=True)
    elif model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name in ('llama-7b-hf', 'llama-13b-hf', 'llama2-7b-hf'):
        hf_model_name = _get_hf_model_name(model_name)
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=None, torch_dtype=torch_dtype, use_safetensors=True)
    elif model_name == "falcon-7b" and MODEL_PATH:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, trust_remote_code=True, torch_dtype=torch_dtype)
    elif model_name == 'roberta-large-mnli':
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    else:
        # Generic HuggingFace Causal LM (e.g. gpt2, facebook/opt-125m, any from_pretrained name)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Unsupported model: {model_name}. Load failed: {e}") from e

    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name == "microsoft/deberta-large-mnli":
        return AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    if model_name == "roberta-large-mnli":
        return AutoTokenizer.from_pretrained("roberta-large-mnli")
    if model_name in ('llama-7b-hf', 'llama-13b-hf', 'llama2-7b-hf'):
        hf_model_name = _get_hf_model_name(model_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=None, use_fast=use_fast)
        tokenizer.eos_token_id = getattr(tokenizer, 'eos_token_id', 2)
        tokenizer.bos_token_id = getattr(tokenizer, 'bos_token_id', 1)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.decode(tokenizer.eos_token_id)
        return tokenizer
    if model_name == "falcon-7b" and MODEL_PATH:
        return AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), trust_remote_code=True, cache_dir=None, use_fast=use_fast)
    # Generic: any HuggingFace model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, trust_remote_code=True)
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.pad_token_id = getattr(tokenizer, 'eos_token_id', 0)
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, 'eos_token', None) else tokenizer.decode(tokenizer.pad_token_id)
    return tokenizer