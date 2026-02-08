import functools
import json
import os

import datasets
import pandas as pd
from datasets import Dataset

import _settings


def _save_dataset():
    # Create a simple TruthfulQA dataset structure
    save_path = f'{_settings.DATA_FOLDER}/truthfulqa_dataset'
    if not os.path.exists(save_path):
        # For now, create a minimal dataset structure
        # In a real implementation, you would load actual TruthfulQA data
        dataset = {}
        
        dataset['question'] = []
        dataset['answer'] = []
        dataset['additional_answers'] = []
        dataset['id'] = []
        
        # Create some sample data for testing
        sample_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical symbol for gold?"
        ]
        
        sample_answers = [
            "Paris",
            "William Shakespeare", 
            "Au"
        ]
        
        for i, (question, answer) in enumerate(zip(sample_questions, sample_answers)):
            dataset['question'].append(question)
            dataset['answer'].append(answer)
            dataset['additional_answers'].append([answer])  # Single additional answer for now
            dataset['id'].append(f'truthfulqa_{i}')

        dataset_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(dataset_df)
        dataset.save_to_disk(save_path)
    
    return save_path


@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(_save_dataset())
    return {_['id']: _['question'] for _ in dataset}


def get_dataset(tokenizer, split='validation'):
    dataset = datasets.load_from_disk(_save_dataset())
    
    def encode_truthfulqa(example):
        example['prompt'] = f"Q: {example['question']} A:"
        return tokenizer(example['prompt'], truncation=False, padding=False)

    dataset = dataset.map(encode_truthfulqa, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    
    return dataset


def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['.', '\n']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    else:
        raise NotImplementedError
    
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids]
    
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)


if __name__ == '__main__':
    import models
    dataset = get_dataset(models.load_tokenizer())
