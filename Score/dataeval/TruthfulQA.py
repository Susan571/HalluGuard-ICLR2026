import functools
import json
import os

import datasets
import pandas as pd
from datasets import Dataset, load_dataset

import _settings


def _save_dataset():
    save_path = f'{_settings.DATA_FOLDER}/truthfulqa_dataset'
    if not os.path.exists(save_path):
        raw = load_dataset("truthful_qa", "generation", split="validation",
                           trust_remote_code=True)

        dataset = {
            'question': [],
            'answer': [],
            'additional_answers': [],
            'id': [],
        }

        for i, item in enumerate(raw):
            dataset['question'].append(item['question'])
            dataset['answer'].append(item['best_answer'])
            correct = item.get('correct_answers', [])
            if not correct:
                correct = [item['best_answer']]
            dataset['additional_answers'].append(correct)
            dataset['id'].append(f'truthfulqa_{i}')

        dataset_df = pd.DataFrame.from_dict(dataset)
        ds = Dataset.from_pandas(dataset_df)
        ds.save_to_disk(save_path)

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
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'],
                       output_all_columns=True)

    return dataset


def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    else:
        raise NotImplementedError

    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][-1]] for eos_token in question_framing_ids]

    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)


if __name__ == '__main__':
    import models
    dataset = get_dataset(models.load_tokenizer())
