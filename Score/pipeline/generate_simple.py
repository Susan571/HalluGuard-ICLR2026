import argparse
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import dataeval.TruthfulQA as TruthfulQA
import models
import utils
from func.metric import *
from halluguard_true import compute_halluguard_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-7b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument('--halluguard_layer', type=int, default=-1, help='Layer index for sigma_max proxy')
parser.add_argument('--halluguard_param_subset', type=str, default='last_block', help='Param subset for NTK gradients')

args = parser.parse_args()
# Handle model names with forward slashes for file naming
safe_model_name = args.model.replace('/', '_')
os.makedirs(_settings.GENERATION_FOLDER, exist_ok=True)
log_path = os.path.join(_settings.GENERATION_FOLDER, "logInfo_{}_{}.txt".format(safe_model_name, args.dataset))
logInfo = open(log_path, mode="w", encoding="utf-8")

def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    elif data_name == 'coqa':
        return coqa.get_dataset
    elif data_name == 'nq_open':
        return nq_open.get_dataset
    elif data_name == "SQuAD":
        return SQuAD.get_dataset
    elif data_name == "TruthfulQA":
        return TruthfulQA.get_dataset
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    elif data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    elif data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    elif data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    elif data_name == 'TruthfulQA':
        generation_config = TruthfulQA._generate_config(tokenizer)
    else:
        generation_config = {}
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    generation_config['pad_token_id'] = getattr(tokenizer, 'eos_token_id', 0)
    return generation_config

def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    
    # Simplified BERTScore without sentence-transformers
    bertscore = BERTScore(model_name_or_path="bert-base-uncased", device="cpu" if device == "cpu" else "cuda")

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        attention_mask = batch.get('attention_mask')
        if attention_mask is None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        else:
            attention_mask = attention_mask.to(device)
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        generation_config = transformers.GenerationConfig(**generation_config)
        
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            with torch.no_grad():
                dict_outputs = model.generate(input_ids, attention_mask=attention_mask,
                                            num_beams=1,
                                            do_sample=False,
                                            generation_config=generation_config,
                                            output_hidden_states = True,
                                            return_dict_in_generate=True,
                                            output_scores=True)

            scores = dict_outputs.scores
            perplexity = get_perplexity_score(scores)
            energy_score = get_energy_score(scores)
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]

        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            with torch.no_grad():
                dict_outputs = model.generate(input_ids, attention_mask=attention_mask,
                                num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                                do_sample=True, top_p=args.top_p, top_k=args.top_k,
                                temperature=args.temperature, generation_config=generation_config,
                                output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                                )

            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            scores = dict_outputs.scores
            predictive_entropy = get_lenghthNormalized_entropy(scores, num_tokens) 
            hidden_states = dict_outputs.hidden_states
            eigenIndicator, eigenValue = getEigenIndicator_v0(hidden_states, num_tokens, device)
        num_gens -= len(generation)

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        if 'most_likely_generations' not in locals():
            most_likely_generations = generations[0]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        lexical_similarity = getLexicalSim(generated_texts)
        
        # Simplified BERTScore calculation
        try:
            sent_bertscore = getAvgBertScore(bertscore, best_generated_text, generated_texts)
        except:
            sent_bertscore = 0.0  # Fallback if BERTScore fails
        
        # Simplified eigen indicator without sentence-transformers
        try:
            eigenIndicatorOutput = 0.0  # Placeholder
            eigenValue_O = 0.0  # Placeholder
        except:
            eigenIndicatorOutput = 0.0
            eigenValue_O = 0.0
            
        # Simplified NTK-S3 score without sentence-transformers
        try:
            ntks3IndicatorOutput = 0.0  # Placeholder
            ntks3InfoOutput = {}  # Placeholder
        except:
            ntks3IndicatorOutput = 0.0
            ntks3InfoOutput = {}

        # remember the data
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        curr_seq.update(
            dict(
                energy=energy_score
            )
        )
        curr_seq.update(
            dict(
                lexical_similarity=lexical_similarity
            )
        )
        curr_seq.update(
            dict(
                sent_bertscore=sent_bertscore
            )
        )
        curr_seq.update(
            dict(
                entropy=predictive_entropy
            )
        )
        curr_seq.update(
            dict(
                eigenIndicator=eigenIndicator
            )
        )
        curr_seq.update(
            dict(
                eigenIndicatorOutput=eigenIndicatorOutput
            )
        )
        try:
            hallu = compute_halluguard_score(
                model=model,
                input_ids=input_ids[0],
                generated_ids=most_likely_generations,
                attention_mask=None,
                layer_idx=args.halluguard_layer,
                param_subset=args.halluguard_param_subset,
                sigma_mode="lipschitz",
            )
            ntks3Indicator = hallu["score"]
            curr_seq.update(dict(halluguard_score=hallu["score"]))
        except Exception as e:
            ntks3Indicator = 0.0
            curr_seq.update(dict(halluguard_score=None, halluguard_error=str(e)))
        curr_seq.update(dict(ntks3Indicator=ntks3Indicator))
        curr_seq.update(
            dict(
                ntks3IndicatorOutput=ntks3IndicatorOutput
            )
        )
        if args.dataset == 'coqa' or args.dataset == "TruthfulQA":
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
        
        # Print information
        print("Question:", batch['question'][0])
        print("AnswerGT:", batch['answer'][0])
        print("MostLikelyAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True))
        print("Batch_Generations:", generated_texts)
        print("Perplexity:", perplexity)
        print("Energy:", energy_score)
        print("NormalizedEntropy: ", predictive_entropy)
        print("LexicalSimilarity: ", lexical_similarity)
        print("EigenScore: ", eigenIndicator)
        print("EigenValue:", eigenValue)
        print("EigenScore-Output: ", eigenIndicatorOutput)
        print("NTK-S3 Score: ", ntks3Indicator)
        print("NTK-S3 Score-Output: ", ntks3IndicatorOutput)
        print("HALLUGUARD: ", curr_seq.get("halluguard_score"))

        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Question:", batch['question'][0], file=logInfo)
        print("GTAns:", batch['answer'][0], file=logInfo)
        print("BestAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=logInfo)
        print("BatchGenerations:", generated_texts, file=logInfo)
        print("Perplexity:", perplexity, file=logInfo)
        print("Energy:", energy_score, file=logInfo)
        print("NormalizedEntropy: ", predictive_entropy, file=logInfo)
        print("LexicalSimilarity: ", lexical_similarity, file=logInfo)
        print("SentBERTScore: ", sent_bertscore, file=logInfo)
        print("EigenScore: ", eigenIndicator, file=logInfo)
        print("EigenValue:", eigenValue, file=logInfo)
        print("EigenScore-Output: ", eigenIndicatorOutput, file=logInfo)
        print("NTK-S3 Score: ", ntks3Indicator, file=logInfo)
        print("NTK-S3 Score-Output: ", ntks3IndicatorOutput, file=logInfo)
        print("HALLUGUARD: ", curr_seq.get("halluguard_score"), file=logInfo)
        print("\n","\n","\n", file=logInfo)
    return sequences

def get_num_tokens(generation):
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens

def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        original_model_name = args.model
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(original_model_name if 'original_model_name' in locals() else args.model, args, seed=args.seed, old_sequences=old_sequences)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
