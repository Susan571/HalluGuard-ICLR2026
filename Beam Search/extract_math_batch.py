import json
import argparse
from utils import *
from datasets import load_dataset
from tqdm import tqdm

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='UWNSL/MATH_training_split_long_cot')
    argparser.add_argument('--save_path', type=str, default='data')
    argparser.add_argument('--model_parsed', type=str, default="gpt4")  
    args = argparser.parse_args()

    dataset = load_dataset(args.dataset, split="train")
    if "/" in args.model_parsed:
        model_parsed_name = args.model_parsed.split("/")[1]
    else:
        model_parsed_name = args.model_parsed
    dataset_name = args.dataset.split("/")[1]
    dataset_parsed = json.load(open("{}/{}_{}_all_output.json".format(args.save_path, dataset_name, model_parsed_name)))
    assert len(dataset) == len(dataset_parsed)

    dataset_parsed_new = []
    for item, item_parsed in tqdm(zip(dataset, dataset_parsed)):
        problem = item["problem"]
        instruction_parse = item_parsed["instruction"]
        assert problem in instruction_parse
        
        cot_extracted = item_parsed["output"]
        item["cot_extracted"] = cot_extracted
        try:
            cot_extracted = cot_extracted.split("</think>")[1]
        except:
            cot_extracted = cot_extracted

        # try:
        #     cot_extracted_dict = extract_dict(cot_extracted)
        # except Exception as E:
        #     print(E)
        #     continue
        try:
            cot_extracted_dict = extract_dict(cot_extracted)
        except:
            continue
        if cot_extracted_dict is None:
            continue
        item["cot_extracted_dict"] = cot_extracted_dict
        dataset_parsed_new.append(item)

    with open("{}/{}_{}_parsed.json".format(args.save_path, dataset_name, model_parsed_name), "w") as f:
        json.dump(dataset_parsed_new, f, indent=2)


if __name__ == "__main__":
    main()