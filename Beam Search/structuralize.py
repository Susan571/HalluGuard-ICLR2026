import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
import json
from utils import *
import pickle
from tqdm import tqdm

template='''Given a question and two response, your task is judging whether the final answer of the first response is the same as the final answer of the second.
Your output should be in a dictionary looks like:
{{
    "rationale": // string, analyze the whether the final answers of the two responses are the same,
    "judgment": // string, yes or no,
}}

## Instruction: {}

## Response 1: {}

## Response 2: {}

## Output: 
'''

def extract_between(s3: str, s1: str, s2: str, begin_from: int) -> str:
    start_idx = s3.find(s1, begin_from)
    if start_idx == -1:
        return "", None, None  # s1 not found
    
    # start_idx += len(s1)  # Move past s1
    end_idx = s3.find(s2, start_idx)
    
    if end_idx == -1:
        return "", None, None  # s2 not found after s1
    
    return s3[start_idx:end_idx], start_idx, end_idx

def step_match(item):
    data = item["cot_extracted_dict"]
    cot = item["solution"]
    k_prev = ""
    begin_from = 0
    reasoning_steps = []
    for i, (k, v) in enumerate(data.items()):
        # print("current:", cot[cot.find(k_prev):])
        if k[-3:] == "...":
            k = k[:-3]
        # while k not in cot[cot.find(k_prev):]:
        #     k = k[:-1]
        # print("original k:", k)
        if k not in cot[cot.find(k_prev):]:
            k = most_similar_substring(k, cot[cot.find(k_prev):])
        if i == 0:
            k_prev = k
            continue
        # print("k_prev:", k_prev)
        # print("k:",k)
        ret, start_idx, end_idx = extract_between(cot, k_prev, k, begin_from)
        # print(start_idx, end_idx, begin_from)
        # print(ret == "")
        # print(ret, start_idx, end_idx)
        if ret == "" or start_idx != begin_from:
            return None
        begin_from = end_idx
        k_prev = k
        reasoning_steps.append(ret)
    
    reasoning_steps.append(cot[begin_from:])

    return reasoning_steps

def build_graph(item, reasoning_steps, use_model):
    cot_extracted_dict = item["cot_extracted_dict"]
    cot = item["solution"]
    problem = item["problem"]
    G = nx.DiGraph()
    nodes = {}
    step_indices = {}
    step_counter = 1  # Start from 1
    extra_edges = []  # edge from the node before explore and to the final answer
    
    # Add an empty starting node
    G.add_node(0, node_type="start", content="")
    nodes[0] = ("start", "start")
    prev_node = 0  # Set the start node as previous
    
    reasoning_step_now = ""

    for (text, category), reasoning_step in zip(cot_extracted_dict.items(), reasoning_steps):
        if "explore" in category:
            node_type = category.split("-")[0]
        else:
            node_type = category
        node_id = step_counter
        nodes[node_id] = (reasoning_step, category)
        step_indices[reasoning_step] = node_id
        G.add_node(node_id, node_type=node_type, content=reasoning_step)
        
        if category.startswith("explore-"):
            try:
                n = int(category.split("-")[-1])
                if n in nodes:
                    G.add_edge(prev_node, n, edge_type="normal")  # Connect (m-1) to step n
                    G.add_edge(n, node_id, edge_type="normal")  # Connect step n to step m (explore)
            except ValueError:
                pass  # Invalid format, skip extra linkage

            if use_model:
                prompt = template.format(problem, cot, reasoning_step_now)
                ret = generate_openai(model="gpt-4o-2024-11-20", messages=prompt)
                extracted_dict = extract_dict(ret)
                if "judgment" in extracted_dict and extracted_dict["judgment"] == "yes":
                    extra_edges.append(prev_node)
            else:
                extra_edges.append(prev_node)

        elif prev_node is not None:
            G.add_edge(prev_node, node_id, edge_type="normal")
        
        prev_node = node_id
        step_counter += 1
        reasoning_step_now += reasoning_step

    for node_id in extra_edges:
        # print(node_id)
        G.add_edge(node_id, prev_node, edge_type="skip")
    
    return G

def visualize_graph(G,i):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    
    categories = set(nx.get_node_attributes(G, 'node_type').values())
    color_map = {cat: (i if not cat.startswith("explore-") else len(categories)) for i, cat in enumerate(categories)}
    node_colors = [color_map[G.nodes[n]['node_type']] for n in G.nodes]
    
    nx.draw(G, pos, with_labels=False, node_color=node_colors, cmap=plt.cm.Set1, node_size=1500, edge_color='gray')
    
    plt.title("Reasoning Structure Graph")
    plt.savefig("vis/test_graph_{}.png".format(i))

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='UWNSL/MATH_training_split_long_cot')
    argparser.add_argument('--save_path', type=str, default='data')
    argparser.add_argument('--model', type=str, default='gpt4')
    argparser.add_argument('--use_model', action='store_true')
    args = argparser.parse_args()

    dataset_name = args.dataset.split("/")[1]
    if "/" in args.model:
        model_name = args.model.split("/")[1]
    else:
        model_name = args.model
    dataset_parsed = json.load(open("{}/{}_{}_parsed.json".format(args.save_path, dataset_name, model_name)))

    folder_path="{}/graph/{}_{}".format(args.save_path, dataset_name, model_name)
    os.makedirs(folder_path, exist_ok=True)

    not_parsed = 0

    for i, item in tqdm(enumerate(dataset_parsed)):
        reasoning_steps = step_match(item)
        if reasoning_steps is None:
            not_parsed += 1
            continue
        graph = build_graph(item, reasoning_steps, args.use_model)
        # visualize_graph(graph, i)
        with open("{}/graph/{}_{}/{}.pkl".format(args.save_path, dataset_name, model_name, i), "wb") as f:
            pickle.dump(graph, f)
    print(not_parsed)
if __name__ == "__main__":
    main()