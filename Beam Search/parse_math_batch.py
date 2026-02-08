import time
import json
from openai import OpenAI
import argparse
from copy import deepcopy
import os
from datasets import load_dataset

# Optional HF cache path via environment variable (set outside this script)

template='''Given a chain-of-thought reasoning process, your task is to identify and categorize each reasoning step into one of the following categories:

1. continue reasoning: encourages the model to build upon its current reasoning trajectory by generating the next intermediate step.
2. reflect: prompts the model to pause and verify the correctness of current reasoning path or calculation results. Marked by some keywords like "wait" and "verify".
3. explore: signals the model to explore a alternative and new solution. Marked by some keywords like "alternative" and "another approach".

Here is an example to follow:

## Instruction: Let $\mathcal{C}$ be the hyperbola $y^2 - x^2 = 1$. Given a point $P_0$ on the $x$-axis, we construct a sequence of points $(P_n)$ on the $x$-axis in the following manner: let $\ell_n$ be the line with slope $n$ passing through $P_n$, then $P_{n+1}$ is the orthogonal projection of the point of intersection of $\ell_n$ and $\mathcal C$ onto the $x$-axis. (If $P_n = 0$, then the sequence simply terminates.) Find the sum of the $x$-coordinates of all points $P_n$ if $P_0 = (1,0)$. Your answer should use only integers and $\ln$..

## Response: Given the hyperbola \(\mathcal{C}\) defined by the equation \(y^2 - x^2 = 1\), we start with the point \(P_0 = (1,0)\). The line \(\ell_0\) passing through \(P_0\) with slope \(0\) is the horizontal line \(y = 0\). The intersection of \(\ell_0\) with \(\mathcal{C}\) is the point \((1,0)\) and \((-1,0)\). The orthogonal projection of \((-1,0)\) onto the \(x\)-axis is \((-1,0)\), so \(P_1 = (-1,0)\).\n\nBut wait. The intersection of \(\ell_0\) with \(\mathcal{C}\) is not \((1,0)\) and \((-1,0)\); it is only \((0,1)\) and \((0,-1)\).\n\nAs an alternative, The correct intersection point to consider is \((0,1)\), and the orthogonal projection of \((0,1)\) onto the \(x\)-axis is \((0,0)\), so \(P_1 = (0,0)\).\n\nLet's re-evaluate the problem correctly. Given the hyperbola \(\mathcal{C}\) defined by the equation \(y^2 - x^2 = 1\), we start with the point \(P_0 = (1,0)\). The line \(\ell_0\) passing through \(P_0\) with slope \(n\) is given by \(y = n(x-1)\). We need to find the intersection of \(\ell_0\) with \(\mathcal{C}\).\n\nSubstitute \(y = n(x-1)\) into the hyperbola equation: \[ (n(x-1))^2 - x^2 = 1 \] \[ n^2(x^2 - 2x + 1) - x^2 = 1 \] \[ n^2x^2 - 2n^2x + n^2 - x^2 = 1 \] \[ (n^2 - 1)x^2 - 2n^2x + n^2 - 1 = 0 \] This is a quadratic equation in \(x\).\n\nUsing the quadratic formula \(x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\), where \(a = n^2 - 1\), \(b = -2n^2\), and \(c = n^2 - 1\), we get: \[ x = \frac{2n^2 \pm \sqrt{(2n^2)^2 - 4(n^2 - 1)(n^2 - 1)}}{2(n^2 - 1)} \] \[ x = \frac{2n^2 \pm \sqrt{4n^4 - 4(n^4 - 2n^2 + 1)}}{2(n^2 - 1)} \] \[ x = \frac{2n^2 \pm \sqrt{4n^4 - 4n^4 + 8n^2 - 4}}{2(n^2 - 1)} \] \[ x = \frac{2n^2 \pm \sqrt{8n^2 - 4}}{2(n^2 - 1)} \] \[ x = \frac{2n^2 \pm 2\sqrt{2n^2 - 1}}{2(n^2 - 1)} \] \[ x = \frac{n^2 \pm \sqrt{2n^2 - 1}}{n^2 - 1} \] The two solutions are \(x = \frac{n^2 + \sqrt{2n^2 - 1}}{n^2 - 1}\) and \(x = \frac{n^2 - \sqrt{2n^2 - 1}}{n^2 - 1}\). Since \(P_0 = (1,0)\) is on the right side of the hyperbola, we take the negative root to ensure the sequence converges to zero.\n\nThus, the \(x\)-coordinate of \(P_{n+1}\) is: \[ x_{n+1} = \frac{n^2 - \sqrt{2n^2 - 1}}{n^2 - 1} \] We can simplify this expression further: \[ x_{n+1} = 1 - \frac{\sqrt{2n^2 - 1}}{n^2 - 1} \] To find the sum of the \(x\)-coordinates of all points \(P_n\), we need to sum the series: \[ \sum_{n=0}^{\infty} x_n \] By examining the pattern, we see that each term \(x_n\) approaches 0 as \(n\) increases. The sum of the series can be found using properties of logarithms and hyperbolic functions. The result is: \[ \sum_{n=0}^{\infty} x_n = \boxed{1} \]

## Output: 
{{
    "Given the hyperbola ...": "continue reasoning",
    "But wait. The intersection of...": "reflect-effective",
    "As an alternative, The correct intersection...": "explore-0",
    "Let's re-evaluate the problem correctly...": "reflect-ineffective",
    "Substitute \(y = n(x-1)\) into...": "continue reasoning",
    "Using the quadratic formula...": "continue reasoning",
    "Thus, the \(x\)-coordinate of...": "continue reasoning"
}}

Noted that each reasoning step is seperated by \n\n. As the demonstration shows, each single reasoning step's brief content and its reasoning category should be a key-value pair in the dictionary. For the explore step, the number followed by "explore" means which step it goes back to and explores from. For example, explore-0 means the reasoning process goes back to the very beginning and tries to find an alternative solution. Only reflecting steps that lead to a different answer should be reflect-effective, otherwise, it should be reflect-ineffective.
Formatted the example above and think step by step to categorize reasoning step for the following case:'''

# Initialize the OpenAI client with your API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set. Please export it before running.")
client = OpenAI(api_key=openai_api_key)


def build_chunk(mode, instruction_data, chunk_size):
    batch_data_all = []
    for i, item in enumerate(instruction_data):
        problem = item["problem"]
        response = item["solution"]
        prompt = template + "\n\n##Instruction: {}\n\n##Response: {}\n\n##Output:".format(problem, response)
        batch_template = {
            "custom_id": str(i), 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-4o-2024-11-20", 
                "messages": [{"role":"user", "content": prompt}],
                "temperature": 0.0
            }
        }
        batch_data_all.append(batch_template)

    file_path_list = []
    chunk_num = int(len(instruction_data)/chunk_size+1)
    for i in range(chunk_num):
        if i == chunk_num-1:
            batch_data_chunk = batch_data_all[i*chunk_size:]
        else:
            batch_data_chunk = batch_data_all[i*chunk_size:(i+1)*chunk_size]

        input_file_path = "data/{}_gpt4_part{}_input.jsonl".format(mode, str(i))
        output_file_path = "data/{}_gpt4_part{}_output.json".format(mode, str(i))
        file_path_list.append((input_file_path, output_file_path))
        with open(input_file_path, "w") as f:
            for item in batch_data_chunk:
                f.write(json.dumps(item)+'\n')

    return file_path_list

def upload_batch_input_file(file_path):
    # Upload the batch input file (in JSONL format)
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    return batch_input_file.id

def create_batch_job(batch_input_file_id):
    # Create a batch job for processing the input file
    ret = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "teacher bias"}
    )
    return ret.id

def check_batch_status(batch_id):
    # Check the status of the batch job
    return client.batches.retrieve(batch_id)

def retrieve_batch_results(output_file_id, result_file_path):
    # Retrieve and save the results of the batch job
    ret_json = []
    file_response = client.files.content(output_file_id)

    # Parse the response line by line and load it into JSON format
    for line in file_response.text.strip().split('\n'):
        ret_json.append(json.loads(line))

    # Write the results to a file
    with open(result_file_path, 'w') as f:
        json.dump(ret_json, f, indent=2)

def extract_result(input_file_path, output_file_path):
    input_file = []
    with open(input_file_path) as f:
        for line in f.readlines():
            input_file.append(json.loads(line))
    output_file = json.load(open(output_file_path))

    synthetic_response_batch = []
    for item_input, item_output in zip(input_file, output_file):
        assert item_input["custom_id"] == item_output["custom_id"]
        instruction = item_input["body"]["messages"][0]["content"]
        output = item_output["response"]["body"]["choices"][0]["message"]["content"]
        synthetic_response_batch.append({
            "instruction": instruction,
            "output": output
        })

    return synthetic_response_batch

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='UWNSL/MATH_training_split_long_cot')
    argparser.add_argument('--chunk_size', type=int, default=1000)  
    args = argparser.parse_args()

    # Step 0: Chunk input file
    # instruction_data = json.load(open("../data/instruction/{}_instruction.json".format(args.mode)))
    # file_path_list = build_chunk(args.mode, instruction_data, args.chunk_size)
    dataset = load_dataset(args.dataset, split="train")
    # dataset = [dataset[i] for i in range(20)]
    dataset_name = args.dataset.split("/")[1]
    file_path_list = build_chunk(dataset_name, dataset, args.chunk_size)

    synthetic_response = []
    for input_file_path, output_file_path in file_path_list:
        # Step 1: Upload the batch input file
        batch_input_file_id = upload_batch_input_file(input_file_path)
        print(f"Batch input file uploaded. Batch File ID: {batch_input_file_id}")

        # Step 2: Create the batch job
        batch_id = create_batch_job(batch_input_file_id)
        print(f"Batch job created. Batch ID: {batch_id}")

        # Step 3: Poll the status every 30 seconds until completion
        while True:
            status = check_batch_status(batch_id)
            print(f"Current status: {status.status}")
            
            if status.status == 'completed':
                print("Batch job completed.")
                break
            elif status.status == 'failed':
                print("Batch job failed.")
                print(status)
                return
            
            time.sleep(30)  # Wait for 30 seconds before checking again

        # Step 4: Retrieve and save the results
        output_file_id = status.output_file_id
        retrieve_batch_results(output_file_id, output_file_path)
        print(f"Results saved to {output_file_path}")

        # Step 5: Extract output from each results file
        synthetic_response_batch = extract_result(input_file_path, output_file_path)
        synthetic_response.extend(synthetic_response_batch)

    # Step 6: Save synthetic response
    with open("data/{}_gpt4_all_output.json".format(dataset_name), "w") as f:
        json.dump(synthetic_response, f, indent=2)

if __name__ == "__main__":
    main()