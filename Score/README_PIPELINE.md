# Running the Pipeline and Model

## Setup

1. **Install dependencies** (from the `Score` directory or repo root):
   ```bash
   pip install -r requirements.txt
   ```
   Or, if you use a venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Paths**  
   Data and outputs use paths under `Score/data/`:
   - `data/datasets/` — put dataset files here (e.g. CoQA: `coqa-dev-v1.0.json`; SQuAD: `dev-v2.0.json`).
   - `data/output/` — run logs and `logInfo_*.txt`.
   - `data/output/<model>_<dataset>_<project_ind>/` — saved generations (`.pkl`) and `args*.json`.

## Run the pipeline

From **repo root** or **Score**:

```bash
# Make the script executable once (optional)
chmod +x Score/run_pipeline.sh

# Example: small run with GPT-2 on CoQA (2 generations per prompt, 1% of data)
./Score/run_pipeline.sh --model gpt2 --dataset coqa --device cuda --num_generations_per_prompt 2 --fraction_of_data_to_use 0.01

# Or from inside Score:
cd Score
./run_pipeline.sh --model gpt2 --dataset coqa --device cuda --num_generations_per_prompt 2 --fraction_of_data_to_use 0.01
```

Without the shell script (same effect):

```bash
cd Score
PYTHONPATH=. python pipeline/generate_simple.py --model gpt2 --dataset coqa --device cuda --num_generations_per_prompt 2 --fraction_of_data_to_use 0.01
```

## Pipeline arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `llama-7b-hf` | Model name: `gpt2`, `facebook/opt-125m`, `facebook/opt-1.3b`, `llama-7b-hf`, or any HuggingFace causal LM ID |
| `--dataset` | `coqa` | `coqa`, `SQuAD`, `nq_open`, `triviaqa`, `TruthfulQA` |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--num_generations_per_prompt` | 10 | Number of sampled generations per question |
| `--fraction_of_data_to_use` | 1.0 | Fraction of dataset (e.g. 0.01 for a quick run) |
| `--decoding_method` | `greedy` | `greedy` or non-greedy uses sampling with `temperature`, `top_p`, `top_k`. |
| `--temperature` | 0.5 | Sampling temperature when not greedy |
| `--seed` | 2023 | Random seed |
| `--halluguard_layer` | -1 | Hidden layer index for σ_max proxy (default last layer) |
| `--halluguard_param_subset` | `last_block` | Parameter subset for NTK gradients (`last_block`, `all`, or `name:<substring>`) |

## Datasets

- **CoQA**: Place `coqa-dev-v1.0.json` in `Score/data/datasets/`. The first run will build a cached dataset under `data/datasets/coqa_dataset/`.
- **SQuAD**: Place `dev-v2.0.json` in `Score/data/datasets/` (or set `DATA_FOLDER`). Same idea for other datasets that use `_settings.DATA_FOLDER`.

## Loading your own model

Any HuggingFace causal LM is supported by name, for example:

- `--model gpt2`
- `--model facebook/opt-125m`
- `--model meta-llama/Llama-2-7b-hf` (if you have access)

The pipeline uses `transformers.AutoModelForCausalLM` and `AutoTokenizer` for unknown names; the model is loaded in `float16` on the requested `--device`.
