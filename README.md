# <a href="https://arxiv.org/abs/2601.18753" style="color: black !important;"> HALLUGUARD: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs [ICLR 2026] </a>

**Xinyue Zeng¹\*, Junhong Lin²\*, Yujun Yan³, Feng Guo¹, Liang Shi¹, Jun Wu⁴, Dawei Zhou¹**  
*\* Equal contribution*

¹ Virginia Tech · ² MIT · ³ Dartmouth · ⁴ Michigan State University

---

## 📌 Abstract

Large Language Models (LLMs) increasingly operate in high-stakes domains, yet their deployment is constrained by **hallucinations**—outputs that are fluent but incorrect or logically inconsistent. Existing hallucination detection methods typically focus on either factual errors or reasoning instability, rely on task-specific heuristics, or require external references, limiting robustness and scalability.

We introduce **HALLUGUARD**, a theory-grounded, NTK-based hallucination detection framework built upon a novel **Hallucination Risk Bound** that formally decomposes hallucinations into **data-driven** and **reasoning-driven** components. This unified view explains how hallucinations emerge and evolve during generation, and enables a single, architecture-agnostic score that detects both failure modes without supervision or external knowledge.

Across 10 benchmarks, 11 baselines, and 9 LLM backbones, HALLUGUARD consistently achieves state-of-the-art hallucination detection, and further improves test-time reasoning accuracy when used for score-guided inference.

---

## 🧠 Core Insight: Hallucinations Are Not One Failure Mode

We show that hallucinations arise from two fundamentally different mechanisms:

| Source | What breaks | When it breaks |
|--------|-------------|----------------|
| **Data-driven hallucinations** | Learned representations misalign with task semantics | Training / fine-tuning |
| **Reasoning-driven hallucinations** | Autoregressive decoding amplifies instability | Inference / multi-step generation |

Crucially, these errors interact and compound over decoding steps, which explains why existing detectors fail under long-horizon reasoning.

---

## 🔍 Hallucination Risk Bound (Theory)

We formalize hallucination as deviation in a semantic hypothesis space and prove a **Hallucination Risk Bound**:

\[
\|u^* - \hat{u}\| \leq \underbrace{\|u^* - \mathbb{E}[\hat{u}]\|}_{\text{data-driven}} + \underbrace{\|\hat{u} - \mathbb{E}[\hat{u}]\|}_{\text{reasoning-driven}}
\]

**Interpretation (No fluff):**

- **Data-driven term** → bounded by NTK spectrum geometry  
- **Reasoning-driven term** → bounded by Jacobian amplification across decoding steps  
- Hallucinations start as representation gaps and escalate via unstable rollouts  

This is the first unified theoretical decomposition of hallucination emergence and evolution.

---

## 🧩 HALLUGUARD Score

To operationalize the bound, we derive a tractable, inference-time score:

\[
\text{HALLUGUARD}(\hat{u}) = \underbrace{\det(K)}_{\text{representation adequacy}} + \underbrace{\log \sigma_{\max}}_{\text{reasoning amplification}} - \underbrace{\log \kappa(K)^2}_{\text{spectral instability penalty}}
\]

where \(K\) is the NTK Gram matrix (over generated outputs), \(\sigma_{\max}\) is the maximum per-step hidden-state Jacobian spectral norm over the rollout, and \(\kappa(K)\) is the condition number of \(K\).

### What each term actually does

| Term | Captures | Why it matters |
|------|----------|-----------------|
| \(\det(K)\) | Semantic coverage | Flags factual / data gaps |
| \(\log \sigma_{\max}\) | Step-wise Jacobian growth | Flags reasoning drift |
| \(2\log \kappa(K)\) (penalty) | NTK conditioning | Penalizes brittle representations |

- ✔ No labels  
- ✔ No task heuristics  
- ✔ No external retrieval  
- ✔ Zero inference overhead  

### Technical specification (code verification)

The score must be assembled exactly as (per paper):

```text
HALLUGUARD = det(K) + log(σ_max) − log(κ(K)^2)
```

- **K**: NTK Gram from Jacobians of log-probabilities of generated tokens w.r.t. a fixed parameter subset; build \(G\) with rows \(g_t = \nabla_\theta f_t / \|g_t\|\) (one per decoding step), then \(K = G G^\top\). Eigenvalues of \(K\) are clamped with a small \(\varepsilon\) before taking \(\det(K)\) via \(\exp(\sum \log \lambda_i)\).
- **σ_max**: Maximum over steps \(t\) of the spectral norm of the hidden-state Jacobian \(J_t = \partial h_t / \partial h_{t-1}\) (or an allowed Lipschitz proxy). Use \(\max_t\), not an average.
- **κ(K)**: \(\lambda_{\max}(K) / (\lambda_{\min}(K) + \varepsilon)\); the penalty term is \(-2\log \kappa(K)\).

Model must be frozen (no fine-tuning); generation must be stochastic (sampling or beam), not greedy.

**Code vs. spec:** Default evaluation scripts now call the true implementation (`halluguard_true.py`) for NTK‑S3/HALLUGUARD. Proxy implementations remain in legacy helper code (e.g., `func/metric.py`) but are not used by default.

---

## 📊 Empirical Results

### Benchmarks

- **Data-grounded QA:** RAGTruth, SQuAD, NQ-Open, HotpotQA  
- **Reasoning:** GSM8K, MATH-500, BBH  
- **Instruction following:** TruthfulQA, Natural, HaluEval  

### Key outcomes

- State-of-the-art **AUROC / AUPRC** across all task families  
- Largest gains on reasoning benchmarks, where existing detectors collapse  
- Stronger improvements on smaller models, where hallucination risk is highest  
- Consistent cross-scale behavior from GPT-2 → Llama-70B  

### Test-time inference (real payoff)

Using HALLUGUARD to guide beam search:

- **+10%** accuracy on MATH-500  
- **+15%** accuracy on Natural  

→ hallucination detection becomes **reasoning control**, not just scoring.

---

## 🗂️ Project Structure

```
.
├── Score/                   # Main evaluation and pipelines
│   ├── halluguard_true.py   # Paper HALLUGUARD score: det(K) + log σ_max − log κ²
│   ├── pipeline/            # Generation (generate.py, generate_simple.py, generate_minimal.py)
│   ├── func/                # Metrics, evalFunc, plot
│   ├── dataeval/            # Data loaders (SQuAD, TruthfulQA, CoQA, NQ-Open, TriviaQA, …)
│   ├── models/              # Model loading, NLI, OpenAI helpers
│   ├── utils/               # Parallel and other utilities
│   ├── data/                # Outputs and run logs
│   ├── metrics_output/      # Benchmark CSV results (SQuAD, GSM8K, MATH-500, …)
│   ├── gpu_evaluation_all.py
│   ├── gpu_evaluation_llm.py
│   ├── evaluation.py
│   └── requirements.txt
├── Beam Search/             # Score-guided beam search and reward models
│   ├── reward_model/        # NTK reward uses halluguard_true (paper formula)
│   └── run/                 # run_score, run_train
└── README.md
```

**Implementation note:** All evaluation scripts and pipelines use the paper HALLUGUARD implementation (`halluguard_true.py`): NTK Gram from per-step parameter gradients, Lipschitz proxy for σ_max, and score det(K) + log σ_max − log κ².

---

## 🚀 Quick Start

### Install

```bash
cd Score
pip install -r requirements.txt
```

### Reproducing results (e.g. GPT-2 + TruthfulQA)

Two evaluation routes are available. Both produce end-to-end results.

#### Route 1 -- Pipeline + evalFunc (recommended)

**Step 1: Generate.** From the repo root:

```bash
cd Score
python pipeline/generate_simple.py \
  --model gpt2 \
  --dataset TruthfulQA \
  --device cuda \
  --num_generations_per_prompt 10 \
  --temperature 0.5 \
  --top_p 0.99 \
  --top_k 10
```

The default `--decoding_method greedy` runs a greedy pass first (for perplexity / energy baselines), then generates N stochastic samples (for diversity metrics and HalluGuard). Results are saved to `Score/data/output/gpt2_TruthfulQA_0/0.pkl`.

**Step 2: Evaluate.** Set `file_name` in `Score/func/evalFunc.py` to point at the pickle, then:

```bash
cd Score/func
python evalFunc.py
```

This reports **AUROC, F1, TPR@5%FPR, and TPR@10%FPR** for every detection method.

#### Route 2 -- gpu\_evaluation\_all.py (standalone)

Runs generation and evaluation in a single script:

```bash
python Score/gpu_evaluation_all.py \
  --models gpt2 \
  --datasets TruthfulQA \
  --device cuda \
  --num_generations 10
```

> **Implementation Note:** In the source code, the HALLUGUARD metric is implemented and labeled as **NTK-S3**.

#### Other datasets and models

Replace `--dataset` / `--datasets` with `coqa`, `SQuAD`, `nq_open`, `triviaqa`, or `TruthfulQA`. Replace `--model` / `--models` with any HuggingFace causal LM (e.g. `meta-llama/Llama-2-7b-hf`, `facebook/opt-6.7b`). TruthfulQA and CoQA are downloaded automatically from HuggingFace; for other datasets place the files under `Score/data/datasets/`. See `Score/README_PIPELINE.md` for the full argument reference.

### Score-guided inference (beam search)

From `Beam Search/`, use the NTK reward model in beam search: `reward_model/ntk_reward.py`, `run/run_score.py`. The reward model imports `halluguard_true` from the `Score/` folder.


---

## 🧠 Why This Matters (Positioning)

HALLUGUARD is not another heuristic detector.  
It is a **theoretically grounded reliability primitive**:

- **Explains** why hallucinations happen  
- **Detects** how they evolve  
- **Controls** where reasoning goes wrong  

This makes it directly extensible to:

- Multi-turn dialogue safety  
- Agentic planning  
- Test-time compute allocation  
- Early-warning hallucination prevention  

---

## 📖 Citation

```bibtex
@misc{zeng2026halluguarddemystifyingdatadrivenreasoningdriven,
      title={HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs}, 
      author={Xinyue Zeng and Junhong Lin and Yujun Yan and Feng Guo and Liang Shi and Jun Wu and Dawei Zhou},
      year={2026},
      eprint={2601.18753},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.18753}, 
}
```
