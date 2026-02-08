# HALLUGUARD: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs [ICLR 2026]

**Xinyue Zeng¹\*, Junhong Lin²\*, Yujun Yan³, Feng Guo¹, Liang Shi¹, Jun Wu⁴, Dawei Zhou¹**  
*\* Equal contribution*

¹ Virginia Tech · ² MIT · ³ Dartmouth · ⁴ Michigan State University

**arXiv:** [2601.18753v1](https://arxiv.org/abs/2601.18753)

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
\text{HALLUGUARD}(\hat{u}) = \underbrace{\det(K)}_{\text{representation adequacy}} + \underbrace{\log \sigma_{\max}}_{\text{reasoning amplification}} - \underbrace{\frac{\log \kappa(K)}{2}}_{\text{spectral instability penalty}}
\]

### What each term actually does

| Term | Captures | Why it matters |
|------|----------|-----------------|
| \(\det(K)\) | Semantic coverage | Flags factual / data gaps |
| \(\log \sigma_{\max}\) | Step-wise Jacobian growth | Flags reasoning drift |
| \(\log \kappa^2(K)\) | NTK conditioning | Penalizes brittle representations |

- ✔ No labels  
- ✔ No task heuristics  
- ✔ No external retrieval  
- ✔ Zero inference overhead  

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
├── theory/                  # Hallucination Risk Bound derivations
├── ntk/                     # NTK spectrum & Jacobian utilities
├── src/
│   ├── halluguard.py        # Core score computation
│   ├── inference.py         # Score-guided decoding
│   ├── eval.py              # Benchmark evaluation
│   └── utils/
├── experiments/             # Benchmark configs
├── results/                 # Logged metrics
└── README.md
```

---

## 🚀 Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Compute HALLUGUARD score

```python
from halluguard import compute_score

score = compute_score(
    model=model,
    input=x,
    generation=y_hat
)
```

### Score-guided inference

```python
from inference import halluguard_beam_search

output = halluguard_beam_search(
    model=model,
    prompt=x,
    beam_size=5
)
```

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
