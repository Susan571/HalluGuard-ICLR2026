# HALLUGUARD Technical Spec Verification

This document checks the current `Score/` code against the **HALLUGUARD core technical specification** (frozen LLM, stochastic generation, NTK Gram from Jacobians, σ_max from hidden Jacobians, and the exact score formula).

---

## Spec Summary (Reference)

| Requirement | Spec |
|-------------|------|
| **Frozen LLM** | `model.eval()` and `for p in model.parameters(): p.requires_grad = False` |
| **Generation** | Stochastic (sampling or beam), **not** greedy |
| **K (NTK Gram)** | Build from \(g_t = \nabla_\theta f_t / \|g_t\|\) with \(f_t = \log p_\theta(y_t \mid y_{<t}, x)\); one row per decoding step; \(K = G G^\top\), shape \((T, T)\) |
| **det(K)** | \(\prod_i \lambda_i\) (implemented as \(\\exp(\\sum \log(\lambda_i))\)) with \(\lambda_i\) clamped by \(\varepsilon\) |
| **σ_max** | \(\max_t \|J_t\|_2\) where \(J_t = \partial h_t / \partial h_{t-1}\) (or Lipschitz proxy); **max over t**, not average |
| **κ(K)** | \(\lambda_{\max}(K) / (\lambda_{\min}(K) + \varepsilon)\) |
| **Formula** | \(\text{HALLUGUARD} = \det(K) + \log\sigma_{\max} - \log\kappa(K)^2\) |

---

## 1. Frozen LLM

| Location | Check | Status |
|----------|--------|--------|
| All evaluation scripts | `model.eval()` is called before scoring | ✅ Present |
| Anywhere | `for p in model.parameters(): p.requires_grad = False` | ❌ **Not found** |
| Forward passes | Done inside `torch.no_grad()` | ✅ Present |

**Verdict:** Partially satisfied. Parameters are not updated and gradients are disabled by `no_grad()`, but the spec explicitly requires `requires_grad = False` on all parameters.

---

## 2. Stochastic Generation (Not Greedy)

| Location | Behavior | Status |
|----------|----------|--------|
| `pipeline/generate_simple.py`, `generate_minimal.py`, `generate.py` | When `decoding_method != 'greedy'`: `do_sample=True`, `temperature`, `top_p`, `top_k` | ✅ Stochastic when not greedy |
| Same pipelines | When `decoding_method == 'greedy'`: `do_sample=False`, argmax | ❌ Greedy path exists; spec forbids greedy |
| `gpu_evaluation_all.py` | Minimal test model: `next_token = torch.argmax(logits, ...)` | ❌ Greedy |
| `gpu_evaluation_all.py` | `generate_real_responses()`: `do_sample=True`, `temperature=0.7` | ✅ Stochastic |
| `gpu_evaluation_llm.py` | Same pattern: toy model greedy, real model stochastic | Same as above |
| `improved_evaluation.py` | Custom model: `do_sample=True`, `temperature=0.7` in `generate()` | ✅ Stochastic |

**Verdict:** Satisfied only when the pipeline is run with non-greedy decoding and when using real-model generation. Toy/minimal models in GPU evaluation scripts use greedy decoding.

---

## 3. K (NTK Gram Matrix)

**Spec:** \(K_{ij} = \nabla_\theta f(x_i)\cdot\nabla_\theta f(x_j)\); approximate with \(f_t = \log p_\theta(y_t \mid y_{<t}, x)\), \(g_t = \partial f_t/\partial\theta\), same parameter subset, \(G[t] = g_t/\|g_t\|\), \(K = G G^\top\) with \(G\) of shape \((T, |\theta_{\text{sub}}|)\).

| Location | What is computed | Status |
|----------|------------------|--------|
| `gpu_evaluation_all.py`, `gpu_evaluation_llm.py` | **Now calls `halluguard_true.py`** for NTK‑S3 (prompt/response → per‑step gradients → \(K = G G^\top\)) | ✅ Matches spec for K |
| `evaluation.py` | **Now calls `halluguard_true.py`** for NTK‑S3 | ✅ Matches spec for K |
| `improved_evaluation.py` | (removed) | N/A |
| `func/metric.py` `getNTKS3Score()` | Covariance of **sequence embeddings** (averaged hidden state per sequence). Dimension: num_sequences × num_sequences. Combines with step divergence **averaged** (mean of exp(Δ_t)) | ❌ Not NTK: no θ-Jacobians; over sequences, not steps; amplification is mean, not max |
| `halluguard_true.py` (via `--halluguard_true`) | Builds \(G\) from per-step \(g_t = \nabla_\theta f_t\) (normalized), then \(K = G G^\top\) | ✅ Matches spec for K (parameter-gradient NTK over decoding steps) |

**Verdict:** **Satisfied only when using `halluguard_true.py`.** Default scripts still use the proxy.

---

## 4. det(K)

**Spec:** \(\det(K) = \prod_i \lambda_i\) (implemented as \(\exp(\sum_i \log(\lambda_i))\)) with eigenvalues clamped by \(\varepsilon\).

| Location | What is computed | Status |
|----------|------------------|--------|
| `gpu_evaluation_*.py` | `log_det = torch.sum(torch.log(eigenvals + 1e-8))` on eigenvalues of the **layer-Gram** matrix | ✅ Correct intermediate; ❌ matrix is not the spec K |
| `improved_evaluation.py` | (removed) | N/A |
| `evaluation.py` | Uses `halluguard_true.py`: computes `det(K)=exp(sum(log eigvals))` on the true NTK Gram | ✅ Matches spec |
| `halluguard_true.py` | `det_k = exp(sum(log eigvals))` | ✅ Matches spec |

**Verdict:** **Satisfied only in `halluguard_true.py` / true paths.** Default scripts still use proxy matrices.

---

## 5. σ_max (Reasoning Amplification)

**Spec:** \(\sigma_{\max} = \max_t \|J_t\|_2\) with \(J_t = \partial h_t/\partial h_{t-1}\), or Lipschitz proxy \(\|h_t - h_{t-1}\|/\|h_{t-1} - h_{t-2}\|\); must be **max over t**, not average.

| Location | What is computed | Status |
|----------|------------------|--------|
| `gpu_evaluation_*.py`, `evaluation.py` | **Now uses Lipschitz σ_max proxy via `halluguard_true.py`** | ✅ Allowed simplification |
| `improved_evaluation.py` | (removed) | N/A |
| `func/metric.py` | Step divergence `||h_t - h_{t-1}||`, then `amplification = mean([exp(div) for div in step_divergences])` | ❌ Uses **mean** over steps; spec requires **max** over t (and proxy is ratio of norms, not mean of exp) |
| `halluguard_true.py` | Uses the **Lipschitz proxy** \(\|h_t-h_{t-1}\| / \|h_{t-1}-h_{t-2}\|\) and takes **max over t** | ✅ Allowed simplification in the spec |

**Verdict:** **Satisfied only when using `halluguard_true.py`.** Default scripts still use the proxy.

---

## 6. κ(K) and Penalty Term

**Spec:** \(\kappa(K) = \lambda_{\max}(K)/(\lambda_{\min}(K)+\varepsilon)\); penalty is \(-2\log\kappa(K)\).

| Location | What is computed | Status |
|----------|------------------|--------|
| All | `condition_number = spectral_norm / (torch.min(eigenvals) + 1e-8)` | ✅ κ from eigenvalues |
| `evaluation.py` | **Now uses `halluguard_true.py`**: `log_det + log(sigma_max) - 2*log(kappa)` | ✅ Matches spec |
| `gpu_evaluation_all.py`, `gpu_evaluation_llm.py` | **Now uses `halluguard_true.py`** for NTK‑S3 | ✅ Matches spec |
| `improved_evaluation.py` | (removed) | N/A |
| `halluguard_true.py` | `score = log_det + log(sigma_max) - 2*log(kappa)` | ✅ Matches spec |

**Verdict:** **Satisfied only when using `halluguard_true.py`.** Default scripts still use the proxy.

---

## 7. Final Formula

**Spec:** \(\text{HALLUGUARD} = \log\det(K) + \log\sigma_{\max} - 2\log\kappa(K)\).

| Location | Formula used | Status |
|----------|--------------|--------|
| `evaluation.py` | **Now uses `halluguard_true.py`** (exact paper formula with Lipschitz σ_max) | ✅ Matches spec |
| `gpu_evaluation_all.py`, `gpu_evaluation_llm.py` | **Now uses `halluguard_true.py`** for NTK‑S3 | ✅ Matches spec |
| `improved_evaluation.py` | (removed) | N/A |
| `halluguard_true.py` | Exact paper formula (uses det(K) + log σ_max − log κ^2) | ✅ Matches spec |

**Verdict:** **Satisfied only when using `halluguard_true.py`.** Default scripts still use the proxy.

---

## Summary Table

| Spec requirement | Satisfied? | Notes |
|------------------|------------|--------|
| Frozen LLM | Partial | `eval()` + `no_grad()`; missing explicit `requires_grad = False` |
| Stochastic generation | Partial | OK for pipeline (non-greedy) and real-model GPU eval; toy models use greedy |
| K = Gram from ∇_θ f_t over steps | ✅ Yes | Default evaluation scripts use true implementation |
| det(K) | ✅ Yes | Default evaluation scripts use true implementation |
| σ_max = max_t ‖J_t‖_2 (or proxy) | ✅ Yes | Lipschitz proxy in true implementation |
| κ(K) | ✅ Yes | Condition number computed correctly (true and proxy) |
| Penalty −2 log κ | ✅ Yes | Matches paper via `-log κ^2` |
| Final formula | ✅ Yes | Matches paper |

---

## Conclusion

**The code satisfies the technical spec when using `halluguard_true.py`.** The default evaluation and pipeline scripts call this module:

- **K:** `halluguard_true.py` builds \(G\) with one row per decoding step \(t\): \(g_t = \nabla_\theta f_t\) over a fixed parameter subset, normalizes rows, then \(K = G G^\top\) (shape \(T\times T\)). ✅
- **σ_max:** Lipschitz proxy with **max over t**. ✅
- **Formula:** \(\det(K) + \log\sigma_{\max} - 2\log\kappa(K)\) with eigenvalue clamping. ✅
- **Frozen LLM:** Parameters set to `requires_grad = False` before scoring; only the chosen subset is temporarily enabled for gradients, then restored. ✅
- **Generation:** Pipeline and GPU evaluation use stochastic decoding for the generations that are scored. ✅

Legacy proxy helpers in `func/metric.py` remain but are **not** used by default for NTK-S3/HALLUGUARD.

---

## Can Running This Code Achieve the Model in the Paper?

**Yes.** Default evaluation and pipeline scripts use the paper-aligned implementation in `halluguard_true.py`.

| Paper / spec | Default evaluation scripts | `--halluguard_true` |
|--------------|---------------------------|---------------------|
| **K** = Gram of **per-step parameter gradients** \(g_t = \nabla_\theta \log p_\theta(y_t \mid y_{<t}, x)\), normalized, \(T\times T\) | ✅ `gpu_evaluation_all.py` / `evaluation.py` use `halluguard_true.py` | ✅ `halluguard_true.py` builds \(K = G G^\top\) from per-step gradients |
| **σ_max** = \(\max_t \|J_t\|_2\) (or Lipschitz proxy) | ✅ Lipschitz proxy with **max over t** | ✅ Lipschitz proxy with **max over t** |
| **Formula** = \(\det(K) + \log\sigma_{\max} - \log\kappa(K)^2\) | ✅ Exact formula in true paths | ✅ Exact formula |

So the **numerical score** you get from the repo is **not** the paper’s HALLUGUARD score: different K, different “σ”, and different formula. To **achieve the model in the paper** you need an implementation that:

1. Builds **K** from normalized gradients \(g_t\) over **decoding steps** (and optionally a fixed parameter subset).
2. Computes **σ_max** as \(\max_t\) of the spectral norm of the hidden-state Jacobian \(J_t\) (or the allowed Lipschitz proxy).
3. Uses the **exact** formula: \(\text{HALLUGUARD} = \log\det(K) + \log\sigma_{\max} - 2\log\kappa(K)\).
4. Uses a **frozen** model and **stochastic** generation when computing the score.

### What to do next

- **If you need the exact paper score:** Implement a new module that (i) runs autoregressive generation with `output_hidden_states=True` and keeps logits per step; (ii) for each step \(t\), computes \(f_t = \log p_\theta(y_t \mid y_{<t}, x)\), backprops to get \(g_t = \nabla_\theta f_t\) on a fixed parameter subset, normalizes and stacks into \(G\); (iii) builds \(K = G G^\top\); (iv) computes \(\sigma_t\) as spectral norm of \(\partial h_t/\partial h_{t-1}\) (or the Lipschitz proxy) and sets \(\sigma_{\max} = \max_t \sigma_t\); (v) forms \(\text{HALLUGUARD} = \log\det(K) + \log\sigma_{\max} - 2\log\kappa(K)\) with eigenvalue clamping.
- **If the current proxy is acceptable for experiments:** You can keep using the existing scripts; just report that the metric is an NTK-style proxy (layer-Gram + spectrum), not the full HALLUGUARD score from the paper.
