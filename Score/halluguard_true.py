import torch


def _get_hf_outputs(outputs):
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    return outputs.logits, outputs.hidden_states


def _select_param_subset(model, strategy="last_block"):
    named_params = list(model.named_parameters())
    if strategy == "all":
        return named_params
    if strategy.startswith("name:"):
        substr = strategy.split(":", 1)[1]
        filtered = [(n, p) for n, p in named_params if substr in n]
        return filtered if filtered else named_params

    # Heuristic: last transformer block for common architectures
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        return list(model.model.layers[-1].named_parameters())
    if hasattr(model, "transformer") and hasattr(model.transformer, "h") and len(model.transformer.h) > 0:
        return list(model.transformer.h[-1].named_parameters())
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        if len(model.model.decoder.layers) > 0:
            return list(model.model.decoder.layers[-1].named_parameters())
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers") and len(model.decoder.layers) > 0:
        return list(model.decoder.layers[-1].named_parameters())

    return named_params


def _compute_sigma_max_lipschitz(hidden_states, prompt_len, gen_len, layer_idx=-1, eps=1e-8):
    if hidden_states is None:
        return torch.tensor(1.0, device="cpu")
    layer = layer_idx if layer_idx >= 0 else (len(hidden_states) - 1)
    hs = hidden_states[layer][0]  # [seq_len, hidden]
    start = prompt_len
    end = prompt_len + gen_len
    if end - start < 3:
        return torch.tensor(1.0, device=hs.device)

    ratios = []
    for t in range(start + 2, end):
        h_t = hs[t]
        h_t1 = hs[t - 1]
        h_t2 = hs[t - 2]
        num = torch.norm(h_t - h_t1)
        den = torch.norm(h_t1 - h_t2)
        ratios.append(num / (den + eps))
    return torch.max(torch.stack(ratios)) if ratios else torch.tensor(1.0, device=hs.device)


def compute_halluguard_score(
    model,
    input_ids,
    generated_ids,
    attention_mask=None,
    layer_idx=-1,
    param_subset="last_block",
    eps=1e-8,
    sigma_mode="lipschitz",
):
    """
    Compute HALLUGUARD score per spec (K from per-step gradients, sigma_max from rollout proxy).
    input_ids: tensor [seq_len_prompt]
    generated_ids: tensor [seq_len_gen]
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated_ids = generated_ids.to(device)
    full_ids = torch.cat([input_ids, generated_ids], dim=0).unsqueeze(0)
    if attention_mask is None:
        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(getattr(model, "config", None), "eos_token_id", 0)
        attention_mask = (full_ids != pad_id).long()
    else:
        # If attention_mask only covers the prompt, extend it for generated tokens.
        if attention_mask.dim() == 2 and attention_mask.shape[1] == input_ids.shape[0]:
            gen_mask = torch.ones((1, generated_ids.shape[0]), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, gen_mask], dim=1)

    # Select parameter subset
    named_params = _select_param_subset(model, strategy=param_subset)
    params = [p for _, p in named_params]

    # Freeze everything, then enable gradients for subset
    orig_requires = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    model.eval()
    with torch.enable_grad():
        outputs = model(full_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits, hidden_states = _get_hf_outputs(outputs)
        log_probs = torch.log_softmax(logits, dim=-1)

        prompt_len = input_ids.shape[0]
        gen_len = generated_ids.shape[0]
        grads = []
        for t in range(gen_len):
            pos = prompt_len + t
            token_id = full_ids[0, pos]
            f_t = log_probs[0, pos - 1, token_id]
            grad_list = torch.autograd.grad(
                f_t,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            flat = []
            for g, p in zip(grad_list, params):
                if g is None:
                    flat.append(torch.zeros_like(p).reshape(-1))
                else:
                    flat.append(g.reshape(-1))
            g_vec = torch.cat(flat)
            g_norm = torch.norm(g_vec) + eps
            grads.append(g_vec / g_norm)

        G = torch.stack(grads)  # [T, P]
        K = torch.matmul(G, G.T)
        eigvals = torch.linalg.eigvalsh(K)
        eigvals = torch.clamp(eigvals, min=eps)
        log_det = torch.sum(torch.log(eigvals))
        det_k = torch.exp(log_det)
        kappa = torch.max(eigvals) / (torch.min(eigvals) + eps)

        if sigma_mode == "lipschitz":
            sigma_max = _compute_sigma_max_lipschitz(hidden_states, prompt_len, gen_len, layer_idx, eps)
        else:
            raise NotImplementedError("Only sigma_mode='lipschitz' is implemented.")

        score = det_k + torch.log(sigma_max + eps) - 2.0 * torch.log(kappa + eps)

    # Restore requires_grad
    for p, orig in zip(model.parameters(), orig_requires):
        p.requires_grad_(orig)

    return {
        "score": score.detach().cpu().item(),
        "det_k": det_k.detach().cpu().item(),
        "log_det": log_det.detach().cpu().item(),
        "sigma_max": sigma_max.detach().cpu().item(),
        "kappa": kappa.detach().cpu().item(),
        "T": gen_len,
    }
