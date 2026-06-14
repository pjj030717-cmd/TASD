"""
TASD-NG: TASD with N-gram PLD draft channel (v5 - optimized).

Key optimization: only bonus token triggers extra target forward.
Non-bonus rounds compute last_t_logit from batch output.
"""
import time
import torch
from transformers import DynamicCache
from .structural_guard import StructuralGuard


def _forward_with_cache(model, input_ids, past_key_values):
    if past_key_values is None:
        past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def _greedy_sample(logits):
    if logits.dim() == 1: return logits.argmax().item()
    return logits[0, -1].argmax().item()


def _find_ngram_match(token_ids, min_len=2, max_len=8):
    ctx = len(token_ids)
    for n in range(max_len, min_len - 1, -1):
        if ctx < n + 1: continue
        pat = tuple(token_ids[-n:])
        for i in range(ctx - n):
            if tuple(token_ids[i:i + n]) == pat:
                return i, i + n
    return None, None


def tasd_ng_decode(
    target_model, draft_model, tokenizer, prompt,
    structure_type="argparse", max_new_tokens=128, draft_len=16,
    top_k_accept=3, ngram_min=2, ngram_max=8, max_ngram_draft=16,
    guard_calibrated=True,
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
    prompt_len = input_ids.shape[1]
    device = target_model.device
    draft_device = next(draft_model.parameters()).device

    guard = StructuralGuard(structure_type=structure_type, calibrated=guard_calibrated)
    token_list = input_ids[0].tolist()

    total_drafted = 0; total_accepted = 0
    target_fwd = 0; draft_fwd = 0
    repair_count = 0; guard_triggers = 0; guard_trims = 0; guard_hard_trims = 0
    ngram_rounds = 0; model_rounds = 0; fallback_rounds = 0
    generated_ids = []; max_iters = max_new_tokens + 50

    torch.cuda.synchronize(); wall_start = time.time()

    with torch.no_grad():
        t_logits_pre, t_past = _forward_with_cache(target_model, input_ids, None)
        d_in = input_ids.to(draft_device)
        d_logits_pre, d_past = _forward_with_cache(draft_model, d_in, None)
    target_fwd += 1; draft_fwd += 1
    last_t_logit = t_logits_pre[0, -1, :]
    last_d_logit = d_logits_pre[0, -1, :].to(device)

    try:
        for _ in range(max_iters):
            remaining = max_new_tokens - len(generated_ids)
            if remaining <= 0: break
            old_gen_len = len(generated_ids)

            # ── Draft ──
            ms, me = _find_ngram_match(token_list, ngram_min, ngram_max)
            used_ngram = False
            if ms is not None:
                ds, de = me, min(me + min(max_ngram_draft, remaining), len(token_list))
                if de > ds:
                    draft_tokens = token_list[ds:de][:draft_len]
                    used_ngram = True; ngram_rounds += 1

            if not used_ngram:
                model_rounds += 1
                draft_tokens = []
                d_logit = last_d_logit.clone()
                for _ in range(min(draft_len, remaining)):
                    tok = _greedy_sample(d_logit)
                    draft_tokens.append(tok)
                    if tok == tokenizer.eos_token_id: break
                    t_in = torch.tensor([[tok]], device=draft_device)
                    with torch.no_grad():
                        d_out, d_past = _forward_with_cache(draft_model, t_in, d_past)
                    draft_fwd += 1; d_logit = d_out[0, -1, :].to(device)

            if not draft_tokens:
                fallback_rounds += 1; repair_count += 1
                with torch.no_grad():
                    rl, t_past = _forward_with_cache(target_model,
                        torch.tensor([[last_t_logit.argmax().item()]], device=device), t_past)
                target_fwd += 1; tok = _greedy_sample(rl)
                generated_ids.append(tok); token_list.append(tok)
                last_t_logit = rl[0, -1, :]; last_d_logit = rl[0, -1, :].to(device)
                d_past.crop(prompt_len + old_gen_len)
                with torch.no_grad():
                    _, d_past = _forward_with_cache(draft_model, torch.tensor([[tok]], device=draft_device), d_past)
                draft_fwd += 1
                continue

            # ── Verification ──
            draft_tensor = torch.tensor([draft_tokens], device=device)
            with torch.no_grad():
                t_logits, t_past = _forward_with_cache(target_model, draft_tensor, t_past)
            target_fwd += 1

            # ── Acceptance ──
            accept_mask = []
            for i, d_tok in enumerate(draft_tokens):
                logit_i = last_t_logit if i == 0 else t_logits[0, i - 1]
                t_arg = logit_i.argmax().item()
                topk = logit_i.topk(max(1, top_k_accept)).indices.tolist()
                accept_mask.append(d_tok == t_arg or d_tok in topk)

            n_acc = 0
            for a in accept_mask:
                if a: n_acc += 1
                else: break

            total_accepted += n_acc; total_drafted += len(draft_tokens)
            candidate_ids = draft_tokens[:n_acc]

            # If nothing accepted: fallback to single AR step
            if n_acc == 0:
                fallback_rounds += 1; repair_count += 1
                with torch.no_grad():
                    rl, t_past = _forward_with_cache(target_model,
                        torch.tensor([[last_t_logit.argmax().item()]], device=device), t_past)
                target_fwd += 1; tok = _greedy_sample(rl)
                generated_ids.append(tok); token_list.append(tok)
                last_t_logit = rl[0, -1, :]; last_d_logit = rl[0, -1, :].to(device)
                d_past.crop(prompt_len + old_gen_len)
                with torch.no_grad():
                    _, d_past = _forward_with_cache(draft_model, torch.tensor([[tok]], device=draft_device), d_past)
                draft_fwd += 1
                continue

            # ── Guard check ──
            guard_triggered = False
            if candidate_ids:
                check_ids = generated_ids + candidate_ids
                check_text = tokenizer.decode(check_ids, skip_special_tokens=True)
                safe, safe_count, risk_type = guard.check(check_text, check_ids, tokenizer)
                if not safe and safe_count > 0:
                    guard_triggered = True; guard_triggers += 1
                    guard_trims += len(check_ids) - max(safe_count, 0)
                    if risk_type and "off_structure" in str(risk_type):
                        guard_hard_trims += len(check_ids) - max(safe_count, 0)
                    candidate_ids = check_ids[len(generated_ids):max(safe_count, 0)]
                    n_acc = len(candidate_ids)

            # ── Bonus (only if all accepted and no guard trim) ──
            bonus = []
            has_bonus = False
            if n_acc == len(draft_tokens) and n_acc > 0 and n_acc < remaining and not guard_triggered:
                bonus_tok = _greedy_sample(t_logits[0, n_acc - 1])
                bonus = [bonus_tok]; has_bonus = True

            final_tokens = candidate_ids + bonus
            generated_ids.extend(final_tokens)
            token_list.extend(final_tokens)

            # ── KV sync ──
            t_past.crop(prompt_len + len(generated_ids))
            d_past.crop(prompt_len + old_gen_len)
            if final_tokens:
                with torch.no_grad():
                    dl_out, d_past = _forward_with_cache(draft_model,
                        torch.tensor([final_tokens], device=draft_device), d_past)
                draft_fwd += 1; last_d_logit = dl_out[0, -1, :].to(device)

            # ── Update last_t_logit ──
            if has_bonus:
                # Need extra target forward for bonus token
                with torch.no_grad():
                    tl_out, t_past = _forward_with_cache(target_model,
                        torch.tensor([bonus], device=device), t_past)
                target_fwd += 1; last_t_logit = tl_out[0, -1, :]
            elif n_acc > 0:
                last_t_logit = t_logits[0, n_acc - 1]
            # else: keep last_t_logit unchanged (shouldn't happen normally)

            if tokenizer.eos_token_id in final_tokens:
                epos = generated_ids.index(tokenizer.eos_token_id)
                generated_ids = generated_ids[:epos]; break

    except Exception as e:
        print(f"  [WARN] TASD-NG: {e}")

    torch.cuda.synchronize(); wall = time.time() - wall_start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "elapsed_time": wall,
        "tokens_per_second": (prompt_len+len(generated_ids))/wall if wall>0 else 0,
        "generated_text": text, "generated_tokens": generated_ids,
        "stats": {
            "accept_rate": total_accepted/total_drafted if total_drafted>0 else 0,
            "total_drafted": total_drafted, "total_accepted": total_accepted,
            "repair_count": repair_count, "target_fwd": target_fwd, "draft_fwd": draft_fwd,
            "guard_trigger_count": guard_triggers, "trim_count": guard_trims,
            "hard_trim_count": guard_hard_trims,
            "ngram_rounds": ngram_rounds, "model_draft_rounds": model_rounds,
            "fallback_rounds": fallback_rounds,
        },
    }
