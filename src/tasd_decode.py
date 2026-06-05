"""
TASD: Training-free Structure-Aware Speculative Decoding (with proper KV cache).

Core flow:
1. Pre-fill KV cache for prompt on both target and draft models
2. Draft model generates multi-block tokens incrementally (using its KV cache)
3. Target model verifies draft tokens in one batch forward (using its KV cache)
4. Prefix acceptance with window-based logic
5. Structural Guard checks for risks (token-level trim)
6. Accept safe prefix, TRIM KV caches to accepted length, continue

Key optimization:
- Verification forward already computes KV cache for ALL draft tokens
- We TRIM the cache to accepted length instead of re-forwarding
- This eliminates redundant forward passes

Missing items addressed:
1. draft_blocks / multi-block draft
2. Per-block early stop
3. Block-level accept/reject stats
4. max_new_tokens remaining budget constraint
5. EOS handling with eos_drafted/eos_accepted/stop_reason
6. Repair token recording
7. Prompt seed check
8. Reference not used in decoding (documented)
9. AR/TASD length alignment (handled by caller)
10. Warmup (handled by caller)
11. CUDA synchronize
12. Memory recording (handled by caller)
13. Failure/fallback mechanism
"""
import time
import re
import torch
from transformers import DynamicCache
from .structural_guard import StructuralGuard


def _forward_with_cache(model, input_ids, past_key_values):
    """Run model forward pass, return logits and updated past_key_values as DynamicCache."""
    if past_key_values is None:
        past_key_values = DynamicCache()

    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def _greedy_sample(logits):
    """Greedy sampling from logits. Handles both 1D and 2D tensors."""
    if logits.dim() == 1:
        return logits.argmax().item()
    return logits[0, -1].argmax().item()


def _trim_past_key_values(past_key_values, keep_len):
    """
    Trim past_key_values to keep only the first keep_len tokens.
    Works with both DynamicCache and tuple formats.
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, 'crop'):
        past_key_values.crop(keep_len)
        return past_key_values

    trimmed = []
    for layer_past in past_key_values:
        if isinstance(layer_past, (list, tuple)):
            trimmed_layer = tuple(
                t[:, :, :keep_len, :] if t is not None else None for t in layer_past
            )
            trimmed.append(trimmed_layer)
        elif layer_past is not None:
            trimmed.append(layer_past[:, :, :keep_len, :])
        else:
            trimmed.append(None)

    return tuple(trimmed)


def _check_tokenizer_consistency(target_tokenizer, draft_tokenizer):
    """
    Check that target and draft tokenizers share the same vocab and encoding.
    Returns (same_vocab, same_encoding).
    """
    same_vocab = target_tokenizer.get_vocab() == draft_tokenizer.get_vocab()
    test_str = "def hello_world(x: int) -> str:\n    return str(x)"
    target_ids = target_tokenizer.encode(test_str)
    draft_ids = draft_tokenizer.encode(test_str)
    same_encoding = target_ids == draft_ids
    return same_vocab, same_encoding


def _check_prompt_seed(prompt, structure_type):
    """
    Check if prompt contains enough structural seed for the given structure_type.
    Returns (valid, seed_count, reason).
    """
    if structure_type == "argparse":
        patterns = [r"add_argument", r"ArgumentParser", r"click\.option", r"click\.argument"]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, prompt))
        valid = count >= 2
        reason = f"argparse seed count={count}" if not valid else ""
        return valid, count, reason
    elif structure_type == "dict_config":
        has_dict = "{" in prompt and "}" in prompt
        has_list = "[" in prompt and "]" in prompt
        count = int(has_dict) + int(has_list)
        valid = count >= 1
        reason = "no dict/list structure in prompt" if not valid else ""
        return valid, count, reason
    elif structure_type in ("openmmlab", "openmmlab_config"):
        patterns = [r"model\s*=", r"pipeline\s*=", r"dataloader\s*=", r"train_cfg", r"test_cfg"]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, prompt))
        valid = count >= 1
        reason = f"openmmlab seed count={count}" if not valid else ""
        return valid, count, reason
    else:
        return True, 0, ""


def _cuda_sync():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def _is_early_stop_condition(token, tokenizer, structure_type, generated_text_so_far):
    """
    Check if we should stop drafting early based on:
    - EOS token
    - Obvious structure end markers
    - High-risk content detected by lightweight check
    """
    if token == tokenizer.eos_token_id:
        return True, "eos"

    # Check for structure end markers
    decoded = tokenizer.decode([token], skip_special_tokens=True)
    if structure_type == "argparse":
        if decoded.strip().startswith(("def ", "class ", "import ", "from ")):
            return True, "off_structure"
    elif structure_type == "dict_config":
        if decoded.strip().startswith(("def ", "class ", "import ", "from ")):
            return True, "off_structure"

    return False, ""


def tasd_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt,
    structure_type="argparse",
    max_new_tokens=128,
    temperature=0.0,
    draft_len=8,
    top_k_accept=3,
    min_token_prob=1e-4,
    prefix_budget=0.2,
    window_len=2,
    draft_tokenizer=None,
    draft_blocks=2,
    enable_guard=True,
    enable_relaxed_accept=True,
    adaptive_policy=None,
):
    """
    TASD speculative decoding with structural guard, proper KV cache,
    multi-block draft, and comprehensive stats.

    Args:
        enable_guard: If False, skip structural guard check
        enable_relaxed_accept: If False, only accept draft_tok == target_argmax (strict)

    Note on reference: reference is NOT used in decoding. It is only
    passed to the evaluator for structural quality assessment.
    """
    # --- Tokenizer consistency check ---
    effective_draft_tok = draft_tokenizer if draft_tokenizer is not None else tokenizer
    same_vocab, same_encoding = _check_tokenizer_consistency(tokenizer, effective_draft_tok)
    if not same_vocab or not same_encoding:
        raise ValueError(
            f"Tokenizer mismatch detected: same_vocab={same_vocab}, same_encoding={same_encoding}. "
            "Target and draft models must share the same tokenizer."
        )

    # --- Prompt seed check ---
    seed_valid, prompt_seed_count, seed_reason = _check_prompt_seed(prompt, structure_type)
    invalid_seed = not seed_valid

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
    prompt_len = input_ids.shape[1]
    device = target_model.device
    draft_device = next(draft_model.parameters()).device

    guard = StructuralGuard(structure_type=structure_type)

    # Stats
    total_drafted = 0
    total_accepted = 0
    token_accept = 0
    prefix_accept = 0
    prefix_budget_used = 0.0
    window_accept_count = 0
    target_model_forwards = 0
    draft_model_forwards = 0
    repair_count = 0
    consecutive_repair_count = 0
    draft_time_total = 0.0
    target_time_total = 0.0
    trim_reasons = []
    repair_reasons = []
    repair_records = []  # List of {token, text, reason}
    failed = False
    error_type = None
    error_msg = None
    stop_reason = None

    # Multi-block draft stats
    requested_draft_blocks = draft_blocks
    actual_draft_blocks_list = []
    early_stop_reasons = []
    block_accept_count = 0
    block_partial_accept_count = 0
    block_reject_count = 0
    first_reject_block_id = -1

    # EOS stats
    eos_drafted = False
    eos_accepted = False

    # Adaptive policy stats
    adaptive_round_drafted = []
    adaptive_round_accepted = []
    adaptive_round_top3_hits = []
    adaptive_round_top5_hits = []
    adaptive_topk_computed = 0

    generated_ids = []
    max_iterations = max_new_tokens + 50

    # --- Pre-fill KV cache for prompt on both models ---
    _cuda_sync()
    wall_start = time.time()

    with torch.no_grad():
        target_logits_prefill, target_past = _forward_with_cache(target_model, input_ids, None)
        draft_input = input_ids.to(draft_device)
        draft_logits_prefill, draft_past = _forward_with_cache(draft_model, draft_input, None)

    target_model_forwards += 1
    draft_model_forwards += 1

    last_target_logit = target_logits_prefill[0, -1, :]
    last_draft_logit = draft_logits_prefill[0, -1, :].to(device)

    try:
        for iteration in range(max_iterations):
            remaining = max_new_tokens - len(generated_ids)
            if remaining <= 0:
                stop_reason = "max_tokens"
                break

            # --- Adaptive policy: update draft_len / top_k_accept ---
            if adaptive_policy is not None:
                draft_len, top_k_accept = adaptive_policy.get_params()

            # If too many consecutive repairs, degrade to AR
            if consecutive_repair_count >= 5:
                repair_reasons.append(f"consecutive_repair_limit:{consecutive_repair_count}")
                stop_reason = "consecutive_repair_limit"
                current_ids = torch.cat(
                    [input_ids, torch.tensor([generated_ids], device=device)], dim=1
                ) if generated_ids else input_ids
                with torch.no_grad():
                    ar_output = target_model.generate(
                        current_ids,
                        max_new_tokens=remaining,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = ar_output[0][current_ids.shape[1]:].tolist()
                generated_ids.extend(new_tokens)
                break

            # --- Multi-block draft phase ---
            draft_start = time.time()
            _cuda_sync()

            all_draft_tokens = []
            all_draft_logits_list = []
            block_boundaries = []  # (start_idx, end_idx) for each block
            actual_blocks = 0
            block_early_stop = False
            block_early_stop_reason = ""

            with torch.no_grad():
                next_token = _greedy_sample(last_draft_logit)

                for block_id in range(draft_blocks):
                    # Check remaining budget
                    already_this_round = len(all_draft_tokens)
                    effective_remaining = remaining - already_this_round
                    if effective_remaining <= 0:
                        block_early_stop = True
                        block_early_stop_reason = "budget_exhausted"
                        break

                    tokens_this_block = min(draft_len, effective_remaining)
                    block_start_idx = len(all_draft_tokens)

                    for _ in range(tokens_this_block):
                        all_draft_tokens.append(next_token)

                        # Check EOS
                        if next_token == tokenizer.eos_token_id:
                            eos_drafted = True
                            break

                        # Check early stop conditions
                        should_stop, stop_reason_str = _is_early_stop_condition(
                            next_token, tokenizer, structure_type,
                            tokenizer.decode(generated_ids + all_draft_tokens, skip_special_tokens=True)
                        )
                        if should_stop:
                            block_early_stop = True
                            block_early_stop_reason = stop_reason_str
                            break

                        # Incremental forward pass with KV cache
                        next_tensor = torch.tensor([[next_token]], device=draft_device)
                        draft_logits, draft_past = _forward_with_cache(
                            draft_model, next_tensor, draft_past
                        )
                        draft_model_forwards += 1
                        all_draft_logits_list.append(draft_logits[0, -1, :].to(device))
                        next_token = _greedy_sample(draft_logits)

                    block_end_idx = len(all_draft_tokens)
                    block_boundaries.append((block_start_idx, block_end_idx))
                    actual_blocks += 1

                    if block_early_stop:
                        break

                    # If we hit EOS in this block, stop drafting more blocks
                    if eos_drafted:
                        block_early_stop_reason = "eos"
                        break

            _cuda_sync()
            draft_time_total += time.time() - draft_start

            actual_draft_blocks_list.append(actual_blocks)
            if block_early_stop:
                early_stop_reasons.append(block_early_stop_reason)

            draft_tokens = all_draft_tokens
            draft_logits_list = all_draft_logits_list

            if not draft_tokens:
                repair_count += 1
                consecutive_repair_count += 1
                repair_start = time.time()
                _cuda_sync()
                with torch.no_grad():
                    repair_logits, target_past = _forward_with_cache(
                        target_model,
                        torch.tensor([[last_target_logit.argmax().item()]], device=device),
                        target_past,
                    )
                    next_token = _greedy_sample(repair_logits)
                _cuda_sync()
                target_time_total += time.time() - repair_start
                target_model_forwards += 1
                generated_ids.append(next_token)
                last_target_logit = repair_logits[0, -1, :]
                last_draft_logit = repair_logits[0, -1, :]
                repair_text = tokenizer.decode([next_token], skip_special_tokens=True)
                repair_records.append({
                    "token": next_token,
                    "text": repair_text,
                    "reason": "all_draft_rejected",
                })
                repair_reasons.append("all_draft_rejected")
                continue

            # --- Target verification with KV cache ---
            target_start = time.time()
            _cuda_sync()

            draft_tensor = torch.tensor([draft_tokens], device=device)
            target_logits, target_past = _forward_with_cache(
                target_model, draft_tensor, target_past
            )
            _cuda_sync()
            target_time_total += time.time() - target_start
            target_model_forwards += 1

            # Verify each draft token
            accept_mask = []
            round_top3_hits = 0
            round_top5_hits = 0
            for i, draft_tok in enumerate(draft_tokens):
                if i == 0:
                    logit_for_token = last_target_logit
                else:
                    logit_idx = i - 1
                    if logit_idx >= target_logits.shape[1]:
                        accept_mask.append(False)
                        continue
                    logit_for_token = target_logits[0, logit_idx]

                target_argmax = logit_for_token.argmax().item()
                if draft_tok == target_argmax:
                    accept_mask.append(True)
                elif enable_relaxed_accept:
                    probs = torch.softmax(logit_for_token, dim=-1)
                    _, topk_indices = torch.topk(probs, top_k_accept)
                    if draft_tok in topk_indices.tolist():
                        accept_mask.append(True)
                    elif probs[draft_tok].item() >= min_token_prob:
                        accept_mask.append(True)
                    else:
                        accept_mask.append(False)

                # --- Top-k hit tracking (for adaptive policy) ---
                if adaptive_policy is not None:
                    probs = torch.softmax(logit_for_token, dim=-1)
                    _, top3_idx = torch.topk(probs, 3)
                    _, top5_idx = torch.topk(probs, 5)
                    if draft_tok in top3_idx.tolist():
                        round_top3_hits += 1
                    if draft_tok in top5_idx.tolist():
                        round_top5_hits += 1

            total_drafted += len(draft_tokens)

            # --- Block-level accept/reject stats ---
            for block_start, block_end in block_boundaries:
                block_accepts = sum(accept_mask[block_start:block_end])
                block_total = block_end - block_start
                if block_total == 0:
                    continue

                if block_accepts == block_total:
                    block_accept_count += 1
                elif block_accepts > 0:
                    block_partial_accept_count += 1
                    if first_reject_block_id < 0:
                        first_reject_block_id = len(actual_draft_blocks_list) - 1
                else:
                    block_reject_count += 1
                    if first_reject_block_id < 0:
                        first_reject_block_id = len(actual_draft_blocks_list) - 1

            # --- Compute accepted prefix ---
            strict_prefix_len = 0
            for i, accepted in enumerate(accept_mask):
                if accepted:
                    strict_prefix_len = i + 1
                else:
                    break

            accepted_tokens = draft_tokens[:strict_prefix_len]
            token_accept += strict_prefix_len

            # Window acceptance beyond strict prefix
            if enable_relaxed_accept and strict_prefix_len < len(draft_tokens):
                window_start = strict_prefix_len
                while window_start < len(draft_tokens):
                    window_end = min(window_start + window_len, len(draft_tokens))
                    window_accepts = sum(accept_mask[window_start:window_end])
                    window_total = window_end - window_start
                    if window_accepts >= window_total * 0.5:
                        accepted_tokens.extend(draft_tokens[window_start:window_end])
                        window_accept_count += 1
                        window_start = window_end
                    else:
                        break

            # Prefix budget acceptance
            if enable_relaxed_accept and len(accepted_tokens) < len(draft_tokens):
                remaining_draft = draft_tokens[len(accepted_tokens):]
                for idx, tok in enumerate(remaining_draft):
                    pos_in_draft = len(accepted_tokens) + idx
                    if pos_in_draft == 0:
                        logit_for_tok = last_target_logit
                    else:
                        logit_idx = pos_in_draft - 1
                        if logit_idx >= target_logits.shape[1]:
                            break
                        logit_for_tok = target_logits[0, logit_idx]

                    log_probs = torch.log_softmax(logit_for_tok, dim=-1)
                    target_best_logprob = log_probs.max().item()
                    draft_token_logprob = log_probs[tok].item()
                    risk = max(0.0, target_best_logprob - draft_token_logprob)

                    if prefix_budget_used + risk <= prefix_budget:
                        accepted_tokens.append(tok)
                        prefix_budget_used += risk
                    else:
                        break

            prefix_accept += len(accepted_tokens)

            # --- Guard check ---
            if enable_guard:
                accepted_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
                safe, guard_keep_count, risk_type = guard.check(
                    accepted_text, tokens=accepted_tokens, tokenizer=tokenizer
                )

                if not safe:
                    guard.trim_count += 1
                    trim_reasons.append(risk_type)
                    accepted_tokens = accepted_tokens[:guard_keep_count]

            accepted_count = len(accepted_tokens)

            # --- Apply accepted tokens or repair ---
            if accepted_tokens:
                generated_ids.extend(accepted_tokens)
                consecutive_repair_count = 0

                # Check if EOS was accepted
                if tokenizer.eos_token_id in accepted_tokens:
                    eos_accepted = True
                    stop_reason = "eos"

                new_cache_len = prompt_len + len(generated_ids)
                target_past = _trim_past_key_values(target_past, new_cache_len)
                draft_past = _trim_past_key_values(draft_past, new_cache_len)

                # Update logits
                if accepted_count <= target_logits.shape[1]:
                    last_target_logit = target_logits[0, accepted_count - 1, :]
                else:
                    last_target_logit = target_logits[0, -1, :]

                if accepted_count <= len(draft_logits_list):
                    last_draft_logit = draft_logits_list[accepted_count - 1]
                else:
                    last_draft_logit = draft_logits_list[-1] if draft_logits_list else last_draft_logit
            else:
                repair_count += 1
                consecutive_repair_count += 1
                repair_start = time.time()
                _cuda_sync()
                with torch.no_grad():
                    next_token_tensor = torch.tensor(
                        [[last_target_logit.argmax().item()]], device=device
                    )
                    repair_logits, target_past = _forward_with_cache(
                        target_model, next_token_tensor, target_past
                    )
                    next_token = _greedy_sample(repair_logits)
                _cuda_sync()
                target_time_total += time.time() - repair_start
                target_model_forwards += 1
                generated_ids.append(next_token)
                last_target_logit = repair_logits[0, -1, :]
                last_draft_logit = repair_logits[0, -1, :]
                repair_text = tokenizer.decode([next_token], skip_special_tokens=True)
                repair_records.append({
                    "token": next_token,
                    "text": repair_text,
                    "reason": "all_draft_rejected",
                })
                repair_reasons.append("all_draft_rejected")

            total_accepted += accepted_count

            # --- Record round stats for adaptive policy ---
            if adaptive_policy is not None:
                # safe is only defined inside enable_guard block; default True if guard disabled
                guard_safe = locals().get("safe", True)
                guard_triggered_this_round = 1 if (enable_guard and not guard_safe) else 0
                has_off_structure = any(
                    r in ("off_structure", "def_class_import", "import_outside", "class_def")
                    for r in trim_reasons[-1:] if trim_reasons
                )
                adaptive_policy.record_round(
                    drafted=len(draft_tokens),
                    accepted=accepted_count,
                    top3_hits=round_top3_hits,
                    top5_hits=round_top5_hits,
                    repair_count=1 if accepted_count == 0 else 0,
                    guard_triggers=guard_triggered_this_round,
                    off_structure=has_off_structure,
                )
                adaptive_topk_computed += 1

            # Check for EOS in generated
            if tokenizer.eos_token_id in generated_ids:
                eos_pos = generated_ids.index(tokenizer.eos_token_id)
                generated_ids = generated_ids[:eos_pos]
                if stop_reason is None:
                    stop_reason = "eos"
                break

            # Check max tokens
            if len(generated_ids) >= max_new_tokens:
                stop_reason = "max_tokens"
                break

    except Exception as e:
        failed = True
        error_type = type(e).__name__
        error_msg = str(e)
        if stop_reason is None:
            stop_reason = f"error:{error_type}"

    if stop_reason is None:
        stop_reason = "max_iterations"

    # --- Compute final metrics ---
    _cuda_sync()
    wall_time = time.time() - wall_start
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_length = len(generated_ids)

    draft_time_share = draft_time_total / wall_time if wall_time > 0 else 0.0
    tps = generated_length / wall_time if wall_time > 0 else 0.0
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0

    stats = {
        # Speed
        "wall_time": round(wall_time, 4),
        "tokens_per_second": round(tps, 2),
        "generated_length": generated_length,
        # Draft / Target
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate": round(accept_rate, 4),
        "target_model_forwards": target_model_forwards,
        "draft_model_forwards": draft_model_forwards,
        "draft_time_total": round(draft_time_total, 4),
        "target_time_total": round(target_time_total, 4),
        "draft_time_share": round(draft_time_share, 4),
        # Quality budget
        "token_accept": token_accept,
        "prefix_accept": prefix_accept,
        "prefix_budget_used": round(prefix_budget_used, 6),
        "window_accept_count": window_accept_count,
        "window_len": window_len,
        "top_k_accept": top_k_accept,
        "min_token_prob": min_token_prob,
        # Guard
        "guard_trigger_count": guard.trigger_count,
        "trim_count": guard.trim_count,
        "repair_count": repair_count,
        "consecutive_repair_count": consecutive_repair_count,
        "trim_reasons": trim_reasons,
        "repair_reasons": repair_reasons,
        "repair_records": repair_records,
        # Multi-block draft
        "requested_draft_blocks": requested_draft_blocks,
        "actual_draft_blocks": actual_draft_blocks_list,
        "early_stop_reasons": early_stop_reasons,
        "block_accept_count": block_accept_count,
        "block_partial_accept_count": block_partial_accept_count,
        "block_reject_count": block_reject_count,
        "first_reject_block_id": first_reject_block_id,
        # EOS
        "eos_drafted": eos_drafted,
        "eos_accepted": eos_accepted,
        "stop_reason": stop_reason,
        # Prompt seed
        "prompt_seed_count": prompt_seed_count,
        "invalid_seed": invalid_seed,
        # Exception
        "failed": failed,
        "error_type": error_type,
        "error_msg": error_msg,
        # Adaptive policy
        "adaptive_topk_computed": adaptive_topk_computed,
        "adaptive_policy_summary": adaptive_policy.get_summary() if adaptive_policy is not None else None,
    }

    return {
        "generated_text": generated_text,
        "tokens_per_second": round(tps, 2),
        "generated_tokens": generated_length,
        "elapsed_time": round(wall_time, 4),
        "stats": stats,
    }
