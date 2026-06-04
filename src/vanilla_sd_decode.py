"""
Greedy Speculative Decoding (greedy SD, no structural guard).

Greedy speculative decoding flow:
1. Draft model generates N tokens greedily
2. Target model verifies all N tokens in one batch forward
3. Accept longest prefix where draft_token == target_argmax
4. If all N accepted, sample one more from target
5. If rejected at position k, use target's token at position k
6. Repeat until max_new_tokens reached

No structural guard, no prefix budget, no window logic.
Note: This is a greedy variant of speculative decoding, not the full
rejection-sampling version from the original paper.
"""
import time
import torch
from transformers import DynamicCache


def _forward_with_cache(model, input_ids, past_key_values):
    """Run model forward pass, return logits and updated past_key_values."""
    if past_key_values is None:
        past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def _greedy_sample(logits):
    """Greedy sampling from logits."""
    if logits.dim() == 1:
        return logits.argmax().item()
    return logits[0, -1].argmax().item()


def greedy_sd_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    draft_len=5,
):
    """
    Greedy speculative decoding.

    Args:
        target_model: target model (larger)
        draft_model: draft model (smaller)
        tokenizer: shared tokenizer
        prompt: input prompt string
        max_new_tokens: max tokens to generate
        draft_len: number of tokens to draft per round

    Returns:
        generated_text, tokens_per_second, generated_tokens, stats
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
    prompt_len = input_ids.shape[1]
    device = target_model.device
    draft_device = next(draft_model.parameters()).device

    # Stats
    total_drafted = 0
    total_accepted = 0
    target_forwards = 0
    draft_forwards = 0
    draft_time_total = 0.0
    target_time_total = 0.0
    round_stats = []

    generated_ids = []
    max_iterations = max_new_tokens + 50

    # Pre-fill KV cache for both models
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_start = time.time()

    with torch.no_grad():
        target_logits_prefill, target_past = _forward_with_cache(target_model, input_ids, None)
        draft_input = input_ids.to(draft_device)
        draft_logits_prefill, draft_past = _forward_with_cache(draft_model, draft_input, None)

    target_forwards += 1
    draft_forwards += 1

    # last_target_logit predicts the first token to be generated
    last_target_logit = target_logits_prefill[0, -1, :]
    last_draft_logit = draft_logits_prefill[0, -1, :].to(device)

    for iteration in range(max_iterations):
        remaining = max_new_tokens - len(generated_ids)
        if remaining <= 0:
            break

        # --- Draft phase ---
        draft_start = time.time()
        draft_tokens = []

        with torch.no_grad():
            next_token = _greedy_sample(last_draft_logit)

            for _ in range(min(draft_len, remaining)):
                draft_tokens.append(next_token)

                tok_tensor = torch.tensor([[next_token]], device=draft_device)
                draft_logits, draft_past = _forward_with_cache(draft_model, tok_tensor, draft_past)
                draft_forwards += 1
                last_draft_logit = draft_logits[0, -1, :].to(device)
                next_token = _greedy_sample(last_draft_logit)

        _cuda_sync()
        draft_time_total += time.time() - draft_start

        # --- Target verification phase ---
        target_start = time.time()

        # Forward draft tokens through target with KV cache
        # target_past currently covers: prompt + all previously generated tokens
        # After forwarding draft_tokens, we get logits for each position
        # logits[0, j, :] predicts the token at position (current_seq_len + j + 1)
        # which corresponds to draft_tokens[j+1]
        # So to verify draft_tokens[j]:
        #   - j=0: use last_target_logit (predicts first new token)
        #   - j>0: use target_logits[0, j-1, :]
        draft_tensor = torch.tensor([draft_tokens], device=device)
        with torch.no_grad():
            target_logits, target_past = _forward_with_cache(target_model, draft_tensor, target_past)
            target_forwards += 1

        _cuda_sync()
        target_time_total += time.time() - target_start

        # --- Acceptance check ---
        accepted_count = 0
        round_accept_details = []

        for j in range(len(draft_tokens)):
            if j == 0:
                verify_logits = last_target_logit
            else:
                verify_logits = target_logits[0, j - 1, :]

            target_argmax = verify_logits.argmax().item()
            match = (draft_tokens[j] == target_argmax)
            round_accept_details.append({
                "j": j,
                "draft_token": draft_tokens[j],
                "target_argmax": target_argmax,
                "match": match,
            })

            if match:
                accepted_count += 1
            else:
                break

        total_drafted += len(draft_tokens)
        total_accepted += accepted_count
        round_stats.append({
            "round": iteration,
            "drafted": len(draft_tokens),
            "accepted": accepted_count,
            "details": round_accept_details[:5],
        })

        # Add accepted tokens
        for j in range(accepted_count):
            generated_ids.append(draft_tokens[j])

        # --- Handle acceptance outcome ---
        if accepted_count == len(draft_tokens):
            # All draft tokens accepted: sample one more from target
            # The logit for the next token is at target_logits[0, -1, :]
            # (or last_target_logit if no draft tokens were generated)
            if len(draft_tokens) > 0:
                extra_logit = target_logits[0, -1, :]
            else:
                extra_logit = last_target_logit

            extra_token = _greedy_sample(extra_logit)
            generated_ids.append(extra_token)
            total_accepted += 1
            total_drafted += 1

            # Update KV cache with extra token and get next logit in ONE forward
            extra_tensor = torch.tensor([[extra_token]], device=device)
            with torch.no_grad():
                new_logits, target_past = _forward_with_cache(target_model, extra_tensor, target_past)
                target_forwards += 1

            draft_extra_tensor = torch.tensor([[extra_token]], device=draft_device)
            with torch.no_grad():
                _, draft_past = _forward_with_cache(draft_model, draft_extra_tensor, draft_past)
                draft_forwards += 1

            last_target_logit = new_logits[0, -1, :]
            last_draft_logit = last_target_logit.to(draft_device)

        elif accepted_count == 0:
            # No tokens accepted: use target's prediction for first position
            fallback_token = _greedy_sample(last_target_logit)
            generated_ids.append(fallback_token)

            # Update KV cache with fallback token and get next logit in ONE forward
            fallback_tensor = torch.tensor([[fallback_token]], device=device)
            with torch.no_grad():
                new_logits, target_past = _forward_with_cache(target_model, fallback_tensor, target_past)
                target_forwards += 1

            draft_fallback_tensor = torch.tensor([[fallback_token]], device=draft_device)
            with torch.no_grad():
                _, draft_past = _forward_with_cache(draft_model, draft_fallback_tensor, draft_past)
                draft_forwards += 1

            last_target_logit = new_logits[0, -1, :]
            last_draft_logit = last_target_logit.to(draft_device)

        else:
            # Partial acceptance: use target's token at rejection position
            # The rejection happened at position accepted_count
            # target_logits[0, accepted_count - 1, :] predicts draft_tokens[accepted_count]
            # We use target's argmax as the replacement
            replacement_token = target_logits[0, accepted_count - 1, :].argmax().item()
            generated_ids.append(replacement_token)
            total_accepted += 1

            # Update KV cache with replacement token and get next logit in ONE forward
            replacement_tensor = torch.tensor([[replacement_token]], device=device)
            with torch.no_grad():
                new_logits, target_past = _forward_with_cache(target_model, replacement_tensor, target_past)
                target_forwards += 1

            draft_replacement_tensor = torch.tensor([[replacement_token]], device=draft_device)
            with torch.no_grad():
                _, draft_past = _forward_with_cache(draft_model, draft_replacement_tensor, draft_past)
                draft_forwards += 1

            last_target_logit = new_logits[0, -1, :]
            last_draft_logit = last_target_logit.to(draft_device)

    # Decode result
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - wall_start

    generated_tokens = len(generated_ids)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = generated_tokens / elapsed if elapsed > 0 else 0.0

    accept_rate = total_accepted / max(total_drafted, 1)

    stats = {
        "wall_time": round(elapsed, 4),
        "tokens_per_second": round(tps, 4),
        "generated_length": generated_tokens,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate": round(accept_rate, 4),
        "target_model_forwards": target_forwards,
        "draft_model_forwards": draft_forwards,
        "draft_time_total": round(draft_time_total, 4),
        "target_time_total": round(target_time_total, 4),
        "draft_time_share": round(draft_time_total / max(elapsed, 1e-6), 4),
        "round_stats": round_stats,
    }

    return {
        "generated_text": generated_text,
        "tokens_per_second": round(tps, 2),
        "generated_tokens": generated_tokens,
        "elapsed_time": round(elapsed, 4),
        "stats": stats,
    }


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
