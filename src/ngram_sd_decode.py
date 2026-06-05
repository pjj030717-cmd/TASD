"""
N-gram / Prompt Lookup Speculative Decoding.

Training-free speculative decoding using n-gram pattern matching from
prompt + generated history to produce draft tokens. No draft model needed.

Algorithm per round:
  1. Take the last `match_len` tokens from the full context (prompt + generated)
  2. Search for the longest matching n-gram in the full context (excl. last position)
  3. Extract tokens immediately following the match as draft candidates (up to max_draft_tokens)
  4. Verify all draft tokens with target model in one batch forward (greedy argmax match)
  5. Accept longest prefix where draft_token == target_argmax
  6. If all accepted, sample one bonus token from target
  7. If no match found, fall back to single-token AR step
"""
import time
import torch
from transformers import DynamicCache


def _forward_with_cache(model, input_ids, past_key_values, device):
    """Forward with KV cache. Forces all inputs to the same device."""
    if past_key_values is None:
        past_key_values = DynamicCache()
    input_ids = input_ids.to(device)
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def _find_ngram_match(token_ids, min_len=3, max_len=8):
    """
    Find the longest n-gram match in the token sequence.

    Looks at the last L tokens (L in [min_len, max_len]) and searches
    for the longest match in the prefix (all tokens except the last occurrence).

    Args:
        token_ids: list of token IDs (prompt + generated so far)
        min_len: minimum n-gram length to consider
        max_len: maximum n-gram length to check

    Returns:
        (match_start, match_end) indices in token_ids, or (None, None) if no match.
        match_end is the position after the matched tokens.
        The draft tokens are token_ids[match_end : match_end + max_draft].
    """
    context = token_ids
    ctx_len = len(context)

    for n in range(max_len, min_len - 1, -1):
        if ctx_len < n + 1:
            continue

        # The last n tokens (the pattern to match)
        pattern = tuple(context[-n:])

        # Search in all positions except the last one
        # Use tuple-based hash lookup for O(N) search
        # Build position map once per round (acceptable at n <= 8)
        for i in range(ctx_len - n):
            if tuple(context[i:i + n]) == pattern:
                # Found match at position i
                return i, i + n

    return None, None


def ngram_sd_decode(
    target_model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    ngram_min=3,
    ngram_max=8,
    max_draft_tokens=16,
    temperature=0.0,
):
    """
    N-gram speculative decoding.

    Args:
        target_model: target model (Qwen2.5-14B)
        tokenizer: tokenizer for the target model
        prompt: input prompt string
        max_new_tokens: max tokens to generate
        ngram_min: minimum n-gram length for matching
        ngram_max: maximum n-gram length for matching
        max_draft_tokens: max tokens to draft per round
        temperature: sampling temperature (0.0 = greedy)

    Returns:
        dict with generated_text, tokens_per_second, generated_tokens, stats
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    device = target_model.device

    # Convert to list for n-gram matching
    token_list = input_ids[0].tolist()
    generated_tokens = 0
    total_time = 0.0
    total_draft_rounds = 0
    total_accepted = 0
    total_drafted = 0
    match_rounds = 0  # rounds where n-gram found a match

    # Pre-fill: run prompt through target model once to build initial KV cache
    with torch.no_grad():
        _, past_kv = _forward_with_cache(target_model, input_ids, None, device)

    start_time = time.time()

    while generated_tokens < max_new_tokens:
        total_draft_rounds += 1

        # 1. Try n-gram match from full context
        match_start, match_end = _find_ngram_match(token_list, ngram_min, ngram_max)

        if match_start is None:
            # No match: fall back to single-token AR
            with torch.no_grad():
                logits, past_kv = _forward_with_cache(
                    target_model,
                    torch.tensor([[token_list[-1]]], device=device),
                    past_kv, device)
            next_token = logits[0, -1].argmax().item()
            token_list.append(next_token)
            generated_tokens += 1
            total_accepted += 1
            total_drafted += 1
            continue

        match_rounds += 1

        # 2. Extract draft tokens (tokens following the match)
        draft_start = match_end
        draft_end = min(match_end + max_draft_tokens, len(token_list))
        draft_tokens = token_list[draft_start:draft_end]

        if len(draft_tokens) == 0:
            # Match is at the very end, no tokens to draft
            with torch.no_grad():
                logits, past_kv = _forward_with_cache(
                    target_model,
                    torch.tensor([[token_list[-1]]], device=device),
                    past_kv, device)
            next_token = logits[0, -1].argmax().item()
            token_list.append(next_token)
            generated_tokens += 1
            total_accepted += 1
            total_drafted += 1
            continue

        k = len(draft_tokens)
        draft_ids = torch.tensor([draft_tokens], device=device)  # [1, k]

        # 3. Target forward: only draft tokens (no seed).
        # past_kv covers prompt + old_gen (= PL + OG entries).
        # Forward [d0, ..., d_{k-1}]; logits[j] predicts after d[j-1]:
        #   logits[0, 0] → predicts after old_gen → verify d[0]
        #   logits[0, j] → predicts after d[j-1] → verify d[j]  (j ≥ 1)
        with torch.no_grad():
            logits, past_kv = _forward_with_cache(
                target_model, draft_ids, past_kv, device)

        # 4. Greedy verification
        target_tokens = logits[0, :].argmax(dim=-1)  # [k]
        accepted_mask = (draft_ids[0, :] == target_tokens)  # [k], bool
        accepted_mask = accepted_mask.cpu().tolist()

        # 5. Apply accepted tokens (causal: stop at first rejection)
        num_accepted = 0
        for i, ok in enumerate(accepted_mask):
            if ok:
                token_list.append(draft_tokens[i])
                num_accepted += 1
            else:
                token_list.append(target_tokens[i].item())
                num_accepted += 1
                break

        total_accepted += num_accepted
        total_drafted += k
        generated_tokens = len(token_list) - prompt_len

        # 6. KV cache sync
        # After forward, past_kv = PL + old_gen + k.
        # token_list after acceptance = PL + old_gen + r + 1.
        # Crop to PL + old_gen + r (accepted drafts only):
        past_kv.crop(prompt_len + generated_tokens - 1)

        if num_accepted == k:
            # All drafts accepted. past_kv ends at d[k-2].
            # Re-forward d[k-1] to get bonus prediction + extend KV to d[k-1].
            d_last = torch.tensor([[draft_tokens[-1]]], device=device)
            with torch.no_grad():
                bonus_logits, past_kv = _forward_with_cache(
                    target_model, d_last, past_kv, device)
            bonus_token = bonus_logits[0, -1].argmax().item()
            token_list.append(bonus_token)
            generated_tokens = len(token_list) - prompt_len
            # Forward bonus to extend KV
            bonus_tensor = torch.tensor([[bonus_token]], device=device)
            with torch.no_grad():
                _, past_kv = _forward_with_cache(
                    target_model, bonus_tensor, past_kv, device)
        else:
            # Rejected. past_kv ends at d[r-1]; token_list[-1] = replacement.
            repl_tensor = torch.tensor([[token_list[-1]]], device=device)
            with torch.no_grad():
                _, past_kv = _forward_with_cache(
                    target_model, repl_tensor, past_kv, device)

        # Check for EOS
        if token_list[-1] == tokenizer.eos_token_id:
            break

    elapsed = time.time() - start_time

    # Decode generated text
    gen_ids = token_list[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    tps = gen_len / elapsed if elapsed > 0 else 0

    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    match_found_rate = match_rounds / total_draft_rounds if total_draft_rounds > 0 else 0
    avg_draft_len = total_drafted / total_draft_rounds if total_draft_rounds > 0 else 0

    return {
        "generated_text": gen_text,
        "tokens_per_second": round(tps, 2),
        "generated_tokens": gen_len,
        "elapsed_time": round(elapsed, 4),
        "stats": {
            "accept_rate": round(accept_rate, 4),
            "draft_rounds": total_draft_rounds,
            "total_accepted": total_accepted,
            "total_drafted": total_drafted,
            "match_found_rate": round(match_found_rate, 4),
            "avg_draft_len": round(avg_draft_len, 2),
            "match_rounds": match_rounds,
        },
    }
