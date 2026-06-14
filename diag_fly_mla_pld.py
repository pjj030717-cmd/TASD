"""
FLY MLA/PLD Diagnostic Smoke Test (5 openmmlab samples).

Checks:
- Is PLA/PLD actually enabled and effective?
- Detailed PLD statistics per sample
- Gamma=8 and Gamma=16 (parameterized)
"""
import json, os, sys, time, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.dirname(__file__))

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 5

# FLY parameters
GAMMAS = [8, 16]
WINDOW_MIN = 3
WINDOW_MAX = 6   # variable-length n-gram: try longest first from 6 down to 3
ENTROPY_THRESHOLD = 0.3

DATA_FILE = "data/ml_config_blocks_openmmlab_80.jsonl"

def compute_sq(pred, ref):
    chars = set("{}[]():,=\n")
    p = [c for c in pred if c in chars]
    r = [c for c in ref if c in chars]
    if not r: return 1.0
    return min(sum(1 for c in p if c in r) / len(r), 1.0)

def _forward_with_cache(model, input_ids, past_key_values):
    if past_key_values is None:
        past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values

def _find_ngram_match(token_ids, min_len=3, max_len=8):
    """Variable-length n-gram: try longest first, then shorter."""
    ctx = token_ids
    ctx_len = len(ctx)
    for n in range(max_len, min_len - 1, -1):
        if ctx_len < n + 1:
            continue
        pattern = tuple(ctx[-n:])
        for i in range(ctx_len - n):
            if tuple(ctx[i:i + n]) == pattern:
                return i, i + n
    return None, None

def _entropy_from_logits(logits):
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum().item()

def fly_diagnostic_decode(target_model, draft_model, tokenizer, prompt,
                          gamma=8, window_min=3, window_max=6,
                          entropy_threshold=0.3, max_new_tokens=128):
    """
    FLY decode with full diagnostic statistics.
    γ = gamma (max draft tokens per round)
    w = window_max (n-gram match window, variable-length from window_min to window_max)
    τ = entropy_threshold
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = target_model.device
    draft_device = next(draft_model.parameters()).device

    token_list = input_ids[0].tolist()
    generated_tokens = 0

    # ── PLD / MLA Stats ──
    mla_enabled = False  # no MLA implementation in current codebase
    pld_enabled = True   # n-gram prompt lookup IS implemented
    pld_attempt_count = 0
    pld_hit_count = 0
    pld_tokens_proposed = 0
    pld_tokens_accepted = 0
    model_draft_fallback_count = 0
    model_tokens_proposed = 0
    model_tokens_accepted = 0
    total_draft_time = 0.0
    total_pld_time = 0.0
    total_target_verify_time = 0.0
    total_model_draft_time = 0.0
    target_forwards = 0

    target_kv = None
    draft_kv = None

    torch.cuda.synchronize()
    wall_start = time.time()

    with torch.no_grad():
        logits, target_kv = _forward_with_cache(target_model, input_ids.to(device), None)
        _, draft_kv = _forward_with_cache(draft_model, input_ids.to(draft_device), None)
        current_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

        while generated_tokens < max_new_tokens:
            # ── Compute entropy for PLD decision ──
            ent = _entropy_from_logits(logits)
            
            draft_tokens = []
            used_pld = False

            t0_draft = time.time()

            if ent < entropy_threshold:
                # Low entropy → try prompt lookup (PLD)
                pld_attempt_count += 1
                t0_pld = time.time()
                match_start, match_end = _find_ngram_match(token_list, window_min, window_max)
                total_pld_time += time.time() - t0_pld

                if match_start is not None:
                    # PLD hit
                    pld_hit_count += 1
                    used_pld = True
                    continuation = token_list[match_end:]
                    draft_tokens = continuation[:min(len(continuation), gamma)]
                    pld_tokens_proposed += len(draft_tokens)
                else:
                    # PLD miss → fallback to model draft
                    pass

            if not used_pld and not draft_tokens:
                # High entropy or PLD miss → model draft
                model_draft_fallback_count += 1
                t0_md = time.time()
                dt = current_token.to(draft_device)
                for _ in range(gamma):
                    d_logits, draft_kv = _forward_with_cache(draft_model, dt, draft_kv)
                    next_id = d_logits[0, -1].argmax().item()
                    draft_tokens.append(next_id)
                    dt = torch.tensor([[next_id]], device=draft_device)
                    if next_id == tokenizer.eos_token_id:
                        break
                total_model_draft_time += time.time() - t0_md
                model_tokens_proposed += len(draft_tokens)

            total_draft_time += time.time() - t0_draft

            if not draft_tokens:
                target_forwards += 1
                logits, target_kv = _forward_with_cache(target_model, current_token, target_kv)
                next_id = logits[0, -1].argmax().item()
                token_list.append(next_id)
                generated_tokens += 1
                current_token = torch.tensor([[next_id]], device=device)
                if next_id == tokenizer.eos_token_id:
                    break
                continue

            # ── Verify phase ──
            target_forwards += 1
            t0_verify = time.time()
            draft_tensor = torch.tensor([draft_tokens], device=device)
            verify_ids = torch.cat([current_token.to(device), draft_tensor], dim=1)
            logits, target_kv = _forward_with_cache(target_model, verify_ids, target_kv)
            total_target_verify_time += time.time() - t0_verify

            accepted_count = 0
            all_accepted = True

            for i in range(len(draft_tokens)):
                target_pred = logits[0, i].argmax().item()
                dr_id = draft_tokens[i]
                if dr_id == target_pred:
                    token_list.append(dr_id)
                    accepted_count += 1
                    generated_tokens += 1
                    dt_in = torch.tensor([[dr_id]], device=draft_device)
                    _, draft_kv = _forward_with_cache(draft_model, dt_in, draft_kv)
                    if dr_id == tokenizer.eos_token_id:
                        all_accepted = False
                        break
                else:
                    token_list.append(target_pred)
                    accepted_count += 1
                    generated_tokens += 1
                    current_token = torch.tensor([[target_pred]], device=device)
                    all_accepted = False
                    break

            if used_pld:
                pld_tokens_accepted += accepted_count
            else:
                model_tokens_accepted += accepted_count

            if token_list[-1] == tokenizer.eos_token_id:
                break

            if all_accepted and generated_tokens < max_new_tokens:
                bonus_id = logits[0, -1].argmax().item()
                token_list.append(bonus_id)
                generated_tokens += 1
                if used_pld:
                    pld_tokens_accepted += 1
                else:
                    model_tokens_accepted += 1
                current_token = torch.tensor([[bonus_id]], device=device)
                _, target_kv = _forward_with_cache(target_model, current_token, target_kv)
                dt_in = torch.tensor([[bonus_id]], device=draft_device)
                _, draft_kv = _forward_with_cache(draft_model, dt_in, draft_kv)
                if bonus_id == tokenizer.eos_token_id:
                    break
            else:
                current_token = torch.tensor([[token_list[-1]]], device=device)

    torch.cuda.synchronize()
    wall_time = time.time() - wall_start

    gen_ids = token_list[input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    tps = len(token_list) / wall_time if wall_time > 0 else 0

    total_proposed = pld_tokens_proposed + model_tokens_proposed
    total_accepted = pld_tokens_accepted + model_tokens_accepted
    accept_rate = total_accepted / max(total_proposed, 1)

    pld_hit_rate = pld_hit_count / max(pld_attempt_count, 1)
    avg_pld_tokens_per_hit = pld_tokens_proposed / max(pld_hit_count, 1)
    avg_model_draft_tokens_per_fallback = model_tokens_proposed / max(model_draft_fallback_count, 1)

    return {
        "tps": tps, "text": gen_text, "gen_len": gen_len, "time": wall_time,
        "accept_rate": accept_rate,
        # MLA/PLD status
        "mla_enabled": mla_enabled,
        "pld_enabled": pld_enabled,
        # PLD stats
        "pld_attempt_count": pld_attempt_count,
        "pld_hit_count": pld_hit_count,
        "pld_hit_rate": pld_hit_rate,
        "pld_tokens_proposed": pld_tokens_proposed,
        "pld_tokens_accepted": pld_tokens_accepted,
        "pld_accept_rate": pld_tokens_accepted / max(pld_tokens_proposed, 1),
        "avg_pld_tokens_per_hit": avg_pld_tokens_per_hit,
        # Model draft stats
        "model_draft_fallback_count": model_draft_fallback_count,
        "model_tokens_proposed": model_tokens_proposed,
        "model_tokens_accepted": model_tokens_accepted,
        "model_accept_rate": model_tokens_accepted / max(model_tokens_proposed, 1),
        "avg_model_draft_tokens_per_fallback": avg_model_draft_tokens_per_fallback,
        # Timing
        "total_draft_time": total_draft_time,
        "total_pld_time": total_pld_time,
        "total_model_draft_time": total_model_draft_time,
        "total_target_verify_time": total_target_verify_time,
        "target_forwards": target_forwards,
        # Ratio
        "pld_draft_ratio": pld_hit_count / max(pld_attempt_count + model_draft_fallback_count, 1),
    }

def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    wall = time.time() - t0
    gen_ids = out[0][inp.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tps = len(out[0]) / wall if wall > 0 else 0
    return {"tps": tps, "text": text, "time": wall, "gen_len": len(gen_ids)}

def main():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    with open(DATA_FILE) as f:
        samples = [json.loads(l.strip()) for l in f.readlines()[:SAMPLE_LIMIT]]

    # Run AR baseline once
    ar_results = []
    print("Running AR baseline...")
    for i, s in enumerate(samples):
        ar = run_ar(target, tokenizer, s["prompt"])
        ar_results.append(ar)
        print(f"  [{i+1}] AR TPS={ar['tps']:.1f}")
    print()

    for gamma in GAMMAS:
        print(f"{'='*70}")
        print(f"FLY Diagnostic: gamma={gamma}, window=3-6 (var), τ={ENTROPY_THRESHOLD}")
        print(f"{'='*70}")
        print(f"{'Sample':<30} {'AR TPS':>7} {'FLY TPS':>7} {'Speedup':>7} {'SQ':>6} "
              f"{'PLD_att':>7} {'PLD_hit':>7} {'PLD_hr':>6} {'PLD_prop':>8} {'PLD_acc':>8} "
              f"{'M_fall':>6}")
        print("-" * 70)

        all_results = []
        for i, s in enumerate(samples):
            ref = s.get("reference", "")
            r = fly_diagnostic_decode(target, draft, tokenizer, s["prompt"],
                                      gamma=gamma, window_min=WINDOW_MIN, window_max=WINDOW_MAX,
                                      entropy_threshold=ENTROPY_THRESHOLD,
                                      max_new_tokens=MAX_NEW_TOKENS)
            ar_tps = ar_results[i]["tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], ref)

            print(f"{s['name']:<30} {ar_tps:>7.1f} {r['tps']:>7.1f} {sp:>7.2f}x {sq:>6.4f} "
                  f"{r['pld_attempt_count']:>7} {r['pld_hit_count']:>7} {r['pld_hit_rate']:>6.3f} "
                  f"{r['pld_tokens_proposed']:>8} {r['pld_tokens_accepted']:>8} "
                  f"{r['model_draft_fallback_count']:>6}")

            all_results.append({"name": s["name"], "ar_tps": ar_tps, "sp": sp, "sq": sq, **r})

        # Summary
        n = len(all_results)
        m = lambda k: sum(r[k] for r in all_results) / n
        msg = lambda k: sum(1 for r in all_results if r[k]) / n
        print()
        print(f"--- gamma={gamma} Summary ---")
        print(f"  MLA enabled:     {all_results[0]['mla_enabled']} (no MLA in codebase)")
        print(f"  PLD enabled:     {all_results[0]['pld_enabled']} (n-gram prompt lookup)")
        print(f"  PLD attempt/round: {m('pld_attempt_count'):.1f}")
        print(f"  PLD hit count:     {m('pld_hit_count'):.1f}")
        print(f"  PLD hit rate:      {sum(r['pld_hit_count'] for r in all_results)/max(1,sum(r['pld_attempt_count'] for r in all_results)):.3f}")
        print(f"  PLD tokens proposed: {m('pld_tokens_proposed'):.1f}")
        print(f"  PLD tokens accepted: {m('pld_tokens_accepted'):.1f}")
        print(f"  Model fallback rounds: {m('model_draft_fallback_count'):.1f}")
        print(f"  Draft ratio (PLD/total): {sum(r['pld_hit_count'] for r in all_results)/max(1,sum(r['pld_hit_count']+r['model_draft_fallback_count'] for r in all_results)):.3f}")
        print(f"  Avg speedup:         {m('sp'):.3f}x")
        print(f"  Avg SQ:              {m('sq'):.4f}")
        print(f"  Below 1.0x:          {sum(1 for r in all_results if r['sp']<1.0)}")
        print()

        # Detailed timing breakdown
        print(f"  Timing breakdown (avg per sample):")
        print(f"    Total wall:           {m('time'):.2f}s")
        print(f"    Draft time:           {m('total_draft_time'):.3f}s ({100*m('total_draft_time')/max(0.01,m('time')):.1f}%)")
        print(f"      - PLD lookup time:  {m('total_pld_time'):.4f}s")
        print(f"      - Model draft time: {m('total_model_draft_time'):.3f}s")
        print(f"    Target verify time:   {m('total_target_verify_time'):.3f}s ({100*m('total_target_verify_time')/max(0.01,m('time')):.1f}%)")
        print(f"    Target forwards:      {m('target_forwards'):.1f}")
        print()

    # ── Final Report ──
    print("=" * 70)
    print("FINAL FLY STATUS REPORT")
    print("=" * 70)
    print()
    print("1. MLA (Multi-Layer Attention):")
    print("   Status: NOT ENABLED")
    print("   Evidence: No enable_mla/use_mla flag, no MLA-related code in entire codebase")
    print("   Impact: Draft and target models maintain separate KV caches; draft KV updates")
    print("           after each accepted token (extra forward, no MLA sharing)")
    print()
    print("2. PLD (Prompt Lookup Decoding):")
    print("   Status: ENABLED (n-gram prompt lookup)")
    print("   Implementation: Variable-length n-gram matching (3-6 tokens, longest-first)")
    print("   Decision: Entropy < τ=0.3 triggers PLD attempt; entropy >= τ goes straight to model draft")
    print("   See detailed stats above for per-sample hit rates and acceptance.")
    print()
    print("3. Classification:")
    print("   This is FLY-style (n-gram + model draft), specifically:")
    print("   -> PLD-enabled (n-gram prompt lookup) ✓")
    print("   -> No MLA (separate KV caches) ✗")
    print("   -> Entropy-gated PLD decision ✓")
    print("   -> Variable-length n-gram matching ✓")
    print()
    print("4. Recommendation:")
    print("   Label as: 'FLY-style (no MLA, n-gram PLD + model draft, τ=0.3)'")
    print("   NOT official FLY (MLA is a core component of FLY paper)")
    print("   Baseline name: FLY (no-MLA, ngram+model)")
    print()

if __name__ == "__main__":
    main()
