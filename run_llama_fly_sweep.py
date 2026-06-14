"""
FLY Parameter Sweep for LLaMA-8B:
Sweep gamma ∈ {8, 16}, window ∈ {4, 6} on 3 benchmarks × 20 samples.
Fixes: entropy_threshold=0.3, temperature=0.0
"""
import json, os, sys, time, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.dirname(__file__))

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 20

BENCHMARKS = [
    ("codesearchnet_dict_config_blocks_80", "dict_config"),
    ("ml_config_blocks_openmmlab_80", "openmmlab_config"),
    ("pipeline_stage_config_80", "pipeline_stage_config"),
]

COMBOS = [
    ("FLY-g8-w4", 8, 4),
    ("FLY-g8-w6", 8, 6),
    ("FLY-g16-w4", 16, 4),
    ("FLY-g16-w6", 16, 6),
]
ENTROPY_THRESHOLD = 0.3

OUT_JSON = "results/llama_fly_sweep.json"
OUT_MD = "results/llama_fly_sweep.md"

def compute_sq(pred, ref):
    chars = set("{}[]():,=\n")
    p = [c for c in pred if c in chars]
    r = [c for c in ref if c in chars]
    if not r: return 1.0
    return min(sum(1 for c in p if c in r) / len(r), 1.0)

def compute_off_structure(text):
    lines = text.split("\n") if text else []
    if not lines: return 0.0
    kw = {"def ", "class ", "import ", "from "}
    return sum(1 for l in lines if any(l.strip().startswith(k) for k in kw)) / len(lines)

def _forward_with_cache(model, input_ids, past_key_values):
    if past_key_values is None:
        past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values

def _find_ngram_match(token_ids, window):
    """Find n-gram match of exactly 'window' tokens. Returns (match_start, match_end) or (None, None)."""
    ctx = token_ids
    if len(ctx) < window + 1:
        return None, None
    pattern = tuple(ctx[-window:])
    for i in range(len(ctx) - window):
        if tuple(ctx[i:i + window]) == pattern:
            return i, i + window
    return None, None

def _entropy_from_logits(logits):
    """Compute softmax entropy for the last token's logits."""
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()
    return entropy

def fly_decode_v2(target_model, draft_model, tokenizer, prompt,
                  gamma=8, window=6, entropy_threshold=0.3, max_new_tokens=128):
    """
    FLY with configurable gamma and window.
    gamma: max draft tokens per round
    window: n-gram match length
    entropy_threshold: use n-gram when entropy < τ, model draft otherwise
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = target_model.device
    draft_device = next(draft_model.parameters()).device

    token_list = input_ids[0].tolist()
    generated_tokens = 0
    total_drafted = 0
    total_accepted = 0
    ngram_rounds = 0
    model_rounds = 0
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
            # Compute entropy to decide n-gram vs model drafting
            ent = _entropy_from_logits(logits)

            draft_tokens = []
            if ent < entropy_threshold:
                # Low entropy → try n-gram lookup
                match_start, match_end = _find_ngram_match(token_list, window)
                if match_start is not None:
                    ngram_rounds += 1
                    continuation = token_list[match_end:]
                    draft_tokens = continuation[:min(len(continuation), gamma)]
                else:
                    # Fall through: no n-gram match found
                    model_rounds += 1
                    dt = current_token.to(draft_device)
                    for _ in range(gamma):
                        d_logits, draft_kv = _forward_with_cache(draft_model, dt, draft_kv)
                        next_id = d_logits[0, -1].argmax().item()
                        draft_tokens.append(next_id)
                        dt = torch.tensor([[next_id]], device=draft_device)
                        if next_id == tokenizer.eos_token_id:
                            break
            else:
                # High entropy → model draft
                model_rounds += 1
                dt = current_token.to(draft_device)
                for _ in range(gamma):
                    d_logits, draft_kv = _forward_with_cache(draft_model, dt, draft_kv)
                    next_id = d_logits[0, -1].argmax().item()
                    draft_tokens.append(next_id)
                    dt = torch.tensor([[next_id]], device=draft_device)
                    if next_id == tokenizer.eos_token_id:
                        break

            if not draft_tokens:
                target_forwards += 1
                logits, target_kv = _forward_with_cache(target_model, current_token, target_kv)
                next_id = logits[0, -1].argmax().item()
                token_list.append(next_id)
                generated_tokens += 1
                total_accepted += 1
                total_drafted += 1
                current_token = torch.tensor([[next_id]], device=device)
                if next_id == tokenizer.eos_token_id:
                    break
                continue

            # ── Verify phase ──
            target_forwards += 1
            draft_tensor = torch.tensor([draft_tokens], device=device)
            verify_ids = torch.cat([current_token.to(device), draft_tensor], dim=1)
            logits, target_kv = _forward_with_cache(target_model, verify_ids, target_kv)

            total_drafted += len(draft_tokens)
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

            total_accepted += accepted_count

            if token_list[-1] == tokenizer.eos_token_id:
                break

            if all_accepted and generated_tokens < max_new_tokens:
                bonus_id = logits[0, -1].argmax().item()
                token_list.append(bonus_id)
                generated_tokens += 1
                total_accepted += 1
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
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0

    return {
        "tps": tps, "text": gen_text, "gen_len": gen_len, "time": wall_time,
        "accept_rate": accept_rate, "total_drafted": total_drafted, "total_accepted": total_accepted,
        "ngram_rounds": ngram_rounds, "model_rounds": model_rounds,
        "target_forwards": target_forwards,
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

    all_results = {}  # {stype: {combo_name: [per_sample_results]}}

    for data_file, stype in BENCHMARKS:
        print(f"\n{'='*50}")
        print(f"Benchmark: {stype}")
        print(f"{'='*50}")

        with open(f"data/{data_file}.jsonl") as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:SAMPLE_LIMIT]]

        # Run AR once (reuse across combos)
        ar_results = []
        print("  Running AR baseline...")
        for i, s in enumerate(samples):
            ar = run_ar(target, tokenizer, s["prompt"])
            ar_results.append({
                "name": s.get("name", f"s{i}"), "ar_tps": round(ar["tps"], 2),
                "reference": s.get("reference", ""),
            })
        mean_ar = sum(r["ar_tps"] for r in ar_results) / len(ar_results)
        print(f"  AR mean TPS: {mean_ar:.1f}")

        all_results[stype] = {"ar_tps_mean": round(mean_ar, 1), "ar_per_sample": ar_results}

        for combo_name, gamma, window in COMBOS:
            print(f"\n  --- {combo_name} (gamma={gamma}, window={window}) ---")
            combo_results = []
            for i, s in enumerate(samples):
                prompt = s["prompt"]
                ref = s.get("reference", "")
                r = fly_decode_v2(target, draft, tokenizer, prompt,
                                  gamma=gamma, window=window,
                                  entropy_threshold=ENTROPY_THRESHOLD,
                                  max_new_tokens=MAX_NEW_TOKENS)
                ar_tps = ar_results[i]["ar_tps"]
                sp = r["tps"] / ar_tps if ar_tps > 0 else 0
                sq = compute_sq(r["text"], ref)
                off = compute_off_structure(r["text"])
                combo_results.append({
                    "name": s.get("name", f"s{i}"),
                    "fly_tps": round(r["tps"], 2),
                    "fly_speedup": round(sp, 3),
                    "fly_sq": round(sq, 4),
                    "fly_off": round(off, 4),
                    "fly_accept": round(r["accept_rate"], 4),
                    "fly_ngram_rds": r["ngram_rounds"],
                    "fly_model_rds": r["model_rounds"],
                    "fly_target_fwds": r["target_forwards"],
                    "ar_tps": ar_tps,
                })
                print(f"    [{i+1}/{SAMPLE_LIMIT}] {s.get('name','?')[:30]}: "
                      f"sp={sp:.2f}x acc={r['accept_rate']:.3f} ngram={r['ngram_rounds']} model={r['model_rounds']}",
                      flush=True)

            all_results[stype][combo_name] = combo_results

            # Quick summary
            n = len(combo_results)
            m_sp = sum(r["fly_speedup"] for r in combo_results) / n
            m_sq = sum(r["fly_sq"] for r in combo_results) / n
            m_acc = sum(r["fly_accept"] for r in combo_results) / n
            below = sum(1 for r in combo_results if r["fly_speedup"] < 1.0)
            print(f"    Summary: sp={m_sp:.3f}x sq={m_sq:.4f} acc={m_acc:.3f} below1={below}")

    # ── Save JSON ──
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {OUT_JSON}")

    # ── Generate MD ──
    with open(OUT_MD, "w") as f:
        f.write("# FLY Parameter Sweep for LLaMA-8B\n\n")
        f.write("**Target**: Llama-3.1-8B-Instruct | **Draft**: Llama-3.2-1B-Instruct\n\n")
        f.write("### Config\n\n")
        f.write(f"- entropy_threshold = {ENTROPY_THRESHOLD} (fixed, FLY paper default)\n")
        f.write(f"- temperature = 0.0 (fixed)\n")
        f.write(f"- Sweep: gamma ∈ {{8, 16}}, window ∈ {{4, 6}}\n")
        f.write(f"- Samples: {SAMPLE_LIMIT} per benchmark × {len(BENCHMARKS)} benchmarks\n\n")
        f.write("**Note**: gamma=16, window=6 ≈ closest to FLY paper defaults for 70B (gamma=15, w=6).\n")
        f.write("Gamma adjusted for 8B target (lower forward cost → smaller gamma).\n\n")

        for stype in [b[1] for b in BENCHMARKS]:
            data = all_results[stype]
            ar_mean = data["ar_tps_mean"]
            f.write(f"## {stype}\n\n")
            f.write(f"AR TPS: {ar_mean:.1f}\n\n")
            f.write("| Combo | FLY TPS | Speedup | Accept | SQ | OffStr | NGram Rds | Model Rds | Target Fwds | Below1.0x |\n")
            f.write("|-------|---------|---------|--------|-----|--------|----------|----------|------------|----------|\n")

            for combo_name, gamma, window in COMBOS:
                cr = data[combo_name]
                n = len(cr)
                m = lambda k: round(sum(r[k] for r in cr) / n, 2) if isinstance(cr[0][k], (int, float)) and not isinstance(cr[0][k], bool) else round(sum(r[k] for r in cr) / n, 1)
                m4 = lambda k: round(sum(r[k] for r in cr) / n, 4)
                below = sum(1 for r in cr if r["fly_speedup"] < 1.0)
                f.write(f"| {combo_name} | {m('fly_tps'):.1f} | {m4('fly_speedup'):.2f}x | {m4('fly_accept'):.3f} | "
                        f"{m4('fly_sq'):.4f} | {m4('fly_off'):.4f} | {m('fly_ngram_rds'):.1f} | "
                        f"{m('fly_model_rds'):.1f} | {m('fly_target_fwds'):.1f} | {below} |\n")
            f.write("\n")

        # Overall summary
        f.write("## Overall (aggregated across 3 benchmarks)\n\n")
        f.write("| Combo | Speedup | SQ | Below1.0x |\n")
        f.write("|-------|---------|-----|----------|\n")
        best_combo = (None, -1)
        for combo_name, gamma, window in COMBOS:
            all_sp = []
            all_sq = []
            all_below = 0
            for stype in [b[1] for b in BENCHMARKS]:
                cr = all_results[stype][combo_name]
                all_sp.extend(r["fly_speedup"] for r in cr)
                all_sq.extend(r["fly_sq"] for r in cr)
                all_below += sum(1 for r in cr if r["fly_speedup"] < 1.0)
            msp = sum(all_sp) / len(all_sp)
            msq = sum(all_sq) / len(all_sq)
            f.write(f"| {combo_name} | {msp:.3f}x | {msq:.4f} | {all_below} |\n")
            if msp > best_combo[1]:
                best_combo = (combo_name, msp)

        f.write(f"\n**Recommended FLY baseline**: {best_combo[0]} (speedup={best_combo[1]:.3f}x)\n")
        f.write("\nThis combo should be used as the corrected FLY baseline for LLaMA-8B comparisons.\n")
        f.write("\n### Notes\n\n")
        f.write("- Entropy threshold τ=0.3 (FLY paper default) unchanged\n")
        f.write("- Only gamma and window swept — parameters directly related to target latency/overhead\n")
        f.write("- FLY paper's gamma=15 for 70B; our 8B target has ~9x lower compute per forward\n")
        f.write("  so smaller gamma (8) can be more efficient on 8B\n")

    print(f"Report saved to {OUT_MD}")
    print("\nDone!")

if __name__ == "__main__":
    main()
