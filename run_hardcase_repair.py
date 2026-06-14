"""
Hard-Case Repair Experiment: AR / FLY / TASD / TASD-F on 24 performance hard cases.

TASD-F config: enable_failure_aware_fallback=True, fallback_tokens=2,
                fallback_guarded=False, progressive_fallback=False,
                boundary_fallback=False, post_fallback_probe=False.
"""

import json
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode

# ─── Config ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 128

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"

DATA_FILES = {
    "argparse": "/root/autodl-tmp/data/codesearchnet_argparse_blocks_80.jsonl",
    "dict_config": "/root/autodl-tmp/data/codesearchnet_dict_config_blocks_80.jsonl",
    "openmmlab_config": "/root/autodl-tmp/data/ml_config_blocks_openmmlab_80.jsonl",
}

BENCHMARK_TO_BID = {
    "Real-Python-Argparse": "argparse",
    "Real-Python-DictConfig": "dict_config",
    "OpenMMLab-Config": "openmmlab_config",
}

# TASD default config (matching 480 main experiment)
TASD_KWARGS = {
    "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
    "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
    "enable_guard": True, "enable_relaxed_accept": True,
}

# TASD-F config: same as TASD + failure-aware fallback
TASDF_KWARGS = {
    **TASD_KWARGS,
    "enable_failure_aware_fallback": True,
}

# FLY config: hybrid n-gram + model SD
FLY_NGRAM_MIN = 3
FLY_NGRAM_MAX = 8
FLY_DRAFT_LEN = 3  # model draft length when no n-gram match
FLY_MAX_DRAFT = 16


# ─── Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path, n):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def _forward_with_cache(model, input_ids, past_key_values):
    if past_key_values is None:
        past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def _find_ngram_match(token_ids, min_len=3, max_len=8):
    """Find longest n-gram match in context. Returns (match_start, match_end) or (None, None)."""
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


def fly_decode(target_model, draft_model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """
    FLY: Hybrid speculative decoding combining n-gram prompt lookup + model drafting.

    1. Try n-gram match in prompt+generated context
    2. If match found: draft from matched continuation
    3. If no match: draft from draft model (greedy)
    4. Verify all via target model greedy matching (longest prefix accept)
    5. If all accepted, sample one bonus token from target
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_start = time.time()

    with torch.no_grad():
        # Pre-fill
        logits, target_kv = _forward_with_cache(target_model, input_ids.to(device), None)
        _, draft_kv = _forward_with_cache(draft_model, input_ids.to(draft_device), None)

        current_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

        while generated_tokens < max_new_tokens:
            # ── Draft phase ──
            draft_tokens = []
            match_start, match_end = _find_ngram_match(token_list, FLY_NGRAM_MIN, FLY_NGRAM_MAX)

            if match_start is not None:
                # Use n-gram continuation as draft
                ngram_rounds += 1
                continuation = token_list[match_end:]
                draft_tokens = continuation[:min(len(continuation), FLY_MAX_DRAFT)]
            else:
                # Use draft model (greedy draft)
                model_rounds += 1
                dt = current_token.to(draft_device)
                for _ in range(FLY_DRAFT_LEN):
                    d_logits, draft_kv = _forward_with_cache(draft_model, dt, draft_kv)
                    next_id = d_logits[0, -1].argmax().item()
                    draft_tokens.append(next_id)
                    dt = torch.tensor([[next_id]], device=draft_device)
                    if next_id == tokenizer.eos_token_id:
                        break

            if not draft_tokens:
                # Fallback: single AR step
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
            target_kv_for_draft_update = target_kv

            total_drafted += len(draft_tokens)
            accepted_count = 0
            all_accepted = True

            # Verify draft_tokens against target's greedy predictions
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

            # Check EOS
            if token_list[-1] == tokenizer.eos_token_id:
                break

            if all_accepted and generated_tokens < max_new_tokens:
                bonus_id = logits[0, -1].argmax().item()
                token_list.append(bonus_id)
                generated_tokens += 1
                total_accepted += 1
                current_token = torch.tensor([[bonus_id]], device=device)
                _, target_kv = _forward_with_cache(target_model, current_token, target_kv)
                # Also advance draft_kv to stay in sync
                dt_in = torch.tensor([[bonus_id]], device=draft_device)
                _, draft_kv = _forward_with_cache(draft_model, dt_in, draft_kv)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time = time.time() - wall_start

    gen_ids = token_list[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tps = generated_tokens / wall_time if wall_time > 0 else 0
    accept_rate = total_accepted / max(total_drafted, 1)

    return {
        "generated_text": gen_text,
        "tokens_per_second": round(tps, 2),
        "tps": round(tps, 2),
        "generated_tokens": generated_tokens,
        "elapsed_time": round(wall_time, 4),
        "accept_rate": round(accept_rate, 4),
        "repair_count": 0,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "ngram_rounds": ngram_rounds,
        "model_rounds": model_rounds,
        "target_forwards": target_forwards,
        "stats": {
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "accept_rate": round(accept_rate, 4),
            "ngram_rounds": ngram_rounds,
            "model_rounds": model_rounds,
        },
    }


def run_ar(target_model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Run baseline AR and return metrics."""
    device = target_model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = target_model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    gen_ids = output[0][input_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_tokens = len(gen_ids)
    tps = n_tokens / elapsed if elapsed > 0 else 0
    return {
        "generated_text": gen_text,
        "tokens_per_second": round(tps, 2),
        "tps": round(tps, 2),
        "generated_tokens": n_tokens,
        "elapsed_time": round(elapsed, 4),
        "accept_rate": 1.0,
        "repair_count": 0,
    }


def run_tasd(target_model, draft_model, tokenizer, prompt, structure_type, extra_kwargs=None):
    """Run TASD or TASD-F and return metrics."""
    kwargs = dict(TASD_KWARGS)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    r = tasd_decode(
        target_model=target_model, draft_model=draft_model, tokenizer=tokenizer,
        prompt=prompt, structure_type=structure_type, max_new_tokens=MAX_NEW_TOKENS,
        **kwargs,
    )
    r["tps"] = r.get("tokens_per_second", 0)
    stats = r.get("stats", {})
    r["accept_rate"] = stats.get("accept_rate", 0)
    r["repair_count"] = stats.get("repair_count", 0)
    r["zero_accept_round_count"] = stats.get("consecutive_repair_count", 0)
    return r


# ─── Metrics ──────────────────────────────────────────────────────────────

def compute_sq(generated_text, reference_text):
    """Character-level overlap of structural chars with reference."""
    struct_chars = set("{}[]():,=\n")
    gen_struct = [c for c in generated_text if c in struct_chars]
    ref_struct = [c for c in reference_text if c in struct_chars]
    if not ref_struct:
        return 1.0
    matches = sum(1 for c in gen_struct if c in ref_struct)
    return min(matches / len(ref_struct), 1.0)


def compute_off_structure(generated_text, structure_type):
    """Proportion of lines starting with off-structure keywords."""
    lines = generated_text.split("\n") if generated_text else []
    if not lines:
        return 0.0
    struct_keywords = {"def ", "class ", "import ", "from "}
    cnt = sum(1 for line in lines if any(line.strip().startswith(kw) for kw in struct_keywords))
    return cnt / len(lines)


def compute_repetition(text):
    """Simple repetition rate: fraction of repeated n-grams (n=4)."""
    if not text or len(text) < 8:
        return 0.0
    n = 4
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if len(ngrams) <= 1:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)


def compute_truncation(text):
    """Detect truncation: ends with incomplete pattern."""
    if not text:
        return 0.0
    truncation_indicators = ["...", "[un", "(\"", "['", "{\"", "type=", "help=", "default="]
    last_80 = text.strip()[-80:] if len(text) > 80 else text.strip()
    for ind in truncation_indicators:
        if last_80.endswith(ind):
            return 1.0
    # Also check if text is very short relative to expected
    if len(text) < 200 and not text.strip().endswith((")", "}", "]", '"', "'")):
        # Maybe truncated
        pass  # Don't flag short texts as truncated by default
    return 0.0


# ─── Report Generation ────────────────────────────────────────────────────

def generate_report(all_results, ar_baselines, out_json, out_md):
    """Generate JSON and MD reports."""
    os.makedirs("results", exist_ok=True)

    # ── JSON ──
    json_output = {
        "config": {
            "n_hard_cases": len(all_results),
            "max_new_tokens": MAX_NEW_TOKENS,
            "methods": ["AR", "FLY", "TASD", "TASD-F"],
            "tasd_kwargs": TASD_KWARGS,
            "tasdf_kwargs": {"enable_failure_aware_fallback": True, "fallback_tokens": 2},
        },
        "summary": {},
        "per_sample": [],
    }

    # Compute summary stats per method
    for method in ["AR", "FLY", "TASD", "TASD-F"]:
        tps_vals = []
        speedups = []
        sq_vals = []
        off_vals = []
        repair_vals = []
        accept_vals = []
        rep_vals = []
        trunc_vals = []
        below_1_count = 0

        for r in all_results:
            data = r[method.lower().replace("-", "_")]
            tps_vals.append(data["tps"])
            sq_vals.append(data.get("sq", 0))
            off_vals.append(data.get("off_structure", 0))
            repair_vals.append(data.get("repair_count", 0))
            accept_vals.append(data.get("accept_rate", 0))
            rep_vals.append(data.get("repetition", 0))
            trunc_vals.append(data.get("truncation", 0))

            if method != "AR":
                ar_tps = ar_baselines.get(r["case_id"], 30)
                sp = data["tps"] / ar_tps if ar_tps > 0 else 0
                speedups.append(sp)
                if sp < 1.0:
                    below_1_count += 1

        n = len(all_results)
        summary = {
            "n": n,
            "mean_tps": round(sum(tps_vals) / n, 2),
            "median_tps": round(sorted(tps_vals)[n // 2], 2),
            "mean_sq": round(sum(sq_vals) / n, 4),
            "mean_off_structure": round(sum(off_vals) / n, 4),
            "mean_repair_count": round(sum(repair_vals) / n, 2),
            "mean_accept_rate": round(sum(accept_vals) / n, 4),
            "mean_repetition": round(sum(rep_vals) / n, 4),
            "mean_truncation": round(sum(trunc_vals) / n, 4),
            "below_1x_count": below_1_count,
        }
        if speedups:
            summary["mean_speedup"] = round(sum(speedups) / len(speedups), 2)
            summary["median_speedup"] = round(sorted(speedups)[len(speedups) // 2], 2)
        json_output["summary"][method] = summary

    # Per-sample data
    for r in all_results:
        entry = {
            "benchmark": r["benchmark"],
            "sample_idx": r["sample_idx"],
            "sample_name": r["sample_name"],
            "source_file": r.get("source_file", ""),
            "ar": r["ar"],
            "fly": r["fly"],
            "tasd": r["tasd"],
            "tasd_f": r["tasd_f"],
            "tasd_speedup": r["tasd_speedup"],
            "tasd_f_speedup": r["tasd_f_speedup"],
            "speedup_delta": round(r["tasd_f_speedup"] - r["tasd_speedup"], 4),
            "repair_delta": round(r["tasd"]["repair_count"] - r["tasd_f"]["repair_count"], 2),
        }
        json_output["per_sample"].append(entry)

    with open(out_json, "w") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {out_json}")

    # ── MD ──
    s = json_output["summary"]
    with open(out_md, "w") as f:
        f.write("# TASD-F Hard-Case Repair Experiment\n\n")
        f.write(f"**Hard cases**: {len(all_results)} performance failures from 480-sample main experiment\n")
        f.write("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n")
        f.write(f"**Settings**: max_new_tokens={MAX_NEW_TOKENS}, temperature=0.0\n\n")
        f.write("**TASD-F config**: `enable_failure_aware_fallback=True`, `fallback_tokens=2`, `fallback_guarded=False`\n\n")

        # Table 1: Overall Summary
        f.write("## Table 1: Hard Cases Overall Summary\n\n")
        f.write("| Method | Mean TPS | Median TPS | Mean Speedup | Median Speedup | Mean SQ | Mean OffStr | Mean Repair | Mean Accept | Mean Repetition | Mean Truncation |\n")
        f.write("|--------|----------|------------|--------------|----------------|---------|-------------|-------------|-------------|-----------------|-----------------|\n")
        for method in ["AR", "FLY", "TASD", "TASD-F"]:
            m = s[method]
            sp_str = f"{m.get('mean_speedup', 0):.2f}x" if "mean_speedup" in m else "-"
            msp_str = f"{m.get('median_speedup', 0):.2f}x" if "median_speedup" in m else "-"
            f.write(f"| {method} | {m['mean_tps']:.1f} | {m['median_tps']:.1f} | "
                    f"{sp_str} | {msp_str} | "
                    f"{m['mean_sq']:.4f} | {m['mean_off_structure']:.4f} | "
                    f"{m['mean_repair_count']:.2f} | {m['mean_accept_rate']:.4f} | "
                    f"{m['mean_repetition']:.4f} | {m['mean_truncation']:.4f} |\n")
        f.write("\n")

        # Table 2: Below-1.0x count
        f.write("## Table 2: Below-1.0x Speedup Count\n\n")
        f.write("| Method | Below-1.0x Count | Total | Rate |\n")
        f.write("|--------|-----------------|-------|------|\n")
        for method in ["FLY", "TASD", "TASD-F"]:
            cnt = s[method]["below_1x_count"]
            f.write(f"| {method} | {cnt} | {len(all_results)} | {cnt/len(all_results):.1%} |\n")
        f.write("\n")

        # Table 3: Per-Sample Repair Effect
        f.write("## Table 3: Per-Sample Repair Effect (TASD -> TASD-F)\n\n")
        f.write("| # | Benchmark | Idx | Sample | TASD Sp | TASD-F Sp | Delta | Repair (TASD) | Repair (TASD-F) | Repair Delta |\n")
        f.write("|---|-----------|-----|--------|---------|-----------|-------|---------------|-----------------|---------------|\n")
        for i, r in enumerate(all_results):
            tasd_sp = r["tasd_speedup"]
            tasdf_sp = r["tasd_f_speedup"]
            delta = tasdf_sp - tasd_sp
            r_tasd = r["tasd"]["repair_count"]
            r_tasdf = r["tasd_f"]["repair_count"]
            r_delta = r_tasd - r_tasdf
            name = r["sample_name"].replace("_", "\\_")[:30]
            bench_short = r["benchmark"].replace("Real-Python-", "").replace("-Config", "")
            f.write(f"| {i+1} | {bench_short} | {r['sample_idx']} | {name} | "
                    f"{tasd_sp:.2f}x | {tasdf_sp:.2f}x | {delta:+.2f}x | "
                    f"{r_tasd} | {r_tasdf} | {r_delta:+d} |\n")
        f.write("\n")

        # Conclusions
        f.write("## Key Findings\n\n")
        tasd_below = s["TASD"]["below_1x_count"]
        tasdf_below = s["TASD-F"]["below_1x_count"]
        improved = tasd_below - tasdf_below

        f.write(f"- **Below-1.0x reduction**: TASD has {tasd_below}/{len(all_results)} below-1.0x cases, "
                f"TASD-F reduces this to {tasdf_below}/{len(all_results)} ({improved} improved)\n")
        f.write(f"- **Repair count**: TASD mean={s['TASD']['mean_repair_count']:.2f}, "
                f"TASD-F mean={s['TASD-F']['mean_repair_count']:.2f}\n")
        f.write(f"- **SQ preservation**: TASD mean={s['TASD']['mean_sq']:.4f}, "
                f"TASD-F mean={s['TASD-F']['mean_sq']:.4f}\n")
        f.write(f"- **Off-structure**: TASD mean={s['TASD']['mean_off_structure']:.4f}, "
                f"TASD-F mean={s['TASD-F']['mean_off_structure']:.4f}\n")

        if improved > 0:
            f.write("\n**Conclusion**: TASD-F successfully reduces the number of below-1.0x hard cases, "
                    "validating it as an effective hard-case repair extension.\n")
        else:
            f.write("\n**Note**: TASD-F does not reduce below-1.0x count in this run. "
                    "It remains an optional extension for specific hard cases, not a replacement for the main TASD method.\n")

    print(f"Report saved to {out_md}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    # Load 24 hard cases
    with open("/tmp/hard_cases_24.json") as f:
        hard_list = json.load(f)

    print(f"Loading {len(hard_list)} hard cases...")

    # Pre-load data files
    data_cache = {}
    for bid, path in DATA_FILES.items():
        data_cache[bid] = load_jsonl(path, 80)

    # Build hard case entries with prompts
    hard_entries = []
    for h in hard_list:
        bid = BENCHMARK_TO_BID.get(h["benchmark"])
        if not bid:
            print(f"WARN: Unknown benchmark {h['benchmark']}, skipping")
            continue
        if h["sample_idx"] >= len(data_cache[bid]):
            print(f"WARN: idx {h['sample_idx']} out of range for {bid}, skipping")
            continue
        sample = data_cache[bid][h["sample_idx"]]
        hard_entries.append({
            "benchmark": h["benchmark"],
            "bid": bid,
            "sample_idx": h["sample_idx"],
            "sample_name": h["sample_name"],
            "source_file": h.get("source_file", sample.get("source", "")),
            "prompt": sample["prompt"],
            "reference": sample.get("reference", ""),
            "structure_type": sample.get("structure_type", bid),
            "prev_tasd_speedup": h["tasd_speedup"],
            "prev_accept_rate": h["accept_rate"],
            "prev_repair_count": h["repair_count"],
            "case_id": f"{bid}_{h['sample_idx']}",
        })
    print(f"Loaded {len(hard_entries)} valid hard cases")

    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, local_files_only=True,
                                                   device_map="auto", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, local_files_only=True,
                                                  device_map="auto", trust_remote_code=True).eval()
    print("Models loaded.")

    # Warm-up
    print("Warming up...")
    _ = run_ar(target, tokenizer, hard_entries[0]["prompt"])
    print("Warm-up done.\n")

    all_results = []
    ar_baselines = {}

    for i, entry in enumerate(hard_entries):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(hard_entries)}] {entry['benchmark']} idx={entry['sample_idx']} "
              f"{entry['sample_name']} (prev TASD speedup: {entry['prev_tasd_speedup']:.3f}x)")
        print(f"{'='*60}")

        prompt = entry["prompt"]
        structure_type = entry["structure_type"]
        reference = entry["reference"]

        result = {
            "benchmark": entry["benchmark"],
            "bid": entry["bid"],
            "sample_idx": entry["sample_idx"],
            "sample_name": entry["sample_name"],
            "source_file": entry["source_file"],
            "case_id": entry["case_id"],
            "tasd_speedup": entry["prev_tasd_speedup"],  # from 480 experiment
        }

        # ── AR ──
        print("  AR...", end=" ", flush=True)
        r_ar = run_ar(target, tokenizer, prompt)
        r_ar["sq"] = compute_sq(r_ar["generated_text"], reference)
        r_ar["off_structure"] = compute_off_structure(r_ar["generated_text"], structure_type)
        r_ar["repetition"] = compute_repetition(r_ar["generated_text"])
        r_ar["truncation"] = compute_truncation(r_ar["generated_text"])
        result["ar"] = r_ar
        ar_tps = r_ar["tps"]
        ar_baselines[entry["case_id"]] = ar_tps
        print(f"TPS={r_ar['tps']:.1f}")

        # ── FLY ──
        print("  FLY...", end=" ", flush=True)
        r_fly = fly_decode(target, draft, tokenizer, prompt)
        r_fly["sq"] = compute_sq(r_fly["generated_text"], reference)
        r_fly["off_structure"] = compute_off_structure(r_fly["generated_text"], structure_type)
        r_fly["repetition"] = compute_repetition(r_fly["generated_text"])
        r_fly["truncation"] = compute_truncation(r_fly["generated_text"])
        r_fly["speedup"] = r_fly["tps"] / ar_tps if ar_tps > 0 else 0
        result["fly"] = r_fly
        print(f"TPS={r_fly['tps']:.1f} Sp={r_fly['speedup']:.2f}x Accept={r_fly['accept_rate']:.3f}")

        # ── TASD ──
        print("  TASD...", end=" ", flush=True)
        r_tasd = run_tasd(target, draft, tokenizer, prompt, structure_type)
        r_tasd["sq"] = compute_sq(r_tasd["generated_text"], reference)
        r_tasd["off_structure"] = compute_off_structure(r_tasd["generated_text"], structure_type)
        r_tasd["repetition"] = compute_repetition(r_tasd["generated_text"])
        r_tasd["truncation"] = compute_truncation(r_tasd["generated_text"])
        r_tasd["speedup"] = r_tasd["tps"] / ar_tps if ar_tps > 0 else 0
        result["tasd"] = r_tasd
        result["tasd_speedup"] = r_tasd["speedup"]
        print(f"TPS={r_tasd['tps']:.1f} Sp={r_tasd['speedup']:.2f}x Accept={r_tasd['accept_rate']:.3f} Repair={r_tasd['repair_count']}")

        # ── TASD-F ──
        print("  TASD-F...", end=" ", flush=True)
        r_tasdf = run_tasd(target, draft, tokenizer, prompt, structure_type,
                           extra_kwargs={"enable_failure_aware_fallback": True})
        r_tasdf["sq"] = compute_sq(r_tasdf["generated_text"], reference)
        r_tasdf["off_structure"] = compute_off_structure(r_tasdf["generated_text"], structure_type)
        r_tasdf["repetition"] = compute_repetition(r_tasdf["generated_text"])
        r_tasdf["truncation"] = compute_truncation(r_tasdf["generated_text"])
        r_tasdf["speedup"] = r_tasdf["tps"] / ar_tps if ar_tps > 0 else 0
        result["tasd_f"] = r_tasdf
        result["tasd_f_speedup"] = r_tasdf["speedup"]
        print(f"TPS={r_tasdf['tps']:.1f} Sp={r_tasdf['speedup']:.2f}x Accept={r_tasdf['accept_rate']:.3f} Repair={r_tasdf['repair_count']}")

        all_results.append(result)

    # Generate reports
    out_json = "results/tasd_hardcase_repair_24.json"
    out_md = "results/tasd_hardcase_repair_24.md"
    generate_report(all_results, ar_baselines, out_json, out_md)
    print("\nDone!")


if __name__ == "__main__":
    main()
