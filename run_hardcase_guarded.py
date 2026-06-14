"""
TASD-F-G Hard-Case Repair: Guarded and Selective Guarded Fallback Variants.

Runs only the new variants (TASD-F guarded, TASD-F selective guarded) on the 24
performance hard cases, reusing AR/FLY/TASD/TASD-F(unguarded) from previous run.
"""

import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

PREV_JSON = "results/tasd_hardcase_repair_24.json"
OUT_JSON = "results/tasd_hardcase_repair_guarded_24.json"
OUT_MD = "results/tasd_hardcase_repair_guarded_24.md"

# TASD base config
TASD_KWARGS = {
    "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
    "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
    "enable_guard": True, "enable_relaxed_accept": True,
}


def load_jsonl(path, n=80):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def run_variant(target_model, draft_model, tokenizer, prompt, structure_type, extra_kwargs):
    kwargs = dict(TASD_KWARGS)
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


def compute_sq(generated_text, reference_text):
    struct_chars = set("{}[]():,=\n")
    gen_struct = [c for c in generated_text if c in struct_chars]
    ref_struct = [c for c in reference_text if c in struct_chars]
    if not ref_struct:
        return 1.0
    matches = sum(1 for c in gen_struct if c in ref_struct)
    return min(matches / len(ref_struct), 1.0)


def compute_off_structure(generated_text, structure_type):
    lines = generated_text.split("\n") if generated_text else []
    if not lines:
        return 0.0
    struct_keywords = {"def ", "class ", "import ", "from "}
    cnt = sum(1 for line in lines if any(line.strip().startswith(kw) for kw in struct_keywords))
    return cnt / len(lines)


def compute_repetition(text):
    if not text or len(text) < 8:
        return 0.0
    n = 4
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if len(ngrams) <= 1:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)


def generate_report(all_results):
    """Generate combined report with all 6 methods."""
    os.makedirs("results", exist_ok=True)

    methods = ["AR", "FLY", "TASD", "TASD-F", "TASD-F-G", "TASD-F-G-Sel"]

    # ── JSON ──
    json_output = {
        "config": {
            "n_hard_cases": len(all_results),
            "max_new_tokens": MAX_NEW_TOKENS,
            "methods": methods,
            "tasd_f_guarded": {"enable_failure_aware_fallback": True, "fallback_guarded": True, "fallback_accept_threshold": 0.5, "fallback_repair_threshold": 2},
            "tasd_f_selective": {"enable_failure_aware_fallback": True, "fallback_guarded": True, "fallback_accept_threshold": 0.2, "fallback_repair_threshold": 3},
        },
        "summary": {},
        "per_sample": [],
    }

    for method in methods:
        tps_vals, speedups, sq_vals, off_vals = [], [], [], []
        repair_vals, accept_vals, rep_vals, trunc_vals = [], [], [], []
        below_1_count = 0

        for r in all_results:
            key = method.lower().replace("-", "_")
            data = r.get(key)
            if data is None:
                continue
            tps_vals.append(data["tps"])
            sq_vals.append(data.get("sq", 0))
            off_vals.append(data.get("off_structure", 0))
            repair_vals.append(data.get("repair_count", 0))
            accept_vals.append(data.get("accept_rate", 0))
            rep_vals.append(data.get("repetition", 0))
            trunc_vals.append(data.get("truncation", 0))

            if method != "AR":
                ar_tps = r["ar"]["tps"]
                sp = data["tps"] / ar_tps if ar_tps > 0 else 0
                speedups.append(sp)
                if sp < 1.0:
                    below_1_count += 1

        n = len(all_results)
        summary = {
            "n": n, "mean_tps": round(sum(tps_vals) / n, 2),
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

    # Per-sample
    for r in all_results:
        entry = {"benchmark": r["benchmark"], "sample_idx": r["sample_idx"],
                 "sample_name": r["sample_name"], "case_id": r["case_id"]}
        for method in methods:
            key = method.lower().replace("-", "_")
            entry[key] = r.get(key)
        # Deltas
        if r.get("tasd_f_g") and r.get("tasd"):
            entry["tasd_f_g_speedup_delta"] = round(r["tasd_f_g"]["speedup"] - r["tasd"]["speedup"], 4)
            entry["tasd_f_g_repair_delta"] = r["tasd"]["repair_count"] - r["tasd_f_g"]["repair_count"]
        if r.get("tasd_f_g_sel") and r.get("tasd"):
            entry["tasd_f_g_sel_speedup_delta"] = round(r["tasd_f_g_sel"]["speedup"] - r["tasd"]["speedup"], 4)
            entry["tasd_f_g_sel_repair_delta"] = r["tasd"]["repair_count"] - r["tasd_f_g_sel"]["repair_count"]
        json_output["per_sample"].append(entry)

    with open(OUT_JSON, "w") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {OUT_JSON}")

    # ── MD ──
    s = json_output["summary"]
    with open(OUT_MD, "w") as f:
        f.write("# TASD-F-G Hard-Case Repair Experiment\n\n")
        f.write(f"**Hard cases**: {len(all_results)} performance failures from 480-sample main experiment\n")
        f.write("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n\n")
        f.write("### Variants\n\n")
        f.write("| Variant | fallback_guarded | accept_threshold | repair_threshold |\n")
        f.write("|---------|-----------------|-------------------|------------------|\n")
        f.write("| TASD-F (unguarded) | False | 0.5 | 2 |\n")
        f.write("| TASD-F-G (guarded) | **True** | 0.5 | 2 |\n")
        f.write("| TASD-F-G-Sel (selective guarded) | **True** | **0.2** | **3** |\n\n")

        # Table 1
        f.write("## Table 1: Hard Cases — All Methods Summary\n\n")
        headers = "Method | Mean TPS | Median TPS | Mean Speedup | Median Speedup | Mean SQ | Mean OffStr | Mean Repair | Mean Accept | Below 1.0x"
        sep = "-------|----------|------------|--------------|----------------|---------|-------------|-------------|-------------|------------"
        f.write(f"| {headers} |\n")
        f.write(f"| {sep} |\n")
        for method in methods:
            m = s[method]
            sp_str = f"{m.get('mean_speedup', 0):.2f}x" if "mean_speedup" in m else "-"
            msp_str = f"{m.get('median_speedup', 0):.2f}x" if "median_speedup" in m else "-"
            f.write(f"| {method} | {m['mean_tps']:.1f} | {m['median_tps']:.1f} | "
                    f"{sp_str} | {msp_str} | {m['mean_sq']:.4f} | {m['mean_off_structure']:.4f} | "
                    f"{m['mean_repair_count']:.2f} | {m['mean_accept_rate']:.4f} | "
                    f"{m['below_1x_count']}/{n} |\n")
        f.write("\n")

        # Table 2: Below-1.0x comparison (TASD vs the F variants)
        f.write("## Table 2: Below-1.0x Count Comparison\n\n")
        f.write("| Variant | Below-1.0x | Rate | vs TASD |\n")
        f.write("|---------|-----------|------|---------|\n")
        tasd_below = s["TASD"]["below_1x_count"]
        for method in ["TASD-F", "TASD-F-G", "TASD-F-G-Sel"]:
            cnt = s[method]["below_1x_count"]
            delta = tasd_below - cnt
            f.write(f"| {method} | {cnt}/{n} | {cnt/n:.1%} | {delta:+d} |\n")
        f.write("\n")

        # Table 3: Per-sample deltas
        f.write("## Table 3: Per-Sample Speedup Deltas (vs TASD baseline)\n\n")
        f.write("| # | Benchmark | Idx | Name | TASD | TASD-F | TASD-F-G | TASD-F-G-Sel |\n")
        f.write("|---|-----------|-----|------|------|--------|----------|--------------|\n")
        for i, r in enumerate(all_results):
            name = r["sample_name"].replace("_", "\\_")[:25]
            bench_short = r["benchmark"].replace("Real-Python-", "").replace("-Config", "")
            vals = []
            for method in ["TASD", "TASD-F", "TASD-F-G", "TASD-F-G-Sel"]:
                key = method.lower().replace("-", "_")
                sp = r.get(key, {}).get("speedup", 0)
                vals.append(f"{sp:.2f}x")
            f.write(f"| {i+1} | {bench_short} | {r['sample_idx']} | {name} | "
                    f"{' | '.join(vals)} |\n")
        f.write("\n")

        # Criteria Check
        f.write("## Criteria Check\n\n")
        tasd = s["TASD"]
        tfg = s.get("TASD-F-G", {})
        tfgs = s.get("TASD-F-G-Sel", {})

        checks = []
        for label, v in [("TASD-F-G", tfg), ("TASD-F-G-Sel", tfgs)]:
            if not v:
                continue
            below_ok = v["below_1x_count"] < tasd["below_1x_count"]
            repair_ok = v["mean_repair_count"] < tasd["mean_repair_count"]
            offstr_ok = v["mean_off_structure"] < 0.05
            speedup_ok = v.get("mean_speedup", 0) >= tasd.get("mean_speedup", 0)
            all_ok = below_ok and repair_ok and offstr_ok and speedup_ok
            checks.append((label, below_ok, repair_ok, offstr_ok, speedup_ok, all_ok))

        f.write("| Variant | Below ↓ | Repair ↓ | OffStr < 0.05 | Speedup ≥ TASD | All Pass |\n")
        f.write("|---------|---------|----------|---------------|----------------|----------|\n")
        for label, below_ok, repair_ok, offstr_ok, speedup_ok, all_ok in checks:
            f.write(f"| {label} | {'YES' if below_ok else 'NO'} | {'YES' if repair_ok else 'NO'} | "
                    f"{'YES' if offstr_ok else 'NO'} | {'YES' if speedup_ok else 'NO'} | "
                    f"{'**PASS**' if all_ok else 'FAIL'} |\n")
        f.write("\n")

        # Conclusions
        f.write("## Key Findings\n\n")
        for label, below_ok, repair_ok, offstr_ok, speedup_ok, all_ok in checks:
            v = s[label]
            f.write(f"### {label}\n\n")
            f.write(f"- Below-1.0x: {v['below_1x_count']}/{n} (TASD: {tasd['below_1x_count']}/{n})\n")
            f.write(f"- Mean Repair: {v['mean_repair_count']:.2f} (TASD: {tasd['mean_repair_count']:.2f})\n")
            f.write(f"- Off-Structure: {v['mean_off_structure']:.4f} (TASD: {tasd['mean_off_structure']:.4f})\n")
            f.write(f"- Mean Speedup: {v.get('mean_speedup', 0):.2f}x (TASD: {tasd.get('mean_speedup', 0):.2f}x)\n")

            if all_ok:
                f.write(f"\n**Verdict**: {label} passes all criteria. Can serve as a positive TASD-F extension.\n\n")
            else:
                failed = []
                if not below_ok: failed.append("below-1.0x count not lower than TASD")
                if not repair_ok: failed.append("repair_count not lower than TASD")
                if not offstr_ok: failed.append("off-structure rate >= 0.05")
                if not speedup_ok: failed.append("mean speedup below TASD")
                f.write(f"\n**Verdict**: {label} fails: {', '.join(failed)}. "
                        f"Not suitable as primary TASD-F extension. May serve as exploratory/negative result.\n\n")

    print(f"Report saved to {OUT_MD}")


def main():
    # Load previous results
    print("Loading previous results...")
    with open(PREV_JSON) as f:
        prev_data = json.load(f)

    # Load data files for prompts/references
    data_cache = {}
    for bid, path in DATA_FILES.items():
        data_cache[bid] = load_jsonl(path, 80)

    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, local_files_only=True,
                                                   device_map="auto", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, local_files_only=True,
                                                  device_map="auto", trust_remote_code=True).eval()
    print("Models loaded.\n")

    # Reuse existing AR/FLY/TASD/TASD-F results, add new variants
    all_results = []
    for ps in prev_data["per_sample"]:
        bid = BENCHMARK_TO_BID.get(ps["benchmark"])
        if not bid or ps["sample_idx"] >= len(data_cache[bid]):
            continue

        entry = {
            "benchmark": ps["benchmark"],
            "sample_idx": ps["sample_idx"],
            "sample_name": ps["sample_name"],
            "case_id": f"{bid}_{ps['sample_idx']}",
            "ar": ps["ar"],
            "fly": ps["fly"],
            "tasd": ps["tasd"],
            "tasd_f": ps["tasd_f"],
            "tasd_speedup": ps["tasd_speedup"],
            "tasd_f_speedup": ps["tasd_f_speedup"],
        }
        all_results.append(entry)

    print(f"Loaded {len(all_results)} hard cases from previous experiment.")

    # Run new variants
    for i, entry in enumerate(all_results):
        prompt = data_cache[BENCHMARK_TO_BID[entry["benchmark"]]][entry["sample_idx"]]["prompt"]
        structure_type = BENCHMARK_TO_BID[entry["benchmark"]]
        reference = data_cache[BENCHMARK_TO_BID[entry["benchmark"]]][entry["sample_idx"]].get("reference", "")

        print(f"\n[{i+1}/{len(all_results)}] {entry['benchmark']} idx={entry['sample_idx']} {entry['sample_name']}")

        if "tasd_f_g" not in entry:
            print("  TASD-F-G (guarded)...", end=" ", flush=True)
            r = run_variant(target, draft, tokenizer, prompt, structure_type, {
                "enable_failure_aware_fallback": True,
                "fallback_guarded": True,
            })
            r["sq"] = compute_sq(r["generated_text"], reference)
            r["off_structure"] = compute_off_structure(r["generated_text"], structure_type)
            r["repetition"] = compute_repetition(r["generated_text"])
            r["truncation"] = 0.0
            r["speedup"] = r["tps"] / entry["ar"]["tps"] if entry["ar"]["tps"] > 0 else 0
            entry["tasd_f_g"] = r
            print(f"TPS={r['tps']:.1f} Sp={r['speedup']:.2f}x Accept={r['accept_rate']:.3f} Repair={r['repair_count']}")

        if "tasd_f_g_sel" not in entry:
            print("  TASD-F-G-Sel (selective guarded)...", end=" ", flush=True)
            r = run_variant(target, draft, tokenizer, prompt, structure_type, {
                "enable_failure_aware_fallback": True,
                "fallback_guarded": True,
                "fallback_accept_threshold": 0.2,
                "fallback_repair_threshold": 3,
            })
            r["sq"] = compute_sq(r["generated_text"], reference)
            r["off_structure"] = compute_off_structure(r["generated_text"], structure_type)
            r["repetition"] = compute_repetition(r["generated_text"])
            r["truncation"] = 0.0
            r["speedup"] = r["tps"] / entry["ar"]["tps"] if entry["ar"]["tps"] > 0 else 0
            entry["tasd_f_g_sel"] = r
            print(f"TPS={r['tps']:.1f} Sp={r['speedup']:.2f}x Accept={r['accept_rate']:.3f} Repair={r['repair_count']}")

    generate_report(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
