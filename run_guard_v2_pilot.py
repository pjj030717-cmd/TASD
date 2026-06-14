"""
Guard-v2 Pilot: Test GuardV2 on 24 perf + 20 quality hard cases.

Methods: TASD, TASD-F-G-Sel, TASD-F-G-Sel + GuardV2
"""
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode

MAX_NEW_TOKENS = 128
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"

DATA_FILES = {
    "argparse": "/root/autodl-tmp/data/codesearchnet_argparse_blocks_80.jsonl",
    "dict_config": "/root/autodl-tmp/data/codesearchnet_dict_config_blocks_80.jsonl",
    "openmmlab_config": "/root/autodl-tmp/data/ml_config_blocks_openmmlab_80.jsonl",
}
BENCHMARK_TO_BID = {
    "Real-Python-Argparse": "argparse", "Real-Python-DictConfig": "dict_config",
    "OpenMMLab-Config": "openmmlab_config",
}
OUT_JSON = "results/guard_v2_pilot_hardcases.json"
OUT_MD = "results/guard_v2_pilot_hardcases.md"

TASD_KWARGS = {"draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
               "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
               "enable_guard": True, "enable_relaxed_accept": True}

TASDF_SEL_KWARGS = {"draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
                    "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
                    "enable_guard": True, "enable_relaxed_accept": True,
                    "enable_failure_aware_fallback": True,
                    "fallback_guarded": True, "fallback_accept_threshold": 0.2,
                    "fallback_repair_threshold": 3}


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


def run_variant(target_model, draft_model, tokenizer, prompt, structure_type, kwargs):
    r = tasd_decode(target_model=target_model, draft_model=draft_model, tokenizer=tokenizer,
                    prompt=prompt, structure_type=structure_type, max_new_tokens=MAX_NEW_TOKENS, **kwargs)
    r["tps"] = r.get("tokens_per_second", 0)
    stats = r.get("stats", {})
    r["accept_rate"] = stats.get("accept_rate", 0)
    r["repair_count"] = stats.get("repair_count", 0)
    r["guard_trigger_count"] = stats.get("guard_trigger_count", 0)
    r["high_risk_count"] = stats.get("guard_v2_high_risk_count", 0)
    return r


def compute_sq(generated_text, reference_text):
    struct_chars = set("{}[]():,=\n")
    gen_struct = [c for c in generated_text if c in struct_chars]
    ref_struct = [c for c in reference_text if c in struct_chars]
    if not ref_struct: return 1.0
    return min(sum(1 for c in gen_struct if c in ref_struct) / len(ref_struct), 1.0)


def compute_off_structure(generated_text, structure_type):
    lines = generated_text.split("\n") if generated_text else []
    if not lines: return 0.0
    kw = {"def ", "class ", "import ", "from "}
    return sum(1 for l in lines if any(l.strip().startswith(k) for k in kw)) / len(lines)


def compute_truncation_rate(text):
    if not text: return 0.0
    endings = ["type=", "default=", "action=", "nargs=", "choices=", "required=",
               "help=", "metavar=", "dest=", "...", "(\"", "['", "{\""]
    last_80 = text.strip()[-80:] if len(text) > 80 else text.strip()
    return 1.0 if any(last_80.endswith(e) for e in endings) else 0.0


def generate_report(all_results):
    os.makedirs("results", exist_ok=True)
    methods = ["TASD", "TASD-F-G-Sel", "TASD-F-G-Sel+GV2"]
    splits = {"perf": [], "quality": []}

    for r in all_results:
        splits[r["split"]].append(r)

    # ── JSON ──
    json_out = {"config": {"n_perf": len(splits["perf"]), "n_quality": len(splits["quality"])},
                "summary": {}, "per_sample": []}
    for split_name, cases in [("perf", splits["perf"]), ("quality", splits["quality"])]:
        json_out["summary"][split_name] = {}
        for method in methods:
            key = method.lower().replace("-", "_").replace("+gv2", "_gv2")
            tps_vals, sq_vals, off_vals, rep_vals = [], [], [], []
            acc_vals, tr_vals, grd_vals, hr_vals = [], [], [], []
            speedups, below_1 = [], 0
            for r in cases:
                d = r.get(key)
                if d is None: continue
                tps_vals.append(d["tps"]); sq_vals.append(d.get("sq", 0))
                off_vals.append(d.get("off_structure", 0))
                rep_vals.append(d.get("repair_count", 0))
                acc_vals.append(d.get("accept_rate", 0))
                tr_vals.append(d.get("truncation", 0))
                grd_vals.append(d.get("guard_trigger_count", 0))
                hr_vals.append(d.get("high_risk_count", 0))
                if method != "TASD" or True:  # always compute speedup
                    sp = d["tps"] / r["ar_tps"] if r["ar_tps"] > 0 else 0
                    speedups.append(sp)
                    if sp < 1.0: below_1 += 1
            n = len(cases)
            m = {"n": n, "mean_tps": round(sum(tps_vals)/n, 2), 
                 "median_tps": round(sorted(tps_vals)[n//2], 2),
                 "mean_sq": round(sum(sq_vals)/n, 4),
                 "mean_off_structure": round(sum(off_vals)/n, 4),
                 "mean_repair_count": round(sum(rep_vals)/n, 2),
                 "mean_accept_rate": round(sum(acc_vals)/n, 4),
                 "mean_truncation": round(sum(tr_vals)/n, 4),
                 "mean_guard_trigger": round(sum(grd_vals)/n, 2),
                 "mean_high_risk": round(sum(hr_vals)/n, 2),
                 "below_1x_count": below_1}
            if speedups:
                m["mean_speedup"] = round(sum(speedups)/len(speedups), 2)
                m["median_speedup"] = round(sorted(speedups)[len(speedups)//2], 2)
            json_out["summary"][split_name][method] = m

    for r in all_results:
        entry = {"benchmark": r["benchmark"], "sample_idx": r["sample_idx"],
                 "split": r["split"], "sample_name": r["sample_name"]}
        for method in methods:
            key = method.lower().replace("-", "_").replace("+gv2", "_gv2")
            entry[key] = r.get(key)
        json_out["per_sample"].append(entry)

    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {OUT_JSON}")

    # ── MD ──
    with open(OUT_MD, "w") as f:
        f.write("# Guard-v2 Pilot Experiment\n\n")
        f.write(f"**Cases**: {len(splits['perf'])} perf + {len(splits['quality'])} quality hard cases\n\n")
        f.write("### GuardV2 Features\n\n")
        f.write("1. Incremental syntax state (bracket_stack, quote_state, comment_state)\n")
        f.write("2. Comment/string awareness (def/class in strings → medium risk, not high)\n")
        f.write("3. Adaptive verification tightening (high risk → top_k_accept=1)\n\n")

        for split_name, label in [("perf", "Performance Hard Cases"), ("quality", "Quality-Flagged Hard Cases")]:
            f.write(f"## {label} ({len(splits[split_name])} cases)\n\n")
            headers = "Method | Mean TPS | Mean Speedup | Below 1.0x | Mean SQ | Mean OffStr | Mean Trunc | Mean Repair | Mean Guard Trig | Mean HighRisk"
            sep = "-------|----------|-------------|-----------|---------|-------------|-----------|-------------|---------------|----------"
            f.write(f"| {headers} |\n| {sep} |\n")
            for method in methods:
                m = json_out["summary"][split_name][method]
                sp = f"{m.get('mean_speedup',0):.2f}x"
                f.write(f"| {method} | {m['mean_tps']:.1f} | {sp} | {m['below_1x_count']} | "
                        f"{m['mean_sq']:.4f} | {m['mean_off_structure']:.4f} | "
                        f"{m['mean_truncation']:.4f} | {m['mean_repair_count']:.2f} | "
                        f"{m['mean_guard_trigger']:.1f} | {m['mean_high_risk']:.1f} |\n")
            f.write("\n")

        # Criteria check
        f.write("## Criteria Check\n\n")
        for split_name, label in [("perf", "Performance Hard Cases"), ("quality", "Quality-Flagged Hard Cases")]:
            f.write(f"### {label}\n\n")
            m_tasd = json_out["summary"][split_name]["TASD"]
            m_f = json_out["summary"][split_name]["TASD-F-G-Sel"]
            m_gv2 = json_out["summary"][split_name]["TASD-F-G-Sel+GV2"]
            off_ok = m_gv2["mean_off_structure"] <= m_f["mean_off_structure"]
            sq_ok = m_gv2["mean_sq"] >= m_f["mean_sq"] * 0.98
            sp_ok = m_gv2.get("mean_speedup", 0) >= m_f.get("mean_speedup", 0) * 0.95
            all_ok = off_ok and sq_ok and sp_ok

            f.write(f"- Off-structure: GuardV2={m_gv2['mean_off_structure']:.4f} vs TASD-F-G-Sel={m_f['mean_off_structure']:.4f} → {'OK' if off_ok else 'FAIL'}\n")
            f.write(f"- SQ: GuardV2={m_gv2['mean_sq']:.4f} vs TASD-F-G-Sel={m_f['mean_sq']:.4f} → {'OK' if sq_ok else 'FAIL'}\n")
            f.write(f"- Speedup: GuardV2={m_gv2.get('mean_speedup',0):.2f}x vs TASD-F-G-Sel={m_f.get('mean_speedup',0):.2f}x → {'OK' if sp_ok else 'FAIL'}\n")
            f.write(f"\n**Verdict**: {'PASS — GuardV2 can enter main method' if all_ok else 'FAIL — GuardV2 retained as future work / negative result'}\n\n")

        # Per-sample speedup deltas
        f.write("## Per-Sample Speedup Deltas\n\n")
        f.write("| # | Split | Benchmark | Idx | Name | TASD | TASD-F-G-Sel | +GV2 | GV2 Delta |\n")
        f.write("|---|-------|-----------|-----|------|------|-------------|------|----------|\n")
        for i, r in enumerate(all_results):
            name = r["sample_name"].replace("_", "\\_")[:20]
            bench_short = r["benchmark"].replace("Real-Python-", "").replace("-Config", "")
            s_tasd = r["tasd"].get("speedup", 0)
            s_f = r["tasd_f_g_sel"].get("speedup", 0)
            s_gv2 = r["tasd_f_g_sel_gv2"].get("speedup", 0)
            delta = s_gv2 - s_f
            f.write(f"| {i+1} | {r['split']} | {bench_short} | {r['sample_idx']} | {name} | "
                    f"{s_tasd:.2f}x | {s_f:.2f}x | {s_gv2:.2f}x | {delta:+.2f}x |\n")

    print(f"Report saved to {OUT_MD}")


def main():
    # Load cases
    print("Loading cases...")
    with open("/tmp/hard_cases_24.json") as f:
        perf_list = json.load(f)
    with open("/tmp/quality_20.json") as f:
        qual_list = json.load(f)

    data_cache = {}
    for bid, path in DATA_FILES.items():
        data_cache[bid] = load_jsonl(path, 80)

    all_entries = []
    for lst, split_label in [(perf_list, "perf"), (qual_list, "quality")]:
        for h in lst:
            bid = BENCHMARK_TO_BID.get(h["benchmark"], h.get("bid", ""))
            if not bid or h["sample_idx"] >= len(data_cache.get(bid, [])):
                continue
            sample = data_cache[bid][h["sample_idx"]]
            all_entries.append({
                "benchmark": h["benchmark"], "bid": bid,
                "sample_idx": h["sample_idx"],
                "sample_name": h.get("sample_name", f"{bid}_{h['sample_idx']}"),
                "prompt": sample["prompt"],
                "reference": sample.get("reference", ""),
                "structure_type": sample.get("structure_type", bid),
                "split": split_label,
            })
    perf_count = len([e for e in all_entries if e['split']=='perf'])
    qual_count = len([e for e in all_entries if e['split']=='quality'])
    print(f"Total entries: {len(all_entries)} ({perf_count} perf, {qual_count} quality)")

    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, local_files_only=True,
                                                   device_map="auto", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, local_files_only=True,
                                                  device_map="auto", trust_remote_code=True).eval()
    print("Models loaded.\n")

    all_results = []
    for i, entry in enumerate(all_entries):
        print(f"\n[{'P' if entry['split']=='perf' else 'Q'}{i+1}] {entry['benchmark']} idx={entry['sample_idx']}")
        prompt = entry["prompt"]
        structure_type = entry["structure_type"]
        reference = entry["reference"]

        result = {"benchmark": entry["benchmark"], "sample_idx": entry["sample_idx"],
                  "sample_name": entry["sample_name"], "split": entry["split"],
                  "case_id": f"{entry['bid']}_{entry['sample_idx']}"}

        # TASD
        print("  TASD...", end=" ", flush=True)
        r = run_variant(target, draft, tokenizer, prompt, structure_type, TASD_KWARGS)
        r["sq"] = compute_sq(r["generated_text"], reference)
        r["off_structure"] = compute_off_structure(r["generated_text"], structure_type)
        r["truncation"] = compute_truncation_rate(r["generated_text"])
        result["ar_tps"] = 31.0  # rough avg
        r["speedup"] = r["tps"] / result["ar_tps"]
        result["tasd"] = r
        print(f"TPS={r['tps']:.1f} Sp={r['speedup']:.2f}x SQ={r['sq']:.3f} Off={r['off_structure']:.3f}")

        # TASD-F-G-Sel
        print("  TASD-F-G-Sel...", end=" ", flush=True)
        r = run_variant(target, draft, tokenizer, prompt, structure_type, TASDF_SEL_KWARGS)
        r["sq"] = compute_sq(r["generated_text"], reference)
        r["off_structure"] = compute_off_structure(r["generated_text"], structure_type)
        r["truncation"] = compute_truncation_rate(r["generated_text"])
        r["speedup"] = r["tps"] / result["ar_tps"]
        result["tasd_f_g_sel"] = r
        print(f"TPS={r['tps']:.1f} Sp={r['speedup']:.2f}x SQ={r['sq']:.3f} Off={r['off_structure']:.3f}")

        # TASD-F-G-Sel + GuardV2
        print("  TASD-F-G-Sel+GV2...", end=" ", flush=True)
        gv2_kwargs = dict(TASDF_SEL_KWARGS)
        gv2_kwargs["guard_v2"] = True
        r = run_variant(target, draft, tokenizer, prompt, structure_type, gv2_kwargs)
        r["sq"] = compute_sq(r["generated_text"], reference)
        r["off_structure"] = compute_off_structure(r["generated_text"], structure_type)
        r["truncation"] = compute_truncation_rate(r["generated_text"])
        r["speedup"] = r["tps"] / result["ar_tps"]
        result["tasd_f_g_sel_gv2"] = r
        print(f"TPS={r['tps']:.1f} Sp={r['speedup']:.2f}x SQ={r['sq']:.3f} Off={r['off_structure']:.3f} HighRisk={r['high_risk_count']}")

        all_results.append(result)

    generate_report(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
