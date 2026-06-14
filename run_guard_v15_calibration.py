"""
Guard-v1.5 Calibration Experiment.
Tests 24 perf hard cases with 4 variants:
  TASD (current guard), TASD-F-G-Sel (current guard),
  TASD-F-G-Sel + Guard-v1.5 (calibrated=True),
  GuardV2 (reference from pilot)
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
OUT_JSON = "results/guard_v15_calibration_24.json"
OUT_MD = "results/guard_v15_calibration_24.md"

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

# Base TASD kwargs (current guard)
TASD_BASE = dict(draft_len=16, draft_blocks=2, top_k_accept=3,
                 min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                 enable_guard=True, enable_relaxed_accept=True)

# TASD-F-G-Sel (current guard)
TFGS_KWARGS = dict(TASD_BASE, enable_failure_aware_fallback=True,
                   fallback_guarded=True, fallback_accept_threshold=0.2,
                   fallback_repair_threshold=3)

# TASD-F-G-Sel + Guard-v1.5 (calibrated)
TFGS_GV15_KWARGS = dict(TFGS_KWARGS, guard_calibrated=True)

# GV2 reference (from pilot)
TFGS_GV2_KWARGS = dict(TFGS_KWARGS, guard_v2=True)

AR_TPS_ESTIMATE = 31.0

def run_variant(target_model, draft_model, tokenizer, prompt, structure_type, kwargs):
    r = tasd_decode(target_model=target_model, draft_model=draft_model, tokenizer=tokenizer,
                    prompt=prompt, structure_type=structure_type, max_new_tokens=MAX_NEW_TOKENS, **kwargs)
    r["tps"] = r.get("tokens_per_second", 0)
    stats = r.get("stats", {})
    r["accept_rate"] = stats.get("accept_rate", 0)
    r["repair_count"] = stats.get("repair_count", 0)
    r["guard_trigger_count"] = stats.get("guard_trigger_count", 0)
    r["trim_count"] = stats.get("trim_count", 0)
    r["hard_trim_count"] = stats.get("hard_trim_count", 0)
    r["repetition_warning_count"] = stats.get("repetition_warning_count", 0)
    r["bracket_warning_count"] = stats.get("bracket_warning_count", 0)
    r["import_warning_count"] = stats.get("import_warning_count", 0)
    r["trim_reasons"] = stats.get("trim_reasons", [])
    return r

def compute_sq(generated_text, reference_text):
    struct_chars = set("{}[]():,=\n")
    gen_struct = [c for c in generated_text if c in struct_chars]
    ref_struct = [c for c in reference_text if c in struct_chars]
    if not ref_struct: return 1.0
    return min(sum(1 for c in gen_struct if c in ref_struct) / len(ref_struct), 1.0)

def compute_off_structure(generated_text):
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

def compute_metrics(entry, variant_name, r):
    sp = r["tps"] / AR_TPS_ESTIMATE
    return {
        "tps": round(r["tps"], 2),
        "speedup": round(sp, 3),
        "sq": round(compute_sq(r["generated_text"], entry["reference"]), 4),
        "off_structure": round(compute_off_structure(r["generated_text"]), 4),
        "truncation": round(compute_truncation_rate(r["generated_text"]), 4),
        "accept_rate": round(r["accept_rate"], 4),
        "repair_count": r["repair_count"],
        "guard_trigger_count": r["guard_trigger_count"],
        "trim_count": r["trim_count"],
        "hard_trim_count": r.get("hard_trim_count", 0),
        "repetition_warning_count": r.get("repetition_warning_count", 0),
        "bracket_warning_count": r.get("bracket_warning_count", 0),
        "import_warning_count": r.get("import_warning_count", 0),
        "high_risk_count": r.get("high_risk_count", 0),
        "trim_reasons": r.get("trim_reasons", []),
    }

def generate_report(all_results):
    os.makedirs("results", exist_ok=True)
    variants = ["TASD", "TASD-F-G-Sel", "+Guard-v1.5", "+GuardV2(ref)"]
    var_keys = ["tasd", "tfg_sel", "tfg_sel_gv15", "tfg_sel_gv2"]

    # Build tables
    table_rows = {}
    for vk in var_keys:
        table_rows[vk] = {k: [] for k in ["tps", "speedup", "sq", "off_structure", "truncation",
                                           "repair_count", "guard_trigger", "trim_count",
                                           "hard_trim", "rep_warn", "bracket_warn", "import_warn",
                                           "high_risk"]}
    for r in all_results:
        for vk in var_keys:
            d = r.get(vk, {})
            if d:
                table_rows[vk]["tps"].append(d["tps"])
                table_rows[vk]["speedup"].append(d["speedup"])
                table_rows[vk]["sq"].append(d["sq"])
                table_rows[vk]["off_structure"].append(d["off_structure"])
                table_rows[vk]["truncation"].append(d["truncation"])
                table_rows[vk]["repair_count"].append(d["repair_count"])
                table_rows[vk]["guard_trigger"].append(d["guard_trigger_count"])
                table_rows[vk]["trim_count"].append(d["trim_count"])
                table_rows[vk]["hard_trim"].append(d.get("hard_trim_count", 0))
                table_rows[vk]["rep_warn"].append(d.get("repetition_warning_count", 0))
                table_rows[vk]["bracket_warn"].append(d.get("bracket_warning_count", 0))
                table_rows[vk]["import_warn"].append(d.get("import_warning_count", 0))
                table_rows[vk]["high_risk"].append(d.get("high_risk_count", 0))

    def mean(lst): return round(sum(lst)/len(lst), 2) if lst else 0
    def mean4(lst): return round(sum(lst)/len(lst), 4) if lst else 0
    def median(lst): return round(sorted(lst)[len(lst)//2], 2) if lst else 0

    # JSON
    json_out = {"config": {"n_cases": len(all_results)}, "summary": {}, "per_sample": all_results}
    for vi, vk in enumerate(var_keys):
        tr = table_rows[vk]
        n = len(tr["tps"])
        if n == 0: continue
        spd = tr["speedup"]
        below_1 = sum(1 for s in spd if s < 1.0)
        json_out["summary"][variants[vi]] = {
            "n": n,
            "mean_tps": mean(tr["tps"]), "median_tps": median(tr["tps"]),
            "mean_speedup": mean(spd), "median_speedup": median(spd),
            "below_1x_count": below_1,
            "mean_sq": mean4(tr["sq"]),
            "mean_off_structure": mean4(tr["off_structure"]),
            "mean_truncation": mean4(tr["truncation"]),
            "mean_repair_count": mean(tr["repair_count"]),
            "mean_guard_trigger": mean(tr["guard_trigger"]),
            "mean_trim_count": mean(tr["trim_count"]),
            "mean_hard_trim": mean(tr["hard_trim"]),
            "mean_repetition_warning": mean(tr["rep_warn"]),
            "mean_bracket_warning": mean(tr["bracket_warn"]),
            "mean_import_warning": mean(tr["import_warn"]),
            "mean_high_risk": mean(tr["high_risk"]),
        }
    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {OUT_JSON}")

    # MD
    with open(OUT_MD, "w") as f:
        f.write("# Guard-v1.5 Calibration Experiment (24 Perf Hard Cases)\n\n")
        f.write("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n\n")
        f.write("### Calibration Rules\n\n")
        f.write("1. **repetition** → warning only (no trim)\n")
        f.write("2. **unbalanced_brackets** → delayed trim (depth>3 & 2+ consecutive rounds); otherwise warning\n")
        f.write("3. **off_structure:import** → warning on DictConfig; hard trim elsewhere\n")
        f.write("4. **duplicate_option** → hard trim for Argparse only\n\n")

        f.write("## Aggregate Results\n\n")
        headers = "Method | Mean TPS | Mean Sp | Below 1.0x | Mean SQ | Mean OffStr | Mean Repair | Guard Trig | Trim Count | Hard Trim | Rep Warn | Bracket Warn | Import Warn"
        sep = "-------|----------|--------|-----------|---------|-------------|------------|-----------|----------|----------|---------|-------------|-----------"
        f.write(f"| {headers} |\n| {sep} |\n")
        for vi, vk in enumerate(var_keys):
            m = json_out["summary"].get(variants[vi])
            if not m: continue
            f.write(f"| {variants[vi]} | {m['mean_tps']:.1f} | {m['mean_speedup']:.2f}x | {m['below_1x_count']} | "
                    f"{m['mean_sq']:.4f} | {m['mean_off_structure']:.4f} | {m['mean_repair_count']:.1f} | "
                    f"{m['mean_guard_trigger']:.1f} | {m['mean_trim_count']:.1f} | {m['mean_hard_trim']:.1f} | "
                    f"{m['mean_repetition_warning']:.1f} | {m['mean_bracket_warning']:.1f} | {m['mean_import_warning']:.1f} |\n")

        m_tfg = json_out["summary"].get("TASD-F-G-Sel", {})
        m_v15 = json_out["summary"].get("+Guard-v1.5", {})

        f.write("\n## Criteria Check\n\n")
        if not m_v15:
            f.write("Guard-v1.5 data unavailable.\n")
        else:
            sp_ok = m_v15.get("mean_speedup", 0) > m_tfg.get("mean_speedup", 0)
            below_ok = m_v15.get("below_1x_count", 99) <= m_tfg.get("below_1x_count", 99)
            off_ok = m_v15.get("mean_off_structure", 0) <= 0.05
            sq_ok = m_v15.get("mean_sq", 0) >= m_tfg.get("mean_sq", 0) - 0.02
            hard_trim_ok = m_v15.get("mean_hard_trim", 0) < m_tfg.get("mean_hard_trim", 0)
            all_ok = sp_ok and below_ok and off_ok and sq_ok

            f.write(f"| Criterion | TASD-F-G-Sel | +Guard-v1.5 | Pass |\n")
            f.write(f"|-----------|-------------|-------------|------|\n")
            f.write(f"| Speedup > TASD-F-G-Sel | {m_tfg.get('mean_speedup',0):.2f}x | {m_v15.get('mean_speedup',0):.2f}x | {'OK' if sp_ok else 'FAIL'} |\n")
            f.write(f"| Below 1.0x <= TASD-F-G-Sel | {m_tfg.get('below_1x_count',0)} | {m_v15.get('below_1x_count',0)} | {'OK' if below_ok else 'FAIL'} |\n")
            f.write(f"| Off-structure <= 0.05 | {m_tfg.get('mean_off_structure',0):.4f} | {m_v15.get('mean_off_structure',0):.4f} | {'OK' if off_ok else 'FAIL'} |\n")
            f.write(f"| SQ >= TASD-F-G-Sel - 0.02 | {m_tfg.get('mean_sq',0):.4f} | {m_v15.get('mean_sq',0):.4f} | {'OK' if sq_ok else 'FAIL'} |\n")
            f.write(f"| Hard trim decreased | {m_tfg.get('mean_hard_trim',0):.1f} | {m_v15.get('mean_hard_trim',0):.1f} | {'OK' if hard_trim_ok else 'FAIL'} |\n")
            f.write(f"\n**Verdict**: {'PASS — Guard-v1.5 can enter main method candidate' if all_ok else 'FAIL — Guard-v1.5 retained as exploratory result'}\n\n")

        # GV2 comparison
        m_gv2 = json_out["summary"].get("+GuardV2(ref)", {})
        if m_v15 and m_gv2:
            f.write("## Guard-v1.5 vs GuardV2 Comparison\n\n")
            f.write(f"| Metric | +Guard-v1.5 | +GuardV2(ref) |\n")
            f.write(f"|--------|-------------|---------------|\n")
            for k, label in [("mean_speedup", "Speedup"), ("mean_off_structure", "OffStr"),
                              ("mean_sq", "SQ"), ("mean_repair_count", "Repair")]:
                f.write(f"| {label} | {m_v15.get(k,0)} | {m_gv2.get(k,0)} |\n")
            f.write("\n")

        f.write("## Per-Sample Details\n\n")
        f.write("| # | Benchmark | Idx | TASD | TFG-Sel | +v1.5 | +GV2 | v1.5 Δsp | v1.5 Δoff | v1.5 HardTrim | v1.5 RepWarn | v1.5 BracketWarn | v1.5 ImportWarn |\n")
        f.write("|---|-----------|-----|------|---------|-------|------|----------|----------|-------------|------------|---------------|-------------|\n")
        for i, r in enumerate(all_results):
            bench = r["benchmark"].replace("Real-Python-", "").replace("-Config", "")[:15]
            sp_tasd = r.get("tasd", {}).get("speedup", 0)
            sp_tfg = r.get("tfg_sel", {}).get("speedup", 0)
            sp_v15 = r.get("tfg_sel_gv15", {}).get("speedup", 0)
            sp_gv2 = r.get("tfg_sel_gv2", {}).get("speedup", 0)
            off_tfg = r.get("tfg_sel", {}).get("off_structure", 0)
            off_v15 = r.get("tfg_sel_gv15", {}).get("off_structure", 0)
            d_v15 = r.get("tfg_sel_gv15", {})
            f.write(f"| {i+1} | {bench} | {r['sample_idx']} | {sp_tasd:.2f}x | {sp_tfg:.2f}x | {sp_v15:.2f}x | {sp_gv2:.2f}x | "
                    f"{sp_v15-sp_tfg:+.2f}x | {off_v15-off_tfg:+.3f} | "
                    f"{d_v15.get('hard_trim_count',0)} | {d_v15.get('repetition_warning_count',0)} | "
                    f"{d_v15.get('bracket_warning_count',0)} | {d_v15.get('import_warning_count',0)} |\n")
        f.write("\n")

        f.write("## Trim Reason Summary\n\n")
        all_reasons = {}
        for label, vk in [("TASD", "tasd"), ("TASD-F-G-Sel", "tfg_sel"),
                           ("+Guard-v1.5", "tfg_sel_gv15"), ("+GuardV2(ref)", "tfg_sel_gv2")]:
            reasons = []
            for r in all_results:
                d = r.get(vk, {})
                reasons.extend(d.get("trim_reasons", []))
            from collections import Counter
            cnt = Counter(reasons)
            all_reasons[label] = dict(cnt.most_common(10))
        f.write("| Reason | TASD | TFG-Sel | +v1.5 | +GV2 |\n")
        f.write("|--------|------|---------|-------|------|\n")
        all_keys = set()
        for v in all_reasons.values():
            all_keys.update(v.keys())
        for key in sorted(all_keys, key=lambda k: -sum(all_reasons[v].get(k, 0) for v in all_reasons)):
            f.write(f"| {key[:60]} | {all_reasons['TASD'].get(key,0)} | {all_reasons['TASD-F-G-Sel'].get(key,0)} | "
                    f"{all_reasons['+Guard-v1.5'].get(key,0)} | {all_reasons['+GuardV2(ref)'].get(key,0)} |\n")
        f.write("\n")

    print(f"Report saved to {OUT_MD}")

def main():
    print("Loading cases...")
    with open("/tmp/hard_cases_24.json") as f:
        perf_list = json.load(f)
    data_cache = {}
    for bid, path in DATA_FILES.items():
        data_cache[bid] = load_jsonl(path, 80)

    entries = []
    for h in perf_list:
        bid = BENCHMARK_TO_BID.get(h["benchmark"], h.get("bid", ""))
        if not bid or h["sample_idx"] >= len(data_cache.get(bid, [])):
            continue
        sample = data_cache[bid][h["sample_idx"]]
        entries.append({
            "benchmark": h["benchmark"], "bid": bid,
            "sample_idx": h["sample_idx"],
            "sample_name": h.get("sample_name", f"{bid}_{h['sample_idx']}"),
            "prompt": sample["prompt"],
            "reference": sample.get("reference", ""),
            "structure_type": sample.get("structure_type", bid),
        })
    print(f"Loaded {len(entries)} cases")

    # Load GV2 reference results from pilot
    with open("results/guard_v2_pilot_hardcases.json") as f:
        pilot = json.load(f)
    pilot_by_case = {}
    for ps in pilot["per_sample"]:
        key = (ps["benchmark"], ps["sample_idx"])
        pilot_by_case[key] = ps

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET_PATH, local_files_only=True,
                                                   device_map="auto", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_PATH, local_files_only=True,
                                                  device_map="auto", trust_remote_code=True).eval()
    print("Models loaded.\n")

    all_results = []
    for i, entry in enumerate(entries):
        bench = entry["benchmark"].replace("Real-Python-", "").replace("-Config", "")
        print(f"\n[{i+1}/24] {bench} idx={entry['sample_idx']}")
        prompt = entry["prompt"]
        stype = entry["structure_type"]

        result = {"benchmark": entry["benchmark"], "sample_idx": entry["sample_idx"],
                  "sample_name": entry["sample_name"], "bid": entry["bid"],
                  "reference": entry["reference"]}

        # 1. TASD (current guard)
        print("  TASD...", end=" ", flush=True)
        r = run_variant(target, draft, tokenizer, prompt, stype, TASD_BASE)
        result["tasd"] = compute_metrics(entry, "tasd", r)
        print(f"Sp={result['tasd']['speedup']:.2f}x Off={result['tasd']['off_structure']:.3f} Trim={result['tasd']['trim_count']}")

        # 2. TASD-F-G-Sel (current guard)
        print("  TASD-F-G-Sel...", end=" ", flush=True)
        r = run_variant(target, draft, tokenizer, prompt, stype, TFGS_KWARGS)
        result["tfg_sel"] = compute_metrics(entry, "tfg_sel", r)
        print(f"Sp={result['tfg_sel']['speedup']:.2f}x Off={result['tfg_sel']['off_structure']:.3f} Trim={result['tfg_sel']['trim_count']}")

        # 3. TASD-F-G-Sel + Guard-v1.5 (calibrated)
        print("  +Guard-v1.5...", end=" ", flush=True)
        r = run_variant(target, draft, tokenizer, prompt, stype, TFGS_GV15_KWARGS)
        result["tfg_sel_gv15"] = compute_metrics(entry, "tfg_sel_gv15", r)
        print(f"Sp={result['tfg_sel_gv15']['speedup']:.2f}x Off={result['tfg_sel_gv15']['off_structure']:.3f} "
              f"HardTrim={result['tfg_sel_gv15']['hard_trim_count']} "
              f"RepWarn={result['tfg_sel_gv15']['repetition_warning_count']} "
              f"BracketWarn={result['tfg_sel_gv15']['bracket_warning_count']} "
              f"ImportWarn={result['tfg_sel_gv15']['import_warning_count']}")

        # 4. GuardV2 reference (from pilot)
        case_key = (entry["benchmark"], entry["sample_idx"])
        pilot_data = pilot_by_case.get(case_key, {})
        gv2_data = pilot_data.get("tasd_f_g_sel_gv2", {})
        result["tfg_sel_gv2"] = {
            "tps": gv2_data.get("tps", 0),
            "speedup": gv2_data.get("speedup", 0),
            "sq": gv2_data.get("sq", 0),
            "off_structure": gv2_data.get("off_structure", 0),
            "truncation": gv2_data.get("truncation", 0),
            "accept_rate": gv2_data.get("accept_rate", 0),
            "repair_count": gv2_data.get("repair_count", 0),
            "guard_trigger_count": gv2_data.get("guard_trigger_count", 0),
            "trim_count": gv2_data.get("trim_count", 0),
            "hard_trim_count": 0,
            "repetition_warning_count": 0,
            "bracket_warning_count": 0,
            "import_warning_count": 0,
            "high_risk_count": gv2_data.get("high_risk_count", 0),
            "trim_reasons": [],
        }
        print(f"  +GV2(ref) Sp={result['tfg_sel_gv2']['speedup']:.2f}x Off={result['tfg_sel_gv2']['off_structure']:.3f}")

        all_results.append(result)

    generate_report(all_results)
    print("\nDone!")

if __name__ == "__main__":
    main()
