#!/usr/bin/env python3
"""
TASD-F Hardcase Pilot: Test failure-aware fallback on below-1.0x TASD samples.

Selected 9 below-1.0x samples + 2-3 normal samples per benchmark = 27 samples total.

Methods: AR, TASD calibrated, TASD-F calibrated
TASD-F params: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=False
"""
import json, os, sys, time, logging
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq, structural_char_f1, bracket_balance_score

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

# Sample selection (name -> benchmark)
SAMPLE_SELECTION = {
    # Below-1.0x hardcases (9)
    "argparse": ["argparse_real_023", "argparse_real_030", "argparse_real_031",
                  "argparse_real_034", "argparse_real_039", "argparse_real_062", "argparse_real_070"],
    "dict_config": ["dict_config_real_019", "dict_config_real_057"],
    # Normal samples (2-3 per benchmark = 18)
}
NORMAL_SELECTION = {
    "argparse": ["argparse_real_015", "argparse_real_004", "argparse_real_041"],
    "dict_config": ["dict_config_real_035", "dict_config_real_032", "dict_config_real_021"],
    "openmmlab_config": ["openmmlab_config_real_014", "openmmlab_config_real_071", "openmmlab_config_real_012"],
    "pipeline_stage_config": ["pipeline_stage_config_076", "pipeline_stage_config_055", "pipeline_stage_config_005"],
    "complex_nested_config": ["complex_nested_config_004", "complex_nested_config_012", "complex_nested_config_028"],
    "rich_cli_option_groups": ["rich_cli_option_groups_030", "rich_cli_option_groups_065", "rich_cli_option_groups_078"],
}

BENCHMARK_MAP = {
    "argparse": ("data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    "dict_config": ("data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    "openmmlab_config": ("data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    "pipeline_stage_config": ("data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
    "complex_nested_config": ("data/complex_nested_config_80.jsonl", "complex_nested_config"),
    "rich_cli_option_groups": ("data/rich_cli_option_groups_80.jsonl", "rich_cli_option_groups"),
}

OUT_JSON = "results/qwen_tasd_f_hardcase_pilot.json"
OUT_MD = "results/qwen_tasd_f_hardcase_pilot.md"


def load_sample(bname, sample_name):
    data_file, _ = BENCHMARK_MAP[bname]
    with open(data_file) as f:
        for line in f:
            s = json.loads(line.strip())
            if s["name"] == sample_name:
                return s
    return None


def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    prompt_len = inp.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    wall = time.time() - t0
    gen_ids = out[0][inp.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    tps = gen_len / wall if wall > 0 else 0.0
    return {"wall": wall, "prompt_len": prompt_len, "gen_len": gen_len,
            "tps": tps, "text": text}


def run_tasd(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    **TASD_COMMON)
    stats = r["stats"]
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "text": r["generated_text"],
            "accept": stats["accept_rate"], "repair": stats.get("repair_count", 0),
            "guard_trig": stats.get("guard_trigger_count", 0),
            "trim": stats.get("trim_count", 0),
            "hard_trim": stats.get("hard_trim_count", 0),
            "rep_warn": stats.get("repetition_warning_count", 0),
            "brack_warn": stats.get("bracket_warning_count", 0)}


def run_tasd_f(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    enable_failure_aware_fallback=True,
                    fallback_guarded=False,
                    fallback_accept_threshold=0.5,
                    fallback_repair_threshold=2,
                    **TASD_COMMON)
    stats = r["stats"]
    fb_summary = stats.get("failure_aware_fallback", {}) or {}
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "text": r["generated_text"],
            "accept": stats["accept_rate"], "repair": stats.get("repair_count", 0),
            "guard_trig": stats.get("guard_trigger_count", 0),
            "trim": stats.get("trim_count", 0),
            "hard_trim": stats.get("hard_trim_count", 0),
            "rep_warn": stats.get("repetition_warning_count", 0),
            "brack_warn": stats.get("bracket_warning_count", 0),
            "fb_count": fb_summary.get("fallback_count", 0),
            "fb_tokens": fb_summary.get("total_fallback_tokens", 0),
            "fb_trigger_count": fb_summary.get("trigger_count", 0)}


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

    all_results = []  # list of per-sample dicts with all methods

    total = sum(len(v) for v in SAMPLE_SELECTION.values()) + sum(len(v) for v in NORMAL_SELECTION.values())
    idx = 0

    for bname in BENCHMARK_MAP:
        samples_to_run = SAMPLE_SELECTION.get(bname, []) + NORMAL_SELECTION.get(bname, [])
        if not samples_to_run:
            continue
        stype = BENCHMARK_MAP[bname][1]
        for name in samples_to_run:
            idx += 1
            sample = load_sample(bname, name)
            if sample is None:
                print(f"  [{idx}/{total}] {bname}/{name}: sample not found, skip")
                continue
            prompt = sample["prompt"]
            ref = sample.get("reference", "")
            print(f"  [{idx}/{total}] {bname}/{name} ...", end=" ", flush=True)

            # AR
            ar = run_ar(target, tokenizer, prompt)
            ar_tps = ar["tps"]

            # TASD
            ts = run_tasd(target, draft, tokenizer, prompt, stype)
            ts_sp = ts["tps"] / ar_tps if ar_tps > 0 else 0.0

            # TASD-F
            tf = run_tasd_f(target, draft, tokenizer, prompt, stype)
            tf_sp = tf["tps"] / ar_tps if ar_tps > 0 else 0.0

            # Quality
            ar_q = compute_composite_sq(ar["text"], ref, stype)
            ts_q = compute_composite_sq(ts["text"], ref, stype)
            tf_q = compute_composite_sq(tf["text"], ref, stype)

            entry = {
                "benchmark": bname, "name": name,
                "AR": {"tps": round(ar_tps, 2), "wall": round(ar["wall"], 3),
                       **ar_q, "text": ar["text"]},
                "TASD": {"tps": round(ts["tps"], 2), "sp": round(ts_sp, 3),
                         "wall": round(ts["wall"], 3), "accept": round(ts["accept"], 4),
                         "repair": ts["repair"], "guard_trig": ts["guard_trig"],
                         "trim": ts["trim"], "hard_trim": ts["hard_trim"],
                         **ts_q, "text": ts["text"]},
                "TASD-F": {"tps": round(tf["tps"], 2), "sp": round(tf_sp, 3),
                           "wall": round(tf["wall"], 3), "accept": round(tf["accept"], 4),
                           "repair": tf["repair"], "guard_trig": tf["guard_trig"],
                           "trim": tf["trim"], "hard_trim": tf["hard_trim"],
                           "fb_count": tf["fb_count"], "fb_tokens": tf["fb_tokens"],
                           **tf_q, "text": tf["text"]},
            }
            all_results.append(entry)

            tag = "BELOW" if ts_sp < 1.0 else "ok"
            print(f"AR={ar_tps:.1f} TASD={ts_sp:.3f}x TASDF={tf_sp:.3f}x [{tag}]")

    # ── Aggregate ──
    below_entries = [e for e in all_results if e["TASD"]["sp"] < 1.0]
    normal_entries = [e for e in all_results if e["TASD"]["sp"] >= 1.0]
    all_entries = all_results

    def compute_stats(entries, method_key):
        sps = sorted([e[method_key]["sp"] for e in entries])
        n = len(sps)
        if n == 0:
            return {}
        mean_sp = sum(sps) / n
        median_sp = sps[n // 2]
        min_sp = sps[0]
        worst10 = sps[:max(1, n // 10)] if n >= 10 else sps[:1]
        worst10_sp = sum(worst10) / len(worst10)
        below_count = sum(1 for s in sps if s < 1.0)
        accepts = [e[method_key].get("accept", 0) for e in entries]
        repairs = [e[method_key].get("repair", 0) for e in entries]
        guards = [e[method_key].get("guard_trig", 0) for e in entries]
        trims = [e[method_key].get("trim", 0) for e in entries]
        sQs = [e[method_key].get("composite_sq", 0) for e in entries]
        off_str = [e[method_key].get("off_structure_rate", 0) for e in entries]
        rep_rate = [e[method_key].get("repetition_rate", 0) for e in entries]
        trunc = [e[method_key].get("is_truncated", 0) for e in entries]
        fb_count = sum(e[method_key].get("fb_count", 0) for e in entries)
        fb_tokens = sum(e[method_key].get("fb_tokens", 0) for e in entries)
        return {
            "n": n,
            "mean_sp": round(mean_sp, 3),
            "median_sp": round(median_sp, 3),
            "min_sp": round(min_sp, 3),
            "worst10_sp": round(worst10_sp, 3),
            "below_count": below_count,
            "accept_avg": round(sum(accepts) / n, 4) if accepts else 0,
            "repair_avg": round(sum(repairs) / n, 1) if repairs else 0,
            "guard_avg": round(sum(guards) / n, 1) if guards else 0,
            "trim_avg": round(sum(trims) / n, 1) if trims else 0,
            "sq_avg": round(sum(sQs) / n, 4) if sQs else 0,
            "off_structure_avg": round(sum(off_str) / n, 4) if off_str else 0,
            "rep_rate_avg": round(sum(rep_rate) / n, 4) if rep_rate else 0,
            "trunc_avg": round(sum(trunc) / n, 4) if trunc else 0,
            "fallback_count": fb_count,
            "fallback_tokens": fb_tokens,
        }

    summary = {
        "config": {
            "methods": "AR / TASD(calibrated) / TASD-F(calibrated)",
            "TASD-F_params": "enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=False",
            "max_new_tokens": MAX_NEW_TOKENS,
            "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
        },
        "below_samples": compute_stats(below_entries, "TASD"),
        "below_tasdf": compute_stats(below_entries, "TASD-F"),
        "normal_samples": compute_stats(normal_entries, "TASD"),
        "normal_tasdf": compute_stats(normal_entries, "TASD-F"),
        "all_samples": compute_stats(all_entries, "TASD"),
        "all_tasdf": compute_stats(all_entries, "TASD-F"),
        "per_sample": all_results,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    # ── MD Report ──
    write_md_report(summary)
    print(f"Saved {OUT_MD}")
    print("Done!")


def write_md_report(s):
    with open(OUT_MD, "w") as f:
        f.write("# TASD-F Hardcase Pilot Report\n\n")
        f.write(f"**Target**: Qwen2.5-14B-Instruct-AWQ  |  **Draft**: Qwen2.5-1.5B-Instruct\n")
        f.write(f"**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3\n\n")
        f.write("**TASD-F params**: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=False\n\n")

        f.write(f"## Sample Composition\n\n")
        f.write(f"- Below-1.0x hardcases: {s['below_samples'].get('n', 0)}\n")
        f.write(f"- Normal samples: {s['normal_samples'].get('n', 0)}\n")
        f.write(f"- Total: {s['all_samples'].get('n', 0)}\n\n")

        f.write("## Below-1.0x Samples (Hardcase)\n\n")
        f.write("| Metric | TASD | TASD-F | Delta |\n")
        f.write("|--------|:----:|:------:|:-----:|\n")
        for metric in ["mean_sp", "median_sp", "min_sp", "worst10_sp", "below_count", "accept_avg",
                       "repair_avg", "guard_avg", "trim_avg", "sq_avg", "off_structure_avg", "rep_rate_avg", "trunc_avg"]:
            ts_v = s["below_samples"].get(metric, "-")
            tf_v = s["below_tasdf"].get(metric, "-")
            if isinstance(ts_v, float) and isinstance(tf_v, float):
                delta = tf_v - ts_v
                delta_str = f"{delta:+.3f}"
            else:
                delta_str = "-"
            f.write(f"| {metric} | {ts_v} | {tf_v} | {delta_str} |\n")
        f.write(f"| fallback_count | - | {s['below_tasdf'].get('fallback_count', 0)} | - |\n")
        f.write(f"| fallback_tokens | - | {s['below_tasdf'].get('fallback_tokens', 0)} | - |\n")

        f.write("\n## Normal Samples (Sanity)\n\n")
        f.write("| Metric | TASD | TASD-F | Delta |\n")
        f.write("|--------|:----:|:------:|:-----:|\n")
        for metric in ["mean_sp", "median_sp", "min_sp", "below_count", "accept_avg",
                       "repair_avg", "guard_avg", "trim_avg", "sq_avg", "off_structure_avg", "rep_rate_avg", "trunc_avg"]:
            ts_v = s["normal_samples"].get(metric, "-")
            tf_v = s["normal_tasdf"].get(metric, "-")
            if isinstance(ts_v, float) and isinstance(tf_v, float):
                delta = tf_v - ts_v
                delta_str = f"{delta:+.3f}"
            else:
                delta_str = "-"
            f.write(f"| {metric} | {ts_v} | {tf_v} | {delta_str} |\n")
        f.write(f"| fallback_count | - | {s['normal_tasdf'].get('fallback_count', 0)} | - |\n")
        f.write(f"| fallback_tokens | - | {s['normal_tasdf'].get('fallback_tokens', 0)} | - |\n")

        f.write("\n## All Samples\n\n")
        f.write("| Metric | TASD | TASD-F | Delta |\n")
        f.write("|--------|:----:|:------:|:-----:|\n")
        for metric in ["mean_sp", "median_sp", "min_sp", "worst10_sp", "below_count", "accept_avg",
                       "repair_avg", "guard_avg", "trim_avg", "sq_avg", "off_structure_avg", "rep_rate_avg", "trunc_avg"]:
            ts_v = s["all_samples"].get(metric, "-")
            tf_v = s["all_tasdf"].get(metric, "-")
            if isinstance(ts_v, float) and isinstance(tf_v, float):
                delta = tf_v - ts_v
                delta_str = f"{delta:+.3f}"
            else:
                delta_str = "-"
            f.write(f"| {metric} | {ts_v} | {tf_v} | {delta_str} |\n")
        f.write(f"| fallback_count | - | {s['all_tasdf'].get('fallback_count', 0)} | - |\n")
        f.write(f"| fallback_tokens | - | {s['all_tasdf'].get('fallback_tokens', 0)} | - |\n")

        f.write("\n## Per-Sample Details\n\n")
        f.write("| Sample | AR TPS | TASD sp | TASD-F sp | TASD acc | TASD-F acc | TASD-F fb | TASD SQ | TASD-F SQ |\n")
        f.write("|--------|:------:|:-------:|:---------:|:--------:|:----------:|:---------:|:-------:|:---------:|\n")
        for e in s["per_sample"]:
            name = e["name"][:25]
            ar_tps = e["AR"]["tps"]
            ts_sp = e["TASD"]["sp"]
            tf_sp = e["TASD-F"]["sp"]
            ts_acc = e["TASD"].get("accept", 0)
            tf_acc = e["TASD-F"].get("accept", 0)
            tf_fb = e["TASD-F"].get("fb_count", 0)
            ts_sq = e["TASD"].get("composite_sq", 0)
            tf_sq = e["TASD-F"].get("composite_sq", 0)
            tag = " **BELOW**" if ts_sp < 1.0 else ""
            f.write(f"| {name}{tag} | {ar_tps:.1f} | {ts_sp:.3f}x | {tf_sp:.3f}x | {ts_acc:.4f} | {tf_acc:.4f} | {tf_fb} | {ts_sq:.4f} | {tf_sq:.4f} |\n")

        f.write("\n## Pass/Fail Criteria\n\n")
        below_ts = s["below_samples"].get("below_count", 0)
        below_tf = s["below_tasdf"].get("below_count", 0)
        ts_mean = s["all_samples"].get("mean_sp", 0)
        tf_mean = s["all_tasdf"].get("mean_sp", 0)
        ts_sq = s["all_samples"].get("sq_avg", 0)
        tf_sq = s["all_tasdf"].get("sq_avg", 0)

        checks = []
        checks.append(("below-1.0x reduced", below_tf < below_ts, f"{below_ts} -> {below_tf}"))
        checks.append(("mean sp not degraded >5%", tf_mean >= ts_mean * 0.95, f"{ts_mean:.3f} -> {tf_mean:.3f}"))
        checks.append(("SQ not degraded >0.02", tf_sq >= ts_sq - 0.02, f"{ts_sq:.4f} -> {tf_sq:.4f}"))

        f.write("| Criterion | Pass | Note |\n")
        f.write("|-----------|:----:|------|\n")
        for name, passed, note in checks:
            f.write(f"| {name} | {'PASS' if passed else 'FAIL'} | {note} |\n")
        f.write("\n")

        f.write(f"## Data\n\n- `{OUT_JSON}`\n")


if __name__ == "__main__":
    main()
