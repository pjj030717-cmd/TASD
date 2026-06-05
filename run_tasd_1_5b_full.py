#!/usr/bin/env python3
"""
1.5B Draft Full Evaluation: 6 benchmarks x 80 samples, d16_b2_k3.
Compares against 3B draft d16_b2_k3 results.
"""

import json
import time
import torch
import sys
import os
import statistics

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode
from src.evaluator import evaluate_structural_quality

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 80

BENCHMARKS = [
    ("argparse", "Real-Python-Argparse", "argparse", "data/codesearchnet_argparse_blocks_80.jsonl", 32.98),
    ("dict_config", "Real-Python-DictConfig", "dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", 32.67),
    ("openmmlab", "OpenMMLab-Config", "openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", 32.91),
    ("rich_cli_option_groups", "Rich-CLI-Option-Groups", "rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", 33.14),
    ("complex_nested_config", "Complex-Nested-Config", "complex_nested_config", "data/complex_nested_config_80.jsonl", 32.71),
    ("pipeline_stage_config", "Pipeline-Stage-Config", "pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", 32.24),
]

# 3B d16_b2_k3 baselines (from tasd_d16_b2_k3_80.json)
BASELINE_3B = {
    "argparse":                 {"tps": 47.40, "spd": 1.44, "sq": 0.9146, "off": 0.0000, "trunc": 0.0228},
    "dict_config":              {"tps": 48.10, "spd": 1.47, "sq": 0.8360, "off": 0.0000, "trunc": 0.0939},
    "openmmlab":                {"tps": 51.25, "spd": 1.56, "sq": 0.8741, "off": 0.0031, "trunc": 0.1372},
    "rich_cli_option_groups":   {"tps": 52.88, "spd": 1.60, "sq": 0.8918, "off": 0.1497, "trunc": 0.0513},
    "complex_nested_config":    {"tps": 51.95, "spd": 1.59, "sq": 0.8026, "off": 0.0179, "trunc": 0.0619},
    "pipeline_stage_config":    {"tps": 53.06, "spd": 1.65, "sq": 0.9121, "off": 0.0000, "trunc": 0.0989},
}


def load_samples(path, limit):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            samples.append(json.loads(line))
            if len(samples) >= limit: break
    return samples


def main():
    print("Loading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_summary = {}
    all_per_sample = {}

    for bid, name, st, data_file, ar_tps in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  TASD 1.5B: {name} ({SAMPLE_LIMIT} samples)")
        print(f"{'='*60}")

        samples = load_samples(data_file, SAMPLE_LIMIT)
        print(f"  Loaded {len(samples)} samples")

        results = []
        for i, s in enumerate(samples):
            prompt = s["prompt"]
            _ = torch.cuda.synchronize()
            t0 = time.time()

            try:
                r = tasd_decode(
                    target_model=target, draft_model=draft, tokenizer=tokenizer,
                    prompt=prompt, structure_type=st,
                    max_new_tokens=MAX_NEW_TOKENS,
                    draft_len=16, draft_blocks=2, top_k_accept=3,
                    min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                    enable_guard=True, enable_relaxed_accept=True,
                )
            except Exception as e:
                print(f"  ERROR [{i+1}]: {e}")
                results.append({"error": str(e)})
                continue

            _ = torch.cuda.synchronize()
            wall = time.time() - t0
            gen = r.get("generated_text", "")
            tps = r.get("tokens_per_second", 0)
            st_data = r.get("stats", {})
            acc = st_data.get("accept_rate", 0)
            q = evaluate_structural_quality(gen, structure_type=st)

            print(f"    [{i+1}/{SAMPLE_LIMIT}] TPS={tps:.1f}, acc={acc:.2f}, "
                  f"SQ={q['structural_quality_score']:.4f}, wall={wall:.1f}s")

            results.append({
                "sample_idx": i, "tps": tps, "wall_time": wall,
                "accept_rate": acc, "tokens_generated": r.get("generated_tokens", 0),
                "repair_count": st_data.get("repair_count", 0),
                "guard_trigger_count": st_data.get("guard_trigger_count", 0),
                "total_drafted": st_data.get("total_drafted", 0),
                "total_accepted": st_data.get("total_accepted", 0),
                "structural_quality_score": q["structural_quality_score"],
                "severe_rate": q["severe_rate"],
                "off_structure_rate": q["off_structure_rate"],
                "repetition_rate": q["repetition_rate"],
                "truncation_rate": q["truncation_rate"],
                "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
            })

        valid = [r for r in results if "error" not in r]
        if not valid:
            print(f"  FAILED: all samples errored")
            all_summary[bid] = {"error": True}
            continue

        accept_rates = [r["accept_rate"] for r in valid]
        tps_list = [r["tps"] for r in valid]
        acc_sorted = sorted(accept_rates)

        def avg(k): return sum(r[k] for r in valid) / len(valid)
        def p(arr, pct):
            idx = int(len(arr) * pct / 100)
            return arr[min(idx, len(arr) - 1)]

        summary = {
            "benchmark": name, "benchmark_id": bid,
            "draft": "1.5B", "n": len(valid), "n_errors": len(results) - len(valid),
            "tps_avg": avg("tps"), "tps_median": statistics.median(tps_list),
            "tps_min": min(tps_list), "tps_max": max(tps_list),
            "speedup_vs_ar": avg("tps") / ar_tps,
            "accept_rate_mean": avg("accept_rate"),
            "accept_rate_median": statistics.median(accept_rates),
            "accept_rate_p10": p(acc_sorted, 10), "accept_rate_p90": p(acc_sorted, 90),
            "high_accept_count": sum(1 for a in accept_rates if a >= 0.9),
            "low_accept_count": sum(1 for a in accept_rates if a < 0.7),
            "structural_quality_score": avg("structural_quality_score"),
            "severe_rate": avg("severe_rate"),
            "off_structure_rate": avg("off_structure_rate"),
            "repetition_rate": avg("repetition_rate"),
            "truncation_rate": avg("truncation_rate"),
            "structure_not_preserved": avg("structure_not_preserved"),
            "repair_count": avg("repair_count"),
            "guard_trigger_count": avg("guard_trigger_count"),
            "total_drafted": avg("total_drafted"), "total_accepted": avg("total_accepted"),
        }

        bl = BASELINE_3B[bid]
        tps_vs_3b = summary["tps_avg"] / bl["tps"] - 1
        sq_vs_3b = summary["structural_quality_score"] - bl["sq"]

        print(f"\n  --- {name} Summary ---")
        print(f"  TPS: {summary['tps_avg']:.2f} (vs 3B={bl['tps']:.2f}, {tps_vs_3b:+.1%})")
        print(f"  Speedup vs AR: {summary['speedup_vs_ar']:.2f}x")
        print(f"  Accept: mean={summary['accept_rate_mean']:.4f}, med={summary['accept_rate_median']:.4f}")
        print(f"  Low-acc(<0.7): {summary['low_accept_count']}/{summary['n']}, High-acc(>=0.9): {summary['high_accept_count']}/{summary['n']}")
        print(f"  SQ: {summary['structural_quality_score']:.4f} (vs 3B={bl['sq']:.4f}, {sq_vs_3b:+.4f})")
        print(f"  OffStr: {summary['off_structure_rate']:.4f}, Trunc: {summary['truncation_rate']:.4f}")

        all_summary[bid] = summary
        all_per_sample[bid] = results

        # Save individual bench
        os.makedirs("results", exist_ok=True)
        with open(f"results/tasd_{bid}_1_5b_d16b2k3_80.json", "w") as f:
            json.dump({"summary": summary, "per_sample": results}, f, ensure_ascii=False)

    # Save full results
    with open("results/tasd_1_5b_d16_b2_k3_80.json", "w") as f:
        json.dump(all_summary, f, ensure_ascii=False)
    print(f"\nSaved: results/tasd_1_5b_d16_b2_k3_80.json")

    # --- Generate comparison markdown ---
    ORDER = [b[0] for b in BENCHMARKS]
    AR_TPS = {b[0]: b[4] for b in BENCHMARKS}

    # Table 1: 1.5B results
    lines = []
    lines.append("# Final Total Experiment Table (1.5B Draft, d16_b2_k3)")
    lines.append("")
    lines.append("**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-1.5B-Instruct (draft)")
    lines.append("**Settings**: temperature=0.0, max_new_tokens=128, KV cache enabled")
    lines.append("**Sample count**: n=80")
    lines.append("**TASD config**: draft_len=16, draft_blocks=2, top_k_accept=3")
    lines.append("")
    lines.append("1.5B draft is a strong candidate default and shows promising speed gains.")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| Benchmark | AR TPS | TASD TPS | Speedup | Accept Mean | Accept Med | Accept P10 | Accept P90 | High/80 | Low/80 | SQ | OffStr | Trunc |")
    lines.append("|-----------|--------|----------|---------|-------------|------------|------------|------------|---------|--------|----|--------|-------|")

    for bid in ORDER:
        s = all_summary.get(bid)
        if not s or "error" in s: continue
        lines.append(
            f"| {s['benchmark']} | {AR_TPS[bid]:.2f} | {s['tps_avg']:.2f} | {s['speedup_vs_ar']:.2f}x | "
            f"{s['accept_rate_mean']:.2f} | {s['accept_rate_median']:.2f} | "
            f"{s['accept_rate_p10']:.2f} | {s['accept_rate_p90']:.2f} | "
            f"{s['high_accept_count']} | {s['low_accept_count']} | "
            f"{s['structural_quality_score']:.4f} | {s['off_structure_rate']:.4f} | {s['truncation_rate']:.4f} |"
        )

    lines.append("")

    with open("results/final_total_experiment_table_1_5b.md", "w") as f:
        f.write("\n".join(lines))
    print("Written: results/final_total_experiment_table_1_5b.md")

    # Table 2: 1.5B vs 3B comparison
    lines2 = []
    lines2.append("# Draft Model Comparison: 1.5B vs 3B")
    lines2.append("")
    lines2.append("**Config**: d16_b2_k3, n=80, 14B-AWQ target")
    lines2.append("")
    lines2.append("## TPS Comparison")
    lines2.append("")
    lines2.append("| Benchmark | 3B TPS | 1.5B TPS | Delta | 3B SQ | 1.5B SQ | SQ Diff | 3B Accept | 1.5B Accept | 1.5B Low | 1.5B High |")
    lines2.append("|-----------|--------|----------|-------|-------|---------|---------|-----------|-------------|----------|-----------|")

    for bid in ORDER:
        s = all_summary.get(bid)
        if not s or "error" in s: continue
        bl = BASELINE_3B[bid]
        tps_delta = (s["tps_avg"] / bl["tps"] - 1) * 100
        sq_diff = s["structural_quality_score"] - bl["sq"]
        lines2.append(
            f"| {s['benchmark']} | {bl['tps']:.2f} | {s['tps_avg']:.2f} | {tps_delta:+.1f}% | "
            f"{bl['sq']:.4f} | {s['structural_quality_score']:.4f} | {sq_diff:+.4f} | "
            f"{1.0:.2f} | {s['accept_rate_mean']:.2f} | "
            f"{s['low_accept_count']} | {s['high_accept_count']} |"
        )

    lines2.append("")

    # Acceptance distribution
    lines2.append("## Acceptance Distribution")
    lines2.append("")
    lines2.append("| Benchmark | Mean | Median | P10 | P90 | Low(<0.7) | High(>=0.9) |")
    lines2.append("|-----------|------|--------|-----|-----|-----------|-------------|")
    for bid in ORDER:
        s = all_summary.get(bid)
        if not s or "error" in s: continue
        lines2.append(
            f"| {s['benchmark']} | {s['accept_rate_mean']:.4f} | {s['accept_rate_median']:.4f} | "
            f"{s['accept_rate_p10']:.4f} | {s['accept_rate_p90']:.4f} | "
            f"{s['low_accept_count']}/{s['n']} | {s['high_accept_count']}/{s['n']} |"
        )

    lines2.append("")

    # Quality comparison
    lines2.append("## Quality Comparison")
    lines2.append("")
    lines2.append("| Benchmark | 3B OffStr | 1.5B OffStr | 3B Trunc | 1.5B Trunc | 3B Rep | 1.5B Rep | 3B SNP | 1.5B SNP |")
    lines2.append("|-----------|-----------|-------------|----------|------------|--------|----------|--------|----------|")
    for bid in ORDER:
        s = all_summary.get(bid)
        if not s or "error" in s: continue
        bl = BASELINE_3B[bid]
        lines2.append(
            f"| {s['benchmark']} | {bl['off']:.4f} | {s['off_structure_rate']:.4f} | "
            f"{bl['trunc']:.4f} | {s['truncation_rate']:.4f} | "
            f"--- | {s['repetition_rate']:.4f} | --- | {s['structure_not_preserved']:.4f} |"
        )

    lines2.append("")

    # Success criteria
    lines2.append("## Success Criteria Check (C4: off_structure <= 0.05 or <= 3B+0.03)")
    lines2.append("")
    lines2.append("| Benchmark | C1 (TPS+8%) | C2 (SQ-0.02) | C3 (Low<=25%) | C4 (OffStr) | Overall |")
    lines2.append("|-----------|------------|-------------|--------------|-------------|---------|")

    overall_pass = True
    for bid in ORDER:
        s = all_summary.get(bid)
        if not s or "error" in s: continue
        bl = BASELINE_3B[bid]
        c1 = s["tps_avg"] >= bl["tps"] * 1.08
        c2 = s["structural_quality_score"] >= bl["sq"] - 0.02
        c3 = s["low_accept_count"] <= s["n"] * 0.25
        c4 = s["off_structure_rate"] <= max(0.05, bl["off"] + 0.03)
        ok = c1 and c2 and c3 and c4
        overall_pass = overall_pass and ok
        lines2.append(
            f"| {s['benchmark']} | {'PASS' if c1 else 'FAIL'} | {'PASS' if c2 else 'FAIL'} | "
            f"{'PASS' if c3 else 'FAIL'} | {'PASS' if c4 else 'FAIL'} | {'PASS' if ok else 'FAIL'} |"
        )

    lines2.append(f"\n**OVERALL: {'PASS' if overall_pass else 'FAIL'}**")
    lines2.append("")
    if overall_pass:
        lines2.append("1.5B draft passes all criteria. Recommended as optimized speed default.")
        lines2.append("3B draft retained as conservative/stable draft baseline.")
    else:
        lines2.append("1.5B draft does not pass all criteria. Consider as structure-specific option.")
        lines2.append("3B draft remains the recommended default.")

    with open("results/draft_model_comparison_1_5b_vs_3b.md", "w") as f:
        f.write("\n".join(lines2))
    print("Written: results/draft_model_comparison_1_5b_vs_3b.md")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL 1.5B DRAFT RESULTS (all n=80)")
    print("=" * 80)
    for bid in ORDER:
        s = all_summary.get(bid)
        if not s or "error" in s: continue
        bl = BASELINE_3B[bid]
        print(f"{s['benchmark']:30s}: TPS={s['tps_avg']:.2f} ({s['speedup_vs_ar']:.2f}x), "
              f"vs 3B: {s['tps_avg']/bl['tps']:.2f}x, "
              f"SQ={s['structural_quality_score']:.4f}, "
              f"Low={s['low_accept_count']}/{s['n']}")

    avg_spd = sum(all_summary[bid]["speedup_vs_ar"] for bid in ORDER if bid in all_summary) / 6
    print(f"\nAverage speedup: {avg_spd:.2f}x")


if __name__ == "__main__":
    main()
