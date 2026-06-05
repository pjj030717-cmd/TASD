#!/usr/bin/env python3
"""
1.5B Draft Evaluation: 3 benchmarks x 20 samples vs 3B baseline.
d16_b2_k3 config, with acceptance distribution analysis.
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
DRAFT_3B = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"
DRAFT_1_5B = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 20

BENCHMARKS = [
    {
        "id": "openmmlab",
        "name": "OpenMMLab-Config",
        "structure_type": "openmmlab_config",
        "data_file": "data/ml_config_blocks_openmmlab_80.jsonl",
        "ar_tps": 32.91,
    },
    {
        "id": "dict_config",
        "name": "Real-Python-DictConfig",
        "structure_type": "dict_config",
        "data_file": "data/codesearchnet_dict_config_blocks_80.jsonl",
        "ar_tps": 32.67,
    },
    {
        "id": "pipeline_stage_config",
        "name": "Pipeline-Stage-Config",
        "structure_type": "pipeline_stage_config",
        "data_file": "data/pipeline_stage_config_80.jsonl",
        "ar_tps": 32.24,
    },
]

# Existing 3B d16_b2_k3 n=80 results for comparison (summary)
BASELINE_3B = {
    "openmmlab": {"tps": 51.25, "spd": 1.56, "sq": 0.8741, "off": 0.0031, "trunc": 0.1372},
    "dict_config": {"tps": 48.10, "spd": 1.47, "sq": 0.8360, "off": 0.0000, "trunc": 0.0939},
    "pipeline_stage_config": {"tps": 53.06, "spd": 1.65, "sq": 0.9121, "off": 0.0000, "trunc": 0.0989},
}


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def load_samples(path, limit):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def run_benchmark(target, draft, tokenizer, bench, draft_label):
    samples = load_samples(bench["data_file"], SAMPLE_LIMIT)
    print(f"  Loaded {len(samples)} samples")
    mem_before = get_gpu_memory()

    results = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        _ = torch.cuda.synchronize()
        t0 = time.time()

        try:
            r = tasd_decode(
                target_model=target, draft_model=draft, tokenizer=tokenizer,
                prompt=prompt, structure_type=bench["structure_type"],
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
        st = r.get("stats", {})
        acc = st.get("accept_rate", 0)
        q = evaluate_structural_quality(gen, structure_type=bench["structure_type"])

        results.append({
            "sample_idx": i,
            "tps": tps,
            "wall_time": wall,
            "accept_rate": acc,
            "repair_count": st.get("repair_count", 0),
            "guard_trigger_count": st.get("guard_trigger_count", 0),
            "total_drafted": st.get("total_drafted", 0),
            "total_accepted": st.get("total_accepted", 0),
            "tokens_generated": r.get("generated_tokens", 0),
            "structural_quality_score": q["structural_quality_score"],
            "severe_rate": q["severe_rate"],
            "off_structure_rate": q["off_structure_rate"],
            "repetition_rate": q["repetition_rate"],
            "truncation_rate": q["truncation_rate"],
            "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
        })

        print(f"    [{i+1}/{SAMPLE_LIMIT}] TPS={tps:.1f}, acc={acc:.2f}, "
              f"SQ={q['structural_quality_score']:.4f}, wall={wall:.1f}s")

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "all samples failed"}

    mem_peak = get_gpu_memory() - mem_before

    accept_rates = [r["accept_rate"] for r in valid]
    tps_list = [r["tps"] for r in valid]
    acc_sorted = sorted(accept_rates)

    def avg(k): return sum(r[k] for r in valid) / len(valid)
    def p(arr, pct):
        idx = int(len(arr) * pct / 100)
        return arr[min(idx, len(arr) - 1)]

    summary = {
        "draft": draft_label,
        "benchmark": bench["name"],
        "n": len(valid),
        "n_errors": len(results) - len(valid),
        "tps_avg": avg("tps"),
        "tps_median": statistics.median(tps_list),
        "tps_min": min(tps_list),
        "tps_max": max(tps_list),
        "speedup_vs_ar": avg("tps") / bench["ar_tps"],
        "accept_rate_mean": avg("accept_rate"),
        "accept_rate_median": statistics.median(accept_rates),
        "accept_rate_p10": p(acc_sorted, 10),
        "accept_rate_p90": p(acc_sorted, 90),
        "low_accept_count": sum(1 for a in accept_rates if a < 0.7),
        "high_accept_count": sum(1 for a in accept_rates if a >= 0.9),
        "structural_quality_score": avg("structural_quality_score"),
        "severe_rate": avg("severe_rate"),
        "off_structure_rate": avg("off_structure_rate"),
        "repetition_rate": avg("repetition_rate"),
        "truncation_rate": avg("truncation_rate"),
        "structure_not_preserved": avg("structure_not_preserved"),
        "repair_count": avg("repair_count"),
        "guard_trigger_count": avg("guard_trigger_count"),
        "total_drafted": avg("total_drafted"),
        "total_accepted": avg("total_accepted"),
        "peak_memory_mb": mem_peak,
        "per_sample": results,
    }

    return summary


def main():
    print("Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_results = []

    for draft_label, draft_path in [("3B", DRAFT_3B), ("1.5B", DRAFT_1_5B)]:
        print(f"\nLoading {draft_label} draft...")
        draft = AutoModelForCausalLM.from_pretrained(
            draft_path, device_map="auto", torch_dtype="auto",
            trust_remote_code=True, local_files_only=True,
        )

        for bench in BENCHMARKS:
            print(f"\n{'='*60}")
            print(f"  {draft_label} Draft | {bench['name']} | n={SAMPLE_LIMIT}")
            print(f"{'='*60}")

            summary = run_benchmark(target, draft, tokenizer, bench, draft_label)

            if "error" in summary:
                print(f"  FAILED: {summary['error']}")
                continue

            print(f"\n  --- {bench['name']} Summary ({draft_label}) ---")
            print(f"  TPS: {summary['tps_avg']:.2f} (med={summary['tps_median']:.1f}, "
                  f"min={summary['tps_min']:.1f}, max={summary['tps_max']:.1f})")
            print(f"  Speedup vs AR: {summary['speedup_vs_ar']:.2f}x")
            print(f"  Accept: mean={summary['accept_rate_mean']:.4f}, med={summary['accept_rate_median']:.4f}, "
                  f"p10={summary['accept_rate_p10']:.4f}, p90={summary['accept_rate_p90']:.4f}")
            print(f"  Low-accept (<0.7): {summary['low_accept_count']}/{summary['n']}, "
                  f"High-accept (>=0.9): {summary['high_accept_count']}/{summary['n']}")
            print(f"  SQ: {summary['structural_quality_score']:.4f}, OffStr: {summary['off_structure_rate']:.4f}, "
                  f"Trunc: {summary['truncation_rate']:.4f}")
            print(f"  Repair: {summary['repair_count']:.1f}, GuardTrig: {summary['guard_trigger_count']:.1f}")
            print(f"  Peak Mem: {summary['peak_memory_mb']:.0f} MB")

            all_results.append(summary)

            # Save per-benchmark results
            out_file = f"results/draft_{draft_label}_{bench['id']}_20.json"
            with open(out_file, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        del draft
        torch.cuda.empty_cache()

    # --- Cross-draft comparison ---
    print("\n" + "=" * 80)
    print("  DRAFT COMPARISON: 3B vs 1.5B")
    print("=" * 80)

    for bench in BENCHMARKS:
        b3 = [r for r in all_results if r["draft"] == "3B" and r["benchmark"] == bench["name"]]
        b15 = [r for r in all_results if r["draft"] == "1.5B" and r["benchmark"] == bench["name"]]
        if not b3 or not b15:
            continue
        r3 = b3[0]
        r15 = b15[0]

        tps_delta = (r15["tps_avg"] / r3["tps_avg"] - 1) * 100
        sq_delta = r15["structural_quality_score"] - r3["structural_quality_score"]

        print(f"\n{bench['name']}:")
        print(f"  TPS:  3B={r3['tps_avg']:.1f}  ->  1.5B={r15['tps_avg']:.1f}  ({tps_delta:+.1f}%)")
        print(f"  Acc:  3B={r3['accept_rate_mean']:.4f}  ->  1.5B={r15['accept_rate_mean']:.4f}")
        print(f"  SQ:   3B={r3['structural_quality_score']:.4f}  ->  1.5B={r15['structural_quality_score']:.4f}  ({sq_delta:+.4f})")
        print(f"  Low-acc: 3B={r3['low_accept_count']}/{r3['n']}  ->  1.5B={r15['low_accept_count']}/{r15['n']}")
        print(f"  High-acc: 3B={r3['high_accept_count']}/{r3['n']}  ->  1.5B={r15['high_accept_count']}/{r15['n']}")

    # --- Success criteria check ---
    print("\n" + "=" * 80)
    print("  SUCCESS CRITERIA")
    print("=" * 80)
    print("  1. Avg TPS >= 3B * 1.05  (5% improvement)")
    print("  2. SQ >= 3B SQ - 0.02     (no significant quality loss)")
    print("  3. Low-accept count <= 25% (no more than 5/20)")
    print("  4. OffStr not worse       (subjective)")

    all_pass = True
    for bench in BENCHMARKS:
        b3 = [r for r in all_results if r["draft"] == "3B" and r["benchmark"] == bench["name"]]
        b15 = [r for r in all_results if r["draft"] == "1.5B" and r["benchmark"] == bench["name"]]
        if not b3 or not b15:
            continue
        r3, r15 = b3[0], b15[0]

        p1 = r15["tps_avg"] >= r3["tps_avg"] * 1.05
        p2 = r15["structural_quality_score"] >= r3["structural_quality_score"] - 0.02
        p3 = r15["low_accept_count"] <= r15["n"] * 0.25
        p4 = r15["off_structure_rate"] <= r3["off_structure_rate"] + 0.01

        status = "PASS" if (p1 and p2 and p3 and p4) else "FAIL"
        all_pass = all_pass and (p1 and p2 and p3 and p4)

        print(f"\n  {bench['name']}: {status}")
        print(f"    C1 (TPS+5%): {p1}  ({r15['tps_avg']:.1f} >= {r3['tps_avg']*1.05:.1f})")
        print(f"    C2 (SQ-0.02): {p2}  ({r15['structural_quality_score']:.4f} >= {r3['structural_quality_score']-.02:.4f})")
        print(f"    C3 (Low<25%): {p3}  ({r15['low_accept_count']}/{r15['n']})")
        print(f"    C4 (OffStr):  {p4}  ({r15['off_structure_rate']:.4f} <= {r3['off_structure_rate']+.01:.4f})")

    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # Save full comparison
    out_json = "results/draft_1_5b_vs_3b_comparison.json"
    with open(out_json, "w") as f:
        json.dump({"results": all_results, "overall_pass": all_pass}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
