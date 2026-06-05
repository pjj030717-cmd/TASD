#!/usr/bin/env python3
"""
Adaptive TASD v2: refined rules on 3 benchmarks x 20 samples (n=20 total per config).
Compares fixed d16_b2_k3 vs AdaptivePolicyV2.

Outputs:
- results/adaptive_v2_detailed.json
- results/adaptive_v2_summary.md
"""
import json, time, torch, sys, os, statistics

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode
from src.adaptive_policy import AdaptivePolicyV2
from src.evaluator import evaluate_structural_quality

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH  = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW     = 128
N_SAMPLES   = 20

BENCHMARKS = [
    ("dict_config", "Real-Python-DictConfig", "dict_config",
     "data/codesearchnet_dict_config_blocks_80.jsonl", 32.67),
    ("openmmlab", "OpenMMLab-Config", "openmmlab_config",
     "data/ml_config_blocks_openmmlab_80.jsonl", 32.91),
    ("pipeline_stage_config", "Pipeline-Stage-Config", "pipeline_stage_config",
     "data/pipeline_stage_config_80.jsonl", 32.24),
]

# Fixed 1.5B d16_b2_k3 n=20 baselines (from v1 fixed run)
FIXED_BASELINE = {
    "dict_config": {"tps": 51.4, "sq": 0.8443, "off": 0.0000, "trunc": 0.0445, "low": 5},
    "openmmlab":   {"tps": 62.8, "sq": 0.8974, "off": 0.0126, "trunc": 0.1554, "low": 2},
    "pipeline_stage_config": {"tps": 65.5, "sq": 0.9581, "off": 0.0303, "trunc": 0.1397, "low": 0},
}


def run_benchmark(target, draft, tokenizer, bid, name, st, data_file, ar_tps):
    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            samples.append(json.loads(line))
            if len(samples) >= N_SAMPLES: break

    print(f"\n{'='*60}")
    print(f"  {name} | adaptive v2 | n={N_SAMPLES}")
    print(f"{'='*60}")

    results = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        policy = AdaptivePolicyV2()

        _ = torch.cuda.synchronize()
        t0 = time.time()

        try:
            r = tasd_decode(
                target_model=target, draft_model=draft, tokenizer=tokenizer,
                prompt=prompt, structure_type=st,
                max_new_tokens=MAX_NEW,
                draft_len=16, draft_blocks=2, top_k_accept=3,
                min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                enable_guard=True, enable_relaxed_accept=True,
                adaptive_policy=policy,
            )
        except Exception as e:
            print(f"  ERROR [{i+1}]: {e}")
            results.append({"error": str(e), "sample_idx": i})
            continue

        _ = torch.cuda.synchronize()
        wall = time.time() - t0

        gen = r.get("generated_text", "")
        tps = r.get("tokens_per_second", 0)
        st_data = r.get("stats", {})
        acc = st_data.get("accept_rate", 0)
        q = evaluate_structural_quality(gen, structure_type=st)

        adp_sum = policy.get_summary()

        entry = {
            "sample_idx": i, "benchmark": name,
            "tps": tps, "wall_time": wall,
            "accept_rate": acc,
            "tokens_generated": r.get("generated_tokens", 0),
            "structural_quality_score": q["structural_quality_score"],
            "severe_rate": q["severe_rate"],
            "off_structure_rate": q["off_structure_rate"],
            "repetition_rate": q["repetition_rate"],
            "truncation_rate": q["truncation_rate"],
            "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
            "repair_count": st_data.get("repair_count", 0),
            "guard_trigger_count": st_data.get("guard_trigger_count", 0),
            "total_drafted": st_data.get("total_drafted", 0),
            "total_accepted": st_data.get("total_accepted", 0),
            "adaptive_change_count": adp_sum["adaptive_change_count"],
            "k5_round_count": adp_sum["k5_round_count"],
            "avg_draft_len": adp_sum["average_draft_len"],
            "short_draft_round_count": adp_sum["short_draft_round_count"],
            "long_draft_round_count": adp_sum["long_draft_round_count"],
        }

        results.append(entry)

        print(f"    [{i+1}/{N_SAMPLES}] TPS={tps:.1f} acc={acc:.2f} SQ={q['structural_quality_score']:.4f}"
              f" chg={adp_sum['adaptive_change_count']} k5={adp_sum['k5_round_count']}"
              f" avgDL={adp_sum['average_draft_len']:.1f} wall={wall:.1f}s")

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": True}

    accept = [r["accept_rate"] for r in valid]
    tps_list = [r["tps"] for r in valid]

    def avg(k): return sum(r[k] for r in valid) / len(valid)

    summary = {
        "benchmark": name, "bid": bid, "n": len(valid),
        "tps_avg": avg("tps"), "tps_median": statistics.median(tps_list),
        "tps_min": min(tps_list), "tps_max": max(tps_list),
        "speedup_vs_ar": avg("tps") / ar_tps,
        "accept_rate_mean": avg("accept_rate"),
        "accept_rate_median": statistics.median(accept),
        "low_accept_count": sum(1 for a in accept if a < 0.7),
        "high_accept_count": sum(1 for a in accept if a >= 0.9),
        "structural_quality_score": avg("structural_quality_score"),
        "off_structure_rate": avg("off_structure_rate"),
        "truncation_rate": avg("truncation_rate"),
        "repetition_rate": avg("repetition_rate"),
        "repair_count": avg("repair_count"),
        "guard_trigger_count": avg("guard_trigger_count"),
        "adaptive_change_count": avg("adaptive_change_count"),
        "k5_round_count": avg("k5_round_count"),
        "avg_draft_len": avg("avg_draft_len"),
        "short_draft_round_count": avg("short_draft_round_count"),
        "long_draft_round_count": avg("long_draft_round_count"),
    }

    return {"summary": summary, "per_sample": results}


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

    all_runs = []

    for bid, name, st, data_file, ar_tps in BENCHMARKS:
        out = run_benchmark(target, draft, tokenizer, bid, name, st, data_file, ar_tps)
        if out and "error" not in out:
            all_runs.append(out)

    # Save detailed JSON
    os.makedirs("results", exist_ok=True)
    with open("results/adaptive_v2_detailed.json", "w") as f:
        json.dump(all_runs, f, ensure_ascii=False)
    print("\nSaved: results/adaptive_v2_detailed.json")

    # Generate markdown
    lines = []
    lines.append("# Adaptive TASD v2 vs Fixed Baseline")
    lines.append("")
    lines.append("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct")
    lines.append("**Config**: adaptive v2 from d16_b2_k3; draft_len ⇆ {8,16,20}, top_k ⇆ {3,5}")
    lines.append("**n**: 20 per benchmark")
    lines.append("")

    lines.append("## Main Comparison (Adaptive v2)")
    lines.append("")
    lines.append("| Benchmark | TPS | Speedup | Accept | Low | SQ | OffStr | Trunc | Chg | k5Rnd | avgDL |")
    lines.append("|-----------|-----|---------|--------|-----|----|--------|-------|-----|-------|-------|")

    for run in all_runs:
        s = run["summary"]
        lines.append(
            f"| {s['benchmark']} | {s['tps_avg']:.1f} | {s['speedup_vs_ar']:.2f}x | "
            f"{s['accept_rate_mean']:.2f} | {s['low_accept_count']}/{s['n']} | "
            f"{s['structural_quality_score']:.4f} | {s['off_structure_rate']:.4f} | "
            f"{s['truncation_rate']:.4f} | {s['adaptive_change_count']:.1f} | "
            f"{s['k5_round_count']:.1f} | {s['avg_draft_len']:.1f} |"
        )

    lines.append("")
    lines.append("## Delta (Adaptive v2 - Fixed)")
    lines.append("")
    lines.append("| Benchmark | TPS Delta | SQ Delta | OffStr Delta | Low Delta | Chg Count | k5 Rounds | Avg DraftLen |")
    lines.append("|-----------|-----------|----------|-------------|-----------|-----------|-----------|-------------|")

    for run in all_runs:
        s = run["summary"]
        bl = FIXED_BASELINE[s["bid"]]
        tps_d = s["tps_avg"] - bl["tps"]
        sq_d = s["structural_quality_score"] - bl["sq"]
        off_d = s["off_structure_rate"] - bl["off"]
        low_d = s["low_accept_count"] - bl["low"]
        lines.append(
            f"| {s['benchmark']} | {tps_d:+.1f} ({tps_d/bl['tps']*100:+.1f}%) | {sq_d:+.4f} | {off_d:+.4f} | "
            f"{low_d:+d} | {s['adaptive_change_count']:.1f} | {s['k5_round_count']:.1f} | "
            f"{s['avg_draft_len']:.1f} |"
        )

    lines.append("")
    lines.append("## Success Criteria")
    lines.append("")

    all_pass = True
    for run in all_runs:
        s = run["summary"]
        bl = FIXED_BASELINE[s["bid"]]
        tps_pct = (s["tps_avg"] / bl["tps"] - 1) * 100
        sq_d = s["structural_quality_score"] - bl["sq"]
        off_d = s["off_structure_rate"] - bl["off"]

        target_tps = ">= +5%" if s["bid"] == "dict_config" else ">= -1%"
        c1_ok = (s["bid"] == "dict_config" and tps_pct >= 5) or \
                (s["bid"] != "dict_config" and tps_pct >= -1)
        c2_ok = sq_d >= -0.02
        c3_ok = off_d <= 0.01
        ok = c1_ok and c2_ok and c3_ok
        all_pass = all_pass and ok

        lines.append(f"| {s['benchmark']} | C1(TPS {target_tps}): {'PASS' if c1_ok else 'FAIL'} ({tps_pct:+.1f}%) |")
        lines.append(f"| | C2(SQ >= -0.02): {'PASS' if c2_ok else 'FAIL'} ({sq_d:+.4f}) |")
        lines.append(f"| | C3(OffStr <= +0.01): {'PASS' if c3_ok else 'FAIL'} ({off_d:+.4f}) |")

    lines.append(f"\n**OVERALL: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")

    if all_pass:
        lines.append("Adaptive TASD v2 passes. Consider expanding to 6 benchmarks x 80.")
    else:
        lines.append("Adaptive TASD v2 does not pass. Retained as future work; main method stays fixed d16_b2_k3.")

    with open("results/adaptive_v2_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("Written: results/adaptive_v2_summary.md")


if __name__ == "__main__":
    main()
