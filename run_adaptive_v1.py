#!/usr/bin/env python3
"""
Adaptive TASD v1: 3 benchmarks x 20 samples, compares fixed vs adaptive.
Uses 1.5B draft, d16_b2_k3 as baseline default.

Outputs:
- results/adaptive_v1_detailed.json  (per-sample data)
- results/adaptive_v1_summary.md     (comparison table)
"""
import json, time, torch, sys, os, statistics

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode
from src.adaptive_policy import AdaptivePolicy
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

# Fixed 1.5B d16_b2_k3 baselines (from tasd_1_5b_d16_b2_k3_80.json)
FIXED_BASELINE = {
    "dict_config": {"tps": 60.00, "sq": 0.8316, "off": 0.0000, "trunc": 0.0590,
                    "low": 10, "repair": 0.31, "guard": 1.79},
    "openmmlab":   {"tps": 63.96, "sq": 0.8876, "off": 0.0059, "trunc": 0.1134,
                    "low": 6, "repair": 0.34, "guard": 0.70},
    "pipeline_stage_config": {"tps": 66.66, "sq": 0.9405, "off": 0.0245, "trunc": 0.1489,
                              "low": 0, "repair": 0.01, "guard": 0.00},
}


def run_benchmark(target, draft, tokenizer, bid, name, st, data_file, ar_tps, adaptive):
    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            samples.append(json.loads(line))
            if len(samples) >= N_SAMPLES: break

    mode = "adaptive" if adaptive else "fixed"
    print(f"\n{'='*60}")
    print(f"  {name} | {mode} | n={N_SAMPLES}")
    print(f"{'='*60}")

    results = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        name_tag = s.get("name", f"sample_{i}")

        policy = AdaptivePolicy() if adaptive else None

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

        entry = {
            "sample_idx": i, "sample_name": name_tag,
            "mode": mode, "benchmark": name,
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
        }

        if adaptive and policy is not None:
            entry["adaptive"] = policy.get_summary()
            entry["draft_len_history"] = policy.draft_len_history
            entry["top_k_history"] = policy.top_k_history
            entry["adaptive_change_count"] = entry["adaptive"]["adaptive_change_count"]
            entry["k5_round_count"] = entry["adaptive"]["k5_round_count"]
            entry["avg_draft_len"] = entry["adaptive"]["average_draft_len"]
            entry["short_draft_round_count"] = entry["adaptive"]["short_draft_round_count"]
            entry["long_draft_round_count"] = entry["adaptive"]["long_draft_round_count"]

        results.append(entry)

        extra = ""
        if adaptive:
            extra = f" chg={entry['adaptive_change_count']} k5={entry['k5_round_count']} avgDL={entry['avg_draft_len']:.1f}"
        print(f"    [{i+1}/{N_SAMPLES}] TPS={tps:.1f} acc={acc:.2f} SQ={q['structural_quality_score']:.4f} wall={wall:.1f}s{extra}")

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": True}

    accept = [r["accept_rate"] for r in valid]
    tps_list = [r["tps"] for r in valid]

    def avg(k): return sum(r[k] for r in valid) / len(valid)

    summary = {
        "mode": mode, "benchmark": name, "bid": bid,
        "n": len(valid), "n_errors": len(results) - len(valid),
        "tps_avg": avg("tps"),
        "tps_median": statistics.median(tps_list),
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
    }

    if adaptive:
        se_entries = [r for r in valid if "adaptive" in r]
        if se_entries:
            summary["adaptive_change_count"] = statistics.mean([r["adaptive_change_count"] for r in se_entries])
            summary["k5_round_count"] = statistics.mean([r["k5_round_count"] for r in se_entries])
            summary["avg_draft_len"] = statistics.mean([r["avg_draft_len"] for r in se_entries])
            summary["short_draft_round_count"] = statistics.mean([r["short_draft_round_count"] for r in se_entries])
            summary["long_draft_round_count"] = statistics.mean([r["long_draft_round_count"] for r in se_entries])

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

    # Run fixed + adaptive for each benchmark
    for adaptive in [False, True]:
        for bid, name, st, data_file, ar_tps in BENCHMARKS:
            out = run_benchmark(target, draft, tokenizer, bid, name, st, data_file, ar_tps, adaptive)
            if out and "error" not in out:
                all_runs.append(out)

    # Save detailed JSON
    os.makedirs("results", exist_ok=True)
    with open("results/adaptive_v1_detailed.json", "w") as f:
        json.dump(all_runs, f, ensure_ascii=False)
    print("\nSaved: results/adaptive_v1_detailed.json")

    # --- Generate comparison markdown ---
    lines = []
    lines.append("# Adaptive TASD v1 vs Fixed Baseline")
    lines.append("")
    lines.append("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct")
    lines.append("**Config**: adaptive starts from d16_b2_k3, then adjusts per round")
    lines.append("**n**: 20 per benchmark")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    lines.append("| Benchmark | Mode | TPS | Speedup | Accept Mean | Low | SQ | OffStr | Trunc | Repair | Guard |")
    lines.append("|-----------|------|-----|---------|-------------|-----|----|--------|-------|--------|-------|")

    for bid, name, st, _, _ in BENCHMARKS:
        fix = [r for r in all_runs if r["summary"]["mode"] == "fixed" and r["summary"]["bid"] == bid]
        adp = [r for r in all_runs if r["summary"]["mode"] == "adaptive" and r["summary"]["bid"] == bid]
        for run in fix + adp:
            s = run["summary"]
            lines.append(
                f"| {s['benchmark']} | {s['mode']} | {s['tps_avg']:.1f} | {s['speedup_vs_ar']:.2f}x | "
                f"{s['accept_rate_mean']:.2f} | {s['low_accept_count']}/{s['n']} | "
                f"{s['structural_quality_score']:.4f} | {s['off_structure_rate']:.4f} | "
                f"{s['truncation_rate']:.4f} | {s['repair_count']:.1f} | {s['guard_trigger_count']:.1f} |"
            )

    lines.append("")
    lines.append("## Delta (Adaptive - Fixed)")
    lines.append("")
    lines.append("| Benchmark | TPS Delta | SQ Delta | OffStr Delta | Low Delta | Chg Count | k5 Rounds | Avg DraftLen | Short Rounds | Long Rounds |")
    lines.append("|-----------|-----------|----------|-------------|-----------|-----------|-----------|-------------|-------------|------------|")

    for bid, name, st, _, _ in BENCHMARKS:
        fix = [r for r in all_runs if r["summary"]["mode"] == "fixed" and r["summary"]["bid"] == bid]
        adp = [r for r in all_runs if r["summary"]["mode"] == "adaptive" and r["summary"]["bid"] == bid]
        if not fix or not adp: continue
        fs, aps = fix[0]["summary"], adp[0]["summary"]
        tps_d = aps["tps_avg"] - fs["tps_avg"]
        sq_d = aps["structural_quality_score"] - fs["structural_quality_score"]
        off_d = aps["off_structure_rate"] - fs["off_structure_rate"]
        low_d = aps["low_accept_count"] - fs["low_accept_count"]

        chg = aps.get("adaptive_change_count", 0)
        k5 = aps.get("k5_round_count", 0)
        avg_dl = aps.get("avg_draft_len", 16)
        short = aps.get("short_draft_round_count", 0)
        long = aps.get("long_draft_round_count", 0)

        lines.append(
            f"| {name} | {tps_d:+.1f} | {sq_d:+.4f} | {off_d:+.4f} | {low_d:+d} | "
            f"{chg:.1f} | {k5:.1f} | {avg_dl:.1f} | {short:.1f} | {long:.1f} |"
        )

    lines.append("")
    lines.append("## Success Criteria")
    lines.append("")
    lines.append("| Criterion | Details |")
    lines.append("|-----------|---------|")

    all_pass = True
    for bid, name, st, _, _ in BENCHMARKS:
        fix = [r for r in all_runs if r["summary"]["mode"] == "fixed" and r["summary"]["bid"] == bid]
        adp = [r for r in all_runs if r["summary"]["mode"] == "adaptive" and r["summary"]["bid"] == bid]
        if not fix or not adp: continue
        fs, aps = fix[0]["summary"], adp[0]["summary"]

        tps_d = (aps["tps_avg"] / fs["tps_avg"] - 1) * 100
        sq_d = aps["structural_quality_score"] - fs["structural_quality_score"]
        chg = aps.get("adaptive_change_count", 0)

        p1 = tps_d >= 5 or (aps["low_accept_count"] < fs["low_accept_count"])
        p2 = sq_d >= -0.02
        p3 = chg > 0

        status = "PASS" if p1 and p2 and p3 else "FAIL"
        all_pass = all_pass and p1 and p2 and p3

        lines.append(
            f"| {name} | C1(TPS+5% or Low↓): {'PASS' if p1 else 'FAIL'} ({tps_d:+.1f}%, Low {aps['low_accept_count']} vs {fs['low_accept_count']}) |"
        )
        lines.append(
            f"| | C2(SQ-0.02): {'PASS' if p2 else 'FAIL'} ({sq_d:+.4f}) |"
        )
        lines.append(
            f"| | C4(Changes>0): {'PASS' if p3 else 'FAIL'} (changes={chg:.1f}) |"
        )

    lines.append(f"\n**OVERALL: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")

    if all_pass:
        lines.append("Adaptive TASD v1 passes success criteria. Recommended to expand to 6 benchmarks x 80 samples.")
    else:
        lines.append("Adaptive TASD v1 does not pass all criteria. Retained as future work.")

    with open("results/adaptive_v1_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("Written: results/adaptive_v1_summary.md")


if __name__ == "__main__":
    main()
