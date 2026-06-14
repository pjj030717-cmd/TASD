#!/usr/bin/env python3
"""
TASD-F Qwen 6×80 Full Experiment.

Reuses AR TPS from quality checkpoint. Runs TASD-F calibrated on all 480 samples.
Saves incremental checkpoints. Generates TASD vs TASD-F comparison report.
"""
import json, os, sys, time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

TASDFG_PARAMS = dict(
    enable_failure_aware_fallback=True,
    fallback_guarded=True,
    fallback_accept_threshold=0.5,
    fallback_repair_threshold=2,
)

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested_config"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli_option_groups"),
]

CHECKPOINT_DIR = "results/qwen_6x80_checkpoints"
OUT_JSON = "results/qwen_tasd_f_6x80.json"
OUT_MD = "results/qwen_tasd_f_6x80.md"


def run_tasd_f(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    **TASDF_PARAMS, **TASD_COMMON)
    stats = r["stats"]
    fb_summary = stats.get("failure_aware_fallback", {}) or {}
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "text": r["generated_text"],
            "accept": stats["accept_rate"],
            "repair": stats.get("repair_count", 0),
            "guard_trig": stats.get("guard_trigger_count", 0),
            "trim": stats.get("trim_count", 0),
            "hard_trim": stats.get("hard_trim_count", 0),
            "rep_warn": stats.get("repetition_warning_count", 0),
            "brack_warn": stats.get("bracket_warning_count", 0),
            "fb_count": fb_summary.get("fallback_count", 0),
            "fb_tokens": fb_summary.get("total_fallback_tokens", 0),
            "fb_trigger_count": fb_summary.get("trigger_count", 0)}


def main():
    # Load AR data from quality JSON
    with open("results/qwen_5method_6x80_quality.json") as f:
        quality_data = json.load(f)
    ar_per_sample = quality_data["per_sample"]

    # Load existing TASD data
    with open("results/qwen_5method_6x80.json") as f:
        old_data = json.load(f)
    old_ps = old_data["per_sample"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    all_data = {}
    aggregate = {}

    for bname, data_file, stype in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bname}")
        print(f"{'='*60}")

        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:80]]
        n = len(samples)
        prompts = [s["prompt"] for s in samples]
        names = [s["name"] for s in samples]
        refs = [s.get("reference", "") for s in samples]

        # AR TPS from quality data
        ar_data = ar_per_sample.get(bname, {}).get("AR", [])
        ar_tps_list = [r["ar_tps"] for r in ar_data]

        all_data[bname] = {"AR": ar_data, "TASD": old_ps.get(bname, {}).get("TASD", [])}

        # ── TASD-F ──
        print(f"  TASD-F ({n} samples)...", end=" ", flush=True)
        ts_results = []
        for i in range(n):
            r = run_tasd_f(target, draft, tokenizer, prompts[i], stype)
            sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0.0
            q = compute_composite_sq(r["text"], refs[i], stype)
            ts_results.append({
                "name": names[i],
                "tps": round(r["tps"], 2),
                "sp": round(sp, 3),
                "wall": round(r["wall"], 3),
                "accept": round(r["accept"], 4),
                "repair": r["repair"],
                "guard_trig": r["guard_trig"],
                "trim": r["trim"],
                "hard_trim": r["hard_trim"],
                "rep_warn": r["rep_warn"],
                "brack_warn": r["brack_warn"],
                "fb_count": r["fb_count"],
                "fb_tokens": r["fb_tokens"],
                "text": r["text"],
                **q,
            })
            if (i + 1) % 20 == 0:
                sp_mu = sum(r_["sp"] for r_ in ts_results) / (i + 1)
                sq_mu = sum(r_["composite_sq"] for r_ in ts_results) / (i + 1)
                fb_total = sum(r_["fb_count"] for r_ in ts_results)
                print(f"{i+1}(sp={sp_mu:.2f}x,sq={sq_mu:.3f},fb={fb_total})...", end=" ", flush=True)

        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(os.path.join(CHECKPOINT_DIR, f"{bname}_TASDF.json"), "w") as f:
            json.dump(ts_results, f, indent=2, ensure_ascii=False)

        all_data[bname]["TASD-F"] = ts_results

        sps = [r["sp"] for r in ts_results]
        below = sum(1 for s in sps if s < 1.0)
        sp_mean = sum(sps) / n
        sQ_mean = sum(r["composite_sq"] for r in ts_results) / n
        fb_total = sum(r["fb_count"] for r in ts_results)

        aggregate[bname] = {"n": n}
        aggregate[bname]["TASD-F"] = {
            "sp_avg": round(sp_mean, 3), "sq_avg": round(sQ_mean, 4),
            "below": below, "fb_count": fb_total,
        }
        aggregate[bname]["AR"] = {
            "tps_avg": round(sum(ar_tps_list) / n, 1),
            "sq_avg": round(sum(r.get("composite_sq", 0) for r in ar_data) / n, 4) if ar_data else 0,
        }
        # TASD from old data
        old_ts = old_ps.get(bname, {}).get("TASD", [])
        if old_ts:
            old_below = sum(1 for r in old_ts if r["sp"] < 1.0)
            old_sp = sum(r["sp"] for r in old_ts) / n
            aggregate[bname]["TASD"] = {
                "sp_avg": round(old_sp, 3),
                "below": old_below,
            }

        print(f"\n  TASD: sp={aggregate[bname].get('TASD',{}).get('sp_avg',0):.3f}x below={aggregate[bname].get('TASD',{}).get('below',0)}")
        print(f"  TASD-F: sp={sp_mean:.3f}x below={below} fb={fb_total} sq={sQ_mean:.4f}")

    # ── Save ──
    output = {
        "config": {
            "target": "Qwen2.5-14B-Instruct-AWQ",
            "draft": "Qwen2.5-1.5B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS,
            "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "enable_failure_aware_fallback": True, "fallback_tokens": 2,
            "fallback_guarded": False, "guard_calibrated": True,
        },
        "per_benchmark": aggregate,
        "per_sample": all_data,
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    # ── MD ──
    write_md_report(output)
    print(f"Saved {OUT_MD}")
    print("Done!")


def write_md_report(output):
    agg = output["per_benchmark"]
    cfg = output["config"]
    bnames = [b[0] for b in BENCHMARKS]

    with open(OUT_MD, "w") as f:
        f.write("# TASD-F Qwen 6×80 Full Experiment\n\n")
        f.write(f"**Target**: {cfg['target']}  |  **Draft**: {cfg['draft']}\n")
        f.write(f"**Config**: max_new_tokens={cfg['max_new_tokens']}, draft_len={cfg['draft_len']}, draft_blocks={cfg['draft_blocks']}, top_k_accept={cfg['top_k_accept']}\n\n")
        f.write("**TASD-F**: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=False, guard_calibrated=True\n\n")

        # Overall
        f.write("## TASD vs TASD-F Overall\n\n")
        f.write("| Method | Speedup | Below 1.0x | SQ | Fallbacks |\n")
        f.write("|--------|:-------:|:----------:|:--:|:---------:|\n")

        ts_sps = []; tf_sps = []; ts_below_total = 0; tf_below_total = 0
        ts_sqs = []; tf_sqs = []; tf_fb_total = 0

        for bname in bnames:
            if "TASD" in agg[bname]:
                ts_sps.append(agg[bname]["TASD"]["sp_avg"])
                ts_below_total += agg[bname]["TASD"].get("below", 0)
            if "TASD-F" in agg[bname]:
                tf_sps.append(agg[bname]["TASD-F"]["sp_avg"])
                tf_below_total += agg[bname]["TASD-F"].get("below", 0)
                tf_sqs.append(agg[bname]["TASD-F"].get("sq_avg", 0))
                tf_fb_total += agg[bname]["TASD-F"].get("fb_count", 0)

        ts_mean_sp = sum(ts_sps) / len(ts_sps) if ts_sps else 0
        tf_mean_sp = sum(tf_sps) / len(tf_sps) if tf_sps else 0
        tf_mean_sq = sum(tf_sqs) / len(tf_sqs) if tf_sqs else 0

        f.write(f"| **TASD** | **{ts_mean_sp:.3f}x** | {ts_below_total}/480 | - | - |\n")
        f.write(f"| **TASD-F** | **{tf_mean_sp:.3f}x** | {tf_below_total}/480 | {tf_mean_sq:.4f} | {tf_fb_total} |\n\n")

        # Detailed per-benchmark
        f.write("## Per-Benchmark Comparison\n\n")
        f.write("| Benchmark | TASD sp | TASD-F sp | TASD below | TASD-F below | TASD-F SQ | TASD-F FB |\n")
        f.write("|-----------|:-------:|:---------:|:----------:|:------------:|:---------:|:---------:|\n")
        for bname in bnames:
            ts = agg[bname].get("TASD", {})
            tf = agg[bname].get("TASD-F", {})
            f.write(f"| {bname} | {ts.get('sp_avg',0):.3f}x | **{tf.get('sp_avg',0):.3f}x** | {ts.get('below',0)} | {tf.get('below',0)} | {tf.get('sq_avg',0):.4f} | {tf.get('fb_count',0)} |\n")
        f.write("\n")

        # Per-sample TASD-F details
        f.write("## TASD-F Per-Benchmark Details\n\n")
        for bname in bnames:
            if "TASD-F" not in agg[bname]:
                continue
            samples = output["per_sample"].get(bname, {}).get("TASD-F", [])
            if not samples:
                continue
            n = len(samples)
            sps = sorted([s["sp"] for s in samples])
            worst10 = sps[:max(1, n // 10)]
            f.write(f"### {bname} ({n} samples)\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|:-----:|\n")
            f.write(f"| mean sp | {sum(sps)/n:.3f}x |\n")
            f.write(f"| median sp | {sps[n//2]:.3f}x |\n")
            f.write(f"| min sp | {sps[0]:.3f}x |\n")
            f.write(f"| worst-10 sp | {sum(worst10)/len(worst10):.3f}x |\n")
            f.write(f"| below-1.0x | {sum(1 for s in sps if s<1.0)} |\n")
            f.write(f"| mean accept | {sum(s['accept'] for s in samples)/n:.4f} |\n")
            f.write(f"| mean repair | {sum(s['repair'] for s in samples)/n:.1f} |\n")
            f.write(f"| mean guard_trig | {sum(s['guard_trig'] for s in samples)/n:.1f} |\n")
            f.write(f"| mean trim | {sum(s['trim'] for s in samples)/n:.1f} |\n")
            f.write(f"| mean SQ | {sum(s['composite_sq'] for s in samples)/n:.4f} |\n")
            f.write(f"| total fb | {sum(s['fb_count'] for s in samples)} |\n\n")

        # Criteria
        f.write("## Pass/Fail Criteria\n\n")
        f.write("| Criterion | Result | Note |\n")
        f.write("|-----------|:------:|------|\n")

        # 1. mean sp not degraded >3%
        ok1 = tf_mean_sp >= ts_mean_sp * 0.97
        f.write(f"| mean sp >= TASD×0.97 | {'PASS' if ok1 else 'FAIL'} | {ts_mean_sp:.3f} → {tf_mean_sp:.3f} |\n")

        # 2. below reduced
        ok2 = tf_below_total < ts_below_total
        f.write(f"| below-1.0x reduced | {'PASS' if ok2 else 'FAIL'} | {ts_below_total} → {tf_below_total} |\n")

        # 3. worst-10 check from all samples
        all_ts_sp = []
        all_tf_sp = []
        for bname in bnames:
            ts = output["per_sample"].get(bname, {}).get("TASD", [])
            tf = output["per_sample"].get(bname, {}).get("TASD-F", [])
            all_ts_sp.extend(s["sp"] for s in ts)
            all_tf_sp.extend(s["sp"] for s in tf)
        all_ts_sp.sort(); all_tf_sp.sort()
        n_total = len(all_ts_sp)
        ts_w10 = sum(all_ts_sp[:n_total//10]) / (n_total//10) if n_total >= 10 else min(all_ts_sp)
        tf_w10 = sum(all_tf_sp[:n_total//10]) / (n_total//10) if n_total >= 10 else min(all_tf_sp)
        ok3 = tf_w10 >= ts_w10 * 0.9
        f.write(f"| worst-10 sp >= TASD×0.9 | {'PASS' if ok3 else 'FAIL'} | {ts_w10:.3f} → {tf_w10:.3f} |\n")

        # 4. SQ not degraded
        ts_sq_vals = [s["composite_sq"] for s in output["per_sample"]["argparse"]["AR"]]  # AR SQ from quality
        ts_sq_all = []
        tf_sq_all = []
        for bname in bnames:
            ts_sq_all.extend(s.get("composite_sq", 0) for s in output["per_sample"].get(bname, {}).get("TASD", []))
            tf_sq_all.extend(s.get("composite_sq", 0) for s in output["per_sample"].get(bname, {}).get("TASD-F", []))
        ts_sq_mean = sum(ts_sq_all)/len(ts_sq_all) if ts_sq_all else 0
        tf_sq_mean = sum(tf_sq_all)/len(tf_sq_all) if tf_sq_all else 0
        ok4 = tf_sq_mean >= ts_sq_mean - 0.02
        f.write(f"| SQ >= TASD-0.02 | {'PASS' if ok4 else 'FAIL'} | {ts_sq_mean:.4f} → {tf_sq_mean:.4f} |\n")

        # 5. off_structure
        ts_off = [s.get("off_structure_rate",0) for bname in bnames for s in output["per_sample"].get(bname,{}).get("TASD",[])]
        tf_off = [s.get("off_structure_rate",0) for bname in bnames for s in output["per_sample"].get(bname,{}).get("TASD-F",[])]
        ts_off_m = sum(ts_off)/len(ts_off) if ts_off else 0
        tf_off_m = sum(tf_off)/len(tf_off) if tf_off else 0
        ok5 = tf_off_m <= ts_off_m * 1.1
        f.write(f"| off_structure <= TASD×1.1 | {'PASS' if ok5 else 'FAIL'} | {ts_off_m:.4f} → {tf_off_m:.4f} |\n")

        all_ok = ok1 and ok2 and ok3 and ok4 and ok5
        f.write(f"\n**Overall**: {'ALL PASS' if all_ok else 'SOME FAIL'}\n\n")

        f.write(f"## Data Files\n\n- `{OUT_JSON}`\n- `{CHECKPOINT_DIR}/`\n")


if __name__ == "__main__":
    main()
