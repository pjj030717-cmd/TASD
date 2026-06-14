#!/usr/bin/env python3
"""
TASD-F-G Qwen 6×80 Full Experiment (Guarded Fallback).

fallback_guarded=True — structural guard applied during failure-aware AR fallback.
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

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested_config"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli_option_groups"),
]

CHECKPOINT_DIR = "results/qwen_6x80_checkpoints"
OUT_JSON = "results/qwen_tasd_fg_6x80.json"
OUT_MD = "results/qwen_tasd_fg_6x80.md"


def run_tasd_fg(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    enable_failure_aware_fallback=True,
                    fallback_guarded=True,
                    fallback_accept_threshold=0.5,
                    fallback_repair_threshold=2,
                    **TASD_COMMON)
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
    with open("results/qwen_5method_6x80_quality.json") as f:
        quality_data = json.load(f)
    ar_per_sample = quality_data["per_sample"]

    with open("results/qwen_5method_6x80.json") as f:
        old_data = json.load(f)
    old_ps = old_data["per_sample"]

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

        ar_data = ar_per_sample.get(bname, {}).get("AR", [])
        ar_tps_list = [r["ar_tps"] for r in ar_data]

        all_data[bname] = {"AR": ar_data, "TASD": old_ps.get(bname, {}).get("TASD", [])}

        # ── TASD-F-G ──
        print(f"  TASD-F-G ({n} samples)...", end=" ", flush=True)
        ts_results = []
        for i in range(n):
            r = run_tasd_fg(target, draft, tokenizer, prompts[i], stype)
            sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0.0
            q = compute_composite_sq(r["text"], refs[i], stype)
            ts_results.append({
                "name": names[i], "tps": round(r["tps"], 2), "sp": round(sp, 3),
                "wall": round(r["wall"], 3), "accept": round(r["accept"], 4),
                "repair": r["repair"], "guard_trig": r["guard_trig"],
                "trim": r["trim"], "hard_trim": r["hard_trim"],
                "rep_warn": r["rep_warn"], "brack_warn": r["brack_warn"],
                "fb_count": r["fb_count"], "fb_tokens": r["fb_tokens"],
                "text": r["text"], **q,
            })
            if (i + 1) % 20 == 0:
                sp_mu = sum(e["sp"] for e in ts_results) / (i + 1)
                sq_mu = sum(e["composite_sq"] for e in ts_results) / (i + 1)
                fb_total = sum(e["fb_count"] for e in ts_results)
                print(f"{i+1}(sp={sp_mu:.2f}x,sq={sq_mu:.3f},fb={fb_total})...", end=" ", flush=True)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(os.path.join(CHECKPOINT_DIR, f"{bname}_TASDFG.json"), "w") as f:
            json.dump(ts_results, f, indent=2, ensure_ascii=False)

        all_data[bname]["TASD-F-G"] = ts_results

        sps = [e["sp"] for e in ts_results]
        below = sum(1 for s in sps if s < 1.0)
        sp_mean = sum(sps) / n
        sq_mean = sum(e["composite_sq"] for e in ts_results) / n
        fb_total = sum(e["fb_count"] for e in ts_results)

        aggregate[bname] = {"n": n}
        aggregate[bname]["TASD-F-G"] = {
            "sp_avg": round(sp_mean, 3), "sq_avg": round(sq_mean, 4),
            "below": below, "fb_count": fb_total,
        }
        aggregate[bname]["AR"] = {
            "tps_avg": round(sum(ar_tps_list) / n, 1),
            "sq_avg": round(sum(r.get("composite_sq", 0) for r in ar_data) / n, 4) if ar_data else 0,
        }
        old_ts = old_ps.get(bname, {}).get("TASD", [])
        if old_ts:
            aggregate[bname]["TASD"] = {
                "sp_avg": round(sum(e["sp"] for e in old_ts) / n, 3),
                "below": sum(1 for e in old_ts if e["sp"] < 1.0),
            }

        print(f"\n  TASD: sp={aggregate[bname]['TASD']['sp_avg']:.3f}x below={aggregate[bname]['TASD']['below']}")
        print(f"  TASD-F-G: sp={sp_mean:.3f}x below={below} fb={fb_total} sq={sq_mean:.4f}")

    # ── Save ──
    output = {
        "config": {
            "target": "Qwen2.5-14B-Instruct-AWQ", "draft": "Qwen2.5-1.5B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS, "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "enable_failure_aware_fallback": True, "fallback_tokens": 2,
            "fallback_guarded": True, "guard_calibrated": True,
        },
        "per_benchmark": aggregate, "per_sample": all_data,
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    write_md_report(output)
    print(f"Saved {OUT_MD}")
    print("Done!")


def write_md_report(output):
    agg = output["per_benchmark"]
    cfg = output["config"]
    bnames = [b[0] for b in BENCHMARKS]

    with open(OUT_MD, "w") as f:
        f.write("# TASD-F-G Qwen 6×80 Full Experiment (Guarded Fallback)\n\n")
        f.write(f"**Target**: {cfg['target']}  |  **Draft**: {cfg['draft']}\n")
        f.write(f"**Config**: max_new_tokens={cfg['max_new_tokens']}, draft_len={cfg['draft_len']}, draft_blocks={cfg['draft_blocks']}, top_k_accept={cfg['top_k_accept']}\n\n")
        f.write("**TASD-F-G**: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=True, guard_calibrated=True\n\n")

        f.write("## TASD vs TASD-F-G Overall\n\n")
        f.write("| Method | Speedup | Below 1.0x | SQ | Fallbacks |\n")
        f.write("|--------|:-------:|:----------:|:--:|:---------:|\n")

        ts_sps = []; tfg_sps = []; ts_below_t = 0; tfg_below_t = 0; tfg_sqs = []; tfg_fb_t = 0
        for bn in bnames:
            if "TASD" in agg[bn]:
                ts_sps.append(agg[bn]["TASD"]["sp_avg"])
                ts_below_t += agg[bn]["TASD"]["below"]
            if "TASD-F-G" in agg[bn]:
                tfg_sps.append(agg[bn]["TASD-F-G"]["sp_avg"])
                tfg_below_t += agg[bn]["TASD-F-G"]["below"]
                tfg_sqs.append(agg[bn]["TASD-F-G"]["sq_avg"])
                tfg_fb_t += agg[bn]["TASD-F-G"]["fb_count"]

        ts_m = sum(ts_sps)/len(ts_sps) if ts_sps else 0
        tfg_m = sum(tfg_sps)/len(tfg_sps) if tfg_sps else 0
        tfg_sq_m = sum(tfg_sqs)/len(tfg_sqs) if tfg_sqs else 0
        f.write(f"| **TASD** | **{ts_m:.3f}x** | {ts_below_t}/480 | - | - |\n")
        f.write(f"| **TASD-F-G** | **{tfg_m:.3f}x** | {tfg_below_t}/480 | {tfg_sq_m:.4f} | {tfg_fb_t} |\n\n")

        f.write("## Per-Benchmark Comparison\n\n")
        f.write("| Benchmark | TASD sp | TASD-F-G sp | TASD b<1 | TASD-F-G b<1 | TASD-F-G SQ | TASD-F-G FB |\n")
        f.write("|-----------|:-------:|:-----------:|:--------:|:------------:|:-----------:|:-----------:|\n")
        for bn in bnames:
            ts = agg[bn].get("TASD", {})
            tfg = agg[bn].get("TASD-F-G", {})
            f.write(f"| {bn} | {ts.get('sp_avg',0):.3f}x | **{tfg.get('sp_avg',0):.3f}x** | {ts.get('below',0)} | {tfg.get('below',0)} | {tfg.get('sq_avg',0):.4f} | {tfg.get('fb_count',0)} |\n")
        f.write("\n")

        f.write("## TASD-F-G Per-Benchmark Details\n\n")
        for bn in bnames:
            if "TASD-F-G" not in agg[bn]:
                continue
            samples = output["per_sample"].get(bn, {}).get("TASD-F-G", [])
            if not samples:
                continue
            n = len(samples)
            sps = sorted([e["sp"] for e in samples])
            worst10 = sps[:max(1, n//10)]
            f.write(f"### {bn} ({n} samples)\n\n")
            f.write(f"| Metric | Value |\n|--------|:-----:|\n")
            f.write(f"| mean sp | {sum(sps)/n:.3f}x |\n")
            f.write(f"| median sp | {sps[n//2]:.3f}x |\n")
            f.write(f"| min sp | {sps[0]:.3f}x |\n")
            f.write(f"| worst-10 sp | {sum(worst10)/len(worst10):.3f}x |\n")
            f.write(f"| below-1.0x | {sum(1 for s in sps if s<1.0)} |\n")
            f.write(f"| mean accept | {sum(e['accept'] for e in samples)/n:.4f} |\n")
            f.write(f"| mean repair | {sum(e['repair'] for e in samples)/n:.1f} |\n")
            f.write(f"| mean guard_trig | {sum(e['guard_trig'] for e in samples)/n:.1f} |\n")
            f.write(f"| mean trim | {sum(e['trim'] for e in samples)/n:.1f} |\n")
            f.write(f"| mean SQ | {sum(e['composite_sq'] for e in samples)/n:.4f} |\n")
            f.write(f"| total fb | {sum(e['fb_count'] for e in samples)} |\n\n")

        f.write("## Pass/Fail\n\n")
        f.write("| Criterion | Result | Note |\n")
        f.write("|-----------|:------:|------|\n")

        ok1 = tfg_m >= ts_m * 0.97
        f.write(f"| mean sp >= TASD×0.97 | {'PASS' if ok1 else 'FAIL'} | {ts_m:.3f} → {tfg_m:.3f} |\n")

        ok2 = tfg_below_t < ts_below_t
        f.write(f"| below-1.0x reduced | {'PASS' if ok2 else 'FAIL'} | {ts_below_t} → {tfg_below_t} |\n")

        all_ts_sp = []; all_tfg_sp = []
        for bn in bnames:
            all_ts_sp.extend(e["sp"] for e in output["per_sample"].get(bn,{}).get("TASD",[]))
            all_tfg_sp.extend(e["sp"] for e in output["per_sample"].get(bn,{}).get("TASD-F-G",[]))
        all_ts_sp.sort(); all_tfg_sp.sort()
        n_tot = len(all_ts_sp)
        w10 = n_tot//10
        ts_w10 = sum(all_ts_sp[:w10])/w10 if w10>0 else min(all_ts_sp)
        tfg_w10 = sum(all_tfg_sp[:w10])/w10 if w10>0 else min(all_tfg_sp)
        ok3 = tfg_w10 >= ts_w10 * 0.9
        f.write(f"| worst-10 sp >= TASD×0.9 | {'PASS' if ok3 else 'FAIL'} | {ts_w10:.3f} → {tfg_w10:.3f} |\n")

        ts_sq_all = [e.get("composite_sq",0) for bn in bnames for e in output["per_sample"].get(bn,{}).get("TASD",[])]
        tfg_sq_all = [e.get("composite_sq",0) for bn in bnames for e in output["per_sample"].get(bn,{}).get("TASD-F-G",[])]
        ts_sq_m = sum(ts_sq_all)/len(ts_sq_all) if ts_sq_all else 0
        tfg_sq_m2 = sum(tfg_sq_all)/len(tfg_sq_all) if tfg_sq_all else 0
        ok4 = tfg_sq_m2 >= ts_sq_m - 0.02
        f.write(f"| SQ >= TASD-0.02 | {'PASS' if ok4 else 'FAIL'} | {ts_sq_m:.4f} → {tfg_sq_m2:.4f} |\n")

        ts_off = [e.get("off_structure_rate",0) for bn in bnames for e in output["per_sample"].get(bn,{}).get("TASD",[])]
        tfg_off = [e.get("off_structure_rate",0) for bn in bnames for e in output["per_sample"].get(bn,{}).get("TASD-F-G",[])]
        ts_off_m = sum(ts_off)/len(ts_off) if ts_off else 0
        tfg_off_m = sum(tfg_off)/len(tfg_off) if tfg_off else 0
        ok5 = tfg_off_m <= ts_off_m * 1.1
        f.write(f"| off_structure <= TASD×1.1 | {'PASS' if ok5 else 'FAIL'} | {ts_off_m:.4f} → {tfg_off_m:.4f} |\n")

        all_ok = ok1 and ok2 and ok3 and ok4 and ok5
        f.write(f"\n**Overall**: {'ALL PASS' if all_ok else 'SOME FAIL'}\n\n")
        f.write(f"## Data\n\n- `{OUT_JSON}`\n- `{CHECKPOINT_DIR}/`\n")


if __name__ == "__main__":
    main()
