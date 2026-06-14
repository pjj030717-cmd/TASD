"""
Recompute Qwen 6×80 with new quality metrics.

Regenerates texts for AR and TASD (and optionally FLY), computes
structural_char_F1, bracket_balance_score, structure_type_preservation,
no_repetition_score, and composite SQ.

Saves per-sample texts and all metrics.

Usage: python3 recompute_qwen_6x80_quality.py
"""
import json, os, sys, time, logging
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import (
    compute_composite_sq,
    structural_char_recall,
    structural_char_f1,
    bracket_balance_score,
    structure_type_preservation,
    no_repetition_score,
    off_structure_rate,
    repetition_rate,
    is_truncated,
)

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
OUT_JSON = "results/qwen_5method_6x80_quality.json"
OUT_MD = "results/qwen_5method_6x80_quality.md"

# Which methods to (re)run
RERUN_METHODS = ["AR", "TASD"]  # "FLY" also available but slow


def load_checkpoint(bname, method_label):
    fname = os.path.join(CHECKPOINT_DIR, f"{bname}_{method_label}.json")
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)
    return None


def save_checkpoint(bname, method_label, data):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = os.path.join(CHECKPOINT_DIR, f"{bname}_{method_label}.json")
    with open(fname, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()

    print("Loading draft model...")
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    # Load existing data for speedup values (we reuse wall/tps from checkpoints)
    existing = {}
    with open("results/qwen_5method_6x80.json") as f:
        old_data = json.load(f)
    old_per_sample = old_data["per_sample"]
    old_per_benchmark = old_data["per_benchmark"]

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

        all_data[bname] = {}
        aggregate[bname] = {"n": n}

        # ── AR ──
        if "AR" in RERUN_METHODS:
            print(f"  AR ({n} samples)...", end=" ", flush=True)
            ar_results = []
            for i in range(n):
                r = run_ar(target, tokenizer, prompts[i])
                q = compute_composite_sq(r["text"], refs[i], stype)
                ar_results.append({
                    "name": names[i],
                    "ar_tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3),
                    "prompt_len": r["prompt_len"],
                    "gen_len": r["gen_len"],
                    "text": r["text"],
                    **q,
                })
                if (i + 1) % 20 == 0:
                    print(f"{i+1}...", end=" ", flush=True)
            save_checkpoint(bname, "AR_quality", ar_results)
            all_data[bname]["AR"] = ar_results
            ar_composites = [r["composite_sq"] for r in ar_results]
            ar_f1 = [r["structural_char_f1"] for r in ar_results]
            print(f"done. composite_sq={sum(ar_composites)/n:.4f} f1={sum(ar_f1)/n:.4f}")
        else:
            # Load from existing data (no text, use legacy SQ)
            ar_ckpt = load_checkpoint(bname, "AR")
            if ar_ckpt is None:
                ar_ckpt = old_per_sample[bname]["AR"]
            all_data[bname]["AR"] = ar_ckpt

        # ── AR TPS for speedup ──
        ar_tps_list = [r["ar_tps"] for r in all_data[bname]["AR"]]
        aggregate[bname]["AR"] = {
            "tps_avg": round(sum(ar_tps_list) / n, 1),
            "sq_avg": round(sum(r.get("composite_sq", r.get("sq", 0)) for r in all_data[bname]["AR"]) / n, 4),
            "sp_avg": 1.0, "below": 0,
        }

        # ── TASD ──
        if "TASD" in RERUN_METHODS:
            print(f"  TASD ({n} samples)...", end=" ", flush=True)
            ts_results = []
            for i in range(n):
                r = run_tasd(target, draft, tokenizer, prompts[i], stype)
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
                    "text": r["text"],
                    **q,
                })
                if (i + 1) % 20 == 0:
                    ts_mu = sum(r_["composite_sq"] for r_ in ts_results) / (i + 1)
                    ts_sp = sum(r_["sp"] for r_ in ts_results) / (i + 1)
                    print(f"{i+1}...", end=" ", flush=True)
            save_checkpoint(bname, "TASD_quality", ts_results)
            all_data[bname]["TASD"] = ts_results

            ts_sp = [r["sp"] for r in ts_results]
            ts_composites = [r["composite_sq"] for r in ts_results]
            below = sum(1 for s in ts_sp if s < 1.0)
            print(f"done. composite_sq={sum(ts_composites)/n:.4f} sp={sum(ts_sp)/n:.3f}x below={below}")
            aggregate[bname]["TASD"] = {
                "sp_avg": round(sum(ts_sp) / n, 3),
                "sq_avg": round(sum(ts_composites) / n, 4),
                "below": below,
            }

        # ── Carry over FLY/GSD/Ngram from old data (no text) ──
        for ml in ["GSD", "Ngram", "FLY"]:
            if ml in RERUN_METHODS:
                continue
            if ml in old_per_sample[bname]:
                all_data[bname][ml] = old_per_sample[bname][ml]
                ag = old_per_benchmark[bname][ml]
                aggregate[bname][ml] = ag

        # Print summary
        print(f"  Summary:")
        for ml in ["AR", "TASD", "FLY", "GSD", "Ngram"]:
            if ml in aggregate[bname]:
                a = aggregate[bname][ml]
                sp_str = f"{a.get('sp_avg', 1.0):.3f}x" if ml != "AR" else "1.000x"
                print(f"    {ml:5s}: sp={sp_str:>7s}  sq={a['sq_avg']:.4f}  below={a.get('below', '-')}")

    # ── Save ──
    output = {
        "config": {
            "target": "Qwen2.5-14B-Instruct-AWQ",
            "draft": "Qwen2.5-1.5B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "sq_formula": "0.4*structural_char_F1 + 0.3*bracket_balance_score + 0.2*structure_type_preservation + 0.1*no_repetition_score",
            "tps_note": "All TPS computed as generated_tokens / wall_time, excluding prompt tokens.",
        },
        "per_benchmark": aggregate,
        "per_sample": all_data,
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    # ── MD report ──
    write_md_report(output)
    print(f"Saved {OUT_MD}")
    print("Done!")


def write_md_report(output):
    agg = output["per_benchmark"]
    cfg = output["config"]
    methods = ["AR", "GSD", "Ngram", "FLY", "TASD"]
    labels = {"AR": "AR", "GSD": "Greedy SD", "Ngram": "N-gram SD",
              "FLY": "Official FLY", "TASD": "TASD"}
    bnames = [b[0] for b in BENCHMARKS]

    with open(OUT_MD, "w") as f:
        f.write("# Qwen 6×80 Quality Report (New SQ)\n\n")
        f.write(f"**Target**: {cfg['target']}  |  **Draft**: {cfg['draft']}\n")
        f.write(f"**Config**: max_new_tokens={cfg['max_new_tokens']}, temperature={cfg['temperature']}\n\n")
        f.write(f"> **SQ Formula**: `{cfg['sq_formula']}`\n\n")
        f.write("> **TPS Note**: computed as `generated_tokens / wall_time`\n\n")

        f.write("## SQ Sub-metrics\n\n")
        f.write("| Metric | Weight | Description |\n")
        f.write("|--------|:------:|-------------|\n")
        f.write("| structural_char_F1 | 0.4 | F1 of structural characters ({,},[,],(,),:,,,=,\\n) between pred and ref |\n")
        f.write("| bracket_balance_score | 0.3 | 1.0 if all (), [], {} are balanced, 0.0 otherwise |\n")
        f.write("| structure_type_preservation | 0.2 | 1.0 if output preserves expected structure type |\n")
        f.write("| no_repetition_score | 0.1 | 1.0 - 3×repetition_rate, penalizes repeated blocks |\n\n")

        f.write("## Overall\n\n")
        f.write("| Method | Speedup | SQ | structural_char_F1 | bracket_balance | type_preservation | no_repetition |\n")
        f.write("|--------|:-------:|:--:|:------------------:|:---------------:|:-----------------:|:-------------:|\n")

        # Compute overall averages
        for ml in methods:
            sps = []; sQs = []; f1s = []; bbs = []; tps = []; nrs = []
            for bname in bnames:
                if ml in agg[bname]:
                    a = agg[bname][ml]
                    sp = a.get("sp_avg", 1.0)
                    sq = a.get("sq_avg", 0.0)
                    if ml == "AR":
                        sps.append(1.0)
                    else:
                        sps.append(sp)
                    sQs.append(sq)
            
            if not sps:
                continue
            mean_sp = sum(sps) / len(sps)
            mean_sq = sum(sQs) / len(sQs)
            
            # Get per-sample sub-metrics
            all_f1 = []; all_bb = []; all_tp = []; all_nr = []
            for bname in bnames:
                if ml in output["per_sample"][bname]:
                    for s in output["per_sample"][bname][ml]:
                        all_f1.append(s.get("structural_char_f1", s.get("sq", 0)))
                        all_bb.append(s.get("bracket_balance_score", 1.0))
                        all_tp.append(s.get("structure_type_preservation", 1.0))
                        all_nr.append(s.get("no_repetition_score", 1.0))
            
            f_mean = sum(all_f1)/len(all_f1) if all_f1 else 0
            b_mean = sum(all_bb)/len(all_bb) if all_bb else 0
            t_mean = sum(all_tp)/len(all_tp) if all_tp else 0
            n_mean = sum(all_nr)/len(all_nr) if all_nr else 0
            
            f.write(f"| **{labels[ml]}** | **{mean_sp:.3f}x** | **{mean_sq:.4f}** | {f_mean:.4f} | {b_mean:.4f} | {t_mean:.4f} | {n_mean:.4f} |\n")
        
        f.write("\n")
        f.write("## Per-Benchmark\n\n")
        for bname in bnames:
            f.write(f"### {bname} ({agg[bname]['n']} samples)\n\n")
            f.write("| Method | Sp | SQ | char_F1 | bracket | type_pres | no_rep | off_str | rep_rate | trunc | Below |\n")
            f.write("|--------|:--:|:--:|:-------:|:-------:|:--------:|:------:|:-------:|:--------:|:-----:|:-----:|\n")
            for ml in methods:
                if ml not in agg[bname]:
                    continue
                a = agg[bname][ml]
                sp = a.get("sp_avg", 1.0) if ml != "AR" else 1.0
                sq = a.get("sq_avg", 0.0)
                below = a.get("below", "-") if ml != "AR" else "-"
                
                # Per-sample sub-metric means for this benchmark
                if ml in output["per_sample"][bname]:
                    smps = output["per_sample"][bname][ml]
                    f1_m = sum(s.get("structural_char_f1", s.get("sq", 0)) for s in smps)/len(smps)
                    bb_m = sum(s.get("bracket_balance_score", 1.0) for s in smps)/len(smps)
                    tp_m = sum(s.get("structure_type_preservation", 1.0) for s in smps)/len(smps)
                    nr_m = sum(s.get("no_repetition_score", 1.0) for s in smps)/len(smps)
                    os_m = sum(s.get("off_structure_rate", 0.0) for s in smps)/len(smps)
                    rp_m = sum(s.get("repetition_rate", 0.0) for s in smps)/len(smps)
                    tr_m = sum(s.get("is_truncated", 0.0) for s in smps)/len(smps)
                else:
                    f1_m = 0; bb_m = 0; tp_m = 0; nr_m = 0; os_m = 0; rp_m = 0; tr_m = 0
                
                f.write(f"| {labels[ml]} | {sp:.3f}x | {sq:.4f} | {f1_m:.4f} | {bb_m:.4f} | {tp_m:.4f} | {nr_m:.4f} | {os_m:.4f} | {rp_m:.4f} | {tr_m:.4f} | {below} |\n")
            f.write("\n")

        f.write(f"## Data Files\n\n- `{OUT_JSON}`\n- `{CHECKPOINT_DIR}/`\n")


if __name__ == "__main__":
    main()
