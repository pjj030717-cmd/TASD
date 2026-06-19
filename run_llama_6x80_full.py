#!/usr/bin/env python3
"""
LLaMA 6×80 Full Generalization Experiment.

Target: meta-llama/Llama-3.1-8B-Instruct
Draft: meta-llama/Llama-3.2-1B-Instruct
Config: max_new_tokens=128, 6 benchmarks × 80 samples

Methods: AR, Official FLY, TASD, TASD-FG
"""
import json, os, sys, time, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.ngram_sd_decode import ngram_sd_decode
from src.quality_metrics import compute_composite_sq
from run_hardcase_repair import fly_decode

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
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

CHECKPOINT_DIR = "results/checkpoints_llama_6x80"
OUT_JSON = "results/llama_6x80_full.json"
OUT_MD = "results/llama_6x80_full.md"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    p_len = inp.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    wall = time.time() - t0
    gids = out[0][p_len:]
    text = tokenizer.decode(gids, skip_special_tokens=True)
    tps = len(gids) / wall if wall > 0 else 0
    return {"wall": wall, "tps": tps, "gen_len": len(gids), "text": text}


def run_tasd(target, draft, tokenizer, prompt, stype, enable_fb=False, fb_guarded=False):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    enable_failure_aware_fallback=enable_fb,
                    fallback_guarded=fb_guarded,
                    fallback_accept_threshold=0.5,
                    fallback_repair_threshold=2,
                    **TASD_COMMON)
    s = r["stats"]
    fb_s = s.get("failure_aware_fallback", {}) or {}
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "gen_len": s["generated_length"], "text": r["generated_text"],
            "accept": s["accept_rate"], "repair": s.get("repair_count", 0),
            "guard_trig": s.get("guard_trigger_count", 0),
            "trim": s.get("trim_count", 0),
            "fb_count": fb_s.get("fallback_count", 0),
            "fb_tokens": fb_s.get("total_fallback_tokens", 0)}


def run_gsd(target, draft, tokenizer, prompt, stype):
    """Greedy Speculative Decoding (strict accept, no guard)"""
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=False, enable_relaxed_accept=False,
                    **TASD_COMMON)
    s = r["stats"]
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "gen_len": s["generated_length"], "text": r["generated_text"],
            "accept": s["accept_rate"]}


def run_ngram(target, tokenizer, prompt):
    """N-gram Speculative Decoding (prompt lookup)"""
    r = ngram_sd_decode(target, tokenizer, prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        ngram_min=3, ngram_max=8, max_draft_tokens=16)
    return {"wall": r.get("wall_time", 0), "tps": r["tokens_per_second"],
            "gen_len": r.get("stats", {}).get("generated_length", 0),
            "text": r["generated_text"],
            "accept": r.get("stats", {}).get("accept_rate", 0)}


def run_fly(target, draft, tokenizer, prompt):
    r = fly_decode(target, draft, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    return {"wall": r.get("elapsed_time", 0), "tps": r.get("tokens_per_second", 0),
            "gen_len": r.get("generated_tokens", 0), "text": r.get("generated_text", ""),
            "accept": 0, "repair": 0, "guard_trig": 0, "trim": 0,
            "fb_count": 0, "fb_tokens": 0}


def main():
    print("Loading LLaMA models...")
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
            samples = [json.loads(l.strip()) for l in f.readlines()[:40]]
        n = len(samples)
        prompts = [s["prompt"] for s in samples]
        names = [s["name"] for s in samples]
        refs = [s.get("reference", "") for s in samples]

        all_data[bname] = {}

        # ── AR ──
        ckpt_ar = os.path.join(CHECKPOINT_DIR, f"{bname}_AR.json")
        if os.path.exists(ckpt_ar):
            with open(ckpt_ar) as f:
                ar_results = json.load(f)
            print(f"  AR: loaded checkpoint ({n} samples)")
        else:
            print(f"  AR ({n} samples)...", end=" ", flush=True)
            ar_results = []
            for i in range(n):
                r = run_ar(target, tokenizer, prompts[i])
                q = compute_composite_sq(r["text"], refs[i], stype)
                ar_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "wall": round(r["wall"], 3),
                    "gen_len": r["gen_len"], "text": r["text"], "sp": 1.0, **q
                })
                if (i+1) % 20 == 0:
                    print(f"{i+1}...", end=" ", flush=True)
            with open(ckpt_ar, "w") as f:
                json.dump(ar_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[bname]["AR"] = ar_results
        ar_tps_list = [r["tps"] for r in ar_results]

        # ── GSD (Greedy SD) ──
        ckpt_gsd = os.path.join(CHECKPOINT_DIR, f"{bname}_GSD.json")
        if os.path.exists(ckpt_gsd):
            with open(ckpt_gsd) as f:
                gsd_results = json.load(f)
            print(f"  GSD: loaded checkpoint")
        else:
            print(f"  GSD ({n} samples)...", end=" ", flush=True)
            gsd_results = []
            for i in range(n):
                r = run_gsd(target, draft, tokenizer, prompts[i], stype)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                gsd_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "sp": round(sp, 3),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4),
                    "text": r["text"], **q
                })
                if (i+1) % 20 == 0:
                    sp_mu = sum(x["sp"] for x in gsd_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_gsd, "w") as f:
                json.dump(gsd_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[bname]["GSD"] = gsd_results

        # ── Ngram (N-gram SD) ──
        ckpt_ngram = os.path.join(CHECKPOINT_DIR, f"{bname}_Ngram.json")
        if os.path.exists(ckpt_ngram):
            with open(ckpt_ngram) as f:
                ngram_results = json.load(f)
            print(f"  Ngram: loaded checkpoint")
        else:
            print(f"  Ngram ({n} samples)...", end=" ", flush=True)
            ngram_results = []
            for i in range(n):
                r = run_ngram(target, tokenizer, prompts[i])
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                ngram_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "sp": round(sp, 3),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4),
                    "text": r["text"], **q
                })
                if (i+1) % 20 == 0:
                    sp_mu = sum(x["sp"] for x in ngram_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_ngram, "w") as f:
                json.dump(ngram_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[bname]["Ngram"] = ngram_results

        # ── FLY ──
        ckpt_fly = os.path.join(CHECKPOINT_DIR, f"{bname}_FLY.json")
        if os.path.exists(ckpt_fly):
            with open(ckpt_fly) as f:
                fly_results = json.load(f)
            print(f"  FLY: loaded checkpoint")
        else:
            print(f"  FLY ({n} samples)...", end=" ", flush=True)
            fly_results = []
            for i in range(n):
                r = run_fly(target, draft, tokenizer, prompts[i])
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                fly_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "sp": round(sp, 3),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "text": r["text"], **q
                })
                if (i+1) % 20 == 0:
                    sp_mu = sum(x["sp"] for x in fly_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_fly, "w") as f:
                json.dump(fly_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[bname]["FLY"] = fly_results

        # ── TASD ──
        ckpt_tasd = os.path.join(CHECKPOINT_DIR, f"{bname}_TASD.json")
        if os.path.exists(ckpt_tasd):
            with open(ckpt_tasd) as f:
                tasd_results = json.load(f)
            print(f"  TASD: loaded checkpoint")
        else:
            print(f"  TASD ({n} samples)...", end=" ", flush=True)
            tasd_results = []
            for i in range(n):
                r = run_tasd(target, draft, tokenizer, prompts[i], stype, enable_fb=False, fb_guarded=False)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                tasd_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "sp": round(sp, 3),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4), "repair": r["repair"],
                    "guard_trig": r["guard_trig"], "trim": r["trim"],
                    "fb_count": 0, "fb_tokens": 0,
                    "text": r["text"], **q
                })
                if (i+1) % 20 == 0:
                    sp_mu = sum(x["sp"] for x in tasd_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_tasd, "w") as f:
                json.dump(tasd_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[bname]["TASD"] = tasd_results

        # ── TASD-FG ──
        ckpt_tasdfg = os.path.join(CHECKPOINT_DIR, f"{bname}_TASDFG.json")
        if os.path.exists(ckpt_tasdfg):
            with open(ckpt_tasdfg) as f:
                tasdfg_results = json.load(f)
            print(f"  TASD-FG: loaded checkpoint")
        else:
            print(f"  TASD-FG ({n} samples)...", end=" ", flush=True)
            tasdfg_results = []
            for i in range(n):
                r = run_tasd(target, draft, tokenizer, prompts[i], stype, enable_fb=True, fb_guarded=True)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                tasdfg_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "sp": round(sp, 3),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4), "repair": r["repair"],
                    "guard_trig": r["guard_trig"], "trim": r["trim"],
                    "fb_count": r["fb_count"], "fb_tokens": r["fb_tokens"],
                    "text": r["text"], **q
                })
                if (i+1) % 20 == 0:
                    sp_mu = sum(x["sp"] for x in tasdfg_results)/(i+1)
                    fb_total = sum(x["fb_count"] for x in tasdfg_results)
                    print(f"{i+1}(sp={sp_mu:.2f}x,fb={fb_total})...", end=" ", flush=True)
            with open(ckpt_tasdfg, "w") as f:
                json.dump(tasdfg_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[bname]["TASD-FG"] = tasdfg_results

        # ── Aggregate ──
        methods = ["AR", "GSD", "Ngram", "FLY", "TASD", "TASD-FG"]
        aggregate[bname] = {"n": n}
        for m in methods:
            data = all_data[bname].get(m, [])
            if not data:
                continue
            sps = [x["sp"] for x in data]
            sp_mean = sum(sps)/len(sps)
            sp_med = sorted(sps)[len(sps)//2]
            below = sum(1 for s in sps if s < 1.0)
            sq_r = sum(x.get("sq_r", 0) for x in data)/len(data)
            sq_s = sum(x.get("sq_s", 0) for x in data)/len(data)
            sq = sum(x.get("composite_sq",0) for x in data)/len(data)
            off = sum(x.get("off_structure_rate",0) for x in data)/len(data)
            rep = sum(x.get("repetition_rate",0) for x in data)/len(data)
            trunc = sum(1 for x in data if x.get("is_truncated",False))/len(data)
            fb = sum(x.get("fb_count",0) for x in data)
            sps_sorted = sorted(sps)
            w10 = sum(sps_sorted[:max(1,len(sps_sorted)//10)])/max(1,len(sps_sorted)//10)
            aggregate[bname][m] = {
                "sp_avg": round(sp_mean, 3), "sp_median": round(sp_med, 3),
                "below": below, "sq_r": round(sq_r, 4), "sq_s": round(sq_s, 4),
                "sq_avg": round(sq, 4),
                "off_str": round(off, 4), "rep_rate": round(rep, 4),
                "truncation": round(trunc, 4), "fb_count": fb,
                "worst10": round(w10, 3),
            }

        print(f"\n  {bname} Summary:")
        for m in methods:
            if m in aggregate[bname]:
                a = aggregate[bname][m]
                print(f"    {m:10s}: sp={a['sp_avg']:.3f}x  below={a['below']}  sq={a['sq_avg']:.4f}  off={a['off_str']:.4f}")

    # ── Save ──
    output = {
        "config": {
            "target": "meta-llama/Llama-3.1-8B-Instruct",
            "draft": "meta-llama/Llama-3.2-1B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS,
            "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "enable_failure_aware_fallback": True,
            "fallback_guarded": True, "guard_calibrated": True,
        },
        "per_benchmark": aggregate,
        "per_sample": all_data,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    # ── Generate MD Report ──
    write_md_report(output)
    print(f"Saved {OUT_MD}")
    print("Done!")


def write_md_report(output):
    agg = output["per_benchmark"]
    cfg = output["config"]
    bnames = [b[0] for b in BENCHMARKS]
    methods = ["AR", "GSD", "Ngram", "FLY", "TASD", "TASD-FG"]
    labels = {"AR": "AR", "GSD": "Greedy SD", "Ngram": "N-gram SD",
              "FLY": "Official FLY", "TASD": "TASD", "TASD-FG": "TASD-FG"}

    with open(OUT_MD, "w") as f:
        f.write("# LLaMA 6×80 Full Generalization Experiment\n\n")
        f.write(f"**Target**: {cfg['target']}  |  **Draft**: {cfg['draft']}\n")
        f.write(f"**Config**: max_new_tokens={cfg['max_new_tokens']}, draft_len={cfg['draft_len']}, draft_blocks={cfg['draft_blocks']}, top_k_accept={cfg['top_k_accept']}\n\n")
        f.write("**TASD-FG**: enable_failure_aware_fallback=True, fallback_guarded=True, guard_calibrated=True\n\n")

        # Overall
        f.write("## Overall Results (240 samples)\n\n")
        f.write("| Method | Speedup | Below | Worst-10 | SQ-R | SQ-S | Off-Str | FB |\n")
        f.write("|--------|:-------:|:-----:|:--------:|:----:|:----:|:-------:|:--:|\n")

        overall = {}
        for m in methods:
            sps = [agg[bn][m]["sp_avg"] for bn in bnames if m in agg[bn]]
            sq_rs = [agg[bn][m]["sq_r"] for bn in bnames if m in agg[bn]]
            sq_ss = [agg[bn][m]["sq_s"] for bn in bnames if m in agg[bn]]
            sqs = [agg[bn][m]["sq_avg"] for bn in bnames if m in agg[bn]]
            offs = [agg[bn][m]["off_str"] for bn in bnames if m in agg[bn]]
            w10s = [agg[bn][m]["worst10"] for bn in bnames if m in agg[bn]]
            belows = [agg[bn][m]["below"] for bn in bnames if m in agg[bn]]
            fbs = [agg[bn][m]["fb_count"] for bn in bnames if m in agg[bn]]

            sp_mean = sum(sps)/len(sps)
            sq_r_mean = sum(sq_rs)/len(sq_rs)
            sq_s_mean = sum(sq_ss)/len(sq_ss)
            sq_mean = sum(sqs)/len(sqs)
            off_mean = sum(offs)/len(offs)
            w10_mean = sum(w10s)/len(w10s)
            below_total = sum(belows)
            fb_total = sum(fbs)

            overall[m] = {"sp_mean": sp_mean, "below": below_total,
                         "sq_r": sq_r_mean, "sq_s": sq_s_mean, "sq": sq_mean,
                         "off_str": off_mean, "worst10": w10_mean, "fb": fb_total}

            bold = "**" if m in ("TASD", "TASD-FG") else ""
            fb_str = str(fb_total) if m == "TASD-FG" else "-"
            f.write(f"| {bold}{labels[m]}{bold} | {bold}{sp_mean:.3f}x{bold} | {below_total}/480 | {w10_mean:.3f}x | {sq_r_mean:.4f} | {sq_s_mean:.4f} | {off_mean:.4f} | {fb_str} |\n")
        f.write("\n")

        # Per-benchmark
        f.write("## Per-Benchmark Results\n\n")
        for bn in bnames:
            f.write(f"### {bn} (80)\n\n")
            f.write("| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |\n")
            f.write("|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|\n")
            for m in methods:
                if m not in agg[bn]:
                    continue
                a = agg[bn][m]
                bold = "**" if m in ("TASD", "TASD-FG") else ""
                fb_str = str(a["fb_count"]) if m == "TASD-FG" else "-"
                f.write(f"| {bold}{labels[m]}{bold} | {bold}{a['sp_avg']:.3f}x{bold} | {a['below']} | {a['sq_r']:.4f} | {a['sq_s']:.4f} | {a['off_str']:.4f} | {fb_str} |\n")
            f.write("\n")

        # Comparison with Qwen
        f.write("## Comparison with Qwen Results\n\n")
        f.write("| Model | TASD sp | TASD below | TASD-FG sp | TASD-FG below |\n")
        f.write("|-------|:-------:|:----------:|:----------:|:-------------:|\n")
        tasd_sp = overall["TASD"]["sp_mean"]
        tasd_below = overall["TASD"]["below"]
        tasdfg_sp = overall["TASD-FG"]["sp_mean"]
        tasdfg_below = overall["TASD-FG"]["below"]
        f.write(f"| LLaMA-8B | {tasd_sp:.3f}x | {tasd_below}/480 | {tasdfg_sp:.3f}x | {tasdfg_below}/480 |\n")
        f.write(f"| Qwen-14B | 1.978x | 9/480 | 2.004x | 3/480 |\n\n")

        # Pass/Fail
        f.write("## Pass/Fail Criteria\n\n")
        f.write("| Criterion | Result | Note |\n")
        f.write("|-----------|:------:|------|\n")

        ok1 = tasdfg_sp >= 1.5
        f.write(f"| TASD-FG sp >= 1.5x | {'PASS' if ok1 else 'FAIL'} | {tasdfg_sp:.3f}x |\n")

        ok2 = tasdfg_below <= 10
        f.write(f"| TASD-FG below <= 10 | {'PASS' if ok2 else 'FAIL'} | {tasdfg_below}/480 |\n")

        ok3 = tasdfg_sp >= tasd_sp * 0.95
        f.write(f"| TASD-FG >= TASD×0.95 | {'PASS' if ok3 else 'FAIL'} | {tasd_sp:.3f} → {tasdfg_sp:.3f} |\n")

        ok4 = overall["TASD-FG"]["sq"] >= overall["TASD"]["sq"] - 0.03
        f.write(f"| SQ >= TASD-0.03 | {'PASS' if ok4 else 'FAIL'} | {overall['TASD']['sq']:.4f} → {overall['TASD-FG']['sq']:.4f} |\n")

        all_ok = ok1 and ok2 and ok3 and ok4
        f.write(f"\n**Overall**: {'ALL PASS — Generalization confirmed' if all_ok else 'SOME FAIL — Investigate'}\n\n")

        f.write("## Data\n\n")
        f.write(f"- `{OUT_JSON}`\n")
        f.write(f"- `{CHECKPOINT_DIR}/`\n")


if __name__ == "__main__":
    main()
