#!/usr/bin/env python3
"""
256-Token Extended Experiment: 3 benchmarks × 40 samples.

Target: Qwen2.5-14B-Instruct-AWQ
Draft: Qwen2.5-1.5B-Instruct
Config: max_new_tokens=256

Methods: AR, Official FLY, TASD, TASD-FG
Goal: TASD-FG >= 2.0x on 256-token completions
"""
import json, os, sys, time, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.ngram_sd_decode import ngram_sd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256

TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

BENCHMARKS = [
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
]
SHORT = ["dict_config", "openmmlab", "pipeline"]
N_SAMPLES = 40

CHECKPOINT_DIR = "results/checkpoints_256token_extended"
OUT_JSON = "results/qwen_256token_extended_3x40.json"
OUT_MD = "results/qwen_256token_extended_3x40.md"

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
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=False, enable_relaxed_accept=False,
                    **TASD_COMMON)
    s = r["stats"]
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "gen_len": s["generated_length"], "text": r["generated_text"],
            "accept": s["accept_rate"]}


def run_ngram(target, tokenizer, prompt):
    r = ngram_sd_decode(target, tokenizer, prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        ngram_min=3, ngram_max=8, max_draft_tokens=16)
    return {"wall": r.get("wall_time", 0), "tps": r["tokens_per_second"],
            "gen_len": r.get("stats", {}).get("generated_length", 0),
            "text": r["generated_text"],
            "accept": r.get("stats", {}).get("accept_rate", 0)}


def run_fly(target, draft, tokenizer, prompt, fly_mod, fly_logger, fly_args):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    spd = fly_mod.SPDGenerate(draft_model=draft, target_model=target,
                              tokenizer=tokenizer, cuslog=fly_logger, spd_args=fly_args)
    t0 = time.time()
    full = spd.generate_chunks(inp.input_ids, temperature=0.0)
    wall = time.time() - t0
    gids = full[0][inp.input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gids, skip_special_tokens=True)
    tps = len(gids) / wall if wall > 0 else 0
    return {"wall": wall, "tps": tps, "gen_len": len(gids), "text": text,
            "accept": 0, "repair": 0, "guard_trig": 0, "trim": 0, "fb_count": 0, "fb_tokens": 0}


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

    # FLY setup
    import importlib.util
    fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
    spec_fly = importlib.util.spec_from_file_location("FLy", fly_path)
    FLy_mod = importlib.util.module_from_spec(spec_fly)
    spec_fly.loader.exec_module(FLy_mod)
    fly_logger = logging.getLogger("FLy")
    fly_logger.setLevel(logging.WARNING)
    FLY_ARGS = {"k": 15, "total_gen_tok": MAX_NEW_TOKENS, "enable_fly": True,
                "win_len": 6, "entropy_thre": 0.3, "use_ngram": True,
                "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
                "verbose": False, "abla_no_window": False, "enable_statistics": True}

    all_data = {}

    for bn, data_file, stype in BENCHMARKS:
        sn = SHORT[BENCHMARKS.index((bn, data_file, stype))]

        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:N_SAMPLES]]
        prompts = [s["prompt"] for s in samples]
        refs = [s.get("reference", "") for s in samples]
        names = [s["name"] for s in samples]
        n = len(samples)

        all_data[sn] = {"n": n, "names": names}

        # ── AR ──
        print(f"\n{'='*50}\n{sn} (n={n})\n{'='*50}")
        ckpt_ar = os.path.join(CHECKPOINT_DIR, f"{sn}_AR_256.json")
        if os.path.exists(ckpt_ar):
            with open(ckpt_ar) as f:
                ar_results = json.load(f)
            print(f"  AR: loaded checkpoint")
        else:
            print(f"  AR...", end=" ", flush=True)
            ar_results = []
            for i in range(n):
                r = run_ar(target, tokenizer, prompts[i])
                q = compute_composite_sq(r["text"], refs[i], stype)
                ar_results.append({
                    "name": names[i], "tps": round(r["tps"], 2), "wall": round(r["wall"], 3),
                    "gen_len": r["gen_len"], "text": r["text"], "sp": 1.0, **q
                })
                if (i+1) % 10 == 0:
                    print(f"{i+1}...", end=" ", flush=True)
            with open(ckpt_ar, "w") as f:
                json.dump(ar_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["AR"] = ar_results
        ar_tps = [r["tps"] for r in ar_results]

        # ── GSD (Greedy SD) ──
        ckpt_gsd = os.path.join(CHECKPOINT_DIR, f"{sn}_GSD_256.json")
        if os.path.exists(ckpt_gsd):
            with open(ckpt_gsd) as f:
                gsd_results = json.load(f)
            print(f"  GSD: loaded checkpoint")
        else:
            print(f"  GSD...", end=" ", flush=True)
            gsd_results = []
            for i in range(n):
                r = run_gsd(target, draft, tokenizer, prompts[i], stype)
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                gsd_results.append({
                    "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4),
                    "text": r["text"], **q
                })
                if (i+1) % 10 == 0:
                    sp_mu = sum(x["sp"] for x in gsd_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_gsd, "w") as f:
                json.dump(gsd_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["GSD"] = gsd_results

        # ── Ngram (N-gram SD) ──
        ckpt_ngram = os.path.join(CHECKPOINT_DIR, f"{sn}_Ngram_256.json")
        if os.path.exists(ckpt_ngram):
            with open(ckpt_ngram) as f:
                ngram_results = json.load(f)
            print(f"  Ngram: loaded checkpoint")
        else:
            print(f"  Ngram...", end=" ", flush=True)
            ngram_results = []
            for i in range(n):
                r = run_ngram(target, tokenizer, prompts[i])
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                ngram_results.append({
                    "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4),
                    "text": r["text"], **q
                })
                if (i+1) % 10 == 0:
                    sp_mu = sum(x["sp"] for x in ngram_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_ngram, "w") as f:
                json.dump(ngram_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["Ngram"] = ngram_results

        # ── FLY ──
        ckpt_fly = os.path.join(CHECKPOINT_DIR, f"{sn}_FLY_256.json")
        if os.path.exists(ckpt_fly):
            with open(ckpt_fly) as f:
                fly_results = json.load(f)
            print(f"  FLY: loaded checkpoint")
        else:
            print(f"  FLY...", end=" ", flush=True)
            fly_results = []
            for i in range(n):
                r = run_fly(target, draft, tokenizer, prompts[i], FLy_mod, fly_logger, FLY_ARGS)
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                fly_results.append({
                    "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "text": r["text"], **q
                })
                if (i+1) % 10 == 0:
                    sp_mu = sum(x["sp"] for x in fly_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_fly, "w") as f:
                json.dump(fly_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["FLY"] = fly_results

        # ── TASD ──
        ckpt_tasd = os.path.join(CHECKPOINT_DIR, f"{sn}_TASD_256.json")
        if os.path.exists(ckpt_tasd):
            with open(ckpt_tasd) as f:
                tasd_results = json.load(f)
            print(f"  TASD: loaded checkpoint")
        else:
            print(f"  TASD...", end=" ", flush=True)
            tasd_results = []
            for i in range(n):
                r = run_tasd(target, draft, tokenizer, prompts[i], stype, enable_fb=False, fb_guarded=False)
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                tasd_results.append({
                    "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4), "repair": r["repair"],
                    "guard_trig": r["guard_trig"], "trim": r["trim"],
                    "fb_count": 0, "fb_tokens": 0,
                    "text": r["text"], **q
                })
                if (i+1) % 10 == 0:
                    sp_mu = sum(x["sp"] for x in tasd_results)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt_tasd, "w") as f:
                json.dump(tasd_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["TASD"] = tasd_results

        # ── TASD-FG ──
        ckpt_tasdfg = os.path.join(CHECKPOINT_DIR, f"{sn}_TASDFG_256.json")
        if os.path.exists(ckpt_tasdfg):
            with open(ckpt_tasdfg) as f:
                tasdfg_results = json.load(f)
            print(f"  TASD-FG: loaded checkpoint")
        else:
            print(f"  TASD-FG...", end=" ", flush=True)
            tasdfg_results = []
            for i in range(n):
                r = run_tasd(target, draft, tokenizer, prompts[i], stype, enable_fb=True, fb_guarded=True)
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                tasdfg_results.append({
                    "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3), "gen_len": r["gen_len"],
                    "accept": round(r["accept"], 4), "repair": r["repair"],
                    "guard_trig": r["guard_trig"], "trim": r["trim"],
                    "fb_count": r["fb_count"], "fb_tokens": r["fb_tokens"],
                    "text": r["text"], **q
                })
                if (i+1) % 10 == 0:
                    sp_mu = sum(x["sp"] for x in tasdfg_results)/(i+1)
                    fb_total = sum(x["fb_count"] for x in tasdfg_results)
                    print(f"{i+1}(sp={sp_mu:.2f}x,fb={fb_total})...", end=" ", flush=True)
            with open(ckpt_tasdfg, "w") as f:
                json.dump(tasdfg_results, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["TASD-FG"] = tasdfg_results

    # ── Aggregate ──
    methods = ["AR", "GSD", "Ngram", "FLY", "TASD", "TASD-FG"]
    labels = {"AR": "AR", "GSD": "Greedy SD", "Ngram": "N-gram SD",
              "FLY": "Official FLY", "TASD": "TASD", "TASD-FG": "TASD-FG"}
    agg = {}
    for sn in SHORT:
        agg[sn] = {}
        for m in methods:
            data = all_data[sn].get(m, [])
            if not data:
                continue
            sps = [x["sp"] for x in data]
            sp_mean = sum(sps)/len(sps)
            sp_med = sorted(sps)[len(sps)//2]
            below = sum(1 for s in sps if s < 1.0)
            sq_r = sum(x.get("sq_r", 0) for x in data)/len(data)
            sq_s = sum(x.get("sq_s", 0) for x in data)/len(data)
            sq = sum(x.get("composite_sq", 0) for x in data)/len(data)
            off = sum(x.get("off_structure_rate", 0) for x in data)/len(data)
            rep = sum(x.get("repetition_rate", 0) for x in data)/len(data)
            trunc = sum(1 for x in data if x.get("is_truncated", False))/len(data)
            fb = sum(x.get("fb_count", 0) for x in data)
            avg_len = sum(x.get("gen_len", 0) for x in data)/len(data)
            accept = sum(x.get("accept", 0) for x in data)/len(data) if m not in ("AR", "FLY") else None
            tps_mean = sum(x.get("tps", 0) for x in data)/len(data)
            wall_mean = sum(x.get("wall", 0) for x in data)/len(data)
            sps_sorted = sorted(sps)
            w10 = sum(sps_sorted[:max(1, len(sps_sorted)//10)])/max(1, len(sps_sorted)//10)
            agg[sn][m] = {
                "sp_mean": round(sp_mean, 3), "sp_median": round(sp_med, 3),
                "below": below, "sq_r": round(sq_r, 4), "sq_s": round(sq_s, 4),
                "sq": round(sq, 4), "off_str": round(off, 4),
                "rep_rate": round(rep, 4), "truncation": round(trunc, 4),
                "fb_count": fb, "avg_gen_len": round(avg_len, 1),
                "accept_rate": round(accept, 4) if accept is not None else None,
                "tps_mean": round(tps_mean, 1), "wall_mean": round(wall_mean, 2),
                "worst10": round(w10, 3)
            }

    # Overall
    overall = {}
    for m in methods:
        sps = [agg[sn][m]["sp_mean"] for sn in SHORT if m in agg[sn]]
        overall[m] = {
            "sp_mean": round(sum(sps)/len(sps), 3),
            "sp_median": round(sum(agg[sn][m]["sp_median"] for sn in SHORT if m in agg[sn])/len(SHORT), 3),
            "below": sum(agg[sn][m]["below"] for sn in SHORT if m in agg[sn]),
            "sq_r": round(sum(agg[sn][m]["sq_r"] for sn in SHORT if m in agg[sn])/len(SHORT), 4),
            "sq_s": round(sum(agg[sn][m]["sq_s"] for sn in SHORT if m in agg[sn])/len(SHORT), 4),
            "sq": round(sum(agg[sn][m]["sq"] for sn in SHORT if m in agg[sn])/len(SHORT), 4),
            "off_str": round(sum(agg[sn][m]["off_str"] for sn in SHORT if m in agg[sn])/len(SHORT), 4),
            "rep_rate": round(sum(agg[sn][m]["rep_rate"] for sn in SHORT if m in agg[sn])/len(SHORT), 4),
            "truncation": round(sum(agg[sn][m]["truncation"] for sn in SHORT if m in agg[sn])/len(SHORT), 4),
            "fb_count": sum(agg[sn][m]["fb_count"] for sn in SHORT if m in agg[sn]),
            "avg_gen_len": round(sum(agg[sn][m]["avg_gen_len"] for sn in SHORT if m in agg[sn])/len(SHORT), 1),
            "accept_rate": round(sum(agg[sn][m]["accept_rate"] for sn in SHORT if m in agg[sn] and agg[sn][m]["accept_rate"])/len(SHORT), 4) if m not in ("AR", "FLY") else None,
            "tps_mean": round(sum(agg[sn][m]["tps_mean"] for sn in SHORT if m in agg[sn])/len(SHORT), 1),
            "wall_mean": round(sum(agg[sn][m]["wall_mean"] for sn in SHORT if m in agg[sn])/len(SHORT), 2),
            "worst10": round(sum(agg[sn][m]["worst10"] for sn in SHORT if m in agg[sn])/len(SHORT), 3)
        }

    output = {
        "config": {
            "max_new_tokens": MAX_NEW_TOKENS, "n_per_benchmark": N_SAMPLES,
            "target": "Qwen2.5-14B-Instruct-AWQ", "draft": "Qwen2.5-1.5B-Instruct",
            "temperature": 0.0
        },
        "per_benchmark": agg, "overall": overall, "per_sample": all_data
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── MD Report ──
    with open(OUT_MD, "w") as f:
        f.write("# 256-Token Extended Experiment (3 benchmarks × 40 = 120 samples)\n\n")
        f.write(f"**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct | **max_new_tokens**: {MAX_NEW_TOKENS}\n\n")

        f.write("## Overall\n\n")
        f.write("| Method | Speedup | Median sp | Below | Worst-10 | SQ-R | SQ-S | Off-Str | Rep | Trunc | FB | Gen Len | Accept | TPS | Wall(s) |\n")
        f.write("|--------|:-------:|:---------:|:-----:|:--------:|:----:|:----:|:-------:|:---:|:-----:|:--:|:-------:|:------:|:---:|:-------:|\n")
        for m in methods:
            o = overall[m]
            bold = "**" if m in ("TASD", "TASD-FG") else ""
            acc = f"{o['accept_rate']:.3f}" if o['accept_rate'] else "-"
            fb = str(o['fb_count']) if m == "TASD-FG" else "-"
            f.write(f"| {bold}{labels[m]}{bold} | {bold}{o['sp_mean']:.3f}x{bold} | {o['sp_median']:.3f}x | {o['below']}/120 | {o['worst10']:.3f}x | {o['sq_r']:.4f} | {o['sq_s']:.4f} | {o['off_str']:.4f} | {o['rep_rate']:.4f} | {o['truncation']:.4f} | {fb} | {o['avg_gen_len']:.0f} | {acc} | {o['tps_mean']:.1f} | {o['wall_mean']:.1f} |\n")
        f.write("\n")

        f.write("## Per-Benchmark\n\n")
        for sn in SHORT:
            f.write(f"### {sn} (40)\n\n")
            f.write("| Method | Speedup | Median | Below | Worst | SQ-R | SQ-S | Off-Str | Rep | Trunc | FB | Gen Len | Accept | TPS | Wall |\n")
            f.write("|--------|:-------:|:------:|:-----:|:-----:|:----:|:----:|:-------:|:---:|:-----:|:--:|:-------:|:------:|:---:|:----:|\n")
            for m in methods:
                a = agg[sn][m]
                bold = "**" if m in ("TASD", "TASD-FG") else ""
                acc = f"{a['accept_rate']:.3f}" if a['accept_rate'] else "-"
                fb = str(a['fb_count']) if m == "TASD-FG" else "-"
                f.write(f"| {bold}{labels[m]}{bold} | {bold}{a['sp_mean']:.3f}x{bold} | {a['sp_median']:.3f}x | {a['below']} | {a['worst10']:.3f}x | {a['sq_r']:.4f} | {a['sq_s']:.4f} | {a['off_str']:.4f} | {a['rep_rate']:.4f} | {a['truncation']:.4f} | {fb} | {a['avg_gen_len']:.0f} | {acc} | {a['tps_mean']:.1f} | {a['wall_mean']:.1f} |\n")
            f.write("\n")

        # Comparison with 128-token
        f.write("## Comparison with 128-Token Results\n\n")
        f.write("| Config | TASD sp | TASD below | TASD-FG sp | TASD-FG below |\n")
        f.write("|--------|:-------:|:----------:|:----------:|:-------------:|\n")
        f.write(f"| 128-token (480 samples) | 1.978x | 9/480 | 2.004x | 3/480 |\n")
        tasd_sp = overall["TASD"]["sp_mean"]
        tasd_below = overall["TASD"]["below"]
        tasdfg_sp = overall["TASD-FG"]["sp_mean"]
        tasdfg_below = overall["TASD-FG"]["below"]
        f.write(f"| 256-token (120 samples) | {tasd_sp:.3f}x | {tasd_below}/120 | {tasdfg_sp:.3f}x | {tasdfg_below}/120 |\n\n")

        # Pass/Fail
        f.write("## Pass/Fail\n\n")
        f.write("| Criterion | Result | Note |\n")
        f.write("|-----------|:------:|------|\n")
        ok1 = overall["TASD-FG"]["sp_mean"] >= 2.0
        f.write(f"| TASD-FG sp >= 2.0x | {'PASS' if ok1 else 'FAIL'} | {overall['TASD-FG']['sp_mean']:.3f}x |\n")
        ok2 = overall["TASD-FG"]["below"] <= overall["TASD"]["below"] + 2
        f.write(f"| TASD-FG below not worse | {'PASS' if ok2 else 'FAIL'} | TASD={overall['TASD']['below']} TASD-FG={overall['TASD-FG']['below']} |\n")
        ok3 = overall["TASD-FG"]["off_str"] <= overall["TASD"]["off_str"] * 1.3
        f.write(f"| off_str ok | {'PASS' if ok3 else 'FAIL'} | TASD={overall['TASD']['off_str']:.4f} TASD-FG={overall['TASD-FG']['off_str']:.4f} |\n")
        ok4 = overall["TASD-FG"]["sq"] >= overall["TASD"]["sq"] - 0.03
        f.write(f"| SQ ok | {'PASS' if ok4 else 'FAIL'} | TASD={overall['TASD']['sq']:.4f} TASD-FG={overall['TASD-FG']['sq']:.4f} |\n")
        ok5 = overall["TASD-FG"]["worst10"] >= overall["TASD"]["worst10"]
        f.write(f"| worst-10 not worse | {'PASS' if ok5 else 'FAIL'} | TASD={overall['TASD']['worst10']:.3f}x TASD-FG={overall['TASD-FG']['worst10']:.3f}x |\n")
        all_ok = ok1 and ok2 and ok3 and ok4 and ok5
        f.write(f"\n**Overall**: {'ALL PASS' if all_ok else 'SOME FAIL'}\n")

    print(f"\nSaved {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
