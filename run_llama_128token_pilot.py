#!/usr/bin/env python3
"""LLaMA 128-token pilot: 3 benchmarks x 20 samples. AR, FLY, TASD, TASD-FG."""
import json, os, sys, time, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128

BENCHMARKS = [
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
]
N_SAMPLES = 20

OUT_JSON = "results/llama_128token_pilot_3x20.json"
OUT_MD = "results/llama_128token_pilot_3x20.md"
CHECKPOINT_DIR = "results/checkpoints_llama_128token"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)


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
    r = tasd_decode(target, draft, tokenizer, prompt, structure_type=stype,
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


def run_fly_method(target, draft, tokenizer, prompt, fly_mod, fly_logger, fly_args):
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

    # FLY (k=8 for 8B model)
    import importlib.util
    fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
    spec_fly = importlib.util.spec_from_file_location("FLy", fly_path)
    FLy_mod = importlib.util.module_from_spec(spec_fly)
    spec_fly.loader.exec_module(FLy_mod)
    fly_logger = logging.getLogger("FLy")
    fly_logger.setLevel(logging.WARNING)
    FLY_ARGS = {"k": 8, "total_gen_tok": MAX_NEW_TOKENS, "enable_fly": True,
                "win_len": 6, "entropy_thre": 0.3, "use_ngram": True,
                "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
                "verbose": False, "abla_no_window": False, "enable_statistics": True}

    all_data = {}
    METHODS_RUN = [("TASD", False, False), ("TASD-FG", True, True)]
    LABELS = {"AR":"AR","FLY":"Official FLY","TASD":"TASD","TASD-FG":"TASD-FG"}

    for sn, data_file, stype in BENCHMARKS:
        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:N_SAMPLES]]
        prompts = [s["prompt"] for s in samples]
        refs = [s.get("reference", "") for s in samples]
        names = [s["name"] for s in samples]
        n = len(samples)
        all_data[sn] = {"n": n, "names": names}

        print(f"\n{'='*50}\n{sn} (n={n})\n{'='*50}")

        # AR
        print("  AR...", end=" ", flush=True)
        ar_results = [run_ar(target, tokenizer, p) for p in prompts]
        ar_tps = [r["tps"] for r in ar_results]
        for i, r in enumerate(ar_results):
            q = compute_composite_sq(r["text"], refs[i], stype)
            r.update(q)
            r["name"] = names[i]
            r["sp"] = 1.0
        all_data[sn]["AR"] = ar_results
        print(f"done tps={sum(ar_tps)/n:.1f}")

        # TASD / TASD-FG
        for label, enable_fb, fb_guarded in METHODS_RUN:
            ckpt = os.path.join(CHECKPOINT_DIR, f"{sn}_{label}.json")
            if os.path.exists(ckpt):
                with open(ckpt) as f:
                    res = json.load(f)
                print(f"  {label}: loaded checkpoint")
            else:
                print(f"  {label}...", end=" ", flush=True)
                res = []
                for i in range(n):
                    r = run_tasd(target, draft, tokenizer, prompts[i], stype, enable_fb, fb_guarded)
                    sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                    q = compute_composite_sq(r["text"], refs[i], stype)
                    res.append({"name": names[i], "sp": round(sp,3), "tps": round(r["tps"],2),
                                "wall": round(r["wall"],3), "gen_len": r["gen_len"],
                                "accept": round(r["accept"],4), "repair": r["repair"],
                                "guard_trig": r["guard_trig"], "trim": r["trim"],
                                "fb_count": r["fb_count"], "fb_tokens": r["fb_tokens"],
                                "text": r["text"], **q})
                    if (i+1) % 5 == 0:
                        print(f"{i+1}(sp={sum(x['sp'] for x in res)/(i+1):.2f}x)...", end=" ", flush=True)
                with open(ckpt, "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print("done")
            all_data[sn][label] = res

        # FLY
        ckpt = os.path.join(CHECKPOINT_DIR, f"{sn}_FLY.json")
        if os.path.exists(ckpt):
            with open(ckpt) as f:
                res = json.load(f)
            print("  FLY: loaded checkpoint")
        else:
            print("  FLY...", end=" ", flush=True)
            res = []
            for i in range(n):
                r = run_fly_method(target, draft, tokenizer, prompts[i], FLy_mod, fly_logger, FLY_ARGS)
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                res.append({"name": names[i], "sp": round(sp,3), "tps": round(r["tps"],2),
                            "wall": round(r["wall"],3), "gen_len": r["gen_len"],
                            "accept": 0, "repair": 0, "guard_trig": 0, "trim": 0,
                            "fb_count": 0, "fb_tokens": 0, "text": r["text"], **q})
                if (i+1) % 5 == 0:
                    print(f"{i+1}(sp={sum(x['sp'] for x in res)/(i+1):.2f}x)...", end=" ", flush=True)
            with open(ckpt, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print("done")
        all_data[sn]["FLY"] = res

    # Aggregate
    methods = ["AR", "FLY", "TASD", "TASD-FG"]
    agg = {}
    for sn, _, _ in BENCHMARKS:
        agg[sn] = {}
        for m in methods:
            data = all_data[sn].get(m, [])
            sps = [x["sp"] for x in data]
            n_d = len(sps)
            sp_mean = sum(sps)/n_d
            sp_med = sorted(sps)[n_d//2]
            below = sum(1 for s in sps if s < 1.0)
            sq = sum(x.get("composite_sq",0) for x in data)/n_d
            off = sum(x.get("off_structure_rate",0) for x in data)/n_d
            rep = sum(x.get("repetition_rate",0) for x in data)/n_d
            trunc = sum(1 for x in data if x.get("is_truncated",False))/n_d
            fb = sum(x.get("fb_count",0) for x in data)
            gen_len = sum(x.get("gen_len",0) for x in data)/n_d
            accept = sum(x.get("accept",0) for x in data)/n_d if m not in ("AR","FLY") else None
            tps_m = sum(x.get("tps",0) for x in data)/n_d
            wall_m = sum(x.get("wall",0) for x in data)/n_d
            ss = sorted(sps)
            w10 = sum(ss[:max(1,n_d//10)])/max(1,n_d//10)
            agg[sn][m] = {"sp_mean": round(sp_mean,3), "sp_median": round(sp_med,3),
                          "below": below, "sq": round(sq,4), "off_str": round(off,4),
                          "rep_rate": round(rep,4), "truncation": round(trunc,4),
                          "fb_count": fb, "avg_gen_len": round(gen_len,1),
                          "accept_rate": round(accept,4) if accept else None,
                          "tps_mean": round(tps_m,1), "wall_mean": round(wall_m,2),
                          "worst10": round(w10,3)}

    overall = {}
    for m in methods:
        sn_list = [sn for sn,_,_ in BENCHMARKS]
        o = {"sp_mean": round(sum(agg[sn][m]["sp_mean"] for sn in sn_list)/len(sn_list),3),
             "sp_median": round(sum(agg[sn][m]["sp_median"] for sn in sn_list)/len(sn_list),3),
             "below": sum(agg[sn][m]["below"] for sn in sn_list),
             "sq": round(sum(agg[sn][m]["sq"] for sn in sn_list)/len(sn_list),4),
             "off_str": round(sum(agg[sn][m]["off_str"] for sn in sn_list)/len(sn_list),4),
             "rep_rate": round(sum(agg[sn][m]["rep_rate"] for sn in sn_list)/len(sn_list),4),
             "truncation": round(sum(agg[sn][m]["truncation"] for sn in sn_list)/len(sn_list),4),
             "fb_count": sum(agg[sn][m]["fb_count"] for sn in sn_list),
             "avg_gen_len": round(sum(agg[sn][m]["avg_gen_len"] for sn in sn_list)/len(sn_list),1),
             "accept_rate": round(sum(agg[sn][m]["accept_rate"] for sn in sn_list)/len(sn_list),4) if m not in ("AR","FLY") else None,
             "tps_mean": round(sum(agg[sn][m]["tps_mean"] for sn in sn_list)/len(sn_list),1),
             "wall_mean": round(sum(agg[sn][m]["wall_mean"] for sn in sn_list)/len(sn_list),2),
             "worst10": round(sum(agg[sn][m]["worst10"] for sn in sn_list)/len(sn_list),3)}
        overall[m] = o

    output = {"config": {"max_new_tokens": MAX_NEW_TOKENS, "n_per_benchmark": N_SAMPLES,
                         "target": "Meta-Llama-3.1-8B-Instruct", "draft": "Llama-3.2-1B-Instruct",
                         "temperature": 0.0},
              "per_benchmark": agg, "overall": overall, "per_sample": all_data}
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # MD
    with open(OUT_MD, "w") as f:
        f.write("# LLaMA 128-Token Pilot (3 benchmarks x 20 = 60 samples)\n\n")
        f.write(f"**Target**: Meta-Llama-3.1-8B-Instruct | **Draft**: Llama-3.2-1B-Instruct | **max_new_tokens**: {MAX_NEW_TOKENS}\n\n")

        f.write("## Overall\n\n")
        f.write("| Method | Speedup | Median | Below | Worst-10 | SQ | Off-Str | Rep | Trunc | FB | Gen Len | Accept | TPS | Wall |\n")
        f.write("|--------|:-------:|:------:|:-----:|:--------:|:--:|:-------:|:---:|:-----:|:--:|:-------:|:------:|:---:|:----:|\n")
        for m in methods:
            o = overall[m]
            bold = "**" if m in ("TASD","TASD-FG") else ""
            acc = f"{o['accept_rate']:.3f}" if o['accept_rate'] else "-"
            fb = str(o['fb_count']) if m == "TASD-FG" else "-"
            f.write(f"| {bold}{LABELS[m]}{bold} | {bold}{o['sp_mean']:.3f}x{bold} | {o['sp_median']:.3f}x | {o['below']}/60 | {o['worst10']:.3f}x | {o['sq']:.4f} | {o['off_str']:.4f} | {o['rep_rate']:.4f} | {o['truncation']:.4f} | {fb} | {o['avg_gen_len']:.0f} | {acc} | {o['tps_mean']:.1f} | {o['wall_mean']:.1f} |\n")
        f.write("\n")

        for sn, _, _ in BENCHMARKS:
            f.write(f"### {sn} (20)\n\n")
            f.write("| Method | Speedup | Median | Below | Worst | SQ | Off-Str | Rep | Trunc | FB | Gen Len | Accept | TPS | Wall |\n")
            f.write("|--------|:-------:|:------:|:-----:|:-----:|:--:|:-------:|:---:|:-----:|:--:|:-------:|:------:|:---:|:----:|\n")
            for m in methods:
                a = agg[sn][m]
                bold = "**" if m in ("TASD","TASD-FG") else ""
                acc = f"{a['accept_rate']:.3f}" if a['accept_rate'] else "-"
                fb = str(a['fb_count']) if m == "TASD-FG" else "-"
                f.write(f"| {bold}{LABELS[m]}{bold} | {bold}{a['sp_mean']:.3f}x{bold} | {a['sp_median']:.3f}x | {a['below']} | {a['worst10']:.3f}x | {a['sq']:.4f} | {a['off_str']:.4f} | {a['rep_rate']:.4f} | {a['truncation']:.4f} | {fb} | {a['avg_gen_len']:.0f} | {acc} | {a['tps_mean']:.1f} | {a['wall_mean']:.1f} |\n")
            f.write("\n")

    print(f"\nSaved {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
