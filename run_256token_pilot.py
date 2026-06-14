#!/usr/bin/env python3
"""256-token pilot: 3 benchmarks x 20 samples. AR, FLY, TASD, TASD-FG."""
import json, os, sys, time, logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256

BENCHMARKS = [
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
]
SHORT = ["dict_config","openmmlab","pipeline"]
N_SAMPLES = 20

OUT_JSON = "results/qwen_256token_pilot_3x20.json"
OUT_MD = "results/qwen_256token_pilot_3x20.md"

CHECKPOINT_DIR = "results/checkpoints_256token"
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

    # FLY
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
        print(f"  AR...", end=" ", flush=True)
        ar_results = [run_ar(target, tokenizer, p) for p in prompts]
        ar_tps = [r["tps"] for r in ar_results]
        for i, r in enumerate(ar_results):
            q = compute_composite_sq(r["text"], refs[i], stype)
            r.update(q)
            r["name"] = names[i]
            r["sp"] = 1.0
        all_data[sn]["AR"] = ar_results
        print(f"done (tps={sum(ar_tps)/n:.1f})")

        for method, enable_fb, fb_guarded, label in [
            ("TASD", False, False, "TASD"),
            ("TASD-FG", True, True, "TASD-FG"),
        ]:
            ckpt = os.path.join(CHECKPOINT_DIR, f"{sn}_{label}_256.json")
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
                        sp_mu = sum(x["sp"] for x in res)/(i+1)
                        print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
                with open(ckpt, "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print(f"done")
            all_data[sn][label] = res

        # ── FLY ──
        ckpt = os.path.join(CHECKPOINT_DIR, f"{sn}_FLY_256.json")
        if os.path.exists(ckpt):
            with open(ckpt) as f:
                res = json.load(f)
            print(f"  FLY: loaded checkpoint")
        else:
            print(f"  FLY...", end=" ", flush=True)
            res = []
            for i in range(n):
                r = run_fly(target, draft, tokenizer, prompts[i], FLy_mod, fly_logger, FLY_ARGS)
                sp = r["tps"] / ar_tps[i] if ar_tps[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                res.append({"name": names[i], "sp": round(sp,3), "tps": round(r["tps"],2),
                            "wall": round(r["wall"],3), "gen_len": r["gen_len"],
                            "accept": 0, "repair": 0, "guard_trig": 0, "trim": 0,
                            "fb_count": 0, "fb_tokens": 0, "text": r["text"], **q})
                if (i+1) % 5 == 0:
                    sp_mu = sum(x["sp"] for x in res)/(i+1)
                    print(f"{i+1}(sp={sp_mu:.2f}x)...", end=" ", flush=True)
            with open(ckpt, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"done")
        all_data[sn]["FLY"] = res

    # ── Aggregate ──
    methods = ["AR", "FLY", "TASD", "TASD-FG"]
    labels = {"AR":"AR","FLY":"Official FLY","TASD":"TASD","TASD-FG":"TASD-FG"}
    agg = {}
    for sn in SHORT:
        agg[sn] = {}
        for m in methods:
            data = all_data[sn].get(m, [])
            sps = [x["sp"] for x in data]
            sp_mean = sum(sps)/len(sps)
            sp_med = sorted(sps)[len(sps)//2]
            below = sum(1 for s in sps if s < 1.0)
            sq = sum(x.get("composite_sq",0) for x in data)/len(data)
            off = sum(x.get("off_structure_rate",0) for x in data)/len(data)
            rep = sum(x.get("repetition_rate",0) for x in data)/len(data)
            trunc = sum(1 for x in data if x.get("is_truncated",False))/len(data)
            fb = sum(x.get("fb_count",0) for x in data)
            avg_len = sum(x.get("gen_len",0) for x in data)/len(data)
            accept = sum(x.get("accept",0) for x in data)/len(data) if m not in ("AR","FLY") else None
            tps_mean = sum(x.get("tps",0) for x in data)/len(data)
            wall_mean = sum(x.get("wall",0) for x in data)/len(data)
            sps_sorted = sorted(sps)
            w10 = sum(sps_sorted[:max(1,len(sps_sorted)//10)])/max(1,len(sps_sorted)//10)
            agg[sn][m] = {"sp_mean": round(sp_mean,3), "sp_median": round(sp_med,3),
                          "below": below, "sq": round(sq,4), "off_str": round(off,4),
                          "rep_rate": round(rep,4), "truncation": round(trunc,4),
                          "fb_count": fb, "avg_gen_len": round(avg_len,1),
                          "accept_rate": round(accept,4) if accept is not None else None,
                          "tps_mean": round(tps_mean,1), "wall_mean": round(wall_mean,2),
                          "worst10": round(w10,3)}

    # Overall
    overall = {}
    for m in methods:
        sps = [agg[sn][m]["sp_mean"] for sn in SHORT]
        overall[m] = {"sp_mean": round(sum(sps)/len(sps), 3),
                      "sp_median": round(sum(agg[sn][m]["sp_median"] for sn in SHORT)/len(SHORT),3),
                      "below": sum(agg[sn][m]["below"] for sn in SHORT),
                      "sq": round(sum(agg[sn][m]["sq"] for sn in SHORT)/len(SHORT),4),
                      "off_str": round(sum(agg[sn][m]["off_str"] for sn in SHORT)/len(SHORT),4),
                      "rep_rate": round(sum(agg[sn][m]["rep_rate"] for sn in SHORT)/len(SHORT),4),
                      "truncation": round(sum(agg[sn][m]["truncation"] for sn in SHORT)/len(SHORT),4),
                      "fb_count": sum(agg[sn][m]["fb_count"] for sn in SHORT),
                      "avg_gen_len": round(sum(agg[sn][m]["avg_gen_len"] for sn in SHORT)/len(SHORT),1),
                      "accept_rate": round(sum(agg[sn][m]["accept_rate"] for sn in SHORT)/len(SHORT),4) if m not in ("AR","FLY") else None,
                      "tps_mean": round(sum(agg[sn][m]["tps_mean"] for sn in SHORT)/len(SHORT),1),
                      "wall_mean": round(sum(agg[sn][m]["wall_mean"] for sn in SHORT)/len(SHORT),2),
                      "worst10": round(sum(agg[sn][m]["worst10"] for sn in SHORT)/len(SHORT),3)}

    output = {"config": {"max_new_tokens": MAX_NEW_TOKENS, "n_per_benchmark": N_SAMPLES,
                         "target": "Qwen2.5-14B-Instruct-AWQ", "draft": "Qwen2.5-1.5B-Instruct",
                         "temperature": 0.0},
              "per_benchmark": agg, "overall": overall, "per_sample": all_data}
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── MD ──
    with open(OUT_MD, "w") as f:
        f.write("# 256-Token Pilot (3 benchmarks x 20 = 60 samples)\n\n")
        f.write(f"**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct | **max_new_tokens**: {MAX_NEW_TOKENS}\n\n")

        f.write("## Overall\n\n")
        f.write("| Method | Speedup | Median sp | Below | Worst-10 | SQ | Off-Str | Rep | Trunc | FB | Gen Len | Accept | TPS | Wall(s) |\n")
        f.write("|--------|:-------:|:---------:|:-----:|:--------:|:--:|:-------:|:---:|:-----:|:--:|:-------:|:------:|:---:|:-------:|\n")
        for m in methods:
            o = overall[m]
            bold = "**" if m in ("TASD","TASD-FG") else ""
            acc = f"{o['accept_rate']:.3f}" if o['accept_rate'] else "-"
            fb = str(o['fb_count']) if m == "TASD-FG" else "-"
            f.write(f"| {bold}{labels[m]}{bold} | {bold}{o['sp_mean']:.3f}x{bold} | {o['sp_median']:.3f}x | {o['below']}/60 | {o['worst10']:.3f}x | {o['sq']:.4f} | {o['off_str']:.4f} | {o['rep_rate']:.4f} | {o['truncation']:.4f} | {fb} | {o['avg_gen_len']:.0f} | {acc} | {o['tps_mean']:.1f} | {o['wall_mean']:.1f} |\n")
        f.write("\n")

        f.write("## Per-Benchmark\n\n")
        for sn in SHORT:
            f.write(f"### {sn} (20)\n\n")
            f.write("| Method | Speedup | Median | Below | Worst | SQ | Off-Str | Rep | Trunc | FB | Gen Len | Accept | TPS | Wall |\n")
            f.write("|--------|:-------:|:------:|:-----:|:-----:|:--:|:-------:|:---:|:-----:|:--:|:-------:|:------:|:---:|:----:|\n")
            for m in methods:
                a = agg[sn][m]
                bold = "**" if m in ("TASD","TASD-FG") else ""
                acc = f"{a['accept_rate']:.3f}" if a['accept_rate'] else "-"
                fb = str(a['fb_count']) if m == "TASD-FG" else "-"
                f.write(f"| {bold}{labels[m]}{bold} | {bold}{a['sp_mean']:.3f}x{bold} | {a['sp_median']:.3f}x | {a['below']} | {a['worst10']:.3f}x | {a['sq']:.4f} | {a['off_str']:.4f} | {a['rep_rate']:.4f} | {a['truncation']:.4f} | {fb} | {a['avg_gen_len']:.0f} | {acc} | {a['tps_mean']:.1f} | {a['wall_mean']:.1f} |\n")
            f.write("\n")

        # Check criteria
        f.write("## Pass/Fail\n\n")
        f.write("| Criterion | Result | Note |\n")
        f.write("|-----------|:------:|------|\n")
        ok1 = overall["TASD-FG"]["sp_mean"] >= 2.0
        f.write(f"| TASD-FG sp >= 2.0x | {'PASS' if ok1 else 'FAIL'} | {overall['TASD-FG']['sp_mean']:.3f}x |\n")
        ok2 = overall["TASD-FG"]["below"] <= overall["TASD"]["below"] + 1
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
