"""
Qwen 3×20 Sanity: AR / Greedy SD / N-gram SD / Official FLY / TASD calibrated.

Target: Qwen2.5-14B-Instruct-AWQ
Draft:  Qwen2.5-1.5B-Instruct

Official FLY params (from FLy_Llama3_70b.json, default k=15 for 14B):
  spd_k=15, win_len=6, entropy_thre=0.3, use_ngram=true,
  max_ngram_size=4, num_ngram_pred_tokens=6
"""

import json, os, sys, time, logging, importlib.util
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.ngram_sd_decode import ngram_sd_decode

# ─── Paths ────────────────────────────────────────────────────────────────
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLES_PER = 20

BENCHMARKS = {
    "dict_config":       ("data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    "openmmlab_config":  ("data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    "pipeline_stage_config": ("data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
}

# ─── Official FLY import (avoid vllm deps) ────────────────────────────────
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

# ─── FLY args (Qwen 14B: k=15 as 70B default) ─────────────────────────────
FLY_K15 = {
    "k": 15, "total_gen_tok": MAX_NEW_TOKENS,
    "enable_fly": True, "win_len": 6, "entropy_thre": 0.3,
    "use_ngram": True, "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
    "verbose": False, "abla_no_window": False, "enable_statistics": True,
}

# ─── GSD (guard off), TASD (calibrated guard) ─────────────────────────────
TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

# ─── Metrics ──────────────────────────────────────────────────────────────

def compute_sq(pred, ref):
    chars = set("{}[]():,=\n")
    p = [c for c in pred if c in chars]
    r = [c for c in ref if c in chars]
    if not r: return 1.0
    return min(sum(1 for c in p if c in r) / len(r), 1.0)

def compute_off_structure(text):
    lines = text.split("\n") if text else []
    if not lines: return 0.0
    kw = {"def ", "class ", "import ", "from "}
    return sum(1 for l in lines if any(l.strip().startswith(k) for k in kw)) / len(lines)

# ─── Decode methods ───────────────────────────────────────────────────────

def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    wall = time.time() - t0
    gen_ids = out[0][inp.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"tps": len(out[0]) / wall, "text": text, "gen_len": len(gen_ids), "wall": wall}

def run_gsd(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=False, enable_relaxed_accept=False,
                    **TASD_COMMON)
    return {"tps": r["tokens_per_second"], "text": r["generated_text"],
            "accept": r["stats"]["accept_rate"]}

def run_ngram(target, tokenizer, prompt):
    r = ngram_sd_decode(target, tokenizer, prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        ngram_min=3, ngram_max=8, max_draft_tokens=16)
    return {"tps": r["tokens_per_second"], "text": r["generated_text"],
            "accept": r.get("stats", {}).get("accept_rate", 0),
            "draft_avg": r.get("stats", {}).get("avg_draft_len", 0)}

def run_fly(target, draft, tokenizer, prompt, fly_args, logger):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    spd_gen = SPDGenerate(draft_model=draft, target_model=target,
                          tokenizer=tokenizer, cuslog=logger, spd_args=fly_args)
    torch.cuda.synchronize()
    t0 = time.time()
    full_ids = spd_gen.generate_chunks(input_ids, temperature=0.0)
    torch.cuda.synchronize()
    wall = time.time() - t0
    gen_ids = full_ids[0][prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_acc = spd_gen.num_accepted_tokens.item() if spd_gen._counter_inited else 0
    n_emitted = spd_gen.num_emitted_tokens.item() if spd_gen._counter_inited else len(gen_ids)
    mat = n_acc / n_emitted if n_emitted > 0 else 0
    ngram_accs = getattr(spd_gen, 'debug_ngram_accept_num', [])
    mean_ngram = sum(ngram_accs) / len(ngram_accs) if ngram_accs else 0

    return {"tps": full_ids.shape[1] / wall, "text": text, "gen_len": len(gen_ids),
            "wall": wall, "mat": mat, "ngram_accept": mean_ngram}

def run_tasd(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    **TASD_COMMON)
    stats = r["stats"]
    return {"tps": r["tokens_per_second"], "text": r["generated_text"],
            "accept": stats["accept_rate"], "repair": stats.get("repair_count", 0),
            "guard_trig": stats.get("guard_trigger_count", 0),
            "trim": stats.get("trim_count", 0),
            "off_str": compute_off_structure(r["generated_text"])}

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading Qwen models...")
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

    # Quiet FLY logger
    fly_logger = logging.getLogger("fly")
    fly_logger.setLevel(logging.WARNING)
    if not fly_logger.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        fly_logger.addHandler(h)

    all_bench = {}

    for bname, (data_file, stype) in BENCHMARKS.items():
        print(f"{'='*65}")
        print(f"Benchmark: {bname}  |  stype: {stype}")
        print(f"{'='*65}")

        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:SAMPLES_PER]]

        results = defaultdict(list)

        # ── AR ──
        print("  AR...")
        for i, s in enumerate(samples):
            r = run_ar(target, tokenizer, s["prompt"])
            sq = compute_sq(r["text"], s.get("reference", ""))
            results["AR"].append({"name": s["name"], "ar_tps": round(r["tps"], 2),
                                  "sq": round(sq, 4)})
        mean_ar = sum(r["ar_tps"] for r in results["AR"]) / len(results["AR"])
        print(f"    mean TPS: {mean_ar:.1f}")

        # ── Greedy SD ──
        print("  Greedy SD...")
        for i, s in enumerate(samples):
            r = run_gsd(target, draft, tokenizer, s["prompt"], stype)
            ar_tps = results["AR"][i]["ar_tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], s.get("reference", ""))
            results["GSD"].append({"name": s["name"], "sp": round(sp, 3),
                                   "tps": round(r["tps"], 2), "sq": round(sq, 4),
                                   "accept": round(r["accept"], 4)})
        gsd_sp = sum(r["sp"] for r in results["GSD"]) / len(results["GSD"])
        gsd_below = sum(1 for r in results["GSD"] if r["sp"] < 1.0)
        print(f"    sp={gsd_sp:.3f}x below1={gsd_below}")

        # ── N-gram SD ──
        print("  N-gram SD...")
        for i, s in enumerate(samples):
            r = run_ngram(target, tokenizer, s["prompt"])
            ar_tps = results["AR"][i]["ar_tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], s.get("reference", ""))
            results["Ngram"].append({"name": s["name"], "sp": round(sp, 3),
                                     "tps": round(r["tps"], 2), "sq": round(sq, 4),
                                     "accept": round(r["accept"], 4),
                                     "draft_avg": round(r["draft_avg"], 1)})
        ng_sp = sum(r["sp"] for r in results["Ngram"]) / len(results["Ngram"])
        ng_below = sum(1 for r in results["Ngram"] if r["sp"] < 1.0)
        print(f"    sp={ng_sp:.3f}x below1={ng_below}")

        # ── Official FLY (k=15) ──
        print("  Official FLY (k=15)...")
        for i, s in enumerate(samples):
            r = run_fly(target, draft, tokenizer, s["prompt"], FLY_K15, fly_logger)
            ar_tps = results["AR"][i]["ar_tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], s.get("reference", ""))
            results["FLY_k15"].append({"name": s["name"], "sp": round(sp, 3),
                                       "tps": round(r["tps"], 2), "sq": round(sq, 4),
                                       "mat": round(r["mat"], 2),
                                       "ngram_acc": round(r["ngram_accept"], 1)})
        fly_sp = sum(r["sp"] for r in results["FLY_k15"]) / len(results["FLY_k15"])
        fly_below = sum(1 for r in results["FLY_k15"] if r["sp"] < 1.0)
        print(f"    sp={fly_sp:.3f}x below1={fly_below} MAT={sum(r['mat'] for r in results['FLY_k15'])/len(results['FLY_k15']):.2f}")

        # ── TASD calibrated ──
        print("  TASD (calibrated)...")
        for i, s in enumerate(samples):
            r = run_tasd(target, draft, tokenizer, s["prompt"], stype)
            ar_tps = results["AR"][i]["ar_tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], s.get("reference", ""))
            results["TASD_cal"].append({"name": s["name"], "sp": round(sp, 3),
                                        "tps": round(r["tps"], 2), "sq": round(sq, 4),
                                        "accept": round(r["accept"], 4),
                                        "repair": r["repair"], "guard": r["guard_trig"],
                                        "trim": r["trim"], "off_str": round(r["off_str"], 4)})
        ts_sp = sum(r["sp"] for r in results["TASD_cal"]) / len(results["TASD_cal"])
        ts_below = sum(1 for r in results["TASD_cal"] if r["sp"] < 1.0)
        ts_off = sum(r["off_str"] for r in results["TASD_cal"]) / len(results["TASD_cal"])
        print(f"    sp={ts_sp:.3f}x below1={ts_below} off_str={ts_off:.4f}")

        all_bench[bname] = {
            "stype": stype, "ar_mean": round(mean_ar, 1),
            "results": dict(results),
            "summary": {
                "GSD":   {"sp": round(gsd_sp, 3), "below": gsd_below},
                "Ngram": {"sp": round(ng_sp, 3), "below": ng_below},
                "FLY_k15": {"sp": round(fly_sp, 3), "below": fly_below},
                "TASD_cal": {"sp": round(ts_sp, 3), "below": ts_below, "off_str": round(ts_off, 4)},
            },
        }

    # ── Overall ──
    print(f"\n{'='*70}")
    print(f"Qwen 3×20 Sanity — Overall Summary (5 methods)")
    print(f"{'='*70}")
    print()
    print(f"| Benchmark | AR TPS | GSD | Ngram SD | Official FLY (k=15) | TASD cal |")
    print(f"|-----------|:------:|:---:|:--------:|:-------------------:|:--------:|")
    methods = ["GSD", "Ngram", "FLY_k15", "TASD_cal"]
    overall = {m: {"sp": 0.0, "below": 0} for m in methods}
    n_benches = len(BENCHMARKS)
    for bname in BENCHMARKS:
        s = all_bench[bname]["summary"]
        sps = [str(s[m]["sp"]) + "x" for m in methods]
        print(f"| {bname} | {all_bench[bname]['ar_mean']:.0f} | {sps[0]} | {sps[1]} | {sps[2]} | {sps[3]} |")
        for m in methods:
            overall[m]["sp"] += s[m]["sp"]
            overall[m]["below"] += s[m]["below"]
    ov_sps = [f"{overall[m]['sp']/n_benches:.3f}x" for m in methods]
    ov_b = [str(overall[m]["below"]) for m in methods]
    print(f"| **Overall** | | **{ov_sps[0]}** | **{ov_sps[1]}** | **{ov_sps[2]}** | **{ov_sps[3]}** |")
    print(f"| Below 1.0x  | | {ov_b[0]} | {ov_b[1]} | {ov_b[2]} | {ov_b[3]} |")

    # ── Flight call ──
    fly_ov = overall["FLY_k15"]["sp"] / n_benches
    if fly_ov < 1.0:
        print(f"\n⚠️  Official FLY overall {fly_ov:.3f}x < 1.0x — recommend sweep!")
    else:
        print(f"\n✔ Official FLY overall {fly_ov:.3f}x >= 1.0x")

    # ── Save ──
    os.makedirs("results", exist_ok=True)
    with open("results/qwen_5method_3x20_sanity.json", "w") as f:
        json.dump(all_bench, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results/qwen_5method_3x20_sanity.json")
    print("Done!")

if __name__ == "__main__":
    main()
