"""
Qwen Guard Calibration Sanity: 3 benchmarks × 20 samples.
Compares TASD with guard_calibrated=True (new default) vs False (old behavior).
Goal: confirm calibration doesn't hurt speed/SQ on Qwen, same as LLaMA.
"""

import json, os, sys, time, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLES_PER_BENCHMARK = 20

BENCHMARKS = {
    "dict_config": ("data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    "openmmlab_config": ("data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    "pipeline_stage_config": ("data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
}

TASD_BASE = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
    enable_guard=True, enable_relaxed_accept=True,
)

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

def run_one(target, draft, tokenizer, prompt, structure_type, calibrated):
    kw = dict(TASD_BASE)
    kw["structure_type"] = structure_type
    kw["guard_calibrated"] = calibrated
    r = tasd_decode(target, draft, tokenizer, prompt, **kw)
    return {
        "tps": r["tokens_per_second"],
        "text": r["generated_text"],
        "accept_rate": r["stats"]["accept_rate"],
        "repair": r["stats"].get("repair_count", 0),
        "guard_trig": r["stats"].get("guard_trigger_count", 0),
        "trim": r["stats"].get("trim_count", 0),
        "hard_trim": r["stats"].get("hard_trim_count", 0),
        "rep_warn": r["stats"].get("repetition_warning_count", 0),
        "brack_warn": r["stats"].get("bracket_warning_count", 0),
    }

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

    all_bench_summaries = {}

    for bname, (datafile, stype) in BENCHMARKS.items():
        print(f"{'='*60}")
        print(f"Benchmark: {bname}")
        print(f"{'='*60}")

        with open(datafile) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:SAMPLES_PER_BENCHMARK]]

        # AR baseline
        ar_results = []
        for i, s in enumerate(samples):
            inp = tokenizer(s["prompt"], return_tensors="pt").to(target.device)
            t0 = time.time()
            with torch.no_grad():
                out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
            wall = time.time() - t0
            gen_ids = out[0][inp.input_ids.shape[1]:]
            ar_tps = len(out[0]) / wall if wall > 0 else 0
            ar_results.append({"name": s["name"], "ar_tps": round(ar_tps, 2)})
        mean_ar = sum(r["ar_tps"] for r in ar_results) / len(ar_results)
        print(f"  AR mean TPS: {mean_ar:.1f}")

        # TASD uncalibrated
        print(f"  Running TASD uncalibrated...")
        u_results = []
        for i, s in enumerate(samples):
            ref = s.get("reference", "")
            r = run_one(target, draft, tokenizer, s["prompt"], stype, calibrated=False)
            ar_tps = ar_results[i]["ar_tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], ref)
            off = compute_off_structure(r["text"])
            u_results.append({
                "name": s["name"], "ar_tps": ar_tps,
                "sp": round(sp, 3), "acc": round(r["accept_rate"], 4),
                "sq": round(sq, 4), "off": round(off, 4),
                "trim": r["trim"], "guard": r["guard_trig"], "hard": r["hard_trim"],
            })
        u_mean_sp = sum(r["sp"] for r in u_results) / len(u_results)
        u_below = sum(1 for r in u_results if r["sp"] < 1.0)
        u_mean_trim = sum(r["trim"] for r in u_results) / len(u_results)
        u_mean_guard = sum(r["guard"] for r in u_results) / len(u_results)
        print(f"    sp={u_mean_sp:.3f}x below={u_below} trim={u_mean_trim:.1f} guard={u_mean_guard:.1f}")

        # TASD calibrated
        print(f"  Running TASD calibrated...")
        c_results = []
        for i, s in enumerate(samples):
            ref = s.get("reference", "")
            r = run_one(target, draft, tokenizer, s["prompt"], stype, calibrated=True)
            ar_tps = ar_results[i]["ar_tps"]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], ref)
            off = compute_off_structure(r["text"])
            c_results.append({
                "name": s["name"], "ar_tps": ar_tps,
                "sp": round(sp, 3), "acc": round(r["accept_rate"], 4),
                "sq": round(sq, 4), "off": round(off, 4),
                "trim": r["trim"], "guard": r["guard_trig"], "hard": r["hard_trim"],
            })
        c_mean_sp = sum(r["sp"] for r in c_results) / len(c_results)
        c_below = sum(1 for r in c_results if r["sp"] < 1.0)
        c_mean_trim = sum(r["trim"] for r in c_results) / len(c_results)
        c_mean_guard = sum(r["guard"] for r in c_results) / len(c_results)
        print(f"    sp={c_mean_sp:.3f}x below={c_below} trim={c_mean_trim:.1f} guard={c_mean_guard:.1f}")

        all_bench_summaries[bname] = {
            "ar_tps": round(mean_ar, 1),
            "uncal": {"sp": round(u_mean_sp, 3), "below": u_below, "trim": round(u_mean_trim, 1),
                      "guard": round(u_mean_guard, 1)},
            "cal": {"sp": round(c_mean_sp, 3), "below": c_below, "trim": round(c_mean_trim, 1),
                    "guard": round(c_mean_guard, 1)},
            "delta_sp": round(c_mean_sp - u_mean_sp, 3),
            "delta_below": c_below - u_below,
            "samples": {"uncalibrated": u_results, "calibrated": c_results},
        }

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"Qwen Guard Calibration Sanity — Summary")
    print(f"{'='*70}")
    print(f"")

    # Per benchmark
    print(f"| Benchmark | AR TPS | Uncal SP | Cal SP | Delta | Below↓ | Trim↓ |")
    print(f"|-----------|:------:|:--------:|:------:|:-----:|:------:|:-----:|")
    overall_u_sp = 0
    overall_c_sp = 0
    overall_u_below = 0
    overall_c_below = 0
    n_benches = 0
    for bname in BENCHMARKS:
        s = all_bench_summaries[bname]
        print(f"| {bname} | {s['ar_tps']:.0f} | {s['uncal']['sp']:.3f}x | {s['cal']['sp']:.3f}x | {s['delta_sp']:+.3f}x | {s['uncal']['below']}→{s['cal']['below']} | {s['uncal']['trim']:.1f}→{s['cal']['trim']:.1f} |")
        overall_u_sp += s['uncal']['sp']
        overall_c_sp += s['cal']['sp']
        overall_u_below += s['uncal']['below']
        overall_c_below += s['cal']['below']
        n_benches += 1
    overall_u_sp /= n_benches
    overall_c_sp /= n_benches
    print(f"| **Overall** | | **{overall_u_sp:.3f}x** | **{overall_c_sp:.3f}x** | **{overall_c_sp-overall_u_sp:+.3f}x** | {overall_u_below}→{overall_c_below} | |")

    # Show per-sample deltas where calibration changed speedup significantly
    print(f"\n## Samples with |delta| > 0.05x")
    for bname in BENCHMARKS:
        s = all_bench_summaries[bname]
        u_list = s['samples']['uncalibrated']
        c_list = s['samples']['calibrated']
        deltas = [(u_list[i]['name'], u_list[i]['sp'], c_list[i]['sp'],
                   c_list[i]['sp'] - u_list[i]['sp'],
                   u_list[i]['trim'], c_list[i]['trim'])
                  for i in range(len(u_list))]
        big = [(nm, usp, csp, d, ut, ct) for nm, usp, csp, d, ut, ct in deltas if abs(d) > 0.05]
        if big:
            print(f"  {bname}:")
            for nm, usp, csp, d, ut, ct in sorted(big, key=lambda x: -abs(x[3])):
                print(f"    {nm}: {usp:.3f}x → {csp:.3f}x (delta={d:+.3f}x, trim {ut}→{ct})")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/qwen_calibration_sanity.json", "w") as f:
        json.dump(all_bench_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results/qwen_calibration_sanity.json")
    print("Done!")

if __name__ == "__main__":
    main()
