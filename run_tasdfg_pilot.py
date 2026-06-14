#!/usr/bin/env python3
"""
TASD-F-G Focused Pilot: Guarded-fallback vs unguarded-fallback on hard samples.

Samples: 9 below-1.0 TASD + argparse_074 (false trigger) + 3 normal argparse
Methods: TASD (cached), TASD-F (unguarded), TASD-F-G (guarded)

TASD-F-G key change: fallback_guarded=True
"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq, bracket_balance_score, off_structure_rate, repetition_rate

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

# Sample selection: 9 below-1.0 + argparse_074 (false trigger) + 3 normal
TARGET_SAMPLES = [
    # Below-1.0 hardcases (9)
    ("argparse", "argparse_real_023"),  # TASD=0.312
    ("argparse", "argparse_real_030"),  # TASD=0.209
    ("argparse", "argparse_real_031"),  # TASD=0.104
    ("argparse", "argparse_real_034"),  # TASD=0.269
    ("argparse", "argparse_real_039"),  # TASD=0.177
    ("argparse", "argparse_real_062"),  # TASD=0.296
    ("argparse", "argparse_real_070"),  # TASD=0.343
    ("dict_config", "dict_config_real_019"),   # TASD=0.975
    ("dict_config", "dict_config_real_057"),   # TASD=0.600
    # False trigger (1)
    ("argparse", "argparse_real_074"),  # TASD=1.335, TASD-F=0.971
    # Normal sanity (3)
    ("argparse", "argparse_real_015"),  # TASD=1.957
    ("argparse", "argparse_real_041"),  # TASD=2.022
    ("dict_config", "dict_config_real_035"),   # TASD=2.140
]

BENCHMARK_MAP = {
    "argparse": ("data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    "dict_config": ("data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
}

OUT_JSON = "results/qwen_tasd_fg_pilot.json"
OUT_MD = "results/qwen_tasd_fg_pilot.md"


def load_sample(bname, name):
    data_file, _ = BENCHMARK_MAP[bname]
    with open(data_file) as f:
        for line in f:
            s = json.loads(line.strip())
            if s["name"] == name:
                return s
    return None


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
    return {"wall": wall, "prompt_len": prompt_len, "gen_len": gen_len, "tps": tps, "text": text}


def run_tasd_unguarded_fallback(target, draft, tokenizer, prompt, stype):
    """TASD-F: failure-aware fallback, unguarded"""
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    enable_failure_aware_fallback=True,
                    fallback_guarded=False,
                    fallback_accept_threshold=0.5,
                    fallback_repair_threshold=2,
                    **TASD_COMMON)
    return extract_result(r)


def run_tasd_guarded_fallback(target, draft, tokenizer, prompt, stype):
    """TASD-F-G: failure-aware fallback, guarded"""
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    enable_failure_aware_fallback=True,
                    fallback_guarded=True,
                    fallback_accept_threshold=0.5,
                    fallback_repair_threshold=2,
                    **TASD_COMMON)
    return extract_result(r)


def extract_result(r):
    stats = r["stats"]
    fb_summary = stats.get("failure_aware_fallback", {}) or {}
    return {
        "wall": r["elapsed_time"], "tps": r["tokens_per_second"],
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
        "fb_trigger_count": fb_summary.get("trigger_count", 0),
    }


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

    # Load cached TASD data for comparison
    with open("results/qwen_tasd_f_6x80.json") as f:
        tasdf_full = json.load(f)
    with open("results/qwen_5method_6x80_quality.json") as f:
        quality = json.load(f)

    results = []

    total = len(TARGET_SAMPLES)
    for idx, (bname, name) in enumerate(TARGET_SAMPLES):
        stype = BENCHMARK_MAP[bname][1]
        sample = load_sample(bname, name)
        if sample is None:
            print(f"  [{idx+1}/{total}] {bname}/{name}: NOT FOUND")
            continue
        prompt = sample["prompt"]
        ref = sample.get("reference", "")

        # Get cached TASD speedup
        ts_data = quality['per_sample'][bname]['TASD']
        ts_sp = next((s['sp'] for s in ts_data if s['name'] == name), 0)

        # AR
        ar = run_ar(target, tokenizer, prompt)
        ar_tps = ar["tps"]

        # TASD-F (unguarded)
        tf = run_tasd_unguarded_fallback(target, draft, tokenizer, prompt, stype)
        tf_sp = tf["tps"] / ar_tps if ar_tps > 0 else 0

        # TASD-F-G (guarded)
        tfg = run_tasd_guarded_fallback(target, draft, tokenizer, prompt, stype)
        tfg_sp = tfg["tps"] / ar_tps if ar_tps > 0 else 0

        # Quality
        ar_q = compute_composite_sq(ar["text"], ref, stype)
        tf_q = compute_composite_sq(tf["text"], ref, stype)
        tfg_q = compute_composite_sq(tfg["text"], ref, stype)

        tag = "BELOW" if ts_sp < 1.0 else ("FALSE_TRIG" if name == "argparse_real_074" else "normal")
        print(f"  [{idx+1}/{total}] {name}: AR={ar_tps:.1f} "
              f"TASD={ts_sp:.3f}x "
              f"TASD-F={tf_sp:.3f}x(below={'Y' if tf_sp<1.0 else 'N'},fb={tf['fb_count']}) "
              f"TASD-F-G={tfg_sp:.3f}x(below={'Y' if tfg_sp<1.0 else 'N'},fb={tfg['fb_count']}) "
              f"[{tag}]")

        entry = {
            "benchmark": bname, "name": name,
            "tag": "below" if ts_sp < 1.0 else ("false_trigger" if name == "argparse_real_074" else "normal"),
            "AR": {"tps": round(ar_tps, 2), "wall": round(ar["wall"], 3), **ar_q},
            "TASD": {"sp": round(ts_sp, 3)},
            "TASD-F": {"tps": round(tf["tps"], 2), "sp": round(tf_sp, 3),
                       "wall": round(tf["wall"], 3), "accept": round(tf["accept"], 4),
                       "repair": tf["repair"], "guard_trig": tf["guard_trig"],
                       "trim": tf["trim"], "hard_trim": tf["hard_trim"],
                       "fb_count": tf["fb_count"], "fb_tokens": tf["fb_tokens"],
                       **tf_q},
            "TASD-F-G": {"tps": round(tfg["tps"], 2), "sp": round(tfg_sp, 3),
                         "wall": round(tfg["wall"], 3), "accept": round(tfg["accept"], 4),
                         "repair": tfg["repair"], "guard_trig": tfg["guard_trig"],
                         "trim": tfg["trim"], "hard_trim": tfg["hard_trim"],
                         "fb_count": tfg["fb_count"], "fb_tokens": tfg["fb_tokens"],
                         **tfg_q},
        }
        results.append(entry)

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"config": {
            "target": "Qwen2.5-14B-Instruct-AWQ", "draft": "Qwen2.5-1.5B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS, "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "TASD-F": "enable_failure_aware_fallback=True, fallback_guarded=False",
            "TASD-F-G": "enable_failure_aware_fallback=True, fallback_guarded=True",
        }, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    write_md_report(results)
    print(f"Saved {OUT_MD}")


def write_md_report(results):
    with open(OUT_MD, "w") as f:
        f.write("# TASD-F-G Focused Pilot\n\n")
        f.write("**TASD-F**: unguarded fallback  |  **TASD-F-G**: guarded fallback\n\n")

        f.write("## Per-Sample\n\n")
        f.write("| Sample | AR TPS | TASD sp | TASD-F sp | TASD-F-G sp | TASD-F fb | TASD-F-G fb | TASD-F SQ | TASD-F-G SQ |\n")
        f.write("|--------|:------:|:-------:|:---------:|:-----------:|:---------:|:-----------:|:---------:|:-----------:|\n")
        for e in results:
            name = e["name"][:20]
            ar_tps = e["AR"]["tps"]
            ts_sp = e["TASD"]["sp"]
            tf_sp = e["TASD-F"]["sp"]
            tfg_sp = e["TASD-F-G"]["sp"]
            tf_fb = e["TASD-F"]["fb_count"]
            tfg_fb = e["TASD-F-G"]["fb_count"]
            tf_sq = e["TASD-F"]["composite_sq"]
            tfg_sq = e["TASD-F-G"]["composite_sq"]
            tag = e.get("tag", "")
            ts_mark = "**" if ts_sp < 1.0 else ""
            ts_sp_s = f"{ts_mark}{ts_sp:.3f}x{ts_mark}" if ts_sp < 1.0 else f"{ts_sp:.3f}x"
            tfg_mark = "**" if tfg_sp < 1.0 else ""
            tfg_sp_s = f"{tfg_mark}{tfg_sp:.3f}x{tfg_mark}" if tfg_sp < 1.0 else f"{tfg_sp:.3f}x"
            tf_mark = "**" if tf_sp < 1.0 else ""
            tf_sp_s = f"{tf_mark}{tf_sp:.3f}x{tf_mark}" if tf_sp < 1.0 else f"{tf_sp:.3f}x"
            f.write(f"| {name} [{tag}] | {ar_tps:.1f} | {ts_sp_s} | {tf_sp_s} | {tfg_sp_s} | {tf_fb} | {tfg_fb} | {tf_sq:.4f} | {tfg_sq:.4f} |\n")
        f.write("\n")

        # Aggregate by tag
        for tag, label in [("below", "Below-1.0x (9)"), ("false_trigger", "False Trigger (argparse_074)"), ("normal", "Normal (3)"), (None, "All (13)")]:
            if tag:
                subset = [e for e in results if e.get("tag") == tag]
                if not subset:
                    continue
            else:
                subset = results
            n = len(subset)

            ts_sps = [e["TASD"]["sp"] for e in subset]
            tf_sps = [e["TASD-F"]["sp"] for e in subset]
            tfg_sps = [e["TASD-F-G"]["sp"] for e in subset]

            f.write(f"## {label} (n={n})\n\n")
            f.write("| Metric | TASD | TASD-F | TASD-F-G |\n")
            f.write("|--------|:----:|:------:|:--------:|\n")

            for metric_name, ts_fn, tf_fn, tfg_fn, fmt in [
                ("mean sp", lambda x: sum(x)/len(x), lambda x: sum(x)/len(x), lambda x: sum(x)/len(x), ".3f"),
                ("below-1.0x", lambda x: sum(1 for s in x if s<1.0), lambda x: sum(1 for s in x if s<1.0), lambda x: sum(1 for s in x if s<1.0), "d"),
            ]:
                ts_v = ts_fn(ts_sps)
                tf_v = tf_fn(tf_sps)
                tfg_v = tfg_fn(tfg_sps)
                fmt_str = f"{{:{fmt}}}"
                f.write(f"| {metric_name} | {fmt_str.format(ts_v)} | {fmt_str.format(tf_v)} | {fmt_str.format(tfg_v)} |\n")

            for metric_key, label2 in [
                ("fb_count", "total fb_count"), ("guard_trig", "mean guard_trig"),
                ("trim", "mean trim"), ("repair", "mean repair"),
                ("composite_sq", "mean SQ"), ("off_structure_rate", "mean off_structure"),
                ("repetition_rate", "mean rep_rate"), ("is_truncated", "mean truncation"),
            ]:
                ts_v = sum(e["TASD-F" if "TASD-F" in e else "AR"].get(metric_key, 0) for e in subset) / n if "mean" in label2 else sum(e.get("TASD-F", {}).get(metric_key, 0) for e in subset)
                tf_v = sum(e["TASD-F"].get(metric_key, 0) for e in subset) / n if "mean" in label2 else sum(e["TASD-F"].get(metric_key, 0) for e in subset)
                tfg_v = sum(e["TASD-F-G"].get(metric_key, 0) for e in subset) / n if "mean" in label2 else sum(e["TASD-F-G"].get(metric_key, 0) for e in subset)
                fmt = ".4f" if isinstance(tf_v, float) else "d"
                fmt_str = f"{{:{fmt}}}"
                f.write(f"| {label2} | {fmt_str.format(ts_v)} | {fmt_str.format(tf_v)} | {fmt_str.format(tfg_v)} |\n")
            f.write("\n")

        # Pass/Fail
        below_s = [e for e in results if e["tag"] == "below"]
        ts_b = [e["TASD"]["sp"] for e in below_s]
        tf_b = [e["TASD-F"]["sp"] for e in below_s]
        tfg_b = [e["TASD-F-G"]["sp"] for e in below_s]
        ts_below = sum(1 for s in ts_b if s < 1.0)
        tf_below = sum(1 for s in tf_b if s < 1.0)
        tfg_below = sum(1 for s in tfg_b if s < 1.0)

        all_s = results
        tf_all = [e["TASD-F"]["sp"] for e in all_s]
        tfg_all = [e["TASD-F-G"]["sp"] for e in all_s]
        tf_mean = sum(tf_all) / len(tf_all)
        tfg_mean = sum(tfg_all) / len(tfg_all)

        tf_off = sum(e["TASD-F"].get("off_structure_rate", 0) for e in all_s) / len(all_s)
        tfg_off = sum(e["TASD-F-G"].get("off_structure_rate", 0) for e in all_s) / len(all_s)

        # Check dict_019
        dict019 = [e for e in all_s if e["name"] == "dict_config_real_019"]
        dict019_sp = dict019[0]["TASD-F-G"]["sp"] if dict019 else 0

        # argparse_074
        a074 = [e for e in all_s if e["name"] == "argparse_real_074"]
        a074_sp = a074[0]["TASD-F-G"]["sp"] if a074 else 0

        f.write("## Pass/Fail\n\n")
        f.write("| Criterion | Result | Note |\n")
        f.write("|-----------|:------:|------|\n")

        c1 = tfg_below < ts_below
        f.write(f"| below-1.0x reduced in hardcases | {'PASS' if c1 else 'FAIL'} | {ts_below} → {tfg_below} |\n")

        c2 = dict019_sp >= 1.0
        f.write(f"| dict_019 stays >1.0x | {'PASS' if c2 else 'FAIL'} | TASD-F-G={dict019_sp:.3f}x |\n")

        c3 = a074_sp >= 1.0
        f.write(f"| argparse_074 NOT <1.0x | {'PASS' if c3 else 'FAIL'} | TASD-F-G={a074_sp:.3f}x |\n")

        c4 = tfg_off <= tf_off * 1.05
        f.write(f"| off_structure <= TASD-F×1.05 | {'PASS' if c4 else 'FAIL'} | {tf_off:.4f} → {tfg_off:.4f} |\n")

        c5 = tfg_mean >= tf_mean * 0.97
        f.write(f"| mean sp >= TASD-F×0.97 | {'PASS' if c5 else 'FAIL'} | {tf_mean:.3f} → {tfg_mean:.3f} |\n")

        all_ok = c1 and c2 and c3 and c4 and c5
        f.write(f"\n**Overall**: {'ALL PASS' if all_ok else 'SOME FAIL'}\n\n")


if __name__ == "__main__":
    main()
