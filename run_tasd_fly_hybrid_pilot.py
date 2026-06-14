"""
TASD-FLY Hybrid Pilot: 3 benchmarks × 20 samples × 4 methods.

Methods: AR, Official FLY, TASD calibrated, TASD-FLY Hybrid.

Hybrid strategy:
  - FLY generates with n-gram PLD + model draft (fastest)
  - Post-generation: TASD structural guard checks output
  - If guard triggers: trim to safe boundary
  - If trimmed: fallback to target AR for remaining tokens
"""
import json, os, sys, time, logging, importlib.util
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.structural_guard import StructuralGuard

# ─── Paths ────────────────────────────────────────────────────────────────
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

# ─── Official FLY import ──────────────────────────────────────────────────
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

# ─── Configs ──────────────────────────────────────────────────────────────
FLY_K15 = {
    "k": 15, "total_gen_tok": MAX_NEW_TOKENS,
    "enable_fly": True, "win_len": 6, "entropy_thre": 0.3,
    "use_ngram": True, "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
    "verbose": False, "abla_no_window": False, "enable_statistics": True,
}

TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
]

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

def compute_repetition_rate(text):
    """N-gram based repetition rate: fraction of repeating 4-gram lines."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4: return 0.0
    seen = set()
    rep = 0
    for i in range(len(lines) - 3):
        ng = tuple(lines[i:i+4])
        if ng in seen:
            rep += 1
        seen.add(ng)
    return rep / max(len(lines) - 3, 1)

def compute_truncation(text):
    if not text or not text.strip(): return 1.0
    last_line = text.rstrip().split("\n")[-1].strip()
    return 0.0 if last_line and (last_line[-1] in "})]" or last_line.endswith(",") or last_line.endswith(":")) else 1.0

# ─── Decode Methods ───────────────────────────────────────────────────────

def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    wall = time.time() - t0
    gen_ids = out[0][inp.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"wall": wall, "tps": len(out[0]) / wall, "text": text, "gen_len": len(gen_ids)}

def run_fly(target, draft, tokenizer, prompt, fly_args, logger):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    spd_gen = SPDGenerate(draft_model=draft, target_model=target,
                          tokenizer=tokenizer, cuslog=logger, spd_args=fly_args)
    torch.cuda.synchronize()
    t0 = time.time()
    full_ids = spd_gen.generate_chunks(input_ids, temperature=0.0)
    torch.cuda.synchronize()
    wall = time.time() - t0
    gen_ids = full_ids[0][input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_acc = spd_gen.num_accepted_tokens.item() if spd_gen._counter_inited else 0
    n_emitted = spd_gen.num_emitted_tokens.item() if spd_gen._counter_inited else len(gen_ids)
    mat = n_acc / n_emitted if n_emitted > 0 else 0
    ngram_accs = getattr(spd_gen, 'debug_ngram_accept_num', [])
    mean_ngram = sum(ngram_accs) / len(ngram_accs) if ngram_accs else 0
    return {"wall": wall, "tps": full_ids.shape[1] / wall, "text": text, "gen_len": len(gen_ids),
            "mat": mat, "ngram_accept": mean_ngram}

def run_tasd(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True,
                    guard_calibrated=True,
                    **TASD_COMMON)
    stats = r["stats"]
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"], "text": r["generated_text"],
            "accept": stats["accept_rate"], "repair": stats.get("repair_count", 0),
            "guard_trig": stats.get("guard_trigger_count", 0),
            "trim": stats.get("trim_count", 0),
            "hard_trim": stats.get("hard_trim_count", 0),
            "rep_warn": stats.get("repetition_warning_count", 0),
            "brack_warn": stats.get("bracket_warning_count", 0),
            "off_str": compute_off_structure(r["generated_text"])}

def run_hybrid(target, draft, tokenizer, prompt, stype, fly_args, logger):
    """
    TASD-FLY Hybrid:
    1. Run FLY's generate_chunks to get fast output
    2. Apply TASD calibrated structural guard on the generated text
    3. If guard triggers: trim to safe position
    4. If trimmed: fallback to target AR for remaining tokens
    """
    guard = StructuralGuard(structure_type=stype, calibrated=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    spd_gen = SPDGenerate(draft_model=draft, target_model=target,
                          tokenizer=tokenizer, cuslog=logger, spd_args=fly_args)

    torch.cuda.synchronize()
    t0 = time.time()

    # Step 1: FLY generates
    full_ids = spd_gen.generate_chunks(input_ids, temperature=0.0)
    gen_ids = full_ids[0][input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Step 2: Guard check
    guard_triggered = False
    trim_count = 0
    safe_token_count = len(gen_ids)
    risk_type = None

    safe, safe_count, risk_type = guard.check(text, tokens=gen_ids, tokenizer=tokenizer)
    if not safe and safe_count > 0:
        guard_triggered = True
        trim_count = len(gen_ids) - safe_count
        safe_token_count = safe_count
        gen_ids = gen_ids[:safe_token_count]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Step 3: Fallback — generate remaining with target AR
    fallback_count = 0
    if safe_token_count < MAX_NEW_TOKENS and safe_token_count >= 0:
        fallback_count = 1
        remaining = MAX_NEW_TOKENS - safe_token_count
        # Build input for target: prompt + safe tokens
        safe_ids_tensor = torch.tensor([gen_ids], device=target.device)
        fb_input = torch.cat([input_ids.to(target.device), safe_ids_tensor], dim=1)
        with torch.no_grad():
            fb_out = target.generate(
                fb_input, max_new_tokens=remaining, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        fb_gen_ids = fb_out[0][fb_input.shape[1]:].tolist()
        gen_ids = gen_ids + fb_gen_ids
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    torch.cuda.synchronize()
    wall = time.time() - t0
    total_ids = input_ids.shape[1] + len(gen_ids)

    n_acc = spd_gen.num_accepted_tokens.item() if spd_gen._counter_inited else 0
    n_emitted = spd_gen.num_emitted_tokens.item() if spd_gen._counter_inited else 0

    return {
        "wall": wall, "tps": total_ids / wall, "text": text, "gen_len": len(gen_ids),
        "guard_triggered": guard_triggered, "trim_count": trim_count,
        "risk_type": risk_type or "none",
        "fallback_count": fallback_count,
        "fly_mat": n_acc / n_emitted if n_emitted > 0 else 0,
    }


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    logger = logging.getLogger("pilot")
    logger.setLevel(logging.WARNING)  # Suppress SPDGenerate verbose output

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

    all_results = {}
    all_summaries = {}

    for bname, datapath, stype in BENCHMARKS:
        print(f"{'='*70}")
        print(f"Benchmark: {bname} (stype: {stype})")
        print(f"{'='*70}")

        with open(datapath) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:20]]
        prompts = [s["prompt"] for s in samples]
        names = [s["name"] for s in samples]
        refs = [s.get("reference", "") for s in samples]

        bench_results = {"stype": stype, "samples": []}

        # --- AR baseline ---
        print("  AR baseline...")
        ar_results = []
        for i in range(20):
            r = run_ar(target, tokenizer, prompts[i])
            sq = compute_sq(r["text"], refs[i])
            trunc = compute_truncation(r["text"])
            ar_results.append({
                "name": names[i], "ar_tps": round(r["tps"], 2),
                "sq": round(sq, 4), "trunc": trunc,
                "gen_len": r["gen_len"], "wall": round(r["wall"], 3),
            })
        ar_tps_map = {r["name"]: r["ar_tps"] for r in ar_results}

        # --- Official FLY ---
        print("  Official FLY (k=15)...")
        fly_results = []
        for i in range(20):
            r = run_fly(target, draft, tokenizer, prompts[i], FLY_K15, logger)
            ar_tps = ar_tps_map[names[i]]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], refs[i])
            trunc = compute_truncation(r["text"])
            off_str = compute_off_structure(r["text"])
            rep = compute_repetition_rate(r["text"])
            fly_results.append({
                "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                "sq": round(sq, 4), "trunc": trunc, "off_str": round(off_str, 4),
                "rep": round(rep, 4), "mat": round(r["mat"], 2),
                "ngram_acc": round(r["ngram_accept"], 1),
                "gen_len": r["gen_len"], "wall": round(r["wall"], 3),
            })

        # --- TASD calibrated ---
        print("  TASD calibrated...")
        tasd_results = []
        for i in range(20):
            r = run_tasd(target, draft, tokenizer, prompts[i], stype)
            ar_tps = ar_tps_map[names[i]]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], refs[i])
            trunc = compute_truncation(r["text"])
            off_str = compute_off_structure(r["text"])
            rep = compute_repetition_rate(r["text"])
            tasd_results.append({
                "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                "sq": round(sq, 4), "trunc": trunc, "off_str": round(off_str, 4),
                "rep": round(rep, 4),
                "accept": round(r["accept"], 4),
                "guard_trig": r["guard_trig"], "trim": r["trim"],
                "hard_trim": r["hard_trim"],
                "rep_warn": r["rep_warn"], "brack_warn": r["brack_warn"],
                "wall": round(r["wall"], 3),
            })

        # --- TASD-FLY Hybrid ---
        print("  TASD-FLY Hybrid...")
        hybrid_results = []
        for i in range(20):
            r = run_hybrid(target, draft, tokenizer, prompts[i], stype, FLY_K15, logger)
            ar_tps = ar_tps_map[names[i]]
            sp = r["tps"] / ar_tps if ar_tps > 0 else 0
            sq = compute_sq(r["text"], refs[i])
            trunc = compute_truncation(r["text"])
            off_str = compute_off_structure(r["text"])
            rep = compute_repetition_rate(r["text"])
            hybrid_results.append({
                "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                "sq": round(sq, 4), "trunc": trunc, "off_str": round(off_str, 4),
                "rep": round(rep, 4),
                "guard_triggered": r["guard_triggered"],
                "trim_count": r["trim_count"],
                "risk_type": r["risk_type"],
                "fallback_count": r["fallback_count"],
                "fly_mat": round(r["fly_mat"], 2),
                "gen_len": r["gen_len"], "wall": round(r["wall"], 3),
            })

        bench_results = {
            "stype": stype,
            "AR": ar_results,
            "FLY": fly_results,
            "TASD": tasd_results,
            "Hybrid": hybrid_results,
        }
        all_results[bname] = bench_results

        # Compute summary for this benchmark
        def summarize(results_list, method_name):
            n = len(results_list)
            sp_vals = [r.get("sp", 1.0) for r in results_list]
            sq_vals = [r.get("sq", 0.0) for r in results_list]
            off_vals = [r.get("off_str", 0.0) for r in results_list]
            rep_vals = [r.get("rep", 0.0) for r in results_list]
            trunc_vals = [r.get("trunc", 0.0) for r in results_list]
            return {
                "sp_avg": round(sum(sp_vals) / n, 3),
                "sq_avg": round(sum(sq_vals) / n, 4),
                "off_str_avg": round(sum(off_vals) / n, 4),
                "rep_avg": round(sum(rep_vals) / n, 4),
                "trunc_avg": round(sum(trunc_vals) / n, 4),
                "below": sum(1 for s in sp_vals if s < 1.0),
                "hard_cases": sum(1 for i in range(n)
                                  if sp_vals[i] < 1.0 or sq_vals[i] < 0.5),
            }

        summaries = {}
        for ml, label in [("AR", "AR"), ("FLY", "FLY"), ("TASD", "TASD"), ("Hybrid", "Hybrid")]:
            summaries[label] = summarize(bench_results[ml], label)

        # Add hybrid-specific stats
        hybrid_guard = sum(1 for r in hybrid_results if r["guard_triggered"])
        hybrid_fallback = sum(1 for r in hybrid_results if r["fallback_count"] > 0)
        summaries["Hybrid"]["guard_trigger_count"] = hybrid_guard
        summaries["Hybrid"]["fallback_count"] = hybrid_fallback
        summaries["Hybrid"]["avg_trim"] = round(sum(r["trim_count"] for r in hybrid_results) / 20, 1)

        all_summaries[bname] = summaries

        # Print summary
        print(f"\n  {bname} summary (20 samples):")
        print(f"  {'Method':<12} {'Speedup':>8} {'SQ':>8} {'OffStr':>8} {'Rep':>8} {'Trunc':>8} {'Below':>6} {'Hard':>6}")
        for ml in ["AR", "FLY", "TASD", "Hybrid"]:
            s = summaries[ml]
            extra = ""
            if ml == "Hybrid":
                extra = f"  guard={s['guard_trigger_count']} fallback={s['fallback_count']}"
            print(f"  {ml:<12} {s['sp_avg']:>8.3f} {s['sq_avg']:>8.4f} {s['off_str_avg']:>8.4f} {s['rep_avg']:>8.4f} {s['trunc_avg']:>8.4f} {s['below']:>6} {s['hard_cases']:>6}{extra}")

    # ─── Save results ─────────────────────────────────────────────────
    out_json = "results/tasd_fly_hybrid_pilot_3x20.json"
    out_data = {"per_benchmark": all_results, "summaries": all_summaries}
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_json}")

    # ─── Write report ─────────────────────────────────────────────────
    write_report(out_data)
    print("Done!")


def write_report(out_data):
    import datetime
    all_results = out_data["per_benchmark"]
    all_summaries = out_data["summaries"]

    with open("results/tasd_fly_hybrid_pilot_3x20.md", "w") as f:
        w = f.write

        w("# TASD-FLY Hybrid Pilot Report (3×20)\n\n")
        w(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        w("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n")
        w("**Samples**: 3 benchmarks × 20 = 60 total\n\n")

        w("## Methods\n\n")
        w("| Method | Description |\n")
        w("|--------|-------------|\n")
        w("| AR | Target autoregressive (greedy) |\n")
        w("| Official FLY | AMD FLy SPDGenerate (k=15, win=6, entropy=0.3, ngram=4/6) |\n")
        w("| TASD cal | TASD + Guard-v1.5 calibrated (draft_len=16, blocks=2, top_k=3) |\n")
        w("| **Hybrid** | FLY generates → TASD guard checks → trim + AR fallback if needed |\n\n")

        w("## Hybrid Mechanism\n\n")
        w("1. FLY generates complete output using n-gram PLD + model draft\n")
        w("2. TASD calibrated structural guard scans the output\n")
        w("3. If guard triggers: trim to safe token boundary\n")
        w("4. If trimmed: target AR fallback for remaining tokens (max 128 total)\n\n")

        w("## Per-Benchmark Results\n\n")

        for bname in all_results:
            bench = all_results[bname]
            s = all_summaries[bname]
            ar_tps = bench["AR"][0]["ar_tps"]  # approximate

            w(f"### {bname} (AR TPS ~{ar_tps:.0f}, 20 samples)\n\n")
            w("| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Notes |\n")
            w("|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|-------|\n")

            for ml in ["AR", "FLY", "TASD", "Hybrid"]:
                sm = s[ml]
                notes = ""
                if ml == "Hybrid":
                    notes = f"guard_trig={sm['guard_trigger_count']}/20, fallback={sm['fallback_count']}/20, avg_trim={sm['avg_trim']:.1f}"
                w(f"| **{ml}** | **{sm['sp_avg']:.3f}x** | {sm['sq_avg']:.4f} | {sm['off_str_avg']:.4f} | {sm['rep_avg']:.4f} | {sm['trunc_avg']:.4f} | {sm['below']} | {sm['hard_cases']} | {notes} |\n")

            w("\n")

        # ─── Overall comparison ───
        w("## Overall Comparison (60 samples)\n\n")
        method_labels = ["AR", "FLY", "TASD", "Hybrid"]
        overall = {ml: {"sp": [], "sq": [], "off": [], "rep": [], "trunc": [], "below": 0, "hard": 0}
                   for ml in method_labels}
        hybrid_guard_total = 0
        hybrid_fallback_total = 0

        for bname in all_results:
            s = all_summaries[bname]
            for ml in method_labels:
                overall[ml]["sp"].append(s[ml]["sp_avg"])
                overall[ml]["sq"].append(s[ml]["sq_avg"])
                overall[ml]["off"].append(s[ml]["off_str_avg"])
                overall[ml]["rep"].append(s[ml]["rep_avg"])
                overall[ml]["trunc"].append(s[ml]["trunc_avg"])
                overall[ml]["below"] += s[ml]["below"]
                overall[ml]["hard"] += s[ml]["hard_cases"]
            hybrid_guard_total += s["Hybrid"]["guard_trigger_count"]
            hybrid_fallback_total += s["Hybrid"]["fallback_count"]

        n_bench = len(all_results)
        w("| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard |\n")
        w("|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|\n")
        for ml in method_labels:
            ov = overall[ml]
            sp_avg = sum(ov["sp"]) / n_bench
            sq_avg = sum(ov["sq"]) / n_bench
            off_avg = sum(ov["off"]) / n_bench
            rep_avg = sum(ov["rep"]) / n_bench
            trunc_avg = sum(ov["trunc"]) / n_bench
            w(f"| **{ml}** | **{sp_avg:.3f}x** | {sq_avg:.4f} | {off_avg:.4f} | {rep_avg:.4f} | {trunc_avg:.4f} | {ov['below']}/60 | {ov['hard']}/60 |\n")

        w(f"\n**Hybrid guard triggers**: {hybrid_guard_total}/60  |  **Hybrid fallbacks**: {hybrid_fallback_total}/60\n\n")

        # ─── Judgment ───
        # Compute hybrid vs FLY deltas
        hy_sp = sum(overall["Hybrid"]["sp"]) / n_bench
        fly_sp = sum(overall["FLY"]["sp"]) / n_bench
        tasd_sp = sum(overall["TASD"]["sp"]) / n_bench
        hy_below = overall["Hybrid"]["below"]
        fly_below = overall["FLY"]["below"]
        tasd_below = overall["TASD"]["below"]
        hy_off = sum(overall["Hybrid"]["off"]) / n_bench
        fly_off = sum(overall["FLY"]["off"]) / n_bench

        speed_penalty = (fly_sp - hy_sp) / fly_sp * 100 if fly_sp > 0 else 0

        w("## Judgment\n\n")
        w(f"### Against criteria:\n\n")
        w(f"1. **Hybrid speedup ≥ 95% of FLY**: Hybrid {hy_sp:.3f}x vs FLY {fly_sp:.3f}x → {'PASS' if hy_sp >= fly_sp * 0.95 else 'FAIL'} (penalty={speed_penalty:.1f}%)\n")
        w(f"2. **Hybrid below-1.0x < FLY below-1.0x**: Hybrid {hy_below}/60 vs FLY {fly_below}/60 → {'PASS' if hy_below < fly_below else 'FAIL'}\n")
        w(f"3. **Hybrid off_str ≤ FLY off_str**: Hybrid {hy_off:.4f} vs FLY {fly_off:.4f} → {'PASS' if hy_off <= fly_off + 0.001 else 'NEAR' if hy_off <= fly_off + 0.005 else 'FAIL'}\n")
        w(f"4. **On openmmlab/pipeline, Hybrid near or exceeds TASD**: See per-benchmark details\n\n")

        # Conclusion
        w("### Overall Assessment\n\n")
        if hy_sp >= fly_sp * 0.95:
            w(f"- Hybrid preserves {100-speed_penalty:.0f}% of FLY's speed — **viable**\n")
        else:
            w(f"- Hybrid loses {speed_penalty:.0f}% of FLY's speed — guard overhead is too high for this approach\n")
        w(f"- Hybrid has {hy_below}/{fly_below} below-1.0x vs FLY — {'reliability improved' if hy_below <= fly_below else 'no reliability gain'}\n")
        w(f"- Guard triggers: {hybrid_guard_total}/60 samples ({hybrid_guard_total/60*100:.0f}%) — {'structural risks detected and mitigated' if hybrid_guard_total > 0 else 'no structural risks found in FLY output'}\n\n")

        # Scoring
        passes = 0
        if hy_sp >= fly_sp * 0.95: passes += 1
        if hy_below < fly_below: passes += 1
        if hy_off <= fly_off + 0.005: passes += 1

        if passes >= 3:
            w("**Conclusion: Hybrid is viable.** ")
            w("FLY + TASD guard combination preserves speed with structural safety. ")
            w("Next step: integrate N-gram PLD directly into TASD's draft pipeline for tighter coupling.\n")
        elif passes >= 2:
            w("**Conclusion: Hybrid needs tuning.** ")
            w("Guard triggers cause noticeable speed penalty. Consider guard threshold tuning or selective activation.\n")
        else:
            w("**Conclusion: Post-hoc guarding is not sufficient.** ")
            w("Better to integrate n-gram PLD directly into TASD's draft pipeline rather than post-processing FLY output.\n")

        w("\n## Raw Data\n\n")
        w(f"- `results/tasd_fly_hybrid_pilot_3x20.json`\n")


if __name__ == "__main__":
    main()
