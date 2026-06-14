"""
Re-run DictConfig 20 samples with guard_calibrated=True (Guard-v1.5).
Tests AR, Greedy SD, TASD on dict_config benchmark.
"""
import json, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 20
DATA_FILE = "data/codesearchnet_dict_config_blocks_80.jsonl"

TASD_CALIBRATED = dict(max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
                       top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                       enable_guard=True, enable_relaxed_accept=True,
                       guard_calibrated=True)

TASD_UNCALIBRATED = dict(max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
                         top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                         enable_guard=True, enable_relaxed_accept=True,
                         guard_calibrated=False)

GSD_KWARGS = dict(max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
                  top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                  enable_guard=False, enable_relaxed_accept=False)

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

def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    wall = time.time() - t0
    gen_ids = out[0][inp.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    tps = len(out[0]) / wall if wall > 0 else 0
    return {"tps": tps, "text": text, "time": wall, "gen_len": len(gen_ids)}

def run_gsd(target, draft, tokenizer, prompt):
    r = tasd_decode(target, draft, tokenizer, prompt, structure_type="dict_config", **GSD_KWARGS)
    return {"tps": r["tokens_per_second"], "text": r["generated_text"],
            "accept_rate": r["stats"]["accept_rate"]}

def run_tasd(target, draft, tokenizer, prompt, calibrated):
    kw = TASD_CALIBRATED if calibrated else TASD_UNCALIBRATED
    r = tasd_decode(target, draft, tokenizer, prompt, structure_type="dict_config", **kw)
    return {
        "tps": r["tokens_per_second"], "text": r["generated_text"],
        "accept_rate": r["stats"]["accept_rate"],
        "repair": r["stats"]["repair_count"],
        "guard_trig": r["stats"]["guard_trigger_count"],
        "trim_count": r["stats"]["trim_count"],
        "hard_trim": r["stats"]["hard_trim_count"],
        "repetition_warn": r["stats"]["repetition_warning_count"],
        "bracket_warn": r["stats"]["bracket_warning_count"],
        "import_warn": r["stats"]["import_warning_count"],
        "trim_reasons": r["stats"].get("trim_reasons", []),
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

    with open(DATA_FILE) as f:
        samples = [json.loads(l.strip()) for l in f.readlines()[:SAMPLE_LIMIT]]

    # AR baseline
    print("Running AR baseline...")
    ar_results = []
    for i, s in enumerate(samples):
        r = run_ar(target, tokenizer, s["prompt"])
        ar_results.append({"name": s["name"], "ar_tps": round(r["tps"], 2)})
    mean_ar = sum(r["ar_tps"] for r in ar_results) / len(ar_results)
    print(f"  AR mean TPS: {mean_ar:.1f}")

    # GSD
    print("\nRunning Greedy SD...")
    gsd_results = []
    for i, s in enumerate(samples):
        ref = s.get("reference", "")
        r = run_gsd(target, draft, tokenizer, s["prompt"])
        ar_tps = ar_results[i]["ar_tps"]
        sp = r["tps"] / ar_tps if ar_tps > 0 else 0
        sq = compute_sq(r["text"], ref)
        gsd_results.append({"name": s["name"], "gsd_tps": round(r["tps"], 2),
                            "gsd_speedup": round(sp, 3), "gsd_sq": round(sq, 4),
                            "gsd_accept": round(r["accept_rate"], 4),
                            "ar_tps": ar_tps})
        print(f"  [{i+1}] {s['name'][:30]}: AR={ar_tps:.0f} GSD={sp:.2f}x acc={r['accept_rate']:.3f}")

    # TASD calibrated
    print("\nRunning TASD (guard_calibrated=True)...")
    tasd_results = []
    for i, s in enumerate(samples):
        ref = s.get("reference", "")
        r = run_tasd(target, draft, tokenizer, s["prompt"], calibrated=True)
        ar_tps = ar_results[i]["ar_tps"]
        sp = r["tps"] / ar_tps if ar_tps > 0 else 0
        sq = compute_sq(r["text"], ref)
        off = compute_off_structure(r["text"])
        tasd_results.append({
            "name": s["name"], "ar_tps": ar_tps,
            "tasd_tps": round(r["tps"], 2),
            "tasd_speedup": round(sp, 3),
            "tasd_sq": round(sq, 4),
            "tasd_off_structure": round(off, 4),
            "tasd_accept": round(r["accept_rate"], 4),
            "tasd_repair": r["repair"],
            "tasd_guard_trig": r["guard_trig"],
            "tasd_trim": r["trim_count"],
            "tasd_hard_trim": r["hard_trim"],
            "tasd_rep_warn": r["repetition_warn"],
            "tasd_bracket_warn": r["bracket_warn"],
            "tasd_import_warn": r["import_warn"],
            "tasd_trim_reasons_str": str(r["trim_reasons"]),
            # also include GSD for comparison
            "gsd_speedup": gsd_results[i]["gsd_speedup"],
            "gsd_sq": gsd_results[i]["gsd_sq"],
            "gsd_accept": gsd_results[i]["gsd_accept"],
        })
        tag = "★ BELOW 1.0" if sp < 1.0 else ""
        print(f"  [{i+1}] {s['name'][:30]}: sp={sp:.2f}x acc={r['accept_rate']:.3f} "
              f"trim={r['trim_count']} guard={r['guard_trig']} "
              f"rep_warn={r['repetition_warn']} brack_warn={r['bracket_warn']} "
              f"{tag}")

    # TASD uncalibrated (for comparison)
    print("\nRunning TASD (guard_calibrated=False)...")
    tasd_u_results = []
    for i, s in enumerate(samples):
        ref = s.get("reference", "")
        r = run_tasd(target, draft, tokenizer, s["prompt"], calibrated=False)
        ar_tps = ar_results[i]["ar_tps"]
        sp = r["tps"] / ar_tps if ar_tps > 0 else 0
        sq = compute_sq(r["text"], ref)
        off = compute_off_structure(r["text"])
        tasd_u_results.append({
            "name": s["name"], "ar_tps": ar_tps,
            "tasd_u_tps": round(r["tps"], 2),
            "tasd_u_speedup": round(sp, 3),
            "tasd_u_sq": round(sq, 4),
            "tasd_u_accept": round(r["accept_rate"], 4),
            "tasd_u_guard_trig": r["guard_trig"],
            "tasd_u_trim": r["trim_count"],
        })
        tag = "★ BELOW 1.0" if sp < 1.0 else ""
        print(f"  [{i+1}] {s['name'][:30]}: sp={sp:.2f}x acc={r['accept_rate']:.3f} trim={r['trim_count']} {tag}")

    # ── Summaries ──
    n = len(tasd_results)
    print(f"\n{'='*60}")
    print(f"DictConfig Summary ({n} samples)")
    print(f"{'='*60}")
    m_cal = lambda k: round(sum(r[k] for r in tasd_results) / n, 3)
    m_gsd = lambda k: round(sum(r[k] for r in gsd_results) / n, 3)
    below_cal = sum(1 for r in tasd_results if r["tasd_speedup"] < 1.0)
    below_gsd = sum(1 for r in gsd_results if r["gsd_speedup"] < 1.0)

    print(f"  AR mean TPS:    {mean_ar:.1f}")
    print(f"  GSD:    {m_gsd('gsd_speedup'):.3f}x acc={m_gsd('gsd_accept'):.4f} sq={m_gsd('gsd_sq'):.4f} below1={below_gsd}")
    print(f"  TASD (calibrated):   {m_cal('tasd_speedup'):.3f}x acc={m_cal('tasd_accept'):.4f} sq={m_cal('tasd_sq'):.4f} trim={m_cal('tasd_trim'):.1f} below1={below_cal}")
    print(f"  TASD (uncalibrated): {round(sum(r['tasd_u_speedup'] for r in tasd_u_results)/n,3):.3f}x acc={round(sum(r['tasd_u_accept'] for r in tasd_u_results)/n,4):.4f} trim={round(sum(r['tasd_u_trim'] for r in tasd_u_results)/n,1):.1f}")

    # ── Per-case detail ──
    print(f"\n{'='*60}")
    print(f"Cases with guard_calibrated improvement")
    print(f"{'='*60}")
    print(f"{'Name':<30} {'uncal_sp':>8} {'cal_sp':>8} {'delta':>7} {'uncal_trim':>10} {'cal_trim':>9} {'cal_trim_reasons'}")
    print("-" * 90)
    for i in range(n):
        u_sp = tasd_u_results[i]["tasd_u_speedup"]
        c_sp = tasd_results[i]["tasd_speedup"]
        delta = c_sp - u_sp
        u_trim = tasd_u_results[i]["tasd_u_trim"]
        c_trim = tasd_results[i]["tasd_trim"]
        reasons = tasd_results[i]["tasd_trim_reasons_str"] if c_trim > 0 else "-"
        if abs(delta) > 0.01 or u_trim > 0:
            print(f"{tasd_results[i]['name']:<30} {u_sp:>8.3f} {c_sp:>8.3f} {delta:>+7.3f} {u_trim:>10} {c_trim:>9} {reasons[:60]}")

    # ── Estimate overall speedup ──
    # From pilot: openmmlab TASD=1.38x (20 samples), pipeline_stage TASD=1.32x (20 samples)
    # New dict_config: m_cal('tasd_speedup')
    # Overall = weighted mean of 60 samples
    dc_sp = m_cal('tasd_speedup')
    om_sp = 1.38
    ps_sp = 1.32
    overall_new = (dc_sp * 20 + om_sp * 20 + ps_sp * 20) / 60
    overall_old = (1.14 * 20 + om_sp * 20 + ps_sp * 20) / 60  # old dict_config was 1.14

    print(f"\n{'='*60}")
    print(f"Estimated Overall Speedup")
    print(f"{'='*60}")
    print(f"  DictConfig (calibrated):    {dc_sp:.3f}x (was 1.14x)")
    print(f"  OpenMMLab (unchanged):      {om_sp:.2f}x")
    print(f"  PipelineStage (unchanged):  {ps_sp:.2f}x")
    print(f"  OVERALL (new):  {overall_new:.3f}x {'>= 1.30 ✓' if overall_new >= 1.3 else '< 1.30'}")
    print(f"  OVERALL (old):  {overall_old:.3f}x")
    print(f"  Improvement:    {overall_new - overall_old:+.3f}x")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        "config": "guard_calibrated=True",
        "ar_tps_mean": round(mean_ar, 1),
        "gsd_summary": {"speedup": m_gsd('gsd_speedup'), "accept": m_gsd('gsd_accept'),
                        "sq": m_gsd('gsd_sq'), "below1": below_gsd},
        "tasd_cal_summary": {"speedup": dc_sp, "accept": m_cal('tasd_accept'),
                              "sq": m_cal('tasd_sq'), "below1": below_cal,
                              "mean_trim": m_cal('tasd_trim'),
                              "mean_hard_trim": m_cal('tasd_hard_trim')},
        "estimated_overall": {"new": round(overall_new, 3), "old": round(overall_old, 3)},
        "samples": tasd_results,
    }
    with open("results/llama_dictconfig_calibrated.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results/llama_dictconfig_calibrated.json")
    print("Done!")

if __name__ == "__main__":
    main()
