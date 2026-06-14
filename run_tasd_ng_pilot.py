"""
TASD-NG Pilot: 3 benchmarks × 5 samples × 5 methods.

Methods: AR, Official FLY, TASD calibrated, Post-hoc Hybrid (negative result), TASD-NG.

TASD-NG: TASD with n-gram PLD draft channel.
  - N-gram lookup first each round
  - If found: use matched suffix as draft
  - If not found: fall back to draft model
  - Then: target top-k verification + calibrated guard
"""
import json, os, sys, time, logging, importlib.util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.tasd_ng_decode import tasd_ng_decode
from src.structural_guard import StructuralGuard

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

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
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4: return 0.0
    seen, rep = set(), 0
    for i in range(len(lines) - 3):
        ng = tuple(lines[i:i+4])
        if ng in seen: rep += 1
        seen.add(ng)
    return rep / max(len(lines) - 3, 1)

def compute_truncation(text):
    if not text or not text.strip(): return 1.0
    last = text.rstrip().split("\n")[-1].strip()
    return 0.0 if last and (last[-1] in "})]" or last.endswith(",") or last.endswith(":")) else 1.0

def run_ar(target, tokenizer, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(target.device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = target.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); wall = time.time() - t0
    ids = out[0][inp.input_ids.shape[1]:]
    return {"wall": wall, "tps": len(out[0])/wall, "text": tokenizer.decode(ids,skip_special_tokens=True), "gen_len": len(ids)}

def run_fly(target, draft, tokenizer, prompt, logger):
    inp = tokenizer(prompt, return_tensors="pt").input_ids
    spd = SPDGenerate(draft_model=draft, target_model=target, tokenizer=tokenizer, cuslog=logger, spd_args=FLY_K15)
    torch.cuda.synchronize(); t0 = time.time()
    full = spd.generate_chunks(inp, temperature=0.0)
    torch.cuda.synchronize(); wall = time.time() - t0
    ids = full[0][inp.shape[1]:].tolist()
    n_acc = spd.num_accepted_tokens.item() if spd._counter_inited else 0
    n_emit = spd.num_emitted_tokens.item() if spd._counter_inited else len(ids)
    na = getattr(spd, 'debug_ngram_accept_num', [])
    return {"wall":wall,"tps":full.shape[1]/wall,"text":tokenizer.decode(ids,skip_special_tokens=True),
            "gen_len":len(ids),"mat":n_acc/n_emit if n_emit>0 else 0,
            "ngram_accept":sum(na)/len(na) if na else 0}

def run_tasd(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt, structure_type=stype,
                    enable_guard=True, enable_relaxed_accept=True, guard_calibrated=True, **TASD_COMMON)
    s = r["stats"]
    return {"wall":r["elapsed_time"],"tps":r["tokens_per_second"],"text":r["generated_text"],
            "accept":s["accept_rate"],"repair":s.get("repair_count",0),
            "guard_trig":s.get("guard_trigger_count",0),"trim":s.get("trim_count",0),
            "hard_trim":s.get("hard_trim_count",0),
            "rep_warn":s.get("repetition_warning_count",0),"brack_warn":s.get("bracket_warning_count",0),
            "off_str":compute_off_structure(r["generated_text"])}

def run_tasd_ng(target, draft, tokenizer, prompt, stype):
    r = tasd_ng_decode(target, draft, tokenizer, prompt, structure_type=stype,
                       max_new_tokens=MAX_NEW_TOKENS, draft_len=16,
                       top_k_accept=3, ngram_min=2, ngram_max=8, max_ngram_draft=16,
                       guard_calibrated=True)
    s = r["stats"]
    return {"wall":r["elapsed_time"],"tps":r["tokens_per_second"],"text":r["generated_text"],
            "accept":s["accept_rate"],"repair":s.get("repair_count",0),
            "guard_trig":s.get("guard_trigger_count",0),"trim":s.get("trim_count",0),
            "hard_trim":s.get("hard_trim_count",0),
            "ngram_rounds":s.get("ngram_rounds",0),"model_rounds":s.get("model_draft_rounds",0),
            "fallback_rounds":s.get("fallback_rounds",0),
            "off_str":compute_off_structure(r["generated_text"])}

def main():
    logger = logging.getLogger("pilot"); logger.setLevel(logging.WARNING)
    print("Loading Qwen models...")
    tok = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    all_results = {}; all_summaries = {}

    for bname, datapath, stype in BENCHMARKS:
        print(f"{'='*70}\nBenchmark: {bname}\n{'='*70}")
        with open(datapath) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:5]]
        prompts = [s["prompt"] for s in samples]
        names = [s["name"] for s in samples]
        refs = [s.get("reference","") for s in samples]

        # AR
        print("  AR...")
        ar_res = []; 
        for i in range(5):
            r = run_ar(target, tok, prompts[i])
            ar_res.append({"name":names[i],"ar_tps":round(r["tps"],2),
                           "sq":round(compute_sq(r["text"],refs[i]),4),
                           "trunc":compute_truncation(r["text"]),
                           "gen_len":r["gen_len"],"wall":round(r["wall"],3)})
        ar_map = {x["name"]:x["ar_tps"] for x in ar_res}

        # FLY
        print("  FLY...")
        fly_res = []
        for i in range(5):
            r = run_fly(target, draft, tok, prompts[i], logger)
            at = ar_map[names[i]]; sp = r["tps"]/at if at>0 else 0
            fly_res.append({"name":names[i],"sp":round(sp,3),"tps":round(r["tps"],2),
                            "sq":round(compute_sq(r["text"],refs[i]),4),
                            "trunc":compute_truncation(r["text"]),
                            "off_str":round(compute_off_structure(r["text"]),4),
                            "rep":round(compute_repetition_rate(r["text"]),4),
                            "mat":round(r["mat"],2),"ngram_acc":round(r["ngram_accept"],1),
                            "gen_len":r["gen_len"],"wall":round(r["wall"],3)})

        # TASD
        print("  TASD...")
        tasd_res = []
        for i in range(5):
            r = run_tasd(target, draft, tok, prompts[i], stype)
            at = ar_map[names[i]]; sp = r["tps"]/at if at>0 else 0
            tasd_res.append({"name":names[i],"sp":round(sp,3),"tps":round(r["tps"],2),
                             "sq":round(compute_sq(r["text"],refs[i]),4),
                             "trunc":compute_truncation(r["text"]),
                             "off_str":round(compute_off_structure(r["text"]),4),
                             "rep":round(compute_repetition_rate(r["text"]),4),
                             "accept":round(r["accept"],4),
                             "guard_trig":r["guard_trig"],"trim":r["trim"],
                             "hard_trim":r["hard_trim"],
                             "wall":round(r["wall"],3)})

        # TASD-NG
        print("  TASD-NG...")
        ng_res = []
        for i in range(5):
            r = run_tasd_ng(target, draft, tok, prompts[i], stype)
            at = ar_map[names[i]]; sp = r["tps"]/at if at>0 else 0
            ng_res.append({"name":names[i],"sp":round(sp,3),"tps":round(r["tps"],2),
                           "sq":round(compute_sq(r["text"],refs[i]),4),
                           "trunc":compute_truncation(r["text"]),
                           "off_str":round(compute_off_structure(r["text"]),4),
                           "rep":round(compute_repetition_rate(r["text"]),4),
                           "accept":round(r["accept"],4),
                           "guard_trig":r["guard_trig"],"trim":r["trim"],
                           "ngram_rounds":r["ngram_rounds"],
                           "model_rounds":r["model_rounds"],
                           "fallback_rounds":r["fallback_rounds"],
                           "wall":round(r["wall"],3)})

        bench = {"stype":stype,"AR":ar_res,"FLY":fly_res,"TASD":tasd_res,"TASD_NG":ng_res}
        all_results[bname] = bench

        def summarize(lst):
            n=len(lst); sp=[r.get("sp",1.0) for r in lst]; sq=[r.get("sq",0) for r in lst]
            off=[r.get("off_str",0) for r in lst]; rep=[r.get("rep",0) for r in lst]
            trunc=[r.get("trunc",0) for r in lst]
            return {"sp_avg":round(sum(sp)/n,3),"sq_avg":round(sum(sq)/n,4),
                    "off_str_avg":round(sum(off)/n,4),"rep_avg":round(sum(rep)/n,4),
                    "trunc_avg":round(sum(trunc)/n,4),"below":sum(1 for s in sp if s<1.0),
                    "hard":sum(1 for i in range(n) if sp[i]<1.0 or sq[i]<0.5)}

        sums = {}
        for ml in ["AR","FLY","TASD","TASD_NG"]: sums[ml] = summarize(bench[ml])
        sums["TASD_NG"]["ngram_rounds"] = sum(r["ngram_rounds"] for r in ng_res)
        sums["TASD_NG"]["model_rounds"] = sum(r["model_rounds"] for r in ng_res)
        sums["TASD_NG"]["guard_trig"] = sum(1 for r in ng_res if r["guard_trig"]>0)
        all_summaries[bname] = sums

        print(f"\n  {bname} summary:")
        print(f"  {'Method':<12} {'Sp':>8} {'SQ':>8} {'Off':>8} {'Rep':>8} {'Trunc':>8} {'Below':>6} {'Hard':>6}")
        for ml in ["AR","FLY","TASD","TASD_NG"]:
            s=sums[ml]; extra=""
            if ml=="TASD_NG": extra=f"  ngram={s['ngram_rounds']} model={s['model_rounds']}"
            print(f"  {ml:<12} {s['sp_avg']:>8.3f} {s['sq_avg']:>8.4f} {s['off_str_avg']:>8.4f} {s['rep_avg']:>8.4f} {s['trunc_avg']:>8.4f} {s['below']:>6} {s['hard']:>6}{extra}")

    out = {"per_benchmark":all_results,"summaries":all_summaries}
    with open("results/tasd_ng_pilot_3x20.json","w") as f:
        json.dump(out,f,indent=2,ensure_ascii=False)
    print(f"\nSaved results/tasd_ng_pilot_3x20.json")

    # Report
    write_report(out)
    print("Done!")


def write_report(out):
    import datetime
    res = out["per_benchmark"]; sums = out["summaries"]
    with open("results/tasd_ng_pilot_3x20.md","w") as f:
        w=f.write
        w("# TASD-NG Pilot Report (3×20)\n\n")
        w(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        w("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n")
        w("**Samples**: 3 benchmarks × 20 = 15 total\n\n")
        w("## Methods\n\n")
        w("| Method | Description |\n|--------|-------------|\n")
        w("| AR | Target autoregressive (greedy) |\n")
        w("| FLY | AMD FLy (k=15, win=6, ngram=4/6) |\n")
        w("| TASD | TASD + Guard-v1.5 (draft_len=16, top_k=3) |\n")
        w("| **TASD-NG** | TASD + n-gram PLD draft channel (ngram_min=2, max=8) |\n\n")
        w("## TASD-NG Design\n\n")
        w("Each decoding round:\n")
        w("1. N-gram lookup from full context (min=2, max=8)\n")
        w("2. If match: use matched suffix tokens as draft\n")
        w("3. If no match: fall back to draft model generation\n")
        w("4. Target top-k verification (unchanged TASD)\n")
        w("5. Calibrated structural guard (unchanged TASD)\n\n")

        w("## Per-Benchmark\n\n")
        for bn in res:
            bench=res[bn]; s=sums[bn]
            w(f"### {bn} (5 samples)\n\n")
            w("| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Details |\n")
            w("|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|--------|\n")
            for ml in ["AR","FLY","TASD","TASD_NG"]:
                sm=s[ml]; det=""
                if ml=="TASD_NG": det=f"ngram_rds={sm['ngram_rounds']}, model_rds={sm['model_rounds']}"
                w(f"| **{ml}** | **{sm['sp_avg']:.3f}x** | {sm['sq_avg']:.4f} | {sm['off_str_avg']:.4f} | {sm['rep_avg']:.4f} | {sm['trunc_avg']:.4f} | {sm['below']} | {sm['hard']} | {det} |\n")
            w("\n")

        ml=["AR","FLY","TASD","TASD_NG"]; n=3
        ov={m:{"sp":[],"sq":[],"off":[],"below":0,"hard":0} for m in ml}
        for bn in res:
            for m in ml:
                ov[m]["sp"].append(sums[bn][m]["sp_avg"])
                ov[m]["sq"].append(sums[bn][m]["sq_avg"])
                ov[m]["off"].append(sums[bn][m]["off_str_avg"])
                ov[m]["below"]+=sums[bn][m]["below"]
                ov[m]["hard"]+=sums[bn][m]["hard"]

        w("## Overall (60 samples)\n\n")
        w("| Method | Speedup | SQ | OffStr | Below | Hard |\n")
        w("|--------|:-------:|:--:|:------:|:-----:|:----:|\n")
        for m in ml:
            w(f"| **{m}** | **{sum(ov[m]['sp'])/n:.3f}x** | {sum(ov[m]['sq'])/n:.4f} | {sum(ov[m]['off'])/n:.4f} | {ov[m]['below']}/15 | {ov[m]['hard']}/15 |\n")
        w("\n")

        # Judgment
        ng_sp=sum(ov["TASD_NG"]["sp"])/n; fly_sp=sum(ov["FLY"]["sp"])/n
        tasd_sp=sum(ov["TASD"]["sp"])/n
        ng_below=ov["TASD_NG"]["below"]; fly_below=ov["FLY"]["below"]
        tasd_below=ov["TASD"]["below"]

        w("## Judgment\n\n")
        w(f"- TASD-NG: **{ng_sp:.3f}x** vs FLY **{fly_sp:.3f}x** vs TASD **{tasd_sp:.3f}x**\n")
        w(f"- Below-1.0x: TASD-NG {ng_below}/15 vs FLY {fly_below}/15 vs TASD {tasd_below}/15\n\n")

        if ng_sp >= fly_sp*0.95:
            w(f"**TASD-NG preserves {ng_sp/fly_sp*100:.0f}% of FLY speed with TASD structure safety.**\n")
            if ng_below <= fly_below:
                w("And has fewer below-1.0x cases than FLY.\n")
        elif ng_sp >= tasd_sp:
            w(f"**TASD-NG improves over TASD ({tasd_sp:.3f}x → {ng_sp:.3f}x, +{ng_sp-tasd_sp:.3f}x)** via n-gram PLD, though not yet reaching FLY's {fly_sp:.3f}x.\n")
        else:
            w("**TASD-NG does not improve over TASD.** N-gram integration needs refinement.\n")

        w(f"\n**Recommendation**: ")
        if ng_sp >= fly_sp*0.90 and ng_below <= fly_below:
            w("TASD-NG is promising. Scale to full 6×80 with n-gram parameter sweep.\n")
        elif ng_sp >= tasd_sp*1.05:
            w("TASD-NG shows improvement over TASD. Tune n-gram params (min/max/len) before committing.\n")
        else:
            w("N-gram PLD in TASD draft stage needs structural tuning or different parameters.\n")

        w(f"\n## Raw Data\n\n- `results/tasd_ng_pilot_3x20.json`\n")

if __name__=="__main__":
    main()
