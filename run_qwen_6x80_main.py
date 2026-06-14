"""
Qwen 6×80 Main Experiment: AR / Greedy SD / N-gram SD / Official FLY / TASD calibrated.

Target: Qwen2.5-14B-Instruct-AWQ  |  Draft: Qwen2.5-1.5B-Instruct
max_new_tokens=128, temperature=0.0

Official FLY: k=15, win_len=6, entropy_thre=0.3, ngram=4/6
TASD: guard_calibrated=True, draft_len=16, draft_blocks=2, top_k_accept=3

Saves incremental checkpoints per benchmark.
"""
import json, os, sys, time, logging, importlib.util, shutil
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.ngram_sd_decode import ngram_sd_decode

# ─── Paths ────────────────────────────────────────────────────────────────
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested_config"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli_option_groups"),
]

# ─── Official FLY import ──────────────────────────────────────────────────
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

# ─── Method configs ───────────────────────────────────────────────────────
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

OUT_JSON = "results/qwen_5method_6x80.json"
OUT_MD = "results/qwen_5method_6x80.md"
CHECKPOINT_DIR = "results/qwen_6x80_checkpoints"

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

def compute_truncation(text):
    if not text or not text.strip(): return 1.0
    last_line = text.rstrip().split("\n")[-1].strip()
    return 0.0 if last_line and (last_line[-1] in "})]" or last_line.endswith(",") or last_line.endswith(":")) else 1.0

# ─── Decode methods ───────────────────────────────────────────────────────

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
    return {"wall": wall, "prompt_len": prompt_len, "gen_len": gen_len,
            "tps": tps, "text": text}

def run_gsd(target, draft, tokenizer, prompt, stype):
    r = tasd_decode(target, draft, tokenizer, prompt,
                    structure_type=stype,
                    enable_guard=False, enable_relaxed_accept=False,
                    **TASD_COMMON)
    stats = r["stats"]
    n_drafted = stats.get("total_drafted", 0)
    n_accepted = stats.get("total_accepted", 0)
    n_generated = stats.get("generated_length", 0)
    rounds = n_drafted / TASD_COMMON["draft_len"] if n_drafted > 0 else 0
    avg_accept_per_round = n_accepted / rounds if rounds > 0 else 0
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "text": r["generated_text"],
            "accept": stats["accept_rate"],
            "drafted": n_drafted, "accepted": n_accepted,
            "target_fw": stats.get("target_model_forwards", 0),
            "draft_fw": stats.get("draft_model_forwards", 0),
            "avg_accept_per_round": round(avg_accept_per_round, 1),
            "gen_len": n_generated}

def run_ngram(target, tokenizer, prompt):
    r = ngram_sd_decode(target, tokenizer, prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        ngram_min=3, ngram_max=8, max_draft_tokens=16)
    return {"wall": r.get("wall_time", 0), "tps": r["tokens_per_second"], "text": r["generated_text"],
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
    gen_len = len(gen_ids)
    tps = gen_len / wall if wall > 0 else 0.0
    return {"wall": wall, "prompt_len": prompt_len, "gen_len": gen_len,
            "tps": tps, "text": text,
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

# ─── Main ─────────────────────────────────────────────────────────────────

def save_checkpoint(bname, method_label, results):
    """Save per-method results to checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"{bname}_{method_label}.json"
    with open(os.path.join(CHECKPOINT_DIR, fname), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_checkpoint(bname, method_label):
    fname = os.path.join(CHECKPOINT_DIR, f"{bname}_{method_label}.json")
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)
    return None

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

    fly_logger = logging.getLogger("fly")
    fly_logger.setLevel(logging.WARNING)
    if not fly_logger.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        fly_logger.addHandler(h)

    all_data = {}  # {bname: {method_label: [per-sample]}}

    for bench_idx, (bname, data_file, stype) in enumerate(BENCHMARKS):
        print(f"\n{'#'*65}")
        print(f"## Benchmark {bench_idx+1}/6: {bname}  (stype: {stype})")
        print(f"{'#'*65}")

        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()]

        n = len(samples)
        refs = [s.get("reference", "") for s in samples]
        prompts = [s["prompt"] for s in samples]
        names = [s["name"] for s in samples]
        all_data[bname] = {}

        # ── AR ──
        ar_ckpt = load_checkpoint(bname, "AR")
        if ar_ckpt and len(ar_ckpt) == n:
            print(f"  [SKIP] AR (from checkpoint, {n} samples)")
        else:
            print(f"  AR ({n} samples)...")
            ar_ckpt = []
            for i, prompt in enumerate(prompts):
                r = run_ar(target, tokenizer, prompt)
                sq = compute_sq(r["text"], refs[i])
                trunc = compute_truncation(r["text"])
                ar_ckpt.append({"name": names[i], "ar_tps": round(r["tps"], 2),
                                "sq": round(sq, 4), "trunc": round(trunc, 4),
                                "wall": round(r["wall"], 3)})
            save_checkpoint(bname, "AR", ar_ckpt)
            print(f"    mean TPS: {sum(r['ar_tps'] for r in ar_ckpt)/n:.1f}")
        all_data[bname]["AR"] = ar_ckpt

        ar_tps_list = [r["ar_tps"] for r in ar_ckpt]

        # ── Greedy SD ──
        gsd_ckpt = load_checkpoint(bname, "GSD")
        if gsd_ckpt and len(gsd_ckpt) == n:
            print(f"  [SKIP] Greedy SD (from checkpoint)")
        else:
            print(f"  Greedy SD ({n} samples)...")
            gsd_ckpt = []
            for i in range(n):
                r = run_gsd(target, draft, tokenizer, prompts[i], stype)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                sq = compute_sq(r["text"], refs[i])
                gsd_ckpt.append({"name": names[i], "tps": round(r["tps"], 2),
                                 "sp": round(sp, 3), "sq": round(sq, 4),
                                 "accept": round(r["accept"], 4),
                                 "wall": round(r["wall"], 3)})
                if (i+1) % 20 == 0:
                    print(f"    [{i+1}/{n}] sp={sum(r['sp'] for r in gsd_ckpt)/(i+1):.3f}x")
            save_checkpoint(bname, "GSD", gsd_ckpt)
            print(f"    mean sp: {sum(r['sp'] for r in gsd_ckpt)/n:.3f}x")
        all_data[bname]["GSD"] = gsd_ckpt

        # ── N-gram SD ──
        ng_ckpt = load_checkpoint(bname, "Ngram")
        if ng_ckpt and len(ng_ckpt) == n:
            print(f"  [SKIP] N-gram SD (from checkpoint)")
        else:
            print(f"  N-gram SD ({n} samples)...")
            ng_ckpt = []
            for i in range(n):
                r = run_ngram(target, tokenizer, prompts[i])
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                sq = compute_sq(r["text"], refs[i])
                ng_ckpt.append({"name": names[i], "tps": round(r["tps"], 2),
                                "sp": round(sp, 3), "sq": round(sq, 4),
                                "accept": round(r["accept"], 4),
                                "draft_avg": round(r["draft_avg"], 1),
                                "wall": round(r["wall"], 3)})
                if (i+1) % 20 == 0:
                    print(f"    [{i+1}/{n}] sp={sum(r['sp'] for r in ng_ckpt)/(i+1):.3f}x")
            save_checkpoint(bname, "Ngram", ng_ckpt)
            print(f"    mean sp: {sum(r['sp'] for r in ng_ckpt)/n:.3f}x")
        all_data[bname]["Ngram"] = ng_ckpt

        # ── Official FLY (k=15) ──
        fly_ckpt = load_checkpoint(bname, "FLY")
        if fly_ckpt and len(fly_ckpt) == n:
            print(f"  [SKIP] Official FLY (from checkpoint)")
        else:
            print(f"  Official FLY (k=15) ({n} samples)...")
            fly_ckpt = []
            for i in range(n):
                r = run_fly(target, draft, tokenizer, prompts[i], FLY_K15, fly_logger)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                sq = compute_sq(r["text"], refs[i])
                fly_ckpt.append({"name": names[i], "tps": round(r["tps"], 2),
                                 "sp": round(sp, 3), "sq": round(sq, 4),
                                 "mat": round(r["mat"], 2),
                                 "ngram_acc": round(r["ngram_accept"], 1),
                                 "wall": round(r["wall"], 3),
                                 "gen_len": r["gen_len"]})
                if (i+1) % 10 == 0:
                    fly_mu = sum(r['sp'] for r in fly_ckpt) / (i+1)
                    print(f"    [{i+1}/{n}] sp={fly_mu:.3f}x")
            save_checkpoint(bname, "FLY", fly_ckpt)
            print(f"    mean sp: {sum(r['sp'] for r in fly_ckpt)/n:.3f}x")
        all_data[bname]["FLY"] = fly_ckpt

        # ── TASD calibrated ──
        ts_ckpt = load_checkpoint(bname, "TASD")
        if ts_ckpt and len(ts_ckpt) == n:
            print(f"  [SKIP] TASD (from checkpoint)")
        else:
            print(f"  TASD calibrated ({n} samples)...")
            ts_ckpt = []
            for i in range(n):
                r = run_tasd(target, draft, tokenizer, prompts[i], stype)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                sq = compute_sq(r["text"], refs[i])
                ts_ckpt.append({"name": names[i], "tps": round(r["tps"], 2),
                                "sp": round(sp, 3), "sq": round(sq, 4),
                                "accept": round(r["accept"], 4),
                                "repair": r["repair"], "guard": r["guard_trig"],
                                "trim": r["trim"], "hard_trim": r["hard_trim"],
                                "rep_warn": r["rep_warn"], "brack_warn": r["brack_warn"],
                                "off_str": round(r["off_str"], 4),
                                "wall": round(r["wall"], 3)})
                if (i+1) % 20 == 0:
                    ts_mu = sum(r['sp'] for r in ts_ckpt) / (i+1)
                    print(f"    [{i+1}/{n}] sp={ts_mu:.3f}x")
            save_checkpoint(bname, "TASD", ts_ckpt)
            print(f"    mean sp: {sum(r['sp'] for r in ts_ckpt)/n:.3f}x")
        all_data[bname]["TASD"] = ts_ckpt

    # ─── Aggregate ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Qwen 6×80 — Aggregate Summary")
    print(f"{'='*70}")

    # Compute per-method per-benchmark aggregated stats
    method_labels = ["AR", "GSD", "Ngram", "FLY", "TASD"]
    aggregate = {}
    overall = {m: {"sp": [], "below": 0, "total": 0} for m in method_labels}

    for bname, _, _ in BENCHMARKS:
        ad = all_data[bname]
        n = len(ad["AR"])
        aggregate[bname] = {"n": n}

        for ml in method_labels:
            entries = ad[ml]
            if ml == "AR":
                tps_vals = [r["ar_tps"] for r in entries]
                sq_vals = [r["sq"] for r in entries]
                aggregate[bname][ml] = {
                    "tps_avg": round(sum(tps_vals)/n, 1),
                    "sq_avg": round(sum(sq_vals)/n, 4),
                    "sp_avg": 1.0, "below": 0,
                }
                overall[ml]["sp"].append(1.0)
            else:
                sp_vals = [r["sp"] for r in entries]
                sq_vals = [r["sq"] for r in entries]
                below = sum(1 for s in sp_vals if s < 1.0)
                sp_avg = sum(sp_vals)/n
                aggregate[bname][ml] = {
                    "sp_avg": round(sp_avg, 3),
                    "sq_avg": round(sum(sq_vals)/n, 4),
                    "below": below,
                }
                overall[ml]["sp"].append(sp_avg)
                overall[ml]["below"] += below
                overall[ml]["total"] += n

            print(f"  {bname}/{ml}: sp={aggregate[bname][ml].get('sp_avg', 1.0):.3f}x "
                  f"sq={aggregate[bname][ml]['sq_avg']:.4f} "
                  f"below={aggregate[bname][ml].get('below', 0)}")

    # ── Save full JSON ──
    output = {
        "config": {
            "target": "Qwen2.5-14B-Instruct-AWQ",
            "draft": "Qwen2.5-1.5B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "fly_k": 15, "fly_win_len": 6, "fly_entropy_thre": 0.3,
            "fly_ngram": "4/6",
            "tasd_guard_calibrated": True,
        },
        "per_benchmark": aggregate,
        "per_sample": all_data,
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_JSON}")

    # ── Write MD report ──
    write_md_report(output, overall, method_labels)

def write_md_report(output, overall, method_labels):
    agg = output["per_benchmark"]
    cfg = output["config"]

    with open(OUT_MD, "w") as f:
        f.write("# Qwen 6×80 Main Experiment Report\n\n")
        f.write(f"**Target**: {cfg['target']}  |  **Draft**: {cfg['draft']}\n")
        f.write(f"**Config**: max_new_tokens={cfg['max_new_tokens']}, temperature={cfg['temperature']}\n\n")

        f.write("## Methods\n\n")
        f.write("| Method | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| AR | Target autoregressive (greedy) |\n")
        f.write("| Greedy SD | Target-verify greedy draft (draft_len=16, blocks=2, top_k=3, no guard) |\n")
        f.write("| N-gram SD | Pure n-gram lookup SD (ngram_min=3, max=8, no draft model) |\n")
        f.write(f"| **Official FLY** | AMD FLy SPDGenerate (k={cfg['fly_k']}, win_len={cfg['fly_win_len']}, entropy_thre={cfg['fly_entropy_thre']}, ngram={cfg['fly_ngram']}) |\n")
        f.write("| **TASD** | Structure-aware SD + Guard-v1.5 calibrated (draft_len=16, blocks=2, top_k=3) |\n\n")

        f.write("## Per-Benchmark Results\n\n")
        bnames = [b[0] for b in BENCHMARKS]
        methods_display = ["AR", "Greedy SD", "N-gram SD", "Official FLY", "TASD"]
        ml_to_display = {"AR": "AR", "GSD": "Greedy SD", "Ngram": "N-gram SD", "FLY": "Official FLY", "TASD": "TASD"}

        for bname in bnames:
            ar_tps = agg[bname]["AR"]["tps_avg"]
            f.write(f"### {bname} ({agg[bname]['n']} samples)\n\n")
            f.write(f"Baseline AR TPS: **{ar_tps:.1f}**\n\n")
            f.write("| Method | Speedup | SQ | Accept/MAT | Below 1.0x |\n")
            f.write("|--------|:-------:|:--:|:----------:|:----------:|\n")

            for ml in method_labels:
                a = agg[bname][ml]
                sp_str = f"**{a.get('sp_avg', 1.0):.3f}x**"
                sq_str = f"{a['sq_avg']:.4f}"
                below_str = str(a.get("below", 0))

                if ml == "AR":
                    acc_str = "-"
                elif ml == "FLY":
                    # Compute mean MAT across per-sample data
                    mat_vals = [r["mat"] for r in output["per_sample"][bname]["FLY"]]
                    acc_str = f"{sum(mat_vals)/len(mat_vals):.2f} (MAT)"
                else:
                    # accept rate
                    sample_key = {"GSD": "GSD", "Ngram": "Ngram", "TASD": "TASD"}[ml]
                    acc_vals = [r["accept"] for r in output["per_sample"][bname][sample_key]]
                    acc_str = f"{sum(acc_vals)/len(acc_vals):.3f}"

                f.write(f"| {ml_to_display[ml]} | {sp_str} | {sq_str} | {acc_str} | {below_str} |\n")
            f.write("\n")

        # Overall
        f.write("## Overall (6 benchmarks × 80 samples = 480 samples)\n\n")
        f.write("| Method | Speedup | Below 1.0x |\n")
        f.write("|--------|:-------:|:----------:|\n")
        overall_sp = {}
        for ml in method_labels:
            sps = overall[ml]["sp"]
            ov_sp = sum(sps) / len(sps)
            overall_sp[ml] = ov_sp
            f.write(f"| {ml_to_display[ml]} | **{ov_sp:.3f}x** | {overall[ml]['below']} |\n")
        f.write("\n")

        # Ranking
        ranked = sorted(overall_sp.items(), key=lambda x: -x[1])
        f.write("## Method Ranking\n\n")
        for rank, (ml, sp) in enumerate(ranked, 1):
            f.write(f"{rank}. **{ml_to_display[ml]}** — {sp:.3f}x\n")

        f.write(f"\n## Key Findings\n\n")
        f.write(f"- **TASD calibrated**: {overall_sp['TASD']:.3f}x overall\n")
        f.write(f"- **Official FLY**: {overall_sp['FLY']:.3f}x overall\n")
        f.write(f"- **Greedy SD**: {overall_sp['GSD']:.3f}x overall\n")
        f.write(f"- **N-gram SD alone**: {overall_sp['Ngram']:.3f}x overall\n\n")

        tsp = overall_sp["TASD"]
        gsp = overall_sp["GSD"]
        fsp = overall_sp["FLY"]
        f.write(f"- TASD vs Greedy SD: {tsp:.3f}x vs {gsp:.3f}x (delta={tsp-gsp:+.3f}x)\n")
        f.write(f"- TASD vs Official FLY: {tsp:.3f}x vs {fsp:.3f}x (delta={tsp-fsp:+.3f}x)\n")
        if tsp > gsp and tsp > fsp:
            f.write("- **TASD is the best overall method.**\n")
        f.write("\n")

        f.write("## Data Files\n\n")
        f.write(f"- Full data: `{OUT_JSON}`\n")
        f.write(f"- Checkpoints: `{CHECKPOINT_DIR}/`\n")

    print(f"Report saved to {OUT_MD}")

if __name__ == "__main__":
    main()
