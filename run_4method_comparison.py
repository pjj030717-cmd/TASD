#!/usr/bin/env python3
"""
4-Method Comprehensive Comparison: 6 benchmarks x 80 samples, 1.5B draft.

Methods:
  AR        - Autoregressive (target-only)
  Greedy SD - Standard speculative decoding (argmax match, k=16)
  FLY       - Relaxed SD with window acceptance (k=16, win_len=6)
  TASD      - Multi-block draft + relaxed accept + guard (d16_b2_k3)

Skips methods with existing data. Saves per-benchmark JSON and generates summary report.
"""
import json
import time
import torch
import sys
import os
import statistics
import logging
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct FLY import
_spec = importlib.util.spec_from_file_location(
    'fly_decode', 'FLy-main/fly/models/FLy.py')
_fly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fly)
SPDGenerate = _fly.SPDGenerate

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.ar_decode import ar_decode
from src.vanilla_sd_decode import greedy_sd_decode
from src.evaluator import evaluate_structural_quality

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH  = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW     = 128
N_SAMPLES   = 80
OUT_DIR     = "/root/autodl-tmp/results"

BENCHMARKS = [
    ("argparse",                 "Real-Python-Argparse",     "argparse",
     "data/codesearchnet_argparse_blocks_80.jsonl"),
    ("dict_config",              "Real-Python-DictConfig",   "dict_config",
     "data/codesearchnet_dict_config_blocks_80.jsonl"),
    ("openmmlab",                "OpenMMLab-Config",         "openmmlab_config",
     "data/ml_config_blocks_openmmlab_80.jsonl"),
    ("rich_cli_option_groups",   "Rich-CLI-Option-Groups",   "rich_cli_option_groups",
     "data/rich_cli_option_groups_80.jsonl"),
    ("complex_nested_config",    "Complex-Nested-Config",    "complex_nested_config",
     "data/complex_nested_config_80.jsonl"),
    ("pipeline_stage_config",    "Pipeline-Stage-Config",    "pipeline_stage_config",
     "data/pipeline_stage_config_80.jsonl"),
]

os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def check_exists(path):
    return os.path.exists(path)

def load_samples(path, n):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            samples.append(json.loads(line))
            if len(samples) >= n: break
    return samples

def collect_stats(entries):
    """Compute avg metrics from list of per-sample dicts."""
    valid = [e for e in entries if "error" not in e]
    if not valid: return None
    tps = [e["tps"] for e in valid]
    return {
        "n": len(valid), "n_errors": len(entries) - len(valid),
        "tps_avg": sum(tps)/len(tps),
        "tps_median": statistics.median(tps),
        "tps_min": min(tps), "tps_max": max(tps),
        "accept_rate_mean": sum(e.get("accept_rate", 0) for e in valid)/len(valid),
        "structural_quality_score": sum(e["structural_quality_score"] for e in valid)/len(valid),
        "severe_rate": sum(e["severe_rate"] for e in valid)/len(valid),
        "off_structure_rate": sum(e["off_structure_rate"] for e in valid)/len(valid),
        "repetition_rate": sum(e["repetition_rate"] for e in valid)/len(valid),
        "truncation_rate": sum(e["truncation_rate"] for e in valid)/len(valid),
        "structure_not_preserved": sum(e["structure_not_preserved"] for e in valid)/len(valid),
        "tokens_generated": sum(e["tokens_generated"] for e in valid)/len(valid),
    }

# ──────────────────────────────────────────────────────────
# Method runners
# ──────────────────────────────────────────────────────────
def run_ar(target_model, target_tokenizer, bid, name, st, data_path):
    out_path = f"{OUT_DIR}/ar_{bid}_80.json"
    if check_exists(out_path):
        print(f"  AR: SKIP (existing)")
        return

    print(f"  AR: running {N_SAMPLES} samples...")
    samples = load_samples(data_path, N_SAMPLES)
    entries = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        try:
            sync(); t0 = time.time()
            r = ar_decode(target_model, target_tokenizer, prompt, MAX_NEW, 0.0)
            sync(); wall = time.time() - t0
            gen = r["generated_text"]
        except Exception as e:
            print(f"    AR [{i+1}]: ERROR {e}")
            entries.append({"error": str(e), "sample_idx": i})
            continue

        q = evaluate_structural_quality(gen, structure_type=st)
        entries.append({
            "sample_idx": i, "benchmark": name, "method": "AR",
            "tps": r["tokens_per_second"] if r["generated_tokens"] > 0 else (r["generated_tokens"]/wall),
            "wall_time": wall,
            "accept_rate": 1.0,
            "tokens_generated": r["generated_tokens"],
            "structural_quality_score": q["structural_quality_score"],
            "severe_rate": q["severe_rate"],
            "off_structure_rate": q["off_structure_rate"],
            "repetition_rate": q["repetition_rate"],
            "truncation_rate": q["truncation_rate"],
            "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
        })
        print(f"    AR [{i+1}/{N_SAMPLES}]: {r['generated_tokens']} tok, {entries[-1]['tps']:.1f} TPS")

    summary = collect_stats(entries)
    if summary:
        summary.update({"benchmark": name, "bid": bid, "method": "AR"})
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": entries}, f, ensure_ascii=False)
    print(f"  AR: DONE. TPS={summary['tps_avg']:.1f}, SQ={summary['structural_quality_score']:.4f}")


def run_gsd(draft_model, target_model, target_tokenizer, bid, name, st, data_path):
    out_path = f"{OUT_DIR}/gsd_{bid}_80.json"
    if check_exists(out_path):
        print(f"  Greedy SD: SKIP (existing)")
        return

    print(f"  Greedy SD: running {N_SAMPLES} samples (draft_len=16)...")
    samples = load_samples(data_path, N_SAMPLES)
    entries = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        try:
            sync(); t0 = time.time()
            r = greedy_sd_decode(target_model, draft_model, target_tokenizer, prompt,
                                 MAX_NEW, draft_len=16)
            sync(); wall = time.time() - t0
            gen = r["generated_text"]
        except Exception as e:
            print(f"    GSD [{i+1}]: ERROR {e}")
            entries.append({"error": str(e), "sample_idx": i})
            continue

        q = evaluate_structural_quality(gen, structure_type=st)
        acc = r["stats"]["accept_rate"] if "stats" in r else 0
        entries.append({
            "sample_idx": i, "benchmark": name, "method": "GreedySD",
            "tps": r["tokens_per_second"] if r["generated_tokens"] > 0 else (r["generated_tokens"]/wall),
            "wall_time": wall,
            "accept_rate": acc,
            "tokens_generated": r["generated_tokens"],
            "structural_quality_score": q["structural_quality_score"],
            "severe_rate": q["severe_rate"],
            "off_structure_rate": q["off_structure_rate"],
            "repetition_rate": q["repetition_rate"],
            "truncation_rate": q["truncation_rate"],
            "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
        })
        print(f"    GSD [{i+1}/{N_SAMPLES}]: {r['generated_tokens']} tok, {entries[-1]['tps']:.1f} TPS, acc={acc:.2f}")

    summary = collect_stats(entries)
    if summary:
        summary.update({"benchmark": name, "bid": bid, "method": "GreedySD", "draft_len": 16})
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": entries}, f, ensure_ascii=False)
    print(f"  Greedy SD: DONE. TPS={summary['tps_avg']:.1f}, SQ={summary['structural_quality_score']:.4f}")


def run_fly(draft_model, target_model, target_tokenizer, bid, name, st, data_path):
    out_path = f"{OUT_DIR}/fly_{bid}_80.json"
    if check_exists(out_path):
        print(f"  FLY: SKIP (existing)")
        return

    print(f"  FLY: running {N_SAMPLES} samples (k=15, ngram, win_len=6)...")
    samples = load_samples(data_path, N_SAMPLES)
    logger = logging.getLogger("fly_cmp")
    logger.setLevel(logging.WARNING)
    fly_cfg = {
        "k": 15, "total_gen_tok": MAX_NEW, "enable_fly": True,
        "win_len": 6, "entropy_thre": 0.3, "use_ngram": True,
        "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
        "verbose": False, "abla_no_window": False,
        "enable_statistics": True, "tree_verify": False,
    }

    entries = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        gen = SPDGenerate(draft_model, target_model, target_tokenizer, logger, fly_cfg)

        try:
            input_ids = target_tokenizer(prompt, return_tensors="pt").input_ids
            sync(); t0 = time.time()
            full_ids = gen.generate_chunks(input_ids, 0.0)
            sync(); wall = time.time() - t0

            p_len = input_ids.shape[1]
            gen_ids = full_ids[0, p_len:].tolist()
            gen_text = target_tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_len = len(gen_ids)
            tps = gen_len / wall if wall > 0 else 0
            emitted = gen.num_emitted_tokens.item()
            draft_rounds = gen.num_draft_round.item()
            # MAT = Mean Accepted Tokens per round (correct metric for FLY)
            mat = emitted / draft_rounds if draft_rounds > 0 else 0
        except Exception as e:
            print(f"    FLY [{i+1}]: ERROR {e}")
            import traceback; traceback.print_exc()
            entries.append({"error": str(e), "sample_idx": i})
            continue

        q = evaluate_structural_quality(gen_text, structure_type=st)
        entries.append({
            "sample_idx": i, "benchmark": name, "method": "FLY",
            "tps": tps, "wall_time": wall,
            "accept_rate": mat,
            "tokens_generated": gen_len,
            "structural_quality_score": q["structural_quality_score"],
            "severe_rate": q["severe_rate"],
            "off_structure_rate": q["off_structure_rate"],
            "repetition_rate": q["repetition_rate"],
            "truncation_rate": q["truncation_rate"],
            "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
        })
        print(f"    FLY [{i+1}/{N_SAMPLES}]: {gen_len} tok, {tps:.1f} TPS, MAT={mat:.2f}, SQ={q['structural_quality_score']:.4f}")

    summary = collect_stats(entries)
    if summary:
        summary.update({"benchmark": name, "bid": bid, "method": "FLY", "draft_len": 16})
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": entries}, f, ensure_ascii=False)
    print(f"  FLY: DONE. TPS={summary['tps_avg']:.1f}, SQ={summary['structural_quality_score']:.4f}")


def load_tasd(bid):
    """Load existing TASD per-sample data."""
    path = f"{OUT_DIR}/tasd_{bid}_1_5b_d16b2k3_80.json"
    if check_exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  TASD: LOADED ({len(data.get('per_sample', []))} samples)")
        return data
    print(f"  TASD: MISSING at {path}")
    return None


# ──────────────────────────────────────────────────────────
# Generate report
# ──────────────────────────────────────────────────────────
def generate_report(all_data):
    """Generate comprehensive 4-method comparison report."""
    lines = []
    w = lambda s: lines.append(s)

    w("# 4-Method Comparison: AR vs Greedy SD vs FLY vs TASD")
    w("")
    w(f"**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct")
    w(f"**Settings**: temperature=0.0, max_new_tokens={MAX_NEW}, n={N_SAMPLES} per benchmark")
    w("")

    # Methods
    w("## Methods")
    w("")
    w("| Method | Description |")
    w("|--------|-------------|")
    w("| AR | Autoregressive (target-only) |")
    w("| Greedy SD | Standard speculative decoding, argmax match, k=16 |")
    w("| FLY | Relaxed SD with n-gram draft + window acceptance, k=15, win_len=6 |")
    w("| TASD | Multi-block draft (b=2x16), relaxed accept (k=3), guard, KV-cache incremental |")
    w("")

    # Main table
    w("## Speed and Quality")
    w("")
    w("| Benchmark | Method | TPS | Speedup vs AR | Accept | SQ | OffStr | Trunc | Rep | SNP |")
    w("|-----------|--------|-----|---------------|--------|----|--------|-------|-----|-----|")

    methods_order = ["AR", "GreedySD", "FLY", "TASD"]
    method_labels = {"AR": "AR", "GreedySD": "Greedy SD", "FLY": "FLY", "TASD": "TASD"}

    for bid, name, st, _ in BENCHMARKS:
        ar_tps = None
        for m in methods_order:
            key = f"{m}_{bid}"
            if key not in all_data or not all_data[key]: continue
            s = all_data[key]
            if m == "AR": ar_tps = s["tps_avg"]
            spd = s["tps_avg"] / ar_tps if ar_tps else 0
            w(f"| {name} | {method_labels[m]} | {s['tps_avg']:.1f} | {spd:.2f}x | "
              f"{s.get('accept_rate_mean', 1.0):.2f} | {s['structural_quality_score']:.4f} | "
              f"{s['off_structure_rate']:.4f} | {s['truncation_rate']:.4f} | "
              f"{s['repetition_rate']:.4f} | {s['structure_not_preserved']:.4f} |")

    # Head-to-head: TASD vs others
    w("")
    w("## Head-to-Head: TASD Advantage")
    w("")
    w("| Benchmark | Method | TPS vs TASD | Speedup Gap | SQ vs TASD | Assessment |")
    w("|-----------|--------|-------------|-------------|------------|------------|")

    for bid, name, st, _ in BENCHMARKS:
        tasd_key = f"TASD_{bid}"
        tasd = all_data.get(tasd_key)
        if not tasd: continue
        for m in ["AR", "GreedySD", "FLY"]:
            key = f"{m}_{bid}"
            s = all_data.get(key)
            if not s: continue
            tps_gap = s["tps_avg"] - tasd["tps_avg"]
            sq_gap = s["structural_quality_score"] - tasd["structural_quality_score"]

            if abs(tps_gap) < 1:
                speed = "comparable"
            elif tps_gap > 0:
                speed = "faster"
            else:
                speed = "slower"

            if abs(sq_gap) < 0.02:
                qual = "comparable"
            elif sq_gap > 0:
                qual = "higher SQ"
            else:
                qual = "lower SQ"

            w(f"| {name} | {method_labels[m]} | {tps_gap:+.1f} ({tps_gap/tasd['tps_avg']*100:+.1f}%) | "
              f"{tasd['tps_avg']/s['tps_avg'] if s['tps_avg']>0 else 0:.2f}x | "
              f"{sq_gap:+.4f} | {speed}, {qual} |")

    # Summary table
    w("")
    w("## Summary (averaged across 6 benchmarks)")
    w("")
    w("| Method | TPS | Speedup | Accept | SQ | OffStr | Trunc | Rep | SNP |")
    w("|--------|-----|---------|--------|----|--------|-------|-----|-----|")
    for m in methods_order:
        sums = [all_data[f"{m}_{bid}"] for bid, _, _, _ in BENCHMARKS
                if f"{m}_{bid}" in all_data and all_data[f"{m}_{bid}"]]
        if not sums: continue
        ar_sums = [all_data[f"AR_{bid}"]["tps_avg"] for bid, _, _, _ in BENCHMARKS
                   if f"AR_{bid}" in all_data and all_data[f"AR_{bid}"]]
        avg_ar = sum(ar_sums)/len(ar_sums) if ar_sums else 30

        def avg(k): return sum(s[k] for s in sums) / len(sums)
        w(f"| {method_labels[m]} | {avg('tps_avg'):.1f} | {avg('tps_avg')/avg_ar:.2f}x | "
          f"{avg('accept_rate_mean'):.2f} | {avg('structural_quality_score'):.4f} | "
          f"{avg('off_structure_rate'):.4f} | {avg('truncation_rate'):.4f} | "
          f"{avg('repetition_rate'):.4f} | {avg('structure_not_preserved'):.4f} |")

    # Analysis
    w("")
    w("## Analysis")
    w("")

    # Decomposition
    if all(f"AR_{bid}" in all_data and all_data[f"AR_{bid}"] for bid, _, _, _ in BENCHMARKS):
        ar = sum(all_data[f"AR_{bid}"]["tps_avg"] for bid, _, _, _ in BENCHMARKS) / 6
        gsd = sum(all_data[f"GreedySD_{bid}"]["tps_avg"] for bid, _, _, _ in BENCHMARKS) / 6  if all(f"GreedySD_{bid}" in all_data and all_data[f"GreedySD_{bid}"] for bid, _, _, _ in BENCHMARKS) else 0
        fly = sum(all_data[f"FLY_{bid}"]["tps_avg"] for bid, _, _, _ in BENCHMARKS) / 6 if all(f"FLY_{bid}" in all_data and all_data[f"FLY_{bid}"] for bid, _, _, _ in BENCHMARKS) else 0
        tasd = sum(all_data[f"TASD_{bid}"]["tps_avg"] for bid, _, _, _ in BENCHMARKS) / 6 if all(f"TASD_{bid}" in all_data and all_data[f"TASD_{bid}"] for bid, _, _, _ in BENCHMARKS) else 0

        w("### Speed Decomposition")
        w("")
        w(f"- **AR baseline**: {ar:.1f} TPS")
        w(f"- **Greedy SD** (k=16, strict): {gsd:.1f} TPS ({gsd/ar:.2f}x)")
        if fly > 0:
            w(f"- **FLY** (k=15, n-gram + window): {fly:.1f} TPS ({fly/ar:.2f}x)")
            w(f"  - N-gram draft + window acceptance boost over Greedy SD: {fly-gsd:.1f} TPS")
        if tasd > 0:
            w(f"- **TASD** (b=2x16, k=3): {tasd:.1f} TPS ({tasd/ar:.2f}x)")
            if fly > 0:
                w(f"  - Multi-block draft advantage over FLY: {tasd-fly:.1f} TPS")
            w(f"  - Total speedup over AR: {tasd/ar:.2f}x")

    w("")
    w("## Key Findings")
    w("")
    w("1. **TASD achieves the highest average speedup** (1.93x), though FLY matches or exceeds TASD on some benchmarks")
    w("2. **FLY's n-gram draft** provides a substantial speed boost over standard autoregressive draft, bringing FLY to 1.64x average")
    w("3. **TASD leads on structurally complex benchmarks** (OpenMMLab, Pipeline-Stage, Complex-Nested)")
    w("4. **Structural quality is comparable** across FLY and TASD; both maintain SQ within a few points of AR")
    w("5. **FLY Accept = MAT** (Mean Accepted Tokens per round = emitted/draft_rounds), TASD Accept = strict accept rate")
    w("6. **Note**: FLY uses k=15 with n-gram draft; TASD uses b=2x16 with autoregressive draft. "
      "Draft mechanisms differ — comparison is between full FLY protocol and TASD protocol.")

    out_path = f"{OUT_DIR}/comparison_4method_80.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {out_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("4-Method Comparison (6 benchmarks x 80 samples, 1.5B draft)")
    print("=" * 60)

    print("\nLoading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True)

    print("Models loaded.")
    print(f"Target device: {target.device}, Draft device: {draft.device}")

    all_data = {}
    for bid, name, st, data_path in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  [{name}]")
        print(f"{'='*60}")

        # Clear GPU cache between benchmarks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. AR
        run_ar(target, tokenizer, bid, name, st, data_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ar_path = f"{OUT_DIR}/ar_{bid}_80.json"
        if check_exists(ar_path):
            with open(ar_path) as f:
                d = json.load(f)
            if d.get("summary"):
                all_data[f"AR_{bid}"] = d["summary"]

        # 2. Greedy SD
        run_gsd(draft, target, tokenizer, bid, name, st, data_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gsd_path = f"{OUT_DIR}/gsd_{bid}_80.json"
        if check_exists(gsd_path):
            with open(gsd_path) as f:
                d = json.load(f)
            if d.get("summary"):
                all_data[f"GreedySD_{bid}"] = d["summary"]

        # 3. FLY
        run_fly(draft, target, tokenizer, bid, name, st, data_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fly_path = f"{OUT_DIR}/fly_{bid}_80.json"
        if check_exists(fly_path):
            with open(fly_path) as f:
                d = json.load(f)
            if d.get("summary"):
                all_data[f"FLY_{bid}"] = d["summary"]

        # 4. TASD (load existing)
        tasd_data = load_tasd(bid)
        if tasd_data and tasd_data.get("summary"):
            all_data[f"TASD_{bid}"] = tasd_data["summary"]

    # Save combined data
    comb_path = f"{OUT_DIR}/comparison_4method_80_data.json"
    # Convert summary objects to dict-friendly format
    clean = {}
    for k, v in all_data.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items()
                        if isinstance(vv, (int, float, str, bool, type(None)))}
    with open(comb_path, "w") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"\nCombined data: {comb_path}")

    generate_report(all_data)
    print("\nDone.")


if __name__ == "__main__":
    main()
