#!/usr/bin/env python3
"""
FLY baseline pilot: 3 benchmarks x 20 samples, 1.5B draft, compared to TASD/AR/GSD.

Integrates SPDGenerate from FLy-main as a standalone decoder.
"""

import json
import time
import torch
import sys
import os
import statistics
import logging

sys.path.insert(0, ".")

# Direct import to avoid FLY package dependency chain
import importlib.util
_spec = importlib.util.spec_from_file_location('fly_decode', 'FLy-main/fly/models/FLy.py')
_fly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fly)
SPDGenerate = _fly.SPDGenerate

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluator import evaluate_structural_quality

# --- Paths ---
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH  = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW     = 128
N_SAMPLES   = 20

BENCHMARKS = [
    ("openmmlab", "OpenMMLab-Config", "openmmlab_config",
     "data/ml_config_blocks_openmmlab_80.jsonl", 32.91),
    ("dict_config", "Real-Python-DictConfig", "dict_config",
     "data/codesearchnet_dict_config_blocks_80.jsonl", 32.67),
    ("pipeline_stage_config", "Pipeline-Stage-Config", "pipeline_stage_config",
     "data/pipeline_stage_config_80.jsonl", 32.24),
]

# Fixed baselines (1.5B draft, n=20) for comparison
FIXED_BASELINE = {
    "dict_config": {"tps": 51.4, "sq": 0.8443, "off": 0.0000, "trunc": 0.0445},
    "openmmlab":   {"tps": 62.8, "sq": 0.8974, "off": 0.0126, "trunc": 0.1554},
    "pipeline_stage_config": {"tps": 65.5, "sq": 0.9581, "off": 0.0303, "trunc": 0.1397},
}


def run_fly_decode(draft_model, target_model, tokenizer, prompt, temperature, fly_cfg):
    """Run FLY SPDGenerate on a single prompt."""

    # Create a simple logger
    logger = logging.getLogger("fly_pilot")
    logger.setLevel(logging.WARNING)

    spd_args = {
        "k": fly_cfg["k"],
        "total_gen_tok": fly_cfg["total_gen_tok"],
        "enable_fly": fly_cfg["enable_fly"],
        "win_len": fly_cfg["win_len"],
        "entropy_thre": fly_cfg["entropy_thre"],
        "use_ngram": fly_cfg.get("use_ngram", False),
        "max_ngram_size": fly_cfg.get("max_ngram_size", 3),
        "num_ngram_pred_tokens": fly_cfg.get("num_ngram_pred_tokens", 10),
        "verbose": False,
        "abla_no_window": fly_cfg.get("abla_no_window", False),
        "enable_statistics": True,
        "tree_verify": False,
        "branch_n": 10,
        "max_nodes_per_level": 10,
        "max_nodes_global": 100,
    }

    gen = SPDGenerate(draft_model, target_model, tokenizer, logger, spd_args)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Handle pad==eos edge case
        pass

    _ = torch.cuda.synchronize()
    t0 = time.time()

    full_ids = gen.generate_chunks(input_ids, temperature)

    _ = torch.cuda.synchronize()
    wall = time.time() - t0

    prompt_len = input_ids.shape[1]
    gen_ids = full_ids[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    tps = gen_len / wall if wall > 0 else 0

    # Stats from FLY's internal counters
    accepted = gen.num_accepted_tokens.item()
    emitted = gen.num_emitted_tokens.item()
    draft_rounds = gen.num_draft_round.item()
    accept_rate = accepted / emitted if emitted > 0 else 0.0

    fly_stats = gen.get_statistics() if fly_cfg.get("enable_statistics", False) else None

    return {
        "generated_text": gen_text,
        "generated_tokens": gen_len,
        "tps": tps,
        "wall_time": wall,
        "accept_rate": accept_rate,
        "draft_rounds": draft_rounds,
        "total_accepted": accepted,
        "total_emitted": emitted,
        "fly_stats": fly_stats,
    }


def run_benchmark(target_model, draft_model, tokenizer, bid, name, st, data_file, ar_tps, fly_cfg):
    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            samples.append(json.loads(line))
            if len(samples) >= N_SAMPLES: break

    print(f"\n{'='*60}")
    print(f"  FLY | {name} | n={N_SAMPLES}")
    print(f"{'='*60}")

    results = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]

        try:
            r = run_fly_decode(draft_model, target_model, tokenizer, prompt, 0.0, fly_cfg)
        except Exception as e:
            print(f"  ERROR [{i+1}]: {e}")
            import traceback; traceback.print_exc()
            results.append({"error": str(e), "sample_idx": i})
            continue

        gen = r["generated_text"]
        q = evaluate_structural_quality(gen, structure_type=st)

        entry = {
            "sample_idx": i, "benchmark": name,
            "method": "FLY",
            "tps": r["tps"], "wall_time": r["wall_time"],
            "accept_rate": r["accept_rate"],
            "tokens_generated": r["generated_tokens"],
            "structural_quality_score": q["structural_quality_score"],
            "severe_rate": q["severe_rate"],
            "off_structure_rate": q["off_structure_rate"],
            "repetition_rate": q["repetition_rate"],
            "truncation_rate": q["truncation_rate"],
            "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
        }
        results.append(entry)

        print(f"    [{i+1}/{N_SAMPLES}] TPS={r['tps']:.1f} acc={r['accept_rate']:.2f} "
              f"SQ={q['structural_quality_score']:.4f} wall={r['wall_time']:.1f}s")

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": True}

    tps_list = [r["tps"] for r in valid]

    def avg(k): return sum(r[k] for r in valid) / len(valid)

    summary = {
        "method": "FLY", "benchmark": name, "bid": bid,
        "n": len(valid), "n_errors": len(results) - len(valid),
        "tps_avg": avg("tps"), "tps_median": statistics.median(tps_list),
        "tps_min": min(tps_list), "tps_max": max(tps_list),
        "speedup_vs_ar": avg("tps") / ar_tps,
        "accept_rate_mean": avg("accept_rate"),
        "structural_quality_score": avg("structural_quality_score"),
        "severe_rate": avg("severe_rate"),
        "off_structure_rate": avg("off_structure_rate"),
        "repetition_rate": avg("repetition_rate"),
        "truncation_rate": avg("truncation_rate"),
        "structure_not_preserved": avg("structure_not_preserved"),
        "tokens_generated": avg("tokens_generated"),
    }

    print(f"\n  --- {name} FLY Summary ---")
    print(f"  TPS: {summary['tps_avg']:.2f} ({summary['speedup_vs_ar']:.2f}x)")
    print(f"  Accept: {summary['accept_rate_mean']:.4f}, SQ: {summary['structural_quality_score']:.4f}")
    print(f"  OffStr: {summary['off_structure_rate']:.4f}, Trunc: {summary['truncation_rate']:.4f}")

    return {"summary": summary, "per_sample": results}


def main():
    print("Loading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    fly_configs = [
        # Config A: FLY default (k=16, fly enabled, win_len=6, ngram=False)
        {"label": "FLY", "k": 16, "total_gen_tok": MAX_NEW, "enable_fly": True,
         "win_len": 6, "entropy_thre": 0.3, "use_ngram": False,
         "enable_statistics": True, "abla_no_window": False},
        # Config B: FLY-no-fly (same as Config A but without fly window acceptance = standard SD)
        {"label": "FLY-no-fly", "k": 16, "total_gen_tok": MAX_NEW, "enable_fly": False,
         "win_len": 6, "entropy_thre": 0.3, "use_ngram": False,
         "enable_statistics": True, "abla_no_window": False},
    ]

    all_runs = []
    for fc in fly_configs:
        print(f"\n{'#'*60}")
        print(f"  Config: {fc['label']}")
        print(f"{'#'*60}")
        for bid, name, st, data_file, ar_tps in BENCHMARKS:
            out = run_benchmark(target, draft, tokenizer, bid, name, st, data_file, ar_tps, fc)
            if out and "error" not in out:
                all_runs.append(out)

    # Save detailed
    os.makedirs("results", exist_ok=True)
    with open("results/fly_pilot_detailed.json", "w") as f:
        json.dump(all_runs, f, ensure_ascii=False)
    print("\nSaved: results/fly_pilot_detailed.json")

    # Generate markdown
    lines = []
    lines.append("# FLY Baseline Pilot")
    lines.append("")
    lines.append("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct")
    lines.append("**Config**: FLY (k=16, win_len=6, entropy=0.3) vs FLY-no-fly (standard SD)")
    lines.append("**n**: 20 per benchmark")
    lines.append("")
    lines.append("## FLY vs TASD (1.5B draft, d16_b2_k3)")
    lines.append("")
    lines.append("| Benchmark | Method | TPS | Speedup | Accept | SQ | OffStr | Trunc | Repetition | SNP |")
    lines.append("|-----------|--------|-----|---------|--------|----|--------|-------|------------|-----|")

    for run in all_runs:
        s = run["summary"]
        lines.append(
            f"| {s['benchmark']} | {s['method']} | {s['tps_avg']:.1f} | {s['speedup_vs_ar']:.2f}x | "
            f"{s['accept_rate_mean']:.2f} | {s['structural_quality_score']:.4f} | "
            f"{s['off_structure_rate']:.4f} | {s['truncation_rate']:.4f} | "
            f"{s['repetition_rate']:.4f} | {s['structure_not_preserved']:.4f} |"
        )
        bl = FIXED_BASELINE.get(s["bid"])
        if bl:
            lines.append(
                f"| | *TASD (ref)* | *{bl['tps']:.1f}* | *{bl['tps']/32.67:.2f}x* | — | "
                f"*{bl['sq']:.4f}* | *{bl['off']:.4f}* | *{bl['trunc']:.4f}* | — | — |"
            )

    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append("| Benchmark | FLY vs TASD TPS | FLY vs TASD SQ | FLY OffStr vs TASD | Assessment |")
    lines.append("|-----------|----------------|---------------|-------------------|------------|")

    for run in all_runs:
        s = run["summary"]
        bl = FIXED_BASELINE.get(s["bid"])
        if not bl: continue
        tps_d = s["tps_avg"] - bl["tps"]
        sq_d = s["structural_quality_score"] - bl["sq"]
        off_d = s["off_structure_rate"] - bl["off"]

        # Assessment
        parts = []
        if tps_d > 0:
            parts.append(f"FLY {tps_d:+.1f} TPS faster")
        else:
            parts.append(f"TASD {-tps_d:.1f} TPS faster")
        if sq_d < -0.02:
            parts.append(f"FLY SQ lower by {sq_d:+.4f}")
        elif sq_d > 0.02:
            parts.append(f"FLY SQ higher by {sq_d:+.4f}")
        else:
            parts.append("SQ comparable")
        if off_d > 0.01:
            parts.append(f"FLY more off-structure ({off_d:+.4f})")

        lines.append(
            f"| {s['benchmark']} | {tps_d:+.1f} ({tps_d/bl['tps']*100:+.1f}%) | "
            f"{sq_d:+.4f} | {off_d:+.4f} | {'; '.join(parts)} |"
        )

    lines.append("")

    # Check if any run has FLY enabled
    fly_enabled = [r for r in all_runs if "FLY" in r["summary"]["method"] and "no-fly" not in r["summary"]["method"]]
    if fly_enabled:
        fly_sq = [r["summary"]["off_structure_rate"] for r in fly_enabled]
        lines.append(f"**FLY off_structure_rate**: {fly_sq}")
    
    nofly = [r for r in all_runs if "no-fly" in r["summary"]["method"]]
    if nofly:
        nofly_sq = [r["summary"]["off_structure_rate"] for r in nofly]
        lines.append(f"**FLY-no-fly off_structure_rate**: {nofly_sq}")

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("FLY uses modified rejection sampling for speculative decoding with optional")
    lines.append("window-based acceptance (\"FLy\" mode). It is a training-free method like TASD,")
    lines.append("but does not incorporate structural awareness (guard, structural evaluator).")
    lines.append("Both FLY and TASD use the same KV-cache mechanism and greedy target verification.")

    with open("results/fly_pilot_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("\nWritten: results/fly_pilot_summary.md")


if __name__ == "__main__":
    main()
