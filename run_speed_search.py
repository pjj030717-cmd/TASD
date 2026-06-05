#!/usr/bin/env python3
"""
TASD Speed Parameter Search

7 configurations on 2 benchmarks (20 samples each).
Fixes: enable_guard=True, enable_relaxed_accept=True, prefix_budget=0.2, window_len=2.
Varies: draft_len, draft_blocks, top_k_accept.
"""

import json
import time
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode
from src.structural_guard import StructuralGuard

# --- Config ---
TARGET_MODEL = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_MODEL = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 20

BENCHMARKS = [
    {
        "name": "OpenMMLab-Config",
        "id": "openmmlab",
        "structure_type": "openmmlab_config",
        "data_file": "data/ml_config_blocks_openmmlab_80.jsonl",
        "ar_tps": 32.91,
    },
    {
        "name": "Real-Python-DictConfig",
        "id": "dictconfig",
        "structure_type": "dict_config",
        "data_file": "data/codesearchnet_dict_config_blocks_80.jsonl",
        "ar_tps": 32.67,
    },
]

# 7 sweeps, ordered
SWEEPS = [
    {"label": "d8_b2_k3",    "draft_len": 8,  "draft_blocks": 2, "top_k_accept": 3},
    {"label": "d8_b1_k3",    "draft_len": 8,  "draft_blocks": 1, "top_k_accept": 3},
    {"label": "d8_b3_k3",    "draft_len": 8,  "draft_blocks": 3, "top_k_accept": 3},
    {"label": "d4_b2_k3",    "draft_len": 4,  "draft_blocks": 2, "top_k_accept": 3},
    {"label": "d16_b2_k3",   "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3},
    {"label": "d8_b2_k1",    "draft_len": 8,  "draft_blocks": 2, "top_k_accept": 1},
    {"label": "d8_b2_k5",    "draft_len": 8,  "draft_blocks": 2, "top_k_accept": 5},
]


def load_samples(data_file, limit):
    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def evaluate_structure(generated_text, structure_type):
    guard = StructuralGuard(structure_type=structure_type)
    is_safe, _, _ = guard.check(generated_text, tokens=None, tokenizer=None)

    has_off = any(kw in generated_text for kw in ["def ", "class ", "import ", "from ", "@"])

    lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    repeat_count = 0
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            repeat_count += 1
    repetition_rate = repeat_count / max(len(lines), 1)

    truncated = not (generated_text.rstrip().endswith(")") or
                     generated_text.rstrip().endswith("]") or
                     generated_text.rstrip().endswith("}") or
                     generated_text.rstrip().endswith(","))

    brackets = {"(": ")", "[": "]", "{": "}"}
    stack = []
    severe = False
    for ch in generated_text:
        if ch in brackets:
            stack.append(ch)
        elif ch in brackets.values():
            if not stack or ch != brackets[stack.pop()]:
                severe = True
                break
    if stack:
        severe = True

    return {
        "structural_quality_score": 1.0 if is_safe else 0.7,
        "severe": 1.0 if severe else 0.0,
        "off_structure": 1.0 if has_off else 0.0,
        "repetition": repetition_rate,
        "truncation": 1.0 if truncated else 0.0,
        "structure_not_preserved": 0.0 if is_safe else 1.0,
    }


def run_sweep(model, draft_model, tokenizer, samples, sweep, benchmark):
    results = []
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        _ = torch.cuda.synchronize()
        t0 = time.time()

        try:
            result = tasd_decode(
                target_model=model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                structure_type=benchmark["structure_type"],
                max_new_tokens=MAX_NEW_TOKENS,
                draft_len=sweep["draft_len"],
                draft_blocks=sweep["draft_blocks"],
                top_k_accept=sweep["top_k_accept"],
                min_token_prob=1e-4,
                prefix_budget=0.2,
                window_len=2,
                enable_guard=True,
                enable_relaxed_accept=True,
            )
        except Exception as e:
            print(f"    ERROR sample {i}: {e}")
            results.append({"error": str(e), "tps": 0})
            continue

        _ = torch.cuda.synchronize()
        wall_time = time.time() - t0

        generated = result.get("generated_text", "")
        tps = result.get("tokens_per_second", 0)
        stats = result.get("stats", {})
        accept = stats.get("accept_rate", 0)
        sq = evaluate_structure(generated, benchmark["structure_type"])

        print(f"    [{i+1}/{len(samples)}] TPS={tps:.1f}, accept={accept:.2f}, "
              f"SQ={sq['structural_quality_score']:.3f}, wall={wall_time:.1f}s")

        results.append({
            "sample_idx": i,
            "tps": tps,
            "wall_time": wall_time,
            "accept_rate": accept,
            "total_drafted": stats.get("total_drafted", 0),
            "total_accepted": stats.get("total_accepted", 0),
            "repair_count": stats.get("repair_count", 0),
            "guard_trigger_count": stats.get("guard_trigger_count", 0),
            **sq,
        })

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "all samples failed", "n": 0}

    def avg(key):
        vals = [r[key] for r in valid if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    return {
        "n": len(valid),
        "tps": avg("tps"),
        "speedup_vs_ar": avg("tps") / benchmark["ar_tps"],
        "accept_rate": avg("accept_rate"),
        "structural_quality_score": avg("structural_quality_score"),
        "severe_rate": avg("severe"),
        "off_structure_rate": avg("off_structure"),
        "repetition_rate": avg("repetition"),
        "truncation_rate": avg("truncation"),
        "repair_count": avg("repair_count"),
        "guard_trigger_count": avg("guard_trigger_count"),
        "per_sample": results,
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, device_map="auto", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_summary = []

    for bench in BENCHMARKS:
        bench_id = bench["id"]
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {bench['name']} (n={SAMPLE_LIMIT})")
        print(f"{'='*60}")

        samples = load_samples(bench["data_file"], SAMPLE_LIMIT)
        print(f"  Loaded {len(samples)} samples")

        for sweep in SWEEPS:
            print(f"\n  --- {sweep['label']}: "
                  f"dl={sweep['draft_len']}, db={sweep['draft_blocks']}, k={sweep['top_k_accept']} ---")

            summary = run_sweep(target, draft, tokenizer, samples, sweep, bench)
            if "error" in summary:
                print(f"    FAILED: {summary['error']}")
                continue

            print(f"    TPS: {summary['tps']:.2f}  |  spd: {summary['speedup_vs_ar']:.2f}x  |  "
                  f"acc: {summary['accept_rate']:.4f}  |  SQ: {summary['structural_quality_score']:.4f}  |  "
                  f"repair: {summary['repair_count']:.1f}")

            all_summary.append({
                "benchmark": bench["name"],
                "sweep": sweep["label"],
                "draft_len": sweep["draft_len"],
                "draft_blocks": sweep["draft_blocks"],
                "top_k_accept": sweep["top_k_accept"],
                **{k: v for k, v in summary.items() if k != "per_sample"},
            })

    # --- Generate markdown ---
    lines = []
    lines.append("# TASD Speed Parameter Search")
    lines.append("")
    lines.append(f"**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft)")
    lines.append(f"**Settings**: max_new_tokens={MAX_NEW_TOKENS}, n={SAMPLE_LIMIT}")
    lines.append(f"**Fixed**: enable_guard=True, enable_relaxed_accept=True, prefix_budget=0.2, window_len=2")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Benchmark | Sweep | dl | db | k | TPS | Speedup | Accept | SQ | Repair | GuardTrig |")
    lines.append("|-----------|-------|----|----|---|-----|---------|--------|----|--------|-----------|")

    for row in all_summary:
        lines.append(
            f"| {row['benchmark']} | {row['sweep']} | {row['draft_len']} | {row['draft_blocks']} | {row['top_k_accept']} | "
            f"{row['tps']:.2f} | {row['speedup_vs_ar']:.2f}x | {row['accept_rate']:.4f} | "
            f"{row['structural_quality_score']:.4f} | {row['repair_count']:.1f} | {row['guard_trigger_count']:.1f} |"
        )

    # Pivot by sweep
    lines.append("")
    lines.append("## By Parameter Dimension")
    lines.append("")

    sweeps_order = [s["label"] for s in SWEEPS]
    for bench in BENCHMARKS:
        lines.append(f"### {bench['name']}")
        lines.append(f"| Sweep | dl | db | k | TPS | Speedup | Accept | SQ |")
        lines.append("|-------|----|----|---|-----|---------|--------|----|")
        bench_rows = [r for r in all_summary if r["benchmark"] == bench["name"]]
        bench_rows.sort(key=lambda r: sweeps_order.index(r["sweep"]))
        for r in bench_rows:
            lines.append(
                f"| {r['sweep']} | {r['draft_len']} | {r['draft_blocks']} | {r['top_k_accept']} | "
                f"{r['tps']:.2f} | {r['speedup_vs_ar']:.2f}x | {r['accept_rate']:.4f} | "
                f"{r['structural_quality_score']:.4f} |"
            )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("*To be filled after analysis*")

    md_path = "results/speed_search_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    json_path = "results/speed_search_full.json"
    with open(json_path, "w") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print(f"\n\nWritten: {md_path}")
    print(f"Written: {json_path}")


if __name__ == "__main__":
    main()
