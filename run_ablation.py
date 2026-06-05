#!/usr/bin/env python3
"""
TASD Ablation Runner

Variants (4 total):
  1. TASD-full         : draft_blocks=2, relaxed accept, guard on
  2. TASD-strict-only   : draft_blocks=2, strict accept only (argmax match), guard on
  3. TASD-no-guard      : draft_blocks=2, relaxed accept, guard off
  4. TASD-draft-blocks-1: draft_blocks=1, relaxed accept, guard on

Benchmarks: Real-Python-Argparse, Real-Python-DictConfig, OpenMMLab-Config
Samples: 20 each
"""

import json
import time
import torch
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode
from src.structural_guard import StructuralGuard

# --- Config ---
TARGET_MODEL = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_MODEL = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"
MAX_NEW_TOKENS = 128
SAMPLE_LIMIT = 10

BENCHMARKS = [
    {
        "name": "Real-Python-Argparse",
        "structure_type": "argparse",
        "data_file": "data/codesearchnet_argparse_blocks_80.jsonl",
        "ar_tps": 32.98,
    },
    {
        "name": "Real-Python-DictConfig",
        "structure_type": "dict_config",
        "data_file": "data/codesearchnet_dict_config_blocks_80.jsonl",
        "ar_tps": 32.67,
    },
    {
        "name": "OpenMMLab-Config",
        "structure_type": "openmmlab_config",
        "data_file": "data/ml_config_blocks_openmmlab_80.jsonl",
        "ar_tps": 32.91,
    },
]

VARIANTS = [
    {
        "id": "tasd_full",
        "label": "TASD-full",
        "draft_blocks": 2,
        "top_k_accept": 3,
        "min_token_prob": 1e-4,
        "prefix_budget": 0.2,
        "window_len": 2,
        "enable_guard": True,
        "enable_relaxed_accept": True,
    },
    {
        "id": "tasd_strict_only",
        "label": "TASD-strict-only",
        "draft_blocks": 2,
        "top_k_accept": 3,
        "min_token_prob": 1e-4,
        "prefix_budget": 0.0,
        "window_len": 0,
        "enable_guard": True,
        "enable_relaxed_accept": False,
    },
    {
        "id": "tasd_no_guard",
        "label": "TASD-no-guard",
        "draft_blocks": 2,
        "top_k_accept": 3,
        "min_token_prob": 1e-4,
        "prefix_budget": 0.2,
        "window_len": 2,
        "enable_guard": False,
        "enable_relaxed_accept": True,
    },
    {
        "id": "tasd_draft_blocks_1",
        "label": "TASD-draft-blocks-1",
        "draft_blocks": 1,
        "top_k_accept": 3,
        "min_token_prob": 1e-4,
        "prefix_budget": 0.2,
        "window_len": 2,
        "enable_guard": True,
        "enable_relaxed_accept": True,
    },
]


def load_samples(data_file: str, limit: int):
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


def evaluate_structure(generated_text: str, reference: dict, structure_type: str) -> dict:
    """Compute per-sample structural quality scores."""
    guard = StructuralGuard(structure_type=structure_type)
    tokens = None  # guard operates on text
    is_safe, _, _ = guard.check(generated_text, tokens=tokens, tokenizer=None)

    # Off-structure: def/class/import leaked into structure block
    has_off = any(kw in generated_text for kw in [
        "def ", "class ", "import ", "from ", "@",
    ])

    # Repetition: consecutive identical tokens
    lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    repeat_count = 0
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            repeat_count += 1
    repetition_rate = repeat_count / max(len(lines), 1)

    # Truncation: no closing bracket at end
    truncated = not (generated_text.rstrip().endswith(")") or
                     generated_text.rstrip().endswith("]") or
                     generated_text.rstrip().endswith("}") or
                     generated_text.rstrip().endswith(","))

    # Severe: structural bug (unbalanced brackets)
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


def run_variant(model, draft_model, tokenizer, samples, variant, benchmark):
    """Run one variant on one benchmark."""
    results = []
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        reference = sample.get("reference", {})

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
                draft_blocks=variant["draft_blocks"],
                top_k_accept=variant["top_k_accept"],
                min_token_prob=variant["min_token_prob"],
                prefix_budget=variant["prefix_budget"],
                window_len=variant["window_len"],
                enable_guard=variant["enable_guard"],
                enable_relaxed_accept=variant["enable_relaxed_accept"],
            )
        except Exception as e:
            print(f"  ERROR sample {i}: {e}")
            results.append({"error": str(e), "tps": 0})
            continue

        _ = torch.cuda.synchronize()
        wall_time = time.time() - t0

        generated = result.get("generated_text", "")
        tokens_generated = result.get("generated_tokens", 0)
        tps = result.get("tokens_per_second", 0)
        stats = result.get("stats", {})
        accept = stats.get("accept_rate", 0)

        sq = evaluate_structure(generated, reference, benchmark["structure_type"])

        print(f"    [{i+1}/{len(samples)}] TPS={tps:.1f}, accept={accept:.2f}, "
              f"SQ={sq['structural_quality_score']:.3f}, wall={wall_time:.1f}s")

        results.append({
            "sample_idx": i,
            "tps": tps,
            "wall_time": wall_time,
            "tokens_generated": tokens_generated,
            "accept_rate": stats.get("accept_rate", 0),
            "total_drafted": stats.get("total_drafted", 0),
            "total_accepted": stats.get("total_accepted", 0),
            "repair_count": stats.get("repair_count", 0),
            "guard_trigger_count": stats.get("guard_trigger_count", 0),
            "trim_count": stats.get("trim_count", 0),
            "target_model_forwards": stats.get("target_model_forwards", 0),
            "draft_model_forwards": stats.get("draft_model_forwards", 0),
            **sq,
        })

    # Aggregate
    n = len([r for r in results if "error" not in r])
    if n == 0:
        return {"error": "all samples failed", "n": 0}

    def avg(key):
        vals = [r[key] for r in results if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    return {
        "n": n,
        "tps": avg("tps"),
        "speedup_vs_ar": avg("tps") / benchmark["ar_tps"],
        "accept_rate": avg("accept_rate"),
        "structural_quality_score": avg("structural_quality_score"),
        "severe_rate": avg("severe"),
        "off_structure_rate": avg("off_structure"),
        "repetition_rate": avg("repetition"),
        "truncation_rate": avg("truncation"),
        "structure_not_preserved": avg("structure_not_preserved"),
        "repair_count": avg("repair_count"),
        "guard_trigger_count": avg("guard_trigger_count"),
        "trim_count": avg("trim_count"),
        "total_drafted": avg("total_drafted"),
        "total_accepted": avg("total_accepted"),
        "target_model_forwards": avg("target_model_forwards"),
        "per_sample": results,
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, device_map="auto", torch_dtype="auto", trust_remote_code=True, local_files_only=True
    )
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, device_map="auto", torch_dtype="auto", trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_results = {}
    all_summary = []

    for bench in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {bench['name']} (n={SAMPLE_LIMIT})")
        print(f"{'='*60}")

        samples = load_samples(bench["data_file"], SAMPLE_LIMIT)
        print(f"  Loaded {len(samples)} samples")

        bench_results = {}

        for variant in VARIANTS:
            print(f"\n  --- {variant['label']} ---")
            print(f"      draft_blocks={variant['draft_blocks']}, "
                  f"top_k_accept={variant['top_k_accept']}, "
                  f"guard={'ON' if variant['enable_guard'] else 'OFF'}")

            summary = run_variant(target, draft, tokenizer, samples, variant, bench)

            if "error" in summary:
                print(f"    FAILED: {summary['error']}")
                continue

            print(f"    TPS: {summary['tps']:.2f}  |  "
                  f"speedup: {summary['speedup_vs_ar']:.2f}x  |  "
                  f"accept: {summary['accept_rate']:.4f}  |  "
                  f"SQ: {summary['structural_quality_score']:.4f}  |  "
                  f"repair: {summary['repair_count']:.1f}  |  "
                  f"guard_triggers: {summary['guard_trigger_count']:.1f}")

            bench_results[variant["id"]] = summary
            all_summary.append({
                "benchmark": bench["name"],
                "variant": variant["label"],
                **{k: v for k, v in summary.items() if k != "per_sample"},
            })

        all_results[bench["structure_type"]] = bench_results

        # Save per-benchmark results
        bench_clean = bench["name"].lower().replace("-", "_").replace(" ", "_")
        out_file = f"results/ablation_{bench_clean}_20.json"
        with open(out_file, "w") as f:
            json.dump(bench_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_file}")

    # --- Generate summary markdown ---
    md = []
    md.append("# TASD Ablation Experiment")
    md.append("")
    md.append(f"**Model**: {TARGET_MODEL} (target) + {DRAFT_MODEL} (draft)")
    md.append(f"**Settings**: max_new_tokens={MAX_NEW_TOKENS}, n={SAMPLE_LIMIT} per benchmark")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Benchmark | Variant | TPS | Speedup | Accept | SQ | Severe | OffStr | Repeat | Trunc | Repair | GuardTrig |")
    md.append("|-----------|---------|-----|---------|--------|----|--------|--------|--------|-------|--------|-----------|")

    for row in all_summary:
        md.append(
            f"| {row['benchmark']} | {row['variant']} | "
            f"{row['tps']:.2f} | {row['speedup_vs_ar']:.2f}x | "
            f"{row['accept_rate']:.4f} | {row['structural_quality_score']:.4f} | "
            f"{row['severe_rate']:.4f} | {row['off_structure_rate']:.4f} | "
            f"{row['repetition_rate']:.4f} | {row['truncation_rate']:.4f} | "
            f"{row['repair_count']:.1f} | {row['guard_trigger_count']:.1f} |"
        )

    md.append("")
    md.append("## Key Findings")
    md.append("")
    md.append("*To be filled after analysis*")

    out_md = "results/ablation_summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    print(f"\nSummary written: {out_md}")

    # Save full JSON
    out_json = "results/ablation_full.json"
    with open(out_json, "w") as f:
        json.dump({"summary": all_summary, "detail": all_results}, f, indent=2, ensure_ascii=False)
    print(f"Full results: {out_json}")


if __name__ == "__main__":
    main()
