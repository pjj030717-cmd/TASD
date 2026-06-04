"""
GSD-only runner for Original 3 benchmarks at 80 samples.

Runs sequentially:
  1. argparse 80
  2. dict_config 80
  3. openmmlab 80

Does NOT run AR or TASD. Uses existing AR/TASD 80-sample data from prior runs.
Outputs per-benchmark JSON + combined summary markdown.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vanilla_sd_decode import greedy_sd_decode
from src.evaluator import evaluate_samples

# ============================================================
# Config
# ============================================================

BENCHMARK_MAP = {
    "argparse": "/root/autodl-tmp/data/codesearchnet_argparse_blocks_80.jsonl",
    "dict_config": "/root/autodl-tmp/data/codesearchnet_dict_config_blocks_80.jsonl",
    "openmmlab": "/root/autodl-tmp/data/ml_config_blocks_openmmlab_80.jsonl",
}

BENCHMARK_DISPLAY = {
    "argparse": "Real-Python-Argparse",
    "dict_config": "Real-Python-DictConfig",
    "openmmlab": "OpenMMLab-Config",
}

# Existing AR/TASD 80-sample data (from prior full runs)
EXISTING_AR_TPS = {
    "argparse": 32.98,
    "dict_config": 32.67,
    "openmmlab": 32.91,
}

EXISTING_TASD_TPS = {
    "argparse": 42.92,
    "dict_config": 42.62,
    "openmmlab": 47.34,
}

EXISTING_TASD_SPEEDUP = {
    "argparse": 1.30,
    "dict_config": 1.30,
    "openmmlab": 1.44,
}

MAX_NEW_TOKENS = 128
DRAFT_LEN = 5
WARMUP_SAMPLES = 1
SAMPLE_LIMIT = 80

TARGET_MODEL_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_MODEL_PATH = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"
RESULTS_DIR = "/root/autodl-tmp/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_ORDER = ["argparse", "dict_config", "openmmlab"]


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


# ============================================================
# Load models (once)
# ============================================================

def load_models():
    print(f"Loading target model: {TARGET_MODEL_PATH}")
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()
    mem = _get_gpu_memory_mb()
    print(f"  Target model memory: {mem:.0f} MB")

    print(f"Loading draft model: {DRAFT_MODEL_PATH}")
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH, trust_remote_code=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    draft_model.eval()
    mem = _get_gpu_memory_mb()
    print(f"  Total GPU memory: {mem:.0f} MB")

    return target_model, target_tokenizer, draft_model, draft_tokenizer


def load_benchmark_samples(benchmark_name):
    path = BENCHMARK_MAP[benchmark_name]
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[:SAMPLE_LIMIT]
    print(f"  Loaded {len(samples)} samples from {path}")
    return samples


# ============================================================
# Run GSD on one benchmark
# ============================================================

def run_gsd_benchmark(benchmark_name, target_model, target_tokenizer, draft_model, draft_tokenizer):
    print(f"\n{'='*60}")
    print(f"  GSD: {BENCHMARK_DISPLAY[benchmark_name]} ({SAMPLE_LIMIT} samples)")
    print(f"{'='*60}")

    samples = load_benchmark_samples(benchmark_name)
    display = BENCHMARK_DISPLAY[benchmark_name]

    gsd_results = []
    all_stats = []
    total_time = 0.0
    total_tokens = 0
    failed = []

    t_start = time.time()
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        try:
            _cuda_sync()
            result = greedy_sd_decode(
                target_model, draft_model, target_tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                draft_len=DRAFT_LEN,
            )
            gsd_results.append({
                "sample_idx": i,
                "sample_name": sample.get("name", f"sample_{i}"),
                "prompt": prompt,
                "reference": sample.get("reference", ""),
                "generated_text": result["generated_text"],
                "generated_tokens": result["generated_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "elapsed_time": result["elapsed_time"],
                "stats": result["stats"],
                "is_warmup": i < WARMUP_SAMPLES,
            })
            total_time += result["elapsed_time"]
            total_tokens += result["generated_tokens"]
            all_stats.append(result["stats"])
            is_warm = "[WARMUP]" if i < WARMUP_SAMPLES else ""
            print(f"  GSD [{i+1}/{len(samples)}] {is_warm}: {result['generated_tokens']} tokens, "
                  f"{result['tokens_per_second']:.2f} TPS, accept={result['stats']['accept_rate']:.2f}")
        except Exception as e:
            failed.append({"sample_idx": i, "error": str(e)})
            print(f"  GSD [{i+1}/{len(samples)}]: FAILED - {e}")

    t_end = time.time()

    # Compute aggregate stats (exclude warmup)
    non_warmup = [r for r in gsd_results if not r.get("is_warmup")]
    non_warmup_time = sum(r["elapsed_time"] for r in non_warmup)
    non_warmup_tokens = sum(r["generated_tokens"] for r in non_warmup)
    gsd_tps = non_warmup_tokens / non_warmup_time if non_warmup_time > 0 else 0.0

    # Aggregate GSD stats
    accept_rates = [s["accept_rate"] for s in all_stats]
    avg_accept = sum(accept_rates) / max(len(accept_rates), 1)

    # Speedup vs existing AR
    ar_tps = EXISTING_AR_TPS.get(benchmark_name, 32.0)
    gsd_speedup = gsd_tps / ar_tps if ar_tps > 0 else 0.0

    # Structural quality via evaluator
    # Map benchmark name to structure_type for evaluator
    structure_type_map = {
        "argparse": "argparse",
        "dict_config": "dict_config",
        "openmmlab": "openmmlab_config",
    }
    stype = structure_type_map.get(benchmark_name, "argparse")
    _, eval_result = evaluate_samples(gsd_results, structure_type=stype)
    sq_score = eval_result.get("structural_quality_score", 0.0)
    severe_rate = eval_result.get("severe_rate", 0.0)
    off_struct = eval_result.get("off_structure_rate", 0.0)
    repetition = eval_result.get("repetition_rate", 0.0)
    truncation = eval_result.get("truncation_rate", 0.0)
    not_preserved = eval_result.get("structure_not_preserved", 0.0)

    result = {
        "benchmark": display,
        "benchmark_key": benchmark_name,
        "sample_count": len(gsd_results),
        "warmup_samples": WARMUP_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "draft_len": DRAFT_LEN,
        "gsd_tps": round(gsd_tps, 2),
        "gsd_speedup_vs_ar": round(gsd_speedup, 4),
        "accept_rate": round(avg_accept, 4),
        "structural_quality_score": round(sq_score, 4),
        "severe_rate": round(severe_rate, 4),
        "off_structure_rate": round(off_struct, 4),
        "repetition_rate": round(repetition, 4),
        "truncation_rate": round(truncation, 4),
        "structure_not_preserved": round(not_preserved, 4),
        "failed_count": len(failed),
        "wall_clock_seconds": round(t_end - t_start, 1),
        "ar_tps": ar_tps,
        "tasd_tps": EXISTING_TASD_TPS.get(benchmark_name, 0),
        "tasd_speedup": EXISTING_TASD_SPEEDUP.get(benchmark_name, 0),
    }

    return result, {"results": gsd_results, "failed": failed, "eval": eval_result}


# ============================================================
# Generate summary markdown
# ============================================================

def generate_summary(all_gsd_results):
    """Generate combined table with AR/GSD/TASD for all 6 benchmarks."""
    # Extended benchmarks GSD 80-sample data (from prior run)
    extended_gsd = {
        "rich_cli_option_groups": {"gsd_tps": 27.39, "gsd_speedup": 0.83, "accept_rate": 0.8496, "sq": 0.9159, "off": 0.0712, "trunc": 0.0604},
        "complex_nested_config": {"gsd_tps": 27.76, "gsd_speedup": 0.85, "accept_rate": 0.8541, "sq": 0.7969, "off": 0.0194, "trunc": 0.1156},
        "pipeline_stage_config": {"gsd_tps": 25.75, "gsd_speedup": 0.80, "accept_rate": 0.8160, "sq": 0.9250, "off": 0.0000, "trunc": 0.1933},
    }
    extended_tasd = {
        "rich_cli_option_groups": {"tasd_tps": 49.12, "tasd_speedup": 1.48, "accept": 1.00, "sq": 0.9074, "off": 0.1218, "trunc": 0.0556},
        "complex_nested_config": {"tasd_tps": 48.23, "tasd_speedup": 1.47, "accept": 1.00, "sq": 0.7985, "off": 0.0198, "trunc": 0.0590},
        "pipeline_stage_config": {"tasd_tps": 49.36, "tasd_speedup": 1.53, "accept": 1.00, "sq": 0.9120, "off": 0.0000, "trunc": 0.1272},
    }
    extended_ar = {
        "rich_cli_option_groups": 33.14,
        "complex_nested_config": 32.71,
        "pipeline_stage_config": 32.24,
    }

    lines = []
    lines.append("# Final Total Experiment Table")
    lines.append("")
    lines.append(f"**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft)")
    lines.append(f"**Settings**: temperature=0.0, max_new_tokens=128, KV cache enabled, n=80 all methods")
    lines.append("")
    lines.append("| Benchmark | AR TPS | GSD TPS | GSD Spd | GSD Accept | TASD TPS | TASD Spd | TASD Accept | GSD SQ | TASD SQ | GSD OffStr | TASD OffStr | GSD Trunc | TASD Trunc |")
    lines.append("|-----------|--------|---------|---------|------------|----------|----------|-------------|--------|---------|------------|-------------|-----------|------------|")

    # Original 3
    for bm in RUN_ORDER:
        r = all_gsd_results[bm]
        display = BENCHMARK_DISPLAY[bm]
        lines.append(
            f"| {display} | {r['ar_tps']:.2f} | {r['gsd_tps']:.2f} | {r['gsd_speedup_vs_ar']:.2f}x | {r['accept_rate']:.2f} | "
            f"{r['tasd_tps']:.2f} | {r['tasd_speedup']:.2f}x | — | "
            f"{r['structural_quality_score']:.4f} | — | "
            f"{r['off_structure_rate']:.4f} | — | "
            f"{r['truncation_rate']:.4f} | — |"
        )

    # Extended 3
    ext_order = ["rich_cli_option_groups", "complex_nested_config", "pipeline_stage_config"]
    ext_names = {
        "rich_cli_option_groups": "Rich-CLI-Option-Groups",
        "complex_nested_config": "Complex-Nested-Config",
        "pipeline_stage_config": "Pipeline-Stage-Config",
    }
    for bm in ext_order:
        g = extended_gsd[bm]
        t = extended_tasd[bm]
        ar = extended_ar[bm]
        display = ext_names[bm]
        lines.append(
            f"| {display} | {ar:.2f} | {g['gsd_tps']:.2f} | {g['gsd_speedup']:.2f}x | {g['accept_rate']:.2f} | "
            f"{t['tasd_tps']:.2f} | {t['tasd_speedup']:.2f}x | {t['accept']:.2f} | "
            f"{g['sq']:.4f} | {t['sq']:.4f} | "
            f"{g['off']:.4f} | {t['off']:.4f} | "
            f"{g['trunc']:.4f} | {t['trunc']:.4f} |"
        )

    lines.append("")
    lines.append("*All benchmarks at n=80 for all methods. Original 3 GSD data from this run; extended 3 from prior 80-sample run.*")
    lines.append("")

    path = os.path.join(RESULTS_DIR, "final_total_experiment_table.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWritten: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("GSD-Only Runner for Original 3 Benchmarks (80 samples)")
    print("=" * 60)
    print(f"Run order: {RUN_ORDER}")
    print()

    target_model, target_tokenizer, draft_model, draft_tokenizer = load_models()

    all_results = {}
    for bm in RUN_ORDER:
        result, full_data = run_gsd_benchmark(
            bm, target_model, target_tokenizer, draft_model, draft_tokenizer
        )
        all_results[bm] = result

        # Save per-benchmark JSON
        json_path = os.path.join(RESULTS_DIR, f"gsd_original_{bm}_80.json")
        with open(json_path, "w") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {json_path}")

        # Print summary
        print(f"\n  --- {BENCHMARK_DISPLAY[bm]} GSD Summary ---")
        print(f"  GSD TPS: {result['gsd_tps']:.2f}")
        print(f"  GSD Speedup vs AR: {result['gsd_speedup_vs_ar']:.2f}x")
        print(f"  Accept Rate: {result['accept_rate']:.4f}")
        print(f"  SQ Score: {result['structural_quality_score']:.4f}")
        print(f"  Severe: {result['severe_rate']:.4f}")
        print(f"  Off-Struct: {result['off_structure_rate']:.4f}")
        print(f"  Repeat: {result['repetition_rate']:.4f}")
        print(f"  Trunc: {result['truncation_rate']:.4f}")
        print(f"  Not Preserved: {result['structure_not_preserved']:.4f}")
        print(f"  Wall Clock: {result['wall_clock_seconds']:.1f}s")
        print()

    # Generate final summary table
    generate_summary(all_results)

    # Print final summary
    print("\n" + "=" * 60)
    print("  FINAL GSD SUMMARY (all n=80)")
    print("=" * 60)
    for bm in RUN_ORDER:
        r = all_results[bm]
        display = BENCHMARK_DISPLAY[bm]
        print(f"  {display}: GSD={r['gsd_tps']:.2f} TPS (speedup={r['gsd_speedup_vs_ar']:.2f}x vs AR={r['ar_tps']:.2f}), accept={r['accept_rate']:.4f}, SQ={r['structural_quality_score']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
