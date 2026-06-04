"""
Runner script for TASD KV cache optimization experiment.
Runs AR and TASD on 10 samples per benchmark, compares TPS.

Features:
- Warmup sample (first sample marked as warmup, excluded from TPS avg)
- CUDA synchronize around timing
- Memory recording (peak, after load)
- Per-sample try/except with fallback to AR on failure
- draft_blocks=2 for multi-block draft
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ar_decode import ar_decode
from src.vanilla_sd_decode import greedy_sd_decode
from src.tasd_decode import tasd_decode
from src.evaluator import evaluate_samples


BENCHMARK_MAP = {
    "argparse": "/root/autodl-tmp/data/codesearchnet_argparse_blocks_80.jsonl",
    "dict_config": "/root/autodl-tmp/data/codesearchnet_dict_config_blocks_80.jsonl",
    "openmmlab": "/root/autodl-tmp/data/ml_config_blocks_openmmlab_80.jsonl",
    "pipeline_stage_config": "/root/autodl-tmp/data/pipeline_stage_config_80.jsonl",
    "complex_nested_config": "/root/autodl-tmp/data/complex_nested_config_80.jsonl",
    "rich_cli_option_groups": "/root/autodl-tmp/data/rich_cli_option_groups_80.jsonl",
}

BENCHMARK_DISPLAY_NAMES = {
    "argparse": "Real-Python-Argparse",
    "dict_config": "Real-Python-DictConfig",
    "openmmlab": "OpenMMLab-Config",
    "pipeline_stage_config": "Pipeline-Stage-Config",
    "complex_nested_config": "Complex-Nested-Config",
    "rich_cli_option_groups": "Rich-CLI-Option-Groups",
}

TARGET_MODEL_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_MODEL_PATH = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


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

    after_target_mem = _get_gpu_memory_mb()
    print(f"  Target model memory: {after_target_mem:.0f} MB")

    # Draft model on GPU (14B AWQ ~9.5GB + 3B ~6GB fits in 24GB)
    print(f"Loading draft model: {DRAFT_MODEL_PATH}")
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH, trust_remote_code=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    draft_model.eval()

    after_draft_mem = _get_gpu_memory_mb()
    print(f"  Draft model memory (GPU): {after_draft_mem:.0f} MB")

    return target_model, target_tokenizer, draft_model, draft_tokenizer, {
        "after_target_load_mb": round(after_target_mem, 1),
        "after_draft_load_mb": round(after_draft_mem, 1),
    }


def load_benchmark(benchmark_path):
    samples = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {benchmark_path}")
    return samples


def run_benchmark(benchmark_name, target_model, target_tokenizer, draft_model, draft_tokenizer,
                  sample_limit=10, max_new_tokens=128, warmup_samples=1):
    benchmark_path = BENCHMARK_MAP[benchmark_name]
    samples = load_benchmark(benchmark_path)[:sample_limit]

    structure_type = samples[0].get("structure_type", benchmark_name)
    if structure_type == "openmmlab_config":
        structure_type = "openmmlab_config"

    # Track peak memory
    peak_memory = _get_gpu_memory_mb()

    # --- AR baseline ---
    print(f"\n=== Running AR baseline on {benchmark_name} ===")
    ar_results = []
    ar_total_time = 0.0
    ar_total_tokens = 0
    ar_failed = []

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        try:
            _cuda_sync()
            result = ar_decode(
                target_model, target_tokenizer, prompt,
                max_new_tokens=max_new_tokens,
            )
            ar_results.append({
                "sample_idx": i,
                "sample_name": sample.get("name", f"sample_{i}"),
                "prompt": prompt,
                "reference": sample.get("reference", ""),
                "generated_text": result["generated_text"],
                "generated_tokens": result["generated_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "elapsed_time": result["elapsed_time"],
                "is_warmup": i < warmup_samples,
            })
            ar_total_time += result["elapsed_time"]
            ar_total_tokens += result["generated_tokens"]
            print(f"  AR [{i+1}/{len(samples)}]: {result['generated_tokens']} tokens, {result['tokens_per_second']:.2f} TPS")
        except Exception as e:
            ar_failed.append({"sample_idx": i, "error_type": type(e).__name__, "error_msg": str(e)})
            print(f"  AR [{i+1}/{len(samples)}]: FAILED - {type(e).__name__}: {e}")

        peak_memory = max(peak_memory, _get_gpu_memory_mb())

    # Exclude warmup from TPS calculation
    ar_non_warmup_time = sum(r["elapsed_time"] for r in ar_results if not r.get("is_warmup"))
    ar_non_warmup_tokens = sum(r["generated_tokens"] for r in ar_results if not r.get("is_warmup"))
    ar_tps = ar_non_warmup_tokens / ar_non_warmup_time if ar_non_warmup_time > 0 else 0.0

    # --- Greedy SD ---
    print(f"\n=== Running Greedy SD on {benchmark_name} ===")
    gsd_results = []
    gsd_total_time = 0.0
    gsd_total_tokens = 0
    all_gsd_stats = []
    gsd_failed = []

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        try:
            _cuda_sync()
            result = greedy_sd_decode(
                target_model, draft_model, target_tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                draft_len=5,
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
                "is_warmup": i < warmup_samples,
            })
            gsd_total_time += result["elapsed_time"]
            gsd_total_tokens += result["generated_tokens"]
            all_gsd_stats.append(result["stats"])
            print(f"  GSD [{i+1}/{len(samples)}]: {result['generated_tokens']} tokens, {result['tokens_per_second']:.2f} TPS, accept_rate={result['stats']['accept_rate']:.2f}")
        except Exception as e:
            gsd_failed.append({"sample_idx": i, "error_type": type(e).__name__, "error_msg": str(e)})
            print(f"  GSD [{i+1}/{len(samples)}]: FAILED - {type(e).__name__}: {e}")

        peak_memory = max(peak_memory, _get_gpu_memory_mb())

    # Exclude warmup from TPS calculation
    gsd_non_warmup_time = sum(r["elapsed_time"] for r in gsd_results if not r.get("is_warmup"))
    gsd_non_warmup_tokens = sum(r["generated_tokens"] for r in gsd_results if not r.get("is_warmup"))
    gsd_tps = gsd_non_warmup_tokens / gsd_non_warmup_time if gsd_non_warmup_time > 0 else 0.0

    # Evaluate structural quality for GSD
    gsd_results, gsd_avg_quality = evaluate_samples(gsd_results, structure_type)

    # Aggregate GSD stats
    gsd_avg_stats = {}
    if all_gsd_stats:
        for key in all_gsd_stats[0]:
            values = [s[key] for s in all_gsd_stats if key in s and s[key] is not None]
            if not values:
                gsd_avg_stats[key] = None
                continue
            if isinstance(values[0], list):
                all_items = []
                for v in values:
                    all_items.extend(v)
                gsd_avg_stats[key] = all_items
            elif isinstance(values[0], (int, float)):
                gsd_avg_stats[key] = round(sum(values) / len(values), 4)
            else:
                gsd_avg_stats[key] = values[0]
    gsd_avg_stats.update(gsd_avg_quality)

    # --- TASD ---
    print(f"\n=== Running TASD on {benchmark_name} ===")
    tasd_results = []
    tasd_total_time = 0.0
    tasd_total_tokens = 0
    all_tasd_stats = []
    tasd_failed = []

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        try:
            _cuda_sync()
            result = tasd_decode(
                target_model, draft_model, target_tokenizer, prompt,
                structure_type=structure_type,
                max_new_tokens=max_new_tokens,
                draft_len=8,
                draft_blocks=2,
            )
            tasd_results.append({
                "sample_idx": i,
                "sample_name": sample.get("name", f"sample_{i}"),
                "prompt": prompt,
                "reference": sample.get("reference", ""),
                "generated_text": result["generated_text"],
                "generated_tokens": result["generated_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "elapsed_time": result["elapsed_time"],
                "stats": result["stats"],
                "is_warmup": i < warmup_samples,
            })
            tasd_total_time += result["elapsed_time"]
            tasd_total_tokens += result["generated_tokens"]
            all_tasd_stats.append(result["stats"])
            print(f"  TASD [{i+1}/{len(samples)}]: {result['generated_tokens']} tokens, {result['tokens_per_second']:.2f} TPS, accept_rate={result['stats']['accept_rate']:.2f}")
        except Exception as e:
            tasd_failed.append({"sample_idx": i, "error_type": type(e).__name__, "error_msg": str(e)})
            print(f"  TASD [{i+1}/{len(samples)}]: FAILED - {type(e).__name__}: {e}")

            # Fallback to AR for this sample
            try:
                _cuda_sync()
                fallback_result = ar_decode(
                    target_model, target_tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                )
                tasd_results.append({
                    "sample_idx": i,
                    "sample_name": sample.get("name", f"sample_{i}"),
                    "prompt": prompt,
                    "reference": sample.get("reference", ""),
                    "generated_text": fallback_result["generated_text"],
                    "generated_tokens": fallback_result["generated_tokens"],
                    "tokens_per_second": fallback_result["tokens_per_second"],
                    "elapsed_time": fallback_result["elapsed_time"],
                    "stats": {
                        "failed": True,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "fallback_to_ar": True,
                    },
                    "is_warmup": i < warmup_samples,
                })
                print(f"    -> Fallback to AR: {fallback_result['generated_tokens']} tokens")
            except Exception as e2:
                print(f"    -> Fallback also FAILED: {type(e2).__name__}")

        peak_memory = max(peak_memory, _get_gpu_memory_mb())

    # Exclude warmup from TPS calculation
    tasd_non_warmup_time = sum(r["elapsed_time"] for r in tasd_results if not r.get("is_warmup"))
    tasd_non_warmup_tokens = sum(r["generated_tokens"] for r in tasd_results if not r.get("is_warmup"))
    tasd_tps = tasd_non_warmup_tokens / tasd_non_warmup_time if tasd_non_warmup_time > 0 else 0.0

    # Evaluate structural quality
    tasd_results, avg_quality = evaluate_samples(tasd_results, structure_type)

    # Aggregate TASD stats
    avg_stats = {}
    if all_tasd_stats:
        for key in all_tasd_stats[0]:
            values = [s[key] for s in all_tasd_stats if key in s and s[key] is not None]
            if not values:
                avg_stats[key] = None
                continue
            if isinstance(values[0], list):
                all_items = []
                for v in values:
                    all_items.extend(v)
                avg_stats[key] = all_items
            elif isinstance(values[0], (int, float)):
                avg_stats[key] = round(sum(values) / len(values), 4)
            else:
                avg_stats[key] = values[0]
    avg_stats.update(avg_quality)

    speedup = tasd_tps / ar_tps if ar_tps > 0 else 0.0
    gsd_speedup = gsd_tps / ar_tps if ar_tps > 0 else 0.0

    summary = {
        "benchmark": benchmark_name,
        "sample_count": len(samples),
        "warmup_samples": warmup_samples,
        "ar_tps": round(ar_tps, 2),
        "gsd_tps": round(gsd_tps, 2),
        "tasd_tps": round(tasd_tps, 2),
        "gsd_speedup": round(gsd_speedup, 4),
        "speedup": round(speedup, 4),
        "ar_total_tokens": ar_total_tokens,
        "gsd_total_tokens": gsd_total_tokens,
        "tasd_total_tokens": tasd_total_tokens,
        "ar_total_time": round(ar_total_time, 4),
        "gsd_total_time": round(gsd_total_time, 4),
        "tasd_total_time": round(tasd_total_time, 4),
        "ar_failed_count": len(ar_failed),
        "gsd_failed_count": len(gsd_failed),
        "tasd_failed_count": len(tasd_failed),
        "peak_memory_mb": round(peak_memory, 1),
        "gsd_avg_stats": gsd_avg_stats,
        "tasd_avg_stats": avg_stats,
        "ar_results": ar_results,
        "gsd_results": gsd_results,
        "tasd_results": tasd_results,
        "ar_failed": ar_failed,
        "gsd_failed": gsd_failed,
        "tasd_failed": tasd_failed,
    }

    display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)

    print(f"\n{'='*60}")
    print(f"  {display_name} Summary")
    print(f"{'='*60}")
    print(f"  AR   TPS:  {ar_tps:.2f}")
    print(f"  GSD  TPS:  {gsd_tps:.2f}  (Speedup: {gsd_speedup:.2f}x, accept={gsd_avg_stats.get('accept_rate', 0):.4f})")
    print(f"  TASD TPS: {tasd_tps:.2f}  (Speedup: {speedup:.2f}x, accept={avg_stats.get('accept_rate', 0):.4f})")
    print(f"  Peak Memory: {peak_memory:.0f} MB")
    # Print structural quality metrics for all methods
    sq_keys = [
        'structural_quality_score', 'severe_rate', 'off_structure_rate',
        'repetition_rate', 'truncation_rate', 'structure_not_preserved',
    ]
    for k in sq_keys:
        gsd_v = gsd_avg_stats.get(k)
        tasd_v = avg_stats.get(k)
        if isinstance(gsd_v, bool):
            gsd_v = 1.0 if gsd_v else 0.0
        if isinstance(tasd_v, bool):
            tasd_v = 1.0 if tasd_v else 0.0
        if gsd_v is not None or tasd_v is not None:
            gsd_str = f"{gsd_v:.4f}" if gsd_v is not None else "N/A"
            tasd_str = f"{tasd_v:.4f}" if tasd_v is not None else "N/A"
            print(f"  {k}: GSD={gsd_str}  TASD={tasd_str}")
    # Print guard metrics for TASD
    guard_keys = ['guard_trigger_count', 'trim_count', 'repair_count']
    for k in guard_keys:
        if k in avg_stats:
            v = avg_stats[k]
            if isinstance(v, (int, float)):
                print(f"  TASD {k}: {v:.2f}")
    if ar_failed:
        print(f"  AR failures: {len(ar_failed)}")
    if tasd_failed:
        print(f"  TASD failures: {len(tasd_failed)}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["argparse", "dict_config", "openmmlab"])
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/results/kv_cache_results.json")
    args = parser.parse_args()

    # Load models once
    target_model, target_tokenizer, draft_model, draft_tokenizer, mem_info = load_models()

    all_results = {}
    all_results["_memory_info"] = mem_info
    all_results["_timing_info"] = {
        "timing_includes_model_load": False,
        "cuda_synchronized": True,
        "warmup_samples": args.warmup_samples,
    }

    for bench in args.benchmarks:
        summary = run_benchmark(
            bench, target_model, target_tokenizer, draft_model, draft_tokenizer,
            sample_limit=args.sample_limit,
            max_new_tokens=args.max_new_tokens,
            warmup_samples=args.warmup_samples,
        )
        all_results[bench] = summary

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {args.output}")

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    for bench, summary in all_results.items():
        if bench.startswith("_"):
            continue
        display_name = BENCHMARK_DISPLAY_NAMES.get(bench, bench)
        gsd_stats = summary.get('gsd_avg_stats', {})
        tasd_stats = summary.get('tasd_avg_stats', {})
        print(f"  {display_name:25s}: AR={summary['ar_tps']:6.2f} TPS  "
              f"GSD={summary['gsd_tps']:6.2f} TPS (speedup={summary['gsd_speedup']:.2f}x, accept={gsd_stats.get('accept_rate', 0):.4f})  "
              f"TASD={summary['tasd_tps']:6.2f} TPS (speedup={summary['speedup']:.2f}x, accept={tasd_stats.get('accept_rate', 0):.4f})")
        print(f"    GSD sq_score={gsd_stats.get('structural_quality_score', 0):.4f}  severe={gsd_stats.get('severe_rate', 0):.4f}  "
              f"off_struct={gsd_stats.get('off_structure_rate', 0):.4f}  repeat={gsd_stats.get('repetition_rate', 0):.4f}  "
              f"trunc={gsd_stats.get('truncation_rate', 0):.4f}  not_preserved={gsd_stats.get('structure_not_preserved', 0):.4f}")
        print(f"    TASD sq_score={tasd_stats.get('structural_quality_score', 0):.4f}  severe={tasd_stats.get('severe_rate', 0):.4f}  "
              f"off_struct={tasd_stats.get('off_structure_rate', 0):.4f}  repeat={tasd_stats.get('repetition_rate', 0):.4f}  "
              f"trunc={tasd_stats.get('truncation_rate', 0):.4f}  not_preserved={tasd_stats.get('structure_not_preserved', 0):.4f}  "
              f"guard={tasd_stats.get('guard_trigger_count', 0):.2f}  trim={tasd_stats.get('trim_count', 0):.2f}  repair={tasd_stats.get('repair_count', 0):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
