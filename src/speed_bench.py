import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from tasd_decode import tasd_decode
from speculative_decode import speculative_decode
from autoregressive_decode import autoregressive_decode


def run_speed_bench(
    target_model_path: str,
    draft_model_path: str,
    split: str = "qualitative",
    max_samples: int = None,
    max_new_tokens: int = 128,
    gamma: int = 5,
    epsilon: float = 0.05,
    temperature: float = 0.0,
    output_dir: str = "./results",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading target model: {target_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    target_model.eval()
    
    print(f"Loading draft model: {draft_model_path}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    draft_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading SPEED-Bench ({split})...")
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    dataset = load_dataset("nvidia/SPEED-Bench", split, split="test")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Config: gamma={gamma}, epsilon={epsilon}, max_new_tokens={max_new_tokens}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    category_stats = {}
    
    for idx, sample in enumerate(dataset):
        question_id = sample["question_id"]
        category = sample.get("category", "unknown")
        sub_category = sample.get("sub_category", "unknown")
        turns = sample["turns"]
        
        prompt = turns[0] if isinstance(turns, list) else str(turns)
        
        print(f"\n[{idx+1}/{len(dataset)}] {question_id} ({category}/{sub_category})")
        
        t0 = time.time()
        _, ar_stats = autoregressive_decode(
            target_model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature, verbose=False
        )
        ar_time = time.time() - t0
        
        t0 = time.time()
        _, sd_stats = speculative_decode(
            target_model, draft_model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, gamma=gamma, temperature=temperature, verbose=False
        )
        sd_time = time.time() - t0
        
        t0 = time.time()
        _, tasd_stats = tasd_decode(
            target_model, draft_model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, gamma=gamma, epsilon=epsilon,
            temperature=temperature, verbose=False
        )
        tasd_time = time.time() - t0
        
        result = {
            "question_id": question_id,
            "category": category,
            "sub_category": sub_category,
            "difficulty": sample.get("difficulty", "unknown"),
            "multiturn": sample.get("multiturn", False),
            "autoregressive": {
                "tokens": ar_stats["generated_tokens"],
                "time": ar_time,
                "tokens_per_second": ar_stats["tokens_per_second"],
            },
            "speculative": {
                "tokens": sd_stats["generated_tokens"],
                "time": sd_time,
                "tokens_per_second": sd_stats["tokens_per_second"],
                "acceptance_rate": sd_stats["acceptance_rate"],
                "target_forwards": sd_stats["target_forward_count"],
                "draft_forwards": sd_stats["draft_forward_count"],
            },
            "tasd": {
                "tokens": tasd_stats["generated_tokens"],
                "time": tasd_time,
                "tokens_per_second": tasd_stats["tokens_per_second"],
                "acceptance_rate": tasd_stats["verification_acceptance_rate"],
                "target_forwards": tasd_stats["target_forward_count"],
                "draft_forwards": tasd_stats["draft_forward_count"],
                "epsilon": epsilon,
            },
        }
        results.append(result)
        
        if category not in category_stats:
            category_stats[category] = {"ar": [], "sd": [], "tasd": []}
        
        category_stats[category]["ar"].append(ar_stats["tokens_per_second"])
        category_stats[category]["sd"].append(sd_stats["tokens_per_second"])
        category_stats[category]["tasd"].append(tasd_stats["tokens_per_second"])
        
        print(f"  AR: {ar_stats['tokens_per_second']:.1f} tok/s | "
              f"SD: {sd_stats['tokens_per_second']:.1f} tok/s (acc={sd_stats['acceptance_rate']:.2%}) | "
              f"TASD: {tasd_stats['tokens_per_second']:.1f} tok/s (acc={tasd_stats['verification_acceptance_rate']:.2%})")
        
        if (idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"speed_bench_{split}_checkpoint_{idx+1}.json")
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [Checkpoint saved to {checkpoint_path}]")
    
    summary = compute_summary(results, category_stats)
    
    final_path = os.path.join(output_dir, f"speed_bench_{split}_final.json")
    with open(final_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    
    print_summary(summary)
    
    return results, summary


def compute_summary(results, category_stats):
    ar_tps = [r["autoregressive"]["tokens_per_second"] for r in results]
    sd_tps = [r["speculative"]["tokens_per_second"] for r in results]
    tasd_tps = [r["tasd"]["tokens_per_second"] for r in results]
    
    sd_acc = [r["speculative"]["acceptance_rate"] for r in results]
    tasd_acc = [r["tasd"]["acceptance_rate"] for r in results]
    
    summary = {
        "overall": {
            "samples": len(results),
            "ar_mean_tps": sum(ar_tps) / len(ar_tps) if ar_tps else 0,
            "sd_mean_tps": sum(sd_tps) / len(sd_tps) if sd_tps else 0,
            "tasd_mean_tps": sum(tasd_tps) / len(tasd_tps) if tasd_tps else 0,
            "sd_speedup": (sum(sd_tps) / len(sd_tps)) / (sum(ar_tps) / len(ar_tps)) if ar_tps else 0,
            "tasd_speedup": (sum(tasd_tps) / len(tasd_tps)) / (sum(ar_tps) / len(ar_tps)) if ar_tps else 0,
            "sd_mean_acc": sum(sd_acc) / len(sd_acc) if sd_acc else 0,
            "tasd_mean_acc": sum(tasd_acc) / len(tasd_acc) if tasd_acc else 0,
        },
        "by_category": {},
    }
    
    for cat, stats in category_stats.items():
        if stats["ar"] and stats["sd"] and stats["tasd"]:
            ar_mean = sum(stats["ar"]) / len(stats["ar"])
            sd_mean = sum(stats["sd"]) / len(stats["sd"])
            tasd_mean = sum(stats["tasd"]) / len(stats["tasd"])
            
            summary["by_category"][cat] = {
                "samples": len(stats["ar"]),
                "ar_mean_tps": ar_mean,
                "sd_mean_tps": sd_mean,
                "tasd_mean_tps": tasd_mean,
                "sd_speedup": sd_mean / ar_mean if ar_mean > 0 else 0,
                "tasd_speedup": tasd_mean / ar_mean if ar_mean > 0 else 0,
            }
    
    return summary


def print_summary(summary):
    print(f"\n{'='*80}")
    print(f"  SPEED-Bench 总结")
    print(f"{'='*80}")
    
    overall = summary["overall"]
    print(f"\n总体 ({overall['samples']} samples):")
    print(f"  AR:  {overall['ar_mean_tps']:.1f} tokens/s")
    print(f"  SD:  {overall['sd_mean_tps']:.1f} tokens/s (speedup: {overall['sd_speedup']:.2f}x, acc: {overall['sd_mean_acc']:.2%})")
    print(f"  TASD: {overall['tasd_mean_tps']:.1f} tokens/s (speedup: {overall['tasd_speedup']:.2f}x, acc: {overall['tasd_mean_acc']:.2%})")
    
    if summary["by_category"]:
        print(f"\n按类别:")
        print(f"  {'Category':<20} {'Samples':>7} {'AR':>8} {'SD':>8} {'TASD':>8} {'SD↑':>6} {'TASD↑':>7}")
        print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7}")
        for cat, stats in summary["by_category"].items():
            print(f"  {cat:<20} {stats['samples']:>7} {stats['ar_mean_tps']:>7.1f} {stats['sd_mean_tps']:>7.1f} "
                  f"{stats['tasd_mean_tps']:>7.1f} {stats['sd_speedup']:>5.2f}x {stats['tasd_speedup']:>5.2f}x")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SPEED-Bench for TASD")
    parser.add_argument("--target_model", type=str, default="./models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    parser.add_argument("--draft_model", type=str, default="./models/LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--split", type=str, default="qualitative", choices=["qualitative", "throughput_1k", "throughput_2k", "throughput_8k"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    run_speed_bench(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        epsilon=args.epsilon,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
