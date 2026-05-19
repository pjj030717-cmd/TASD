import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import json
import os
import sys

sys.path.insert(0, '/hy-tmp/my-repo/src')

from tasd_decode import tasd_decode


def get_humaneval_prompt(task):
    return task["prompt"]


def run_tasd_bench(
    target_model_path: str,
    draft_model_path: str,
    max_samples: int = 60,
    max_new_tokens: int = 256,
    gamma: int = 5,
    epsilon: float = 0.05,
    temperature: float = 0.0,
    calibration_path: str = None,
    output_dir: str = "/hy-tmp/my-repo/results",
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
    
    print(f"Loading HumanEval dataset...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Config: gamma={gamma}, epsilon={epsilon}")
    if calibration_path:
        print(f"校准路径: {calibration_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for idx, sample in enumerate(dataset):
        task_id = sample["task_id"]
        prompt = get_humaneval_prompt(sample)
        
        print(f"\n[{idx+1}/{len(dataset)}] {task_id}")
        
        t0 = time.time()
        generated_text, tasd_stats = tasd_decode(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            epsilon=epsilon,
            temperature=temperature,
            verbose=False,
            calibration_path=calibration_path,
        )
        
        result = {
            "task_id": task_id,
            "tasd": tasd_stats,
        }
        results.append(result)
        
        print(f"  TASD: {tasd_stats['tokens_per_second']:.1f} tok/s (acc={tasd_stats['verification_acceptance_rate']:.2%}, tokens={tasd_stats['generated_tokens']})")
        
        if (idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"tasd_humaneval_checkpoint_{idx+1}.json")
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [Checkpoint saved to {checkpoint_path}]")
        
        if idx + 1 >= max_samples:
            break
    
    summary = compute_summary(results)
    
    final_path = os.path.join(output_dir, "tasd_humaneval_final.json")
    with open(final_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    
    print_summary(summary)
    
    return results, summary


def compute_summary(results):
    tasd_tps = [r["tasd"]["tokens_per_second"] for r in results]
    tasd_acc = [r["tasd"]["verification_acceptance_rate"] for r in results]
    tasd_tokens = [r["tasd"]["generated_tokens"] for r in results]
    
    return {
        "samples": len(results),
        "tasd_mean_tps": sum(tasd_tps) / len(tasd_tps) if tasd_tps else 0,
        "tasd_mean_acc": sum(tasd_acc) / len(tasd_acc) if tasd_acc else 0,
        "tasd_mean_tokens": sum(tasd_tokens) / len(tasd_tokens) if tasd_tokens else 0,
    }


def print_summary(summary):
    print(f"\n{'='*80}")
    print(f"  TASD HumanEval 测试结果")
    print(f"{'='*80}")
    
    print(f"\n总体 ({summary['samples']} samples):")
    print(f"  平均生成长度: {summary['tasd_mean_tokens']:.1f} tokens")
    print(f"  TASD 速度: {summary['tasd_mean_tps']:.1f} tokens/s")
    print(f"  TASD 接受率: {summary['tasd_mean_acc']:.2%}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TASD HumanEval benchmark")
    parser.add_argument("--target_model", type=str, default="/hy-tmp/my-repo/models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4")
    parser.add_argument("--draft_model", type=str, default="/hy-tmp/my-repo/models/LLM-Research/Meta-Llama-3___1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--max_samples", type=int, default=60)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--calibration_path", type=str, default=None, help="校准结果文件路径")
    parser.add_argument("--output_dir", type=str, default="/hy-tmp/my-repo/results")
    
    args = parser.parse_args()
    
    run_tasd_bench(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        epsilon=args.epsilon,
        temperature=args.temperature,
        calibration_path=args.calibration_path,
        output_dir=args.output_dir,
    )
