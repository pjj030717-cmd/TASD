import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autoregressive_decode import autoregressive_decode
from speculative_decode import speculative_decode
from tasd_solver import quality_bound, solve_optimal
from gpu_utils import check_gpu_before_use, select_free_gpu


def main():
    parser = argparse.ArgumentParser(description="质量驱动推测解码测试")
    parser.add_argument("--target_model", type=str, default="./models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                        help="Target 模型路径")
    parser.add_argument("--draft_model", type=str, default="./models/LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                        help="Draft 模型路径")
    parser.add_argument("--prompt", type=str,
                        default="Explain the concept of machine learning and its applications in healthcare.",
                        help="输入 Prompt")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="温度参数, 0 表示贪心解码")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="质量预算（TVD 上界）")
    parser.add_argument("--k_max", type=int, default=10,
                        help="最大草稿长度搜索范围")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定使用的 GPU ID")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        gpu_id = select_free_gpu()
        if gpu_id == -1:
            print("错误: 没有可用的 GPU")
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("="*70)
    print("  质量驱动推测解码 (Quality-Driven Speculative Decoding)")
    print("="*70)

    print(f"\n[1/4] 求解最优参数...")
    print(f"  质量预算 δ = {args.delta}")
    print(f"  最大草稿长度 K_max = {args.k_max}")

    eps_star, k_star, best_cost = solve_optimal(
        delta_max=args.delta,
        K_max=args.k_max
    )
    q_bound = quality_bound(eps_star, k_star)

    print(f"  最优宽容度 ε* = {eps_star:.4f}")
    print(f"  最优草稿长度 k* = {k_star}")
    print(f"  最小期望代价 = {best_cost:.4f}")
    print(f"  质量损失上界 = {q_bound:.4f}")

    print(f"\n[2/4] 加载模型...")
    print(f"  Target: {args.target_model}")
    print(f"  Draft:  {args.draft_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    draft_model.eval()

    print(f"\n[3/4] 质量驱动推测解码 (γ={k_star})...")
    spec_text, spec_stats = speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        gamma=k_star,
        temperature=args.temperature,
        verbose=True,
    )

    del draft_model
    torch.cuda.empty_cache()

    print(f"\n[4/4] 自回归解码（基线对比）...")
    ar_text, ar_stats = autoregressive_decode(
        model=target_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=True,
    )

    del target_model
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  对比总结")
    print(f"{'='*70}")
    print(f"{'指标':<30} {'自回归':>15} {'质量驱动推测':>15}")
    print(f"{'-'*70}")
    print(f"{'生成 token 数':<30} {ar_stats['generated_tokens']:>15} {spec_stats['generated_tokens']:>15}")
    print(f"{'总耗时 (s)':<30} {ar_stats['total_time']:>15.3f} {spec_stats['total_time']:>15.3f}")
    print(f"{'生成速度 (tokens/s)':<30} {ar_stats['tokens_per_second']:>15.1f} {spec_stats['tokens_per_second']:>15.1f}")

    if spec_stats['tokens_per_second'] > 0 and ar_stats['tokens_per_second'] > 0:
        speedup = spec_stats['tokens_per_second'] / ar_stats['tokens_per_second']
        print(f"{'实际加速比':<30} {speedup:>15.2f}x")

    print(f"\n质量驱动参数:")
    print(f"  质量预算 δ: {args.delta}")
    print(f"  最优 ε*: {eps_star:.4f}")
    print(f"  最优 k*: {k_star}")
    print(f"  质量损失上界: {q_bound:.4f}")

    print(f"\n推测解码指标:")
    print(f"  Target 前向次数: {spec_stats['target_forward_count']}")
    print(f"  Draft 前向次数: {spec_stats['draft_forward_count']}")
    print(f"  接受率: {spec_stats['acceptance_rate']:.2%}")
    print(f"  等效加速比: {spec_stats['speedup_vs_target_calls']:.2f}x")

    print(f"\n输出一致性:")
    if ar_text == spec_text:
        print(f"  ✓ 贪心模式下输出完全一致!")
    else:
        print(f"  ! 输出不完全一致 (采样模式下正常)")

    print(f"\n测试完成!")


if __name__ == "__main__":
    main()
