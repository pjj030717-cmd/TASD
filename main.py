import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from autoregressive_decode import autoregressive_decode
from speculative_decode import speculative_decode


def main():
    parser = argparse.ArgumentParser(description="标准自回归解码 vs 标准推测解码 对比测试")
    parser.add_argument("--target_model", type=str, default="/root/data/models/Qwen/Qwen2.5-7B-Instruct",
                        help="Target 模型路径")
    parser.add_argument("--draft_model", type=str, default="/root/autodl-tmp/models/Qwen/Qwen2.5-1.5B-Instruct",
                        help="Draft 模型路径")
    parser.add_argument("--prompt", type=str,
                        default="请详细解释一下什么是量子计算，以及它对未来的影响。",
                        help="输入 Prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="最大生成 token 数")
    parser.add_argument("--gamma", type=int, default=5,
                        help="每次推测的候选 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="温度参数, 0 表示贪心解码")
    parser.add_argument("--skip_ar", action="store_true",
                        help="跳过自回归解码")
    parser.add_argument("--skip_spec", action="store_true",
                        help="跳过推测解码")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    print(f"\n加载 tokenizer: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    ar_text, ar_stats = None, None
    spec_text, spec_stats = None, None

    if not args.skip_ar:
        print(f"\n{'#'*60}")
        print(f"# 测试 1: 标准自回归解码")
        print(f"{'#'*60}")
        print(f"\n加载 Target 模型: {args.target_model}")
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        target_model.eval()

        print(f"\n开始自回归解码...")
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

    if not args.skip_spec:
        print(f"\n{'#'*60}")
        print(f"# 测试 2: 标准推测解码")
        print(f"{'#'*60}")

        print(f"\n加载 Target 模型: {args.target_model}")
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        target_model.eval()

        print(f"\n加载 Draft 模型: {args.draft_model}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        draft_model.eval()

        print(f"\n开始推测解码...")
        spec_text, spec_stats = speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            temperature=args.temperature,
            verbose=True,
        )

        del target_model
        del draft_model
        torch.cuda.empty_cache()

    if not args.skip_ar and not args.skip_spec:
        print(f"\n{'#'*60}")
        print(f"# 对比总结")
        print(f"{'#'*60}")
        print(f"{'指标':<30} {'自回归':>15} {'推测解码':>15}")
        print(f"{'-'*60}")
        print(f"{'生成 token 数':<30} {ar_stats['generated_tokens']:>15} {spec_stats['generated_tokens']:>15}")
        print(f"{'总耗时 (s)':<30} {ar_stats['total_time']:>15.3f} {spec_stats['total_time']:>15.3f}")
        print(f"{'生成速度 (tokens/s)':<30} {ar_stats['tokens_per_second']:>15.1f} {spec_stats['tokens_per_second']:>15.1f}")

        if spec_stats['tokens_per_second'] > 0 and ar_stats['tokens_per_second'] > 0:
            speedup = spec_stats['tokens_per_second'] / ar_stats['tokens_per_second']
            print(f"{'实际加速比':<30} {speedup:>15.2f}x")

        print(f"\n推测解码特有指标:")
        print(f"  Target 前向次数: {spec_stats['target_forward_count']}")
        print(f"  Draft 前向次数: {spec_stats['draft_forward_count']}")
        print(f"  接受率: {spec_stats['acceptance_rate']:.2%}")
        print(f"  等效加速比: {spec_stats['speedup_vs_target_calls']:.2f}x")

        print(f"\n结果一致性检查:")
        print(f"  自回归输出 (前100字): {ar_text[:100]}...")
        print(f"  推测解码输出 (前100字): {spec_text[:100]}...")
        if ar_text == spec_text:
            print(f"  ✓ 贪心模式下输出完全一致!")
        else:
            print(f"  ! 输出不完全一致 (采样模式下这是正常的)")

    print(f"\n测试完成!")


if __name__ == "__main__":
    main()