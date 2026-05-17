import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

from autoregressive_decode import autoregressive_decode
from speculative_decode import speculative_decode
from fsd_decode import fsd_decode
from gpu_utils import check_gpu_before_use, select_free_gpu, check_multi_gpus_available, select_multi_gpus


def main():
    parser = argparse.ArgumentParser(description="标准自回归解码 vs 标准推测解码 vs FSD 对比测试")
    parser.add_argument("--target_model", type=str, default="./models/Qwen2.5-72B-Instruct-AWQ",
                        help="Target 模型路径 (4-bit AWQ 量化)")
    parser.add_argument("--draft_model", type=str, default="./models/Qwen2.5-7B-Instruct-GPTQ-Int4",
                        help="Draft 模型路径 (4-bit GPTQ 量化)")
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
    parser.add_argument("--fsd", action="store_true",
                        help="使用 Fuzzy Speculative Decoding (FSD)")
    parser.add_argument("--fsd_div_type", type=str, default="js_div",
                        choices=["js_div", "kl_div", "tv_div"],
                        help="FSD 散度类型")
    parser.add_argument("--fsd_div_threshold", type=float, default=0.4,
                        help="FSD 散度阈值")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定使用的 GPU ID，不指定则自动选择空闲 GPU")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="使用的 GPU 数量，>1 时启用模型并行")
    parser.add_argument("--no_wait", action="store_true",
                        help="不等待 GPU 释放，如果 GPU 被占用则直接退出")
    args = parser.parse_args()

    use_multi_gpu = args.num_gpus > 1

    if use_multi_gpu:
        print(f"\n使用 {args.num_gpus} 张 GPU 进行模型并行")
        selected_gpus = select_multi_gpus(num_gpus=args.num_gpus, min_free_memory_mb=10000)
        
        if len(selected_gpus) < args.num_gpus:
            print(f"错误: 需要 {args.num_gpus} 张 GPU，但只找到 {len(selected_gpus)} 张可用")
            return
        
        gpu_ids_str = ",".join(map(str, selected_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        print(f"已设置 CUDA_VISIBLE_DEVICES={gpu_ids_str}")
        
        if not check_multi_gpus_available(selected_gpus, wait=not args.no_wait):
            print("错误: GPU 不可用，程序退出")
            return
    elif args.gpu is not None:
        gpu_id = args.gpu
        print(f"使用指定的 GPU: {gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        if not check_gpu_before_use(gpu_id, wait=not args.no_wait):
            print(f"错误: GPU {gpu_id} 不可用，程序退出")
            return
    else:
        gpu_id = select_free_gpu()
        if gpu_id == -1:
            print("错误: 没有可用的 GPU，程序退出")
            return
        print(f"自动选择空闲 GPU: {gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        if not check_gpu_before_use(gpu_id, wait=not args.no_wait):
            print(f"错误: GPU {gpu_id} 不可用，程序退出")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")

    print(f"\n加载 tokenizer: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    ar_text, ar_stats = None, None
    spec_text, spec_stats = None, None

    if not args.skip_spec:
        print(f"\n{'#'*60}")
        if args.fsd:
            print(f"# 测试 1: Fuzzy Speculative Decoding (FSD)")
        else:
            print(f"# 测试 1: 标准推测解码")
        print(f"{'#'*60}")

        print(f"\n加载 Target 模型 (4-bit AWQ): {args.target_model}")
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        target_model.eval()

        print(f"\n加载 Draft 模型 (4-bit GPTQ): {args.draft_model}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        draft_model.eval()

        if args.fsd:
            print(f"\n开始 FSD 解码...")
            spec_text, spec_stats = fsd_decode(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                fsd_div_type=args.fsd_div_type,
                fsd_div_threshold=args.fsd_div_threshold,
                verbose=True,
            )
        else:
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

        del draft_model
        torch.cuda.empty_cache()

    if not args.skip_ar:
        print(f"\n{'#'*60}")
        print(f"# 测试 2: 标准自回归解码")
        print(f"{'#'*60}")

        if not args.skip_spec:
            print(f"\n复用已加载的 Target 模型 (4-bit AWQ): {args.target_model}")
        else:
            print(f"\n加载 Target 模型 (4-bit AWQ): {args.target_model}")
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target_model,
                torch_dtype="auto",
                device_map="auto",
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
        if args.fsd:
            print(f"  FSD 接受率: {spec_stats['fsd_acceptance_rate']:.2%}")
        print(f"  总接受率: {spec_stats['acceptance_rate']:.2%}")
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
