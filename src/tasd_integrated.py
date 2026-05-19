import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasd_solver import quality_bound, estimate_rollback_prob, solve_optimal


def kl_divergence_torch(p: torch.Tensor, q: torch.Tensor, smoothing: float = 1e-8) -> float:
    p_smooth = p.clamp(min=smoothing)
    q_smooth = q.clamp(min=smoothing)
    p_norm = p_smooth / p_smooth.sum()
    q_norm = q_smooth / q_smooth.sum()
    kl = (p_norm * (p_norm / q_norm).log()).sum().item()
    return max(0.0, kl)


def tasd_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    gamma: int = 5,
    epsilon: float = 0.05,
    temperature: float = 0.7,
    verbose: bool = True,
):
    device = next(target_model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_ids = []

    target_forward_count = 0
    draft_forward_count = 0
    total_accepted = 0
    total_drafted = 0
    total_relaxed_accepted = 0
    total_kl_values = []

    start_time = time.time()
    target_total_time = 0.0
    draft_total_time = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASD 质量驱动宽松验证推测解码")
        print(f"Target 模型: {target_model.config.name_or_path}")
        print(f"Draft 模型: {draft_model.config.name_or_path}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Gamma (草稿长度): {gamma}")
        print(f"Epsilon (宽容度): {epsilon}")
        print(f"{'='*60}")

    with torch.no_grad():
        target_out = target_model(input_ids=input_ids, use_cache=True)
        target_past = target_out.past_key_values
        target_last_logits = target_out.logits[:, -1, :]

        draft_out = draft_model(input_ids=input_ids, use_cache=True)
        draft_past = draft_out.past_key_values
        draft_last_logits = draft_out.logits[:, -1, :]

    while len(generated_ids) < max_new_tokens:
        draft_tokens = []
        draft_logits_list = []

        for step in range(gamma):
            if len(generated_ids) + len(draft_tokens) >= max_new_tokens:
                break

            if step == 0:
                logits = draft_last_logits
            else:
                t0 = time.time()
                with torch.no_grad():
                    draft_input = torch.tensor([[draft_tokens[-1]]], device=device)
                    draft_out = draft_model(
                        input_ids=draft_input,
                        past_key_values=draft_past,
                        use_cache=True,
                    )
                    draft_past = draft_out.past_key_values
                    logits = draft_out.logits[:, -1, :]
                    draft_forward_count += 1
                    draft_total_time += time.time() - t0

            if temperature == 0.0:
                next_tok = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(next_tok.item())
            draft_logits_list.append(logits.clone())

            if next_tok.item() == tokenizer.eos_token_id:
                break

        if len(draft_tokens) == 0:
            break

        n_draft = len(draft_tokens)

        t0 = time.time()
        with torch.no_grad():
            draft_tensor = torch.tensor([draft_tokens], device=device)
            target_out = target_model(
                input_ids=draft_tensor,
                past_key_values=target_past,
                use_cache=True,
            )
        target_past = target_out.past_key_values
        target_logits = target_out.logits
        target_forward_count += 1
        target_total_time += time.time() - t0

        all_accepted = True
        accepted_count = 0

        for i in range(n_draft):
            draft_tok = draft_tokens[i]

            if i == 0:
                verify_logits = target_last_logits
            else:
                verify_logits = target_logits[:, i - 1, :]

            target_probs = F.softmax(verify_logits / temperature, dim=-1)[0]
            draft_probs = F.softmax(draft_logits_list[i] / temperature, dim=-1)[0]

            kl_div = kl_divergence_torch(target_probs, draft_probs)
            total_kl_values.append(kl_div)

            if kl_div <= epsilon:
                generated_ids.append(draft_tok)
                total_accepted += 1
                total_relaxed_accepted += 1
                accepted_count += 1
            else:
                if temperature == 0.0:
                    target_pred = torch.argmax(verify_logits, dim=-1).item()
                    generated_ids.append(target_pred)
                else:
                    accept_prob = target_probs[draft_tok].item() / draft_probs[draft_tok].item()
                    if torch.rand(1).item() < min(1.0, accept_prob):
                        generated_ids.append(draft_tok)
                        total_accepted += 1
                        accepted_count += 1
                    else:
                        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        next_tok = torch.multinomial(adjusted_probs, num_samples=1).item()
                        generated_ids.append(next_tok)
                
                all_accepted = False
                accepted_count += 1
                break

            if generated_ids[-1] == tokenizer.eos_token_id:
                all_accepted = False
                break

        total_drafted += n_draft

        if len(generated_ids) >= max_new_tokens or \
                (len(generated_ids) > 0 and generated_ids[-1] == tokenizer.eos_token_id):
            break

        if all_accepted:
            target_last_logits = target_logits[:, -1, :]
            t0 = time.time()
            with torch.no_grad():
                last_token = torch.tensor([[draft_tokens[-1]]], device=device)
                draft_out = draft_model(
                    input_ids=last_token,
                    past_key_values=draft_past,
                    use_cache=True,
                )
                draft_past = draft_out.past_key_values
                draft_last_logits = draft_out.logits[:, -1, :]
                draft_forward_count += 1
                draft_total_time += time.time() - t0
        else:
            full_seq = torch.tensor([input_ids[0].tolist() + generated_ids], device=device)
            with torch.no_grad():
                target_out = target_model(input_ids=full_seq, use_cache=True)
            target_past = target_out.past_key_values
            target_last_logits = target_out.logits[:, -1, :]
            target_forward_count += 1
            target_total_time += time.time() - t0

            with torch.no_grad():
                draft_out = draft_model(input_ids=full_seq, use_cache=True)
            draft_past = draft_out.past_key_values
            draft_last_logits = draft_out.logits[:, -1, :]

    total_time = time.time() - start_time

    generated_text = tokenizer.decode(
        input_ids[0].tolist() + generated_ids, skip_special_tokens=True
    )

    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    relaxed_rate = total_relaxed_accepted / total_drafted if total_drafted > 0 else 0
    avg_kl = sum(total_kl_values) / len(total_kl_values) if total_kl_values else 0

    stats = {
        "generated_tokens": len(generated_ids),
        "total_time": total_time,
        "target_forward_count": target_forward_count,
        "draft_forward_count": draft_forward_count,
        "target_total_time": target_total_time,
        "draft_total_time": draft_total_time,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "total_relaxed_accepted": total_relaxed_accepted,
        "acceptance_rate": acceptance_rate,
        "relaxed_acceptance_rate": relaxed_rate,
        "avg_kl_divergence": avg_kl,
        "tokens_per_second": len(generated_ids) / total_time if total_time > 0 else 0,
        "speedup_vs_target_calls": len(generated_ids) / target_forward_count if target_forward_count > 0 else 0,
        "epsilon": epsilon,
    }

    if verbose:
        print(f"\n生成文本:\n{generated_text}\n")
        print(f"统计信息:")
        print(f"  生成 token 数: {stats['generated_tokens']}")
        print(f"  总耗时: {stats['total_time']:.3f}s")
        print(f"  Target 前向次数: {stats['target_forward_count']}")
        print(f"  Draft 前向次数: {stats['draft_forward_count']}")
        print(f"  Target 前向耗时: {stats['target_total_time']:.3f}s")
        print(f"  Draft 前向耗时: {stats['draft_total_time']:.3f}s")
        print(f"  总共推测 token 数: {stats['total_drafted']}")
        print(f"  接受 token 数: {stats['total_accepted']}")
        print(f"  宽松接受 token 数: {stats['total_relaxed_accepted']}")
        print(f"  总接受率: {stats['acceptance_rate']:.2%}")
        print(f"  宽松接受率: {stats['relaxed_acceptance_rate']:.2%}")
        print(f"  平均 KL 散度: {stats['avg_kl_divergence']:.4f}")
        print(f"  宽容度 ε: {epsilon:.4f}")
        print(f"  生成速度: {stats['tokens_per_second']:.1f} tokens/s")
        print(f"  等效加速比: {stats['speedup_vs_target_calls']:.2f}x")

    return generated_text, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TASD 集成演示")
    parser.add_argument("--target_model", type=str, default="./models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    parser.add_argument("--draft_model", type=str, default="./models/LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--prompt", type=str, default="Explain machine learning in simple terms.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=0.05, help="质量预算")
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--gamma", type=int, default=None, help="手动指定草稿长度")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("="*70)
    print("  TASD 集成演示：质量驱动宽松验证推测解码")
    print("="*70)

    print(f"\n[1/4] 求解最优参数...")
    eps_star, k_star, best_cost = solve_optimal(
        delta_max=args.delta,
        K_max=args.k_max
    )
    q_bound = quality_bound(eps_star, k_star)

    if args.gamma is not None:
        k_star = args.gamma
        print(f"  使用手动指定 γ = {k_star}")
    else:
        print(f"  质量预算 δ = {args.delta}")
        print(f"  最优 ε* = {eps_star:.4f}")
        print(f"  最优 k* = {k_star}")
        print(f"  质量损失上界 = {q_bound:.4f}")

    print(f"\n[2/4] 加载模型...")
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

    print(f"\n[3/4] TASD 宽松验证推测解码 (γ={k_star}, ε={eps_star:.4f})...")
    tasd_text, tasd_stats = tasd_decode(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        gamma=k_star,
        epsilon=eps_star,
        temperature=args.temperature,
        verbose=True,
    )

    del draft_model
    torch.cuda.empty_cache()

    print(f"\n[4/4] 自回归解码（基线）...")
    from autoregressive_decode import autoregressive_decode
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
    print(f"{'指标':<30} {'自回归':>12} {'TASD':>12}")
    print(f"{'-'*70}")
    print(f"{'生成 token 数':<30} {ar_stats['generated_tokens']:>12} {tasd_stats['generated_tokens']:>12}")
    print(f"{'总耗时 (s)':<30} {ar_stats['total_time']:>12.3f} {tasd_stats['total_time']:>12.3f}")
    print(f"{'生成速度 (tokens/s)':<30} {ar_stats['tokens_per_second']:>12.1f} {tasd_stats['tokens_per_second']:>12.1f}")
    
    speedup = tasd_stats['tokens_per_second'] / ar_stats['tokens_per_second']
    print(f"{'TASD 加速比':<30} {'-':>12} {speedup:>12.2f}x")

    print(f"\nTASD 质量指标:")
    print(f"  宽容度 ε*: {eps_star:.4f}")
    print(f"  宽松接受率: {tasd_stats['relaxed_acceptance_rate']:.2%}")
    print(f"  平均 KL 散度: {tasd_stats['avg_kl_divergence']:.4f}")
    print(f"  质量损失上界: {q_bound:.4f}")

    print(f"\n测试完成!")


if __name__ == "__main__":
    main()
