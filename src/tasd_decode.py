import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os


def kl_divergence_torch(p, q, smoothing=1e-10):
    """计算两个概率分布之间的 KL 散度"""
    p = p.clamp(min=smoothing)
    q = q.clamp(min=smoothing)
    return torch.sum(p * torch.log(p / q)).item()


def load_calibration_results(calibration_path):
    """从文件加载校准结果"""
    if not calibration_path or not os.path.exists(calibration_path):
        return None, None
    
    with open(calibration_path, "r") as f:
        results = json.load(f)
    
    alpha = results.get("alpha_calibrated", 1.0)
    
    rollback_table = {}
    for key_str, value in results.get("rollback_calibration_table", {}).items():
        parts = key_str.split("_")
        epsilon = float(parts[0])
        k = int(parts[1])
        rollback_table[(epsilon, k)] = value
    
    return alpha, rollback_table


def estimate_rollback_prob(epsilon, k, calibration_table=None):
    """
    估计回滚概率
    
    优先使用校准表，否则使用独立假设近似
    """
    if calibration_table:
        key = (round(epsilon, 3), k)
        if key in calibration_table:
            return calibration_table[key]
    
    return 1.0 - (1.0 - epsilon) ** k


def quality_bound(epsilon, k, alpha=1.0):
    """
    计算质量损失上界
    
    Δ ≤ α · k · √(ε/2)
    
    alpha=1.0 是理论 Pinsker 界
    alpha<1.0 是经验校准值（更紧）
    """
    return alpha * k * (epsilon / 2) ** 0.5


def tasd_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    gamma: int = 5,
    epsilon: float = 0.05,
    temperature: float = 0.0,
    verbose: bool = True,
    calibration_path: str = None,
):
    """
    TASD: 质量驱动的免训练宽松验证推测解码
    
    核心思想：
    - 不要求 draft 和 target 输出完全一致
    - 允许在 epsilon 宽容度内接受 draft token
    - 通过 KL 散度或概率比值判断是否接受
    
    参数:
        epsilon: KL 散度宽容度阈值，控制质量损失上界
        calibration_path: 校准结果文件路径（可选）
    """
    device = next(target_model.parameters()).device

    alpha_calibrated = 1.0
    rollback_table = None
    
    if calibration_path:
        alpha_calibrated, rollback_table = load_calibration_results(calibration_path)
        if verbose and alpha_calibrated != 1.0:
            print(f"使用校准 α = {alpha_calibrated:.4f} (默认 1.0)")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_ids = []

    target_forward_count = 0
    draft_forward_count = 0
    total_accepted = 0
    total_drafted = 0
    total_verified = 0
    total_rollbacks = 0

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
        print(f"质量损失上界: {quality_bound(epsilon, gamma, alpha_calibrated):.4f} (α={alpha_calibrated:.4f})")
        p_rollback_est = estimate_rollback_prob(epsilon, gamma, rollback_table)
        print(f"估计回滚概率: {p_rollback_est:.4f}")
        print(f"{'='*60}")

    with torch.no_grad():
        target_out = target_model(input_ids=input_ids, use_cache=True)
        target_past = target_out.past_key_values
        target_last_logits = target_out.logits[:, -1, :]

        draft_out = draft_model(input_ids=input_ids, use_cache=True)
        draft_past = draft_out.past_key_values

    while len(generated_ids) < max_new_tokens:
        draft_tokens = []

        for step in range(gamma):
            if len(generated_ids) + len(draft_tokens) >= max_new_tokens:
                break

            if step == 0:
                logits = target_last_logits
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

        accepted_count = 0
        all_accepted = True
        round_rollback = False

        for i in range(n_draft):
            draft_tok = draft_tokens[i]
            total_verified += 1

            if i == 0:
                verify_logits = target_last_logits
            else:
                verify_logits = target_logits[:, i - 1, :]

            if temperature == 0.0:
                target_pred = torch.argmax(verify_logits, dim=-1).item()
                
                if target_pred == draft_tok:
                    generated_ids.append(draft_tok)
                    total_accepted += 1
                    accepted_count += 1
                else:
                    p_target = F.softmax(verify_logits, dim=-1)[0, draft_tok].item()
                    p_max = F.softmax(verify_logits, dim=-1).max().item()
                    
                    if p_target >= (1 - epsilon) * p_max:
                        generated_ids.append(draft_tok)
                        total_accepted += 1
                        accepted_count += 1
                    else:
                        generated_ids.append(target_pred)
                        all_accepted = False
                        round_rollback = True
                        accepted_count += 1
                        break
            else:
                p_target = F.softmax(verify_logits / temperature, dim=-1)[0, draft_tok].item()

                t0 = time.time()
                with torch.no_grad():
                    if i == 0:
                        full_seq = torch.tensor([input_ids[0].tolist() + generated_ids], device=device)
                        d_out = draft_model(input_ids=full_seq, use_cache=True)
                    else:
                        d_in = torch.tensor([[draft_tokens[i - 1]]], device=device)
                        d_out = draft_model(
                            input_ids=d_in,
                            past_key_values=draft_past,
                            use_cache=True,
                        )
                    draft_past = d_out.past_key_values
                    d_logits = d_out.logits[:, -1, :]
                    draft_forward_count += 1
                    draft_total_time += time.time() - t0

                p_draft = F.softmax(d_logits / temperature, dim=-1)[0, draft_tok].item()

                if p_draft <= p_target:
                    generated_ids.append(draft_tok)
                    total_accepted += 1
                    accepted_count += 1
                else:
                    accept_prob = p_target / p_draft
                    if accept_prob >= (1 - epsilon) or torch.rand(1).item() < accept_prob:
                        generated_ids.append(draft_tok)
                        total_accepted += 1
                        accepted_count += 1
                    else:
                        target_probs = F.softmax(verify_logits / temperature, dim=-1)
                        draft_probs = F.softmax(d_logits / temperature, dim=-1)
                        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        next_tok = torch.multinomial(adjusted_probs, num_samples=1).item()
                        generated_ids.append(next_tok)
                        all_accepted = False
                        round_rollback = True
                        accepted_count += 1
                        break

            if generated_ids[-1] == tokenizer.eos_token_id:
                all_accepted = False
                round_rollback = True
                break
        
        if round_rollback:
            total_rollbacks += 1

        total_drafted += n_draft

        target_last_logits = target_logits[:, -1, :]

    end_time = time.time()
    total_time = end_time - start_time

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = {
        "generated_tokens": len(generated_ids),
        "time_seconds": total_time,
        "tokens_per_second": len(generated_ids) / total_time if total_time > 0 else 0,
        "target_model_forwards": target_forward_count,
        "draft_model_forwards": draft_forward_count,
        "total_verified": total_verified,
        "total_accepted": total_accepted,
        "total_drafted": total_drafted,
        "total_rollbacks": total_rollbacks,
        "verification_acceptance_rate": total_accepted / total_verified if total_verified > 0 else 0,
        "average_draft_length": total_drafted / target_forward_count if target_forward_count > 0 else 0,
        "rollback_rate": total_rollbacks / target_forward_count if target_forward_count > 0 else 0,
        "quality_bound": quality_bound(epsilon, gamma, alpha_calibrated),
        "alpha_calibrated": alpha_calibrated,
        "p_rollback_estimated": estimate_rollback_prob(epsilon, gamma, rollback_table),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"生成完成")
        print(f"{'='*60}")
        print(f"生成文本长度: {len(generated_ids)} tokens")
        print(f"耗时: {total_time:.2f}s")
        print(f"速度: {stats['tokens_per_second']:.2f} tokens/s")
        print(f"Target 前向次数: {target_forward_count}")
        print(f"Draft 前向次数: {draft_forward_count}")
        print(f"验证接受率: {stats['verification_acceptance_rate']:.2%}")
        print(f"平均草稿长度: {stats['average_draft_length']:.2f}")
        print(f"回滚率: {stats['rollback_rate']:.2%}")
        print(f"质量损失上界: {stats['quality_bound']:.4f}")
        print(f"\n生成文本: {output_text[:200]}...")

    return output_text, stats
