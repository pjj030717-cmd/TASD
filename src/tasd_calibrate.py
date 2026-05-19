import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, '/hy-tmp/my-repo/src')


def compute_tvd(p_logits, q_logits, temperature=1.0):
    """计算两个 logits 分布之间的 TV 距离"""
    p = F.softmax(p_logits / temperature, dim=-1)
    q = F.softmax(q_logits / temperature, dim=-1)
    return 0.5 * torch.sum(torch.abs(p - q)).item()


def run_calibration_step(target_model, draft_model, tokenizer, prompt, epsilon, gamma, max_new_tokens=64, temperature=0.0):
    """
    运行单次校准，收集 TVD 和回滚数据
    
    返回:
        tvd_per_token: 每个 token 的 TVD 列表
        rollback_occurred: 是否发生回滚（布尔值列表，每轮验证一个）
        stats: 基础统计信息
    """
    device = next(target_model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    generated_ids = []
    tvd_per_token = []
    rollback_occurred = []
    
    target_forward_count = 0
    total_accepted = 0
    total_verified = 0
    
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
                with torch.no_grad():
                    draft_input = torch.tensor([[draft_tokens[-1]]], device=device)
                    draft_out = draft_model(
                        input_ids=draft_input,
                        past_key_values=draft_past,
                        use_cache=True,
                    )
                    draft_past = draft_out.past_key_values
                    logits = draft_out.logits[:, -1, :]
            
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
                
                tvd = compute_tvd(verify_logits, verify_logits, temperature=1.0)
                tvd_per_token.append(0.0)
                
                if target_pred == draft_tok:
                    generated_ids.append(draft_tok)
                    total_accepted += 1
                else:
                    p_target = F.softmax(verify_logits, dim=-1)[0, draft_tok].item()
                    p_max = F.softmax(verify_logits, dim=-1).max().item()
                    
                    tvd = 1.0 - p_target / p_max
                    tvd_per_token.append(tvd)
                    
                    if p_target >= (1 - epsilon) * p_max:
                        generated_ids.append(draft_tok)
                        total_accepted += 1
                    else:
                        generated_ids.append(target_pred)
                        round_rollback = True
                        break
            else:
                p_target = F.softmax(verify_logits / temperature, dim=-1)[0, draft_tok].item()
                
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
                
                p_draft = F.softmax(d_logits / temperature, dim=-1)[0, draft_tok].item()
                
                tvd = compute_tvd(verify_logits, d_logits, temperature=temperature)
                tvd_per_token.append(tvd)
                
                if p_draft <= p_target:
                    generated_ids.append(draft_tok)
                    total_accepted += 1
                else:
                    accept_prob = p_target / p_draft
                    if accept_prob >= (1 - epsilon) or torch.rand(1).item() < accept_prob:
                        generated_ids.append(draft_tok)
                        total_accepted += 1
                    else:
                        target_probs = F.softmax(verify_logits / temperature, dim=-1)
                        draft_probs = F.softmax(d_logits / temperature, dim=-1)
                        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        next_tok = torch.multinomial(adjusted_probs, num_samples=1).item()
                        generated_ids.append(next_tok)
                        round_rollback = True
                        break
            
            if generated_ids[-1] == tokenizer.eos_token_id:
                round_rollback = True
                break
        
        rollback_occurred.append(round_rollback)
        target_last_logits = target_logits[:, -1, :]
    
    stats = {
        "generated_tokens": len(generated_ids),
        "target_forwards": target_forward_count,
        "total_accepted": total_accepted,
        "total_verified": total_verified,
        "acceptance_rate": total_accepted / total_verified if total_verified > 0 else 0,
    }
    
    return tvd_per_token, rollback_occurred, stats


def calibrate_alpha(target_model, draft_model, tokenizer, prompts, epsilon_values, gamma=5, rounds_per_epsilon=20, max_new_tokens=64, temperature=0.0):
    """
    校准 α 缩放因子
    
    对每个 ε 值运行多轮，测量实际 TVD，然后拟合 α
    """
    print(f"\n{'='*60}")
    print(f"α 校准实验")
    print(f"{'='*60}")
    print(f"ε 值列表: {epsilon_values}")
    print(f"每 ε 轮数: {rounds_per_epsilon}")
    print(f"γ = {gamma}")
    
    all_actual_tvds = []
    all_theoretical_bounds = []
    
    for epsilon in epsilon_values:
        print(f"\n--- ε = {epsilon} ---")
        round_tvds = []
        
        for round_idx in range(rounds_per_epsilon):
            prompt = prompts[round_idx % len(prompts)]
            tvd_list, _, _ = run_calibration_step(
                target_model, draft_model, tokenizer, 
                prompt, epsilon, gamma, max_new_tokens, temperature
            )
            if tvd_list:
                round_tvds.extend(tvd_list)
            
            if (round_idx + 1) % 5 == 0:
                avg_tvd = sum(round_tvds) / len(round_tvds) if round_tvds else 0
                print(f"  轮次 {round_idx + 1}/{rounds_per_epsilon}, 平均 TVD = {avg_tvd:.4f}")
        
        if round_tvds:
            avg_actual_tvd = sum(round_tvds) / len(round_tvds)
            theoretical_bound = gamma * (epsilon / 2) ** 0.5
            all_actual_tvds.append(avg_actual_tvd)
            all_theoretical_bounds.append(theoretical_bound)
            print(f"  平均实际 TVD: {avg_actual_tvd:.4f}")
            print(f"  理论上界: {theoretical_bound:.4f}")
            print(f"  比值 (实际/理论): {avg_actual_tvd / theoretical_bound:.4f}")
    
    if len(all_actual_tvds) >= 2:
        def objective(alpha):
            errors = []
            for actual, bound in zip(all_actual_tvds, all_theoretical_bounds):
                predicted = alpha * bound
                errors.append((actual - predicted) ** 2)
            return sum(errors) / len(errors)
        
        result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
        alpha_calibrated = result.x
        
        print(f"\n{'='*60}")
        print(f"α 校准结果")
        print(f"{'='*60}")
        print(f"校准后 α = {alpha_calibrated:.4f}")
        print(f"默认 α = 1.0 (Pinsker 界)")
        print(f"缩放比例 = {alpha_calibrated:.2%}")
        
        for actual, bound in zip(all_actual_tvds, all_theoretical_bounds):
            predicted_old = 1.0 * bound
            predicted_new = alpha_calibrated * bound
            print(f"  ε 对应: 实际={actual:.4f}, 旧预测={predicted_old:.4f}, 新预测={predicted_new:.4f}")
        
        return alpha_calibrated
    else:
        print("校准数据不足，返回默认 α = 1.0")
        return 1.0


def calibrate_rollback_prob(target_model, draft_model, tokenizer, prompts, epsilon_values, k_values, rounds_per_config=30, max_new_tokens=64, temperature=0.0):
    """
    校准回滚概率表
    
    对每个 (ε, k) 组合运行多轮，统计实际回滚频率
    """
    print(f"\n{'='*60}")
    print(f"回滚概率校准实验")
    print(f"{'='*60}")
    print(f"ε 值列表: {epsilon_values}")
    print(f"k 值列表: {k_values}")
    print(f"每配置轮数: {rounds_per_config}")
    
    calibration_table = {}
    
    for epsilon in epsilon_values:
        for k in k_values:
            print(f"\n--- ε = {epsilon}, k = {k} ---")
            rollback_count = 0
            total_rounds = 0
            
            for round_idx in range(rounds_per_config):
                prompt = prompts[round_idx % len(prompts)]
                _, rollback_list, _ = run_calibration_step(
                    target_model, draft_model, tokenizer,
                    prompt, epsilon, k, max_new_tokens, temperature
                )
                
                if rollback_list:
                    rollback_count += sum(rollback_list)
                    total_rounds += len(rollback_list)
                
                if (round_idx + 1) % 10 == 0:
                    current_rate = rollback_count / total_rounds if total_rounds > 0 else 0
                    print(f"  轮次 {round_idx + 1}/{rounds_per_config}, 回滚率 = {current_rate:.4f}")
            
            actual_p_rollback = rollback_count / total_rounds if total_rounds > 0 else 0
            theoretical_p_rollback = 1.0 - (1.0 - epsilon) ** k
            
            key = (round(epsilon, 3), k)
            calibration_table[key] = actual_p_rollback
            
            print(f"  实际回滚率: {actual_p_rollback:.4f}")
            print(f"  理论回滚率 (独立假设): {theoretical_p_rollback:.4f}")
            print(f"  比值 (实际/理论): {actual_p_rollback / theoretical_p_rollback:.4f}" if theoretical_p_rollback > 0 else "  N/A")
    
    return calibration_table


def save_calibration_results(alpha, calibration_table, output_path):
    """保存校准结果到 JSON 文件"""
    results = {
        "alpha_calibrated": alpha,
        "rollback_calibration_table": {f"{k[0]}_{k[1]}": v for k, v in calibration_table.items()},
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n校准结果已保存到: {output_path}")
    return output_path


def load_calibration_results(input_path):
    """加载校准结果"""
    with open(input_path, "r") as f:
        results = json.load(f)
    
    alpha = results["alpha_calibrated"]
    calibration_table = {}
    for key_str, value in results["rollback_calibration_table"].items():
        parts = key_str.split("_")
        epsilon = float(parts[0])
        k = int(parts[1])
        calibration_table[(epsilon, k)] = value
    
    return alpha, calibration_table


def run_full_calibration(
    target_model_path,
    draft_model_path,
    calibration_dataset="openai/openai_humaneval",
    calibration_samples=20,
    epsilon_values=None,
    k_values=None,
    rounds_per_epsilon=20,
    rounds_per_config=30,
    max_new_tokens=64,
    temperature=0.0,
    output_path="/hy-tmp/my-repo/results/calibration_results.json",
):
    """
    运行完整的校准流程
    """
    if epsilon_values is None:
        epsilon_values = [0.02, 0.05, 0.08, 0.10, 0.15]
    if k_values is None:
        k_values = [2, 3, 4, 5, 6, 7, 8]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"加载目标模型: {target_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    target_model.eval()
    
    print(f"加载草稿模型: {draft_model_path}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    draft_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"加载校准数据集: {calibration_dataset}")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    dataset = load_dataset(calibration_dataset, split="test")
    
    if calibration_samples is not None:
        dataset = dataset.select(range(min(calibration_samples, len(dataset))))
    
    prompts = []
    for sample in dataset:
        if "prompt" in sample:
            prompts.append(sample["prompt"])
        elif "text" in sample:
            prompts.append(sample["text"])
        else:
            prompts.append(str(sample))
    
    print(f"校准提示数量: {len(prompts)}")
    
    alpha_calibrated = calibrate_alpha(
        target_model, draft_model, tokenizer, prompts,
        epsilon_values, gamma=5, rounds_per_epsilon=rounds_per_epsilon,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    
    rollback_table = calibrate_rollback_prob(
        target_model, draft_model, tokenizer, prompts,
        epsilon_values, k_values, rounds_per_config=rounds_per_config,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    
    save_calibration_results(alpha_calibrated, rollback_table, output_path)
    
    return alpha_calibrated, rollback_table


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TASD 校准脚本")
    parser.add_argument("--target_model", type=str, default="/hy-tmp/my-repo/models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4")
    parser.add_argument("--draft_model", type=str, default="/hy-tmp/my-repo/models/LLM-Research/Meta-Llama-3___1-8B-Instruct-AWQ-INT4")
    parser.add_argument("--calibration_samples", type=int, default=20)
    parser.add_argument("--epsilon_values", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.10, 0.15])
    parser.add_argument("--k_values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--rounds_per_epsilon", type=int, default=20)
    parser.add_argument("--rounds_per_config", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, default="/hy-tmp/my-repo/results/calibration_results.json")
    
    args = parser.parse_args()
    
    run_full_calibration(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        calibration_samples=args.calibration_samples,
        epsilon_values=args.epsilon_values,
        k_values=args.k_values,
        rounds_per_epsilon=args.rounds_per_epsilon,
        rounds_per_config=args.rounds_per_config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_path=args.output_path,
    )
