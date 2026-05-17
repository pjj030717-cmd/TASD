import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    gamma: int = 5,
    temperature: float = 0.0,
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

    start_time = time.time()
    target_total_time = 0.0
    draft_total_time = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"标准推测解码 (KV cache 优化)")
        print(f"Target 模型: {target_model.config.name_or_path}")
        print(f"Draft 模型: {draft_model.config.name_or_path}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Gamma: {gamma}")
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

        all_accepted = True
        accepted_count = 0

        for i in range(n_draft):
            draft_tok = draft_tokens[i]

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
                    generated_ids.append(target_pred)
                    all_accepted = False
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
                    if torch.rand(1).item() < accept_prob:
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

    total_time = time.time() - start_time

    generated_text = tokenizer.decode(
        input_ids[0].tolist() + generated_ids, skip_special_tokens=True
    )

    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0

    stats = {
        "generated_tokens": len(generated_ids),
        "total_time": total_time,
        "target_forward_count": target_forward_count,
        "draft_forward_count": draft_forward_count,
        "target_total_time": target_total_time,
        "draft_total_time": draft_total_time,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
        "tokens_per_second": len(generated_ids) / total_time if total_time > 0 else 0,
        "speedup_vs_target_calls": len(generated_ids) / target_forward_count if target_forward_count > 0 else 0,
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
        print(f"  接受率: {stats['acceptance_rate']:.2%}")
        print(f"  生成速度: {stats['tokens_per_second']:.1f} tokens/s")
        print(f"  等效加速比: {stats['speedup_vs_target_calls']:.2f}x (vs Target 单次调用)")

    return generated_text, stats
