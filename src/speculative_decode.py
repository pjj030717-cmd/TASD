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
        print(f"标准推测解码开始")
        print(f"Target 模型: {target_model.config.name_or_path}")
        print(f"Draft 模型: {draft_model.config.name_or_path}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Gamma: {gamma}")
        print(f"{'='*60}")

    current_ids = input_ids

    while len(generated_ids) < max_new_tokens:
        draft_ids = current_ids.clone()
        draft_past = None
        draft_tokens = []

        for _ in range(gamma):
            if len(generated_ids) + len(draft_tokens) >= max_new_tokens:
                break

            t0 = time.time()
            with torch.no_grad():
                if draft_past is not None:
                    draft_out = draft_model(
                        input_ids=draft_ids[:, -1:],
                        past_key_values=draft_past,
                        use_cache=True,
                    )
                else:
                    draft_out = draft_model(
                        input_ids=draft_ids,
                        use_cache=True,
                    )
            draft_forward_count += 1
            draft_total_time += time.time() - t0

            draft_logits = draft_out.logits[:, -1, :]
            draft_past = draft_out.past_key_values

            if temperature == 0.0:
                next_tok = torch.argmax(draft_logits, dim=-1)
                draft_probs = None
            else:
                draft_probs = F.softmax(draft_logits / temperature, dim=-1)
                next_tok = torch.multinomial(draft_probs, num_samples=1)

            draft_tokens.append(next_tok.item())
            draft_ids = torch.cat([draft_ids, next_tok.unsqueeze(0)], dim=1)

            if next_tok.item() == tokenizer.eos_token_id:
                break

        if len(draft_tokens) == 0:
            break

        verify_ids = torch.cat(
            [current_ids, torch.tensor([draft_tokens], device=device)],
            dim=1,
        )

        t0 = time.time()
        with torch.no_grad():
            target_out = target_model(
                input_ids=verify_ids,
                use_cache=True,
            )
        target_forward_count += 1
        target_total_time += time.time() - t0

        target_logits = target_out.logits[:, current_ids.shape[1] - 1:, :]

        if temperature == 0.0:
            target_preds = torch.argmax(target_logits[:, :-1, :], dim=-1)
            draft_probs_all = None
        else:
            target_probs_all = F.softmax(target_logits[:, :-1, :] / temperature, dim=-1)
            draft_probs_all_list = []
            draft_past2 = None
            draft_ids2 = current_ids.clone()
            for _ in range(len(draft_tokens)):
                with torch.no_grad():
                    if draft_past2 is not None:
                        d_out = draft_model(
                            input_ids=draft_ids2[:, -1:],
                            past_key_values=draft_past2,
                            use_cache=True,
                        )
                    else:
                        d_out = draft_model(
                            input_ids=draft_ids2,
                            use_cache=True,
                        )
                d_logits = d_out.logits[:, -1, :]
                draft_past2 = d_out.past_key_values
                draft_probs_all_list.append(F.softmax(d_logits / temperature, dim=-1))
                draft_ids2 = torch.cat([draft_ids2, torch.tensor([[draft_tokens[_]]], device=device)], dim=1)
            draft_probs_all = torch.stack(draft_probs_all_list, dim=1)

        n_draft = len(draft_tokens)
        all_accepted = True
        accept_count = 0

        for i in range(n_draft):
            draft_tok = draft_tokens[i]

            if temperature == 0.0:
                if target_preds[0, i] == draft_tok:
                    generated_ids.append(draft_tok)
                    total_accepted += 1
                    accept_count += 1
                else:
                    generated_ids.append(target_preds[0, i].item())
                    all_accepted = False
                    accept_count += 1
                    break
            else:
                p_target = target_probs_all[0, i, draft_tok].item()
                p_draft = draft_probs_all[0, i, draft_tok].item()

                if p_draft <= p_target:
                    generated_ids.append(draft_tok)
                    total_accepted += 1
                    accept_count += 1
                else:
                    accept_prob = p_target / p_draft
                    if torch.rand(1).item() < accept_prob:
                        generated_ids.append(draft_tok)
                        total_accepted += 1
                        accept_count += 1
                    else:
                        adjusted_probs = torch.clamp(
                            target_probs_all[0, i] - draft_probs_all[0, i], min=0.0
                        )
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        next_tok = torch.multinomial(adjusted_probs, num_samples=1).item()
                        generated_ids.append(next_tok)
                        all_accepted = False
                        accept_count += 1
                        break

            if generated_ids[-1] == tokenizer.eos_token_id:
                all_accepted = False
                break

        total_drafted += n_draft

        if len(generated_ids) >= max_new_tokens or \
                (len(generated_ids) > 0 and generated_ids[-1] == tokenizer.eos_token_id):
            break

        current_ids = torch.tensor(
            [input_ids[0].tolist() + generated_ids], device=device
        )

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