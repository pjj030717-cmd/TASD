import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def autoregressive_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = None,
    top_p: float = None,
    verbose: bool = True,
):
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    past_key_values = None

    generated_ids = []

    start_time = time.time()
    total_forward_time = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"标准自回归解码开始")
        print(f"模型: {model.config.name_or_path}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"{'='*60}")

    current_input_ids = input_ids

    for step in range(max_new_tokens):
        t0 = time.time()
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=current_input_ids,
                    use_cache=True,
                )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        if temperature == 0.0:
            next_token_id = torch.argmax(logits, dim=-1)
        else:
            scaled_logits = logits / temperature
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
                mask = torch.full_like(scaled_logits, float("-inf"))
                mask.scatter_(-1, top_k_indices, top_k_values)
                scaled_logits = mask
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float("-inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        forward_time = time.time() - t0
        total_forward_time += forward_time

        current_input_ids = next_token_id.unsqueeze(0)
        generated_ids.append(next_token_id.item())

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    total_time = time.time() - start_time

    generated_text = tokenizer.decode(input_ids[0].tolist() + generated_ids, skip_special_tokens=True)

    stats = {
        "generated_tokens": len(generated_ids),
        "total_time": total_time,
        "total_forward_time": total_forward_time,
        "tokens_per_second": len(generated_ids) / total_time if total_time > 0 else 0,
    }

    if verbose:
        print(f"\n生成文本:\n{generated_text}\n")
        print(f"统计信息:")
        print(f"  生成 token 数: {stats['generated_tokens']}")
        print(f"  总耗时: {stats['total_time']:.3f}s")
        print(f"  总前向耗时: {stats['total_forward_time']:.3f}s")
        print(f"  生成速度: {stats['tokens_per_second']:.1f} tokens/s")

    return generated_text, stats