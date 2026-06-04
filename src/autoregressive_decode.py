"""
Autoregressive (AR) decode baseline.
Generates text token-by-token using the target model only.
"""
import time
import torch


def ar_decode(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0):
    """
    Run autoregressive generation.

    Returns:
        generated_text: str
        tokens_per_second: float
        generated_tokens: int
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time
    generated_tokens = outputs.shape[1] - prompt_len
    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    tps = generated_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "generated_text": generated_text,
        "tokens_per_second": tps,
        "generated_tokens": generated_tokens,
        "elapsed_time": elapsed,
    }
