"""
Diagnose dict_config_real_003 guard false positives.
Prints every guard trigger with reason, partial text, trim position.
Tests both calibrated=False (current) and calibrated=True (v1.5).
"""
import json, os, sys, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.structural_guard import StructuralGuard

TARGET_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DRAFT_PATH = "/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct"
MAX_NEW_TOKENS = 128

def run_tasd_diagnostic(target, draft, tokenizer, prompt, structure_type, calibrated):
    """Run TASD with detailed guard logging."""
    from src.tasd_decode import tasd_decode

    result = tasd_decode(
        target, draft, tokenizer, prompt,
        structure_type=structure_type,
        max_new_tokens=MAX_NEW_TOKENS,
        draft_len=16, draft_blocks=2,
        top_k_accept=3, min_token_prob=1e-4,
        prefix_budget=0.2, window_len=2,
        enable_guard=True,
        enable_relaxed_accept=True,
        guard_calibrated=calibrated,
    )
    return result

def main():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    # Load sample
    with open("data/codesearchnet_dict_config_blocks_80.jsonl") as f:
        for line in f:
            s = json.loads(line.strip())
            if s["name"] == "dict_config_real_003":
                break

    prompt = s["prompt"]
    print(f"=== {s['name']} ===")
    print(f"Prompt: {prompt}")
    print(f"Prompt len: {len(prompt)} chars")
    print()

    for calibrated in [False, True]:
        label = "v1.5 CALIBRATED" if calibrated else "ORIGINAL (uncalibrated)"
        print(f"{'='*60}")
        print(f"Running TASD with guard_calibrated={calibrated} ({label})")
        print(f"{'='*60}")

        torch.cuda.synchronize()
        t0 = time.time()
        result = run_tasd_diagnostic(target, draft, tokenizer, prompt, "dict_config", calibrated)
        torch.cuda.synchronize()
        wall = time.time() - t0

        stats = result['stats']
        print(f"\nResults:")
        print(f"  TPS: {stats.get('tokens_per_second', 'N/A')}")
        print(f"  Accept rate: {stats.get('accept_rate', 'N/A')}")
        print(f"  Guard triggers: {stats.get('guard_trigger_count', 'N/A')}")
        print(f"  Hard trim (actual): {stats.get('trim_count', 'N/A')}")
        print(f"    hard_trim_count: {stats.get('hard_trim_count', 'N/A')}")
        print(f"    repetition_warnings: {stats.get('repetition_warning_count', 'N/A')}")
        print(f"    bracket_warnings: {stats.get('bracket_warning_count', 'N/A')}")
        print(f"    import_warnings: {stats.get('import_warning_count', 'N/A')}")
        print(f"  Repair count: {stats.get('repair_count', 'N/A')}")
        print(f"  Generated tokens: {stats.get('generated_length', 'N/A')}")
        print(f"  Wall time: {wall:.2f}s")
        print(f"  Trim reasons (ALL):")
        trim_reasons = stats.get('trim_reasons', [])
        if trim_reasons:
            from collections import Counter
            rc = Counter(trim_reasons)
            for reason, count in rc.most_common():
                print(f"    [{count}x] {reason}")
        else:
            print(f"    (none)")
        print(f"  Generated text: {result['generated_text'][:300]}")
        print()

    # ── Manual guard check on generated text ──
    print("="*60)
    print("Manual guard check on TASD (calibrated) output text")
    print("="*60)
    text = result["generated_text"]
    guard = StructuralGuard(structure_type="dict_config", calibrated=True)
    safe, trim_count, reason = guard.check(text)
    print(f"  Final check: safe={safe}, trim={trim_count}, reason={reason}")
    print(f"  trigger_count={guard.trigger_count}, hard_trim={guard.hard_trim_count}")
    print(f"  repetition_warn={guard.repetition_warning_count}")
    print(f"  bracket_warn={guard.bracket_warning_count}")
    print(f"  import_warn={guard.import_warning_count}")

if __name__ == "__main__":
    main()
