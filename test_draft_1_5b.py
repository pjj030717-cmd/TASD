#!/usr/bin/env python3
"""Quick test: Qwen2.5-1.5B-Instruct as draft vs 3B draft, on OpenMMLab 10 samples."""
import json, time, torch, sys, os
sys.path.insert(0, ".")

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tasd_decode import tasd_decode
from src.evaluator import evaluate_structural_quality

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_3B = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"
DRAFT_1_5B = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
DATA = "data/ml_config_blocks_openmmlab_80.jsonl"
N = 10

print("Loading models...")
target = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, device_map="auto", torch_dtype="auto", trust_remote_code=True, local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load samples
samples = []
with open(DATA) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        samples.append(json.loads(line))
        if len(samples) >= N: break

for draft_label, draft_path in [("3B", DRAFT_3B), ("1.5B", DRAFT_1_5B)]:
    print(f"\n{'='*60}")
    print(f"  Draft: {draft_label} | OpenMMLab-Config | n={N}")
    print(f"{'='*60}")

    draft = AutoModelForCausalLM.from_pretrained(
        draft_path, device_map="auto", torch_dtype="auto", trust_remote_code=True, local_files_only=True,
    )

    results = []
    for i, s in enumerate(samples):
        prompt = s["prompt"]
        _ = torch.cuda.synchronize()
        t0 = time.time()

        try:
            r = tasd_decode(
                target_model=target, draft_model=draft, tokenizer=tokenizer,
                prompt=prompt, structure_type="openmmlab_config",
                max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3,
                min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
                enable_guard=True, enable_relaxed_accept=True,
            )
        except Exception as e:
            print(f"  ERROR [{i+1}]: {e}")
            continue

        _ = torch.cuda.synchronize()
        wall = time.time() - t0
        gen = r.get("generated_text", "")
        tps = r.get("tokens_per_second", 0)
        st = r.get("stats", {})
        acc = st.get("accept_rate", 0)
        q = evaluate_structural_quality(gen, structure_type="openmmlab_config")

        print(f"  [{i+1}/{N}] TPS={tps:.1f}, acc={acc:.2f}, SQ={q['structural_quality_score']:.4f}, wall={wall:.1f}s")
        results.append({"tps": tps, "accept": acc, "sq": q["structural_quality_score"], "wall": wall})

    valid = [r for r in results if r["tps"] > 0]
    avg_tps = sum(r["tps"] for r in valid) / len(valid)
    avg_acc = sum(r["accept"] for r in valid) / len(valid)
    avg_sq = sum(r["sq"] for r in valid) / len(valid)

    print(f"\n  Avg: TPS={avg_tps:.2f}, Accept={avg_acc:.4f}, SQ={avg_sq:.4f}")

    # Clean up draft model to free GPU memory
    del draft
    torch.cuda.empty_cache()

print("\nDone.")
