#!/usr/bin/env python3
"""
TASD-FGQ Pilot: Hard subset experiment
Test QualityGuard on score=0 samples to see if it improves quality
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode

# Load score=0 samples
with open('results/human_quality_review_auto_scores.json') as f:
    scores = json.load(f)

hard_samples = [s for s in scores if s['score'] == 0]
print(f"Found {len(hard_samples)} score=0 samples")

# Load models
print("Loading models...")
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"

target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, local_files_only=True, device_map="auto", trust_remote_code=True
)
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, local_files_only=True, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)

# Load prompts
prompts = {}
for bm in ['argparse', 'dict_config', 'openmmlab_config', 'pipeline_stage_config', 'complex_nested_config', 'rich_cli_option_groups']:
    with open(f'data/{bm}_prompts.json') as f:
        prompts[bm] = json.load(f)

# Run pilot
results = []
for i, sample in enumerate(hard_samples[:60]):  # Max 60 samples
    print(f"\n[{i+1}/{min(60, len(hard_samples))}] {sample['name']} (score={sample['score']})")

    prompt = prompts[sample['benchmark']][sample['name']]['prompt']

    # Run TASD-FG (baseline)
    print("  Running TASD-FG...")
    result_fg = tasd_decode(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=128,
        draft_len=8,
        draft_blocks=2,
        enable_guard=True,
        enable_relaxed_accept=True,
        enable_failure_aware_fallback=True,
        enable_quality_guard=False,
    )

    # Run TASD-FGQ (with QualityGuard)
    print("  Running TASD-FGQ...")
    result_fgq = tasd_decode(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=128,
        draft_len=8,
        draft_blocks=2,
        enable_guard=True,
        enable_relaxed_accept=True,
        enable_failure_aware_fallback=True,
        enable_quality_guard=True,
        quality_guard_ngram=4,
        quality_guard_window=64,
        quality_guard_strict_trigger=2,
        quality_guard_strict_window=3,
        quality_guard_strict_rounds=2,
        quality_guard_low_accept_threshold=1,
        quality_guard_low_accept_patience=2,
        quality_guard_repair_tokens=2,
    )

    results.append({
        'name': sample['name'],
        'benchmark': sample['benchmark'],
        'original_score': sample['score'],
        'fg': {
            'speedup': result_fg['stats']['speedup'],
            'generated_length': result_fg['stats']['generated_length'],
        },
        'fgq': {
            'speedup': result_fgq['stats']['speedup'],
            'generated_length': result_fgq['stats']['generated_length'],
            'quality_guard': result_fgq['stats']['quality_guard'],
        },
    })

    print(f"  FG: speedup={result_fg['stats']['speedup']:.3f}")
    print(f"  FGQ: speedup={result_fgq['stats']['speedup']:.3f}, rep_trim={result_fgq['stats']['quality_guard']['rep_trim_count']}")

# Save results
with open('results/tasd_fgq_hard_subset_pilot.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nPilot complete. Results saved to results/tasd_fgq_hard_subset_pilot.json")

# Summary
avg_speedup_fg = sum(r['fg']['speedup'] for r in results) / len(results)
avg_speedup_fgq = sum(r['fgq']['speedup'] for r in results) / len(results)
total_rep_trim = sum(r['fgq']['quality_guard']['rep_trim_count'] for r in results)

print(f"\nSummary:")
print(f"  TASD-FG avg speedup: {avg_speedup_fg:.3f}")
print(f"  TASD-FGQ avg speedup: {avg_speedup_fgq:.3f}")
print(f"  Speedup loss: {(avg_speedup_fg - avg_speedup_fgq) / avg_speedup_fg * 100:.1f}%")
print(f"  Total rep_trim triggers: {total_rep_trim}")
