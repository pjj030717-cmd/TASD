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
DATA_FILES = {
    'argparse': 'data/codesearchnet_argparse_blocks_80.jsonl',
    'dict_config': 'data/codesearchnet_dict_config_blocks_80.jsonl',
    'openmmlab_config': 'data/ml_config_blocks_openmmlab_80.jsonl',
    'pipeline_stage_config': 'data/pipeline_stage_config_80.jsonl',
    'complex_nested_config': 'data/complex_nested_config_80.jsonl',
    'rich_cli_option_groups': 'data/rich_cli_option_groups_80.jsonl',
}

prompts = {}
for bm, filepath in DATA_FILES.items():
    prompts[bm] = {}
    with open(filepath) as f:
        for line in f:
            sample = json.loads(line)
            prompts[bm][sample['name']] = sample['prompt']

# Run pilot
results = []
for i, sample in enumerate(hard_samples[:60]):  # Max 60 samples
    print(f"\n[{i+1}/{min(60, len(hard_samples))}] {sample['name']} (score={sample['score']})")

    prompt = prompts[sample['benchmark']][sample['name']]

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
            'tokens_per_second': result_fg['tokens_per_second'],
            'generated_length': result_fg['generated_tokens'],
        },
        'fgq': {
            'tokens_per_second': result_fgq['tokens_per_second'],
            'generated_length': result_fgq['generated_tokens'],
            'quality_guard': result_fgq['stats'].get('quality_guard', {}),
        },
    })

    print(f"  FG: tps={result_fg['tokens_per_second']:.1f}")
    print(f"  FGQ: tps={result_fgq['tokens_per_second']:.1f}, rep_trim={result_fgq['stats'].get('quality_guard', {}).get('rep_trim_count', 0)}")

# Save results
with open('results/tasd_fgq_hard_subset_pilot.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nPilot complete. Results saved to results/tasd_fgq_hard_subset_pilot.json")

# Summary
avg_tps_fg = sum(r['fg']['tokens_per_second'] for r in results) / len(results)
avg_tps_fgq = sum(r['fgq']['tokens_per_second'] for r in results) / len(results)
total_rep_trim = sum(r['fgq']['quality_guard'].get('rep_trim_count', 0) for r in results)

print(f"\nSummary:")
print(f"  TASD-FG avg TPS: {avg_tps_fg:.1f}")
print(f"  TASD-FGQ avg TPS: {avg_tps_fgq:.1f}")
print(f"  TPS loss: {(avg_tps_fg - avg_tps_fgq) / avg_tps_fg * 100:.1f}%")
print(f"  Total rep_trim triggers: {total_rep_trim}")
