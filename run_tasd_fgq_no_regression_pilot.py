#!/usr/bin/env python3
"""
TASD-FGQ No-Regression Pilot: score=2 subset
Verify that QualityGuard does not degrade clean samples.
"""
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq

# Load score=2 samples
with open('results/human_quality_review_auto_scores.json') as f:
    scores = json.load(f)

score2_samples = [s for s in scores if s['score'] == 2]
print(f"Found {len(score2_samples)} score=2 samples")

# Select up to 10 per benchmark
from collections import defaultdict
by_benchmark = defaultdict(list)
for s in score2_samples:
    by_benchmark[s['benchmark']].append(s)

selected = []
for bm, samples in by_benchmark.items():
    selected.extend(samples[:10])
print(f"Selected {len(selected)} samples ({min(10, len(score2_samples))} per benchmark)")

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

# Load prompts and references from JSONL files
DATA_FILES = {
    'argparse': 'data/codesearchnet_argparse_blocks_80.jsonl',
    'dict_config': 'data/codesearchnet_dict_config_blocks_80.jsonl',
    'openmmlab_config': 'data/ml_config_blocks_openmmlab_80.jsonl',
    'pipeline_stage_config': 'data/pipeline_stage_config_80.jsonl',
    'complex_nested_config': 'data/complex_nested_config_80.jsonl',
    'rich_cli_option_groups': 'data/rich_cli_option_groups_80.jsonl',
}

prompts = {}
references = {}
for bm, filepath in DATA_FILES.items():
    prompts[bm] = {}
    references[bm] = {}
    with open(filepath) as f:
        for line in f:
            sample = json.loads(line)
            name = sample['name']
            prompts[bm][name] = sample['prompt']
            references[bm][name] = sample['reference']

# Run pilot
results = []
for i, sample in enumerate(selected):
    print(f"\n[{i+1}/{len(selected)}] {sample['name']} (benchmark={sample['benchmark']}, score={sample['score']})")

    prompt = prompts[sample['benchmark']][sample['name']]
    ref = references[sample['benchmark']][sample['name']]

    # Run TASD-FG (baseline)
    print("  Running TASD-FG...")
    t0 = time.time()
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
    t_fg = time.time() - t0

    # Run TASD-FGQ (with QualityGuard)
    print("  Running TASD-FGQ...")
    t0 = time.time()
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
    )
    t_fgq = time.time() - t0

    # Compute quality metrics for both
    fg_text = result_fg['generated_text']
    fgq_text = result_fgq['generated_text']

    fg_metrics = compute_composite_sq(fg_text, ref, structure_type=sample['benchmark'])
    fgq_metrics = compute_composite_sq(fgq_text, ref, structure_type=sample['benchmark'])

    results.append({
        'name': sample['name'],
        'benchmark': sample['benchmark'],
        'original_score': sample['score'],
        'fg': {
            'speedup': result_fg['stats'].get('speedup', result_fg['tokens_per_second'] / max(result_fg['generated_tokens'], 1)),
            'generated_length': result_fg['generated_tokens'],
            'structural_f1': fg_metrics.get('structural_char_f1', 0),
            'bracket_balance': fg_metrics.get('bracket_balance_score', 0),
            'repetition_rate': fg_metrics.get('repetition_rate', 0),
            'off_structure': fg_metrics.get('off_structure_rate', 0),
            'composite_sq': fg_metrics.get('composite_sq', 0),
        },
        'fgq': {
            'speedup': result_fgq['stats'].get('speedup', result_fgq['tokens_per_second'] / max(result_fgq['generated_tokens'], 1)),
            'generated_length': result_fgq['generated_tokens'],
            'structural_f1': fgq_metrics.get('structural_char_f1', 0),
            'bracket_balance': fgq_metrics.get('bracket_balance_score', 0),
            'repetition_rate': fgq_metrics.get('repetition_rate', 0),
            'off_structure': fgq_metrics.get('off_structure_rate', 0),
            'composite_sq': fgq_metrics.get('composite_sq', 0),
            'quality_guard': result_fgq['stats'].get('quality_guard', {}),
        },
    })

    print(f"  FG:  speedup={results[-1]['fg']['speedup']:.3f}, f1={fg_metrics.get('structural_char_f1', 0):.3f}")
    print(f"  FGQ: speedup={results[-1]['fgq']['speedup']:.3f}, f1={fgq_metrics.get('structural_char_f1', 0):.3f}")

# Save JSON
with open('results/tasd_fgq_no_regression_pilot.json', 'w') as f:
    json.dump(results, f, indent=2)

# Generate report
from collections import Counter

# Score distribution (recompute using auto_human_quality_review logic)
def compute_score(metrics):
    """Simplified scoring for report."""
    rep = metrics.get('repetition_rate', 0)
    off = metrics.get('off_structure', 0)
    f1 = metrics.get('structural_f1', 0)
    bracket = metrics.get('bracket_balance', 0)

    # 0-score conditions
    if rep >= 0.50 or off >= 0.25 or f1 < 0.20 or bracket < 0.50:
        return 0

    # severe count
    severe = 0
    if rep >= 0.25: severe += 1
    if off >= 0.10: severe += 1
    if f1 < 0.50: severe += 1
    if bracket < 0.50: severe += 1
    if severe >= 3:
        return 0

    # 2-score conditions
    if f1 >= 0.85 and rep < 0.08 and off < 0.02 and bracket >= 0.95:
        return 2

    return 1

fg_scores = [compute_score(r['fg']) for r in results]
fgq_scores = [compute_score(r['fgq']) for r in results]

fg_score_dist = Counter(fg_scores)
fgq_score_dist = Counter(fgq_scores)

# Score=2 retention
score2_count = sum(1 for s in fg_scores if s == 2)
score2_retained = sum(1 for i in range(len(results)) if fg_scores[i] == 2 and fgq_scores[i] == 2)
retention_rate = score2_retained / max(score2_count, 1)

# Speedup comparison
avg_speedup_fg = sum(r['fg']['speedup'] for r in results) / len(results)
avg_speedup_fgq = sum(r['fgq']['speedup'] for r in results) / len(results)
speed_loss = (avg_speedup_fg - avg_speedup_fgq) / avg_speedup_fg * 100

# Quality guard stats
total_rep_trim = sum(r['fgq']['quality_guard'].get('rep_trim_count', 0) for r in results)
total_strict = sum(r['fgq']['quality_guard'].get('strict_rounds', 0) for r in results)
total_repair = sum(r['fgq']['quality_guard'].get('low_progress_repairs', 0) for r in results)
total_trimmed = sum(r['fgq']['quality_guard'].get('total_trimmed_tokens', 0) for r in results)

# Per-benchmark
bm_results = defaultdict(list)
for r in results:
    bm_results[r['benchmark']].append(r)

report = f"""# TASD-FGQ No-Regression Pilot Report

## Summary
- **Samples**: {len(results)} score=2 samples
- **Score=2 retention rate**: {retention_rate:.1%} ({score2_retained}/{score2_count})
- **TASD-FG avg speedup**: {avg_speedup_fg:.3f}x
- **TASD-FGQ avg speedup**: {avg_speedup_fgq:.3f}x
- **Speed loss**: {speed_loss:.1f}%

## Acceptance Criteria
- [ ] Score=2 retention >= 90%: {'PASS' if retention_rate >= 0.90 else 'FAIL'}
- [ ] Speed loss <= 5%: {'PASS' if speed_loss <= 5 else 'FAIL'}

## Score Distribution

| Score | TASD-FG | TASD-FGQ |
|:-----:|:-------:|:--------:|
| 2 | {fg_score_dist.get(2, 0)} | {fgq_score_dist.get(2, 0)} |
| 1 | {fg_score_dist.get(1, 0)} | {fgq_score_dist.get(1, 0)} |
| 0 | {fg_score_dist.get(0, 0)} | {fgq_score_dist.get(0, 0)} |

## Quality Guard Triggers
- rep_trim_count: {total_rep_trim}
- strict_rounds: {total_strict}
- low_progress_repairs: {total_repair}
- total_trimmed_tokens: {total_trimmed}

## Per-Benchmark Results

| Benchmark | N | FG speedup | FGQ speedup | Speed loss | Score=2 retained |
|-----------|:-:|:----------:|:-----------:|:----------:|:----------------:|
"""

for bm in sorted(bm_results.keys()):
    bm_data = bm_results[bm]
    n = len(bm_data)
    fg_sp = sum(r['fg']['speedup'] for r in bm_data) / n
    fgq_sp = sum(r['fgq']['speedup'] for r in bm_data) / n
    loss = (fg_sp - fgq_sp) / fg_sp * 100 if fg_sp > 0 else 0
    bm_fg_scores = [compute_score(r['fg']) for r in bm_data]
    bm_fgq_scores = [compute_score(r['fgq']) for r in bm_data]
    s2_count = sum(1 for s in bm_fg_scores if s == 2)
    s2_retained = sum(1 for i in range(n) if bm_fg_scores[i] == 2 and bm_fgq_scores[i] == 2)
    report += f"| {bm} | {n} | {fg_sp:.3f}x | {fgq_sp:.3f}x | {loss:.1f}% | {s2_retained}/{s2_count} |\n"

report += f"""
## Conclusion
{'TASD-FGQ passes no-regression criteria. Proceed to hard subset pilot.' if retention_rate >= 0.90 and speed_loss <= 5 else 'TASD-FGQ fails no-regression criteria. Do NOT proceed.'}
"""

with open('results/tasd_fgq_no_regression_pilot.md', 'w') as f:
    f.write(report)

print(f"\n\n=== No-Regression Pilot Complete ===")
print(f"Score=2 retention: {retention_rate:.1%}")
print(f"Speed loss: {speed_loss:.1f}%")
print(f"Reports saved to results/tasd_fgq_no_regression_pilot.json/.md")
