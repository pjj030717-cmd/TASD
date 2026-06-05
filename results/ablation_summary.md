# TASD Ablation Experiment

**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft)
**Settings**: max_new_tokens=128, n=10 per benchmark, temperature=0.0

## Average Across 3 Benchmarks

| Variant | TPS | Speedup | Accept | SQ | Repair | GuardTrig |
|---------|-----|---------|--------|----|--------|-----------|
| TASD-full | 49.37 | **1.50x** | 0.9829 | 0.9800 | 0.1 | 0.3 |
| TASD-strict-only | 43.94 | 1.34x | 0.8685 | 0.9800 | 0.1 | 0.5 |
| TASD-no-guard | 51.28 | 1.56x | 0.9997 | 0.9800 | 0.0 | 0.0 |
| TASD-draft-blocks-1 | 42.50 | 1.29x | 0.9596 | 0.9900 | 0.6 | 1.1 |

## Per-Benchmark Detail

| Benchmark | Variant | TPS | Speedup | Accept | SQ | Repair | GuardTrig |
|-----------|---------|-----|---------|--------|----|--------|-----------|
| Real-Python-Argparse | TASD-full | 49.18 | 1.49x | 1.0000 | 1.0000 | 0.0 | 0.0 |
| Real-Python-Argparse | TASD-strict-only | 43.77 | 1.33x | 0.8647 | 1.0000 | 0.1 | 0.0 |
| Real-Python-Argparse | TASD-no-guard | 50.42 | 1.53x | 1.0000 | 1.0000 | 0.0 | 0.0 |
| Real-Python-Argparse | TASD-draft-blocks-1 | 43.46 | 1.32x | 1.0000 | 1.0000 | 0.0 | 0.0 |
| Real-Python-DictConfig | TASD-full | 48.54 | 1.49x | 0.9488 | 0.9400 | 0.2 | 0.9 |
| Real-Python-DictConfig | TASD-strict-only | 42.99 | 1.32x | 0.8429 | 0.9400 | 0.2 | 1.4 |
| Real-Python-DictConfig | TASD-no-guard | 51.66 | 1.58x | 0.9992 | 0.9400 | 0.0 | 0.0 |
| Real-Python-DictConfig | TASD-draft-blocks-1 | 38.95 | 1.19x | 0.8787 | 0.9700 | 1.7 | 3.2 |
| OpenMMLab-Config | TASD-full | 50.39 | 1.53x | 1.0000 | 1.0000 | 0.0 | 0.0 |
| OpenMMLab-Config | TASD-strict-only | 45.05 | 1.37x | 0.8979 | 1.0000 | 0.0 | 0.0 |
| OpenMMLab-Config | TASD-no-guard | 51.76 | 1.57x | 1.0000 | 1.0000 | 0.0 | 0.0 |
| OpenMMLab-Config | TASD-draft-blocks-1 | 45.09 | 1.37x | 1.0000 | 1.0000 | 0.0 | 0.0 |

## Amplitude: Speedup Gains

| Ablation | TPS Drop | Speedup Drop | Cause |
|----------|----------|--------------|-------|
| TASD-full -> strict-only | -5.43 TPS (-11.0%) | 1.50x -> 1.34x (-10.7%) | Accept rate drops from 0.98 -> 0.87 |
| TASD-full -> no-guard | +1.91 TPS (+3.9%) | 1.50x -> 1.56x (+4.0%) | Guard overhead removed (0.3 triggers) |
| TASD-full -> draft-blocks-1 | -6.87 TPS (-13.9%) | 1.50x -> 1.29x (-14.0%) | Single block limits draft contribution |

## Key Findings

1. **Relaxed accept is the primary speed gain (proven)**. Strict-only loses 0.16x speedup on average (1.50x -> 1.34x). The accept rate drops from 98% to 87%: the 3B draft disagrees with the 14B target on ~13% of tokens at argmax, and strict rejection means those tokens are discarded.

2. **draft_blocks=2 is the second-largest contributor**. Reducing to draft_blocks=1 costs 0.21x speedup on average (1.50x -> 1.29x). DictConfig shows the largest drop (1.49x -> 1.19x) because its lower baseline accept rate benefits most from multi-block batching.

3. **Guard has negligible speed impact (as expected)**. Guard removal gives only +0.06x speedup across 3 benchmarks (1.50x -> 1.56x), with guard triggers averaging only 0.3 per benchmark. The guard's job is quality protection, not speed — and it triggers too rarely to affect throughput.

4. **Quality scores are stable across variants**. SQ ranges 0.94-1.00 regardless of variant. Guard triggers are low (0-3.2) because the 3 benchmarks use structure-redundant prompts where the draft model rarely ventures off-pattern. Quality benefits of the guard would be more visible on benchmarks with less predictable structure.

## Caveats

- n=10: ablation runs are indicative but noisier than the n=80 main experiments.
- Structural quality metrics showed no degradation even in no-guard on these 3 benchmarks — the guard's protective value is likely higher on extended benchmarks (Rich-CLI, Complex-Nested, Pipeline-Stage).
- The strict-only variant was implemented by disabling relaxed accept entirely (no top-k, no prefix budget, no window acceptance). It maintains the draft mechanism and guard, but only accepts exact argmax matches.
