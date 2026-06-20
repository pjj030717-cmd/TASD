# TASD-FG-V Speed/Quality Tradeoff Analysis

**Baselines**: AR recov=84.6%, FLY speedup=1.64x recov=80.8%, TASD-FG speedup=2.00x recov=72.5%

**TASD-FG-V (V1)**: speedup=1.31x recov=91.0%, rerun=25.4%

## 1. Variant Comparison

| Variant | Rerun | Recoverable | Score 0 | Speedup | vs FLY Speed | vs V1 Speed | vs V1 Recov |
|---------|:-----:|:----------:|:------:|:-------:|:------------:|:-----------:|:-----------:|
| AR | — | 406/480 (84.6%) | 74 | 1.00x | 61% | 76% | — |
| FLY | — | 388/480 (80.8%) | 92 | 1.64x | **100%** | 125% | — |
| TASD-FG | — | 348/480 (72.5%) | 132 | 2.00x | **122%** | 152% | — |
| TASD-FG-V (V1) | 25.4% | 437/480 (91.0%) | 43 | 1.31x | 80% | 100% | +0 |
| V-Lite | 22.7% | 437/480 (91.0%) | 43 | 1.36x | 83% | 104% | +0 |
| Benchmark-Aware | 22.7% | 428/480 (89.2%) | 52 | 1.36x | 83% | 104% | -9 |
| Partial Repair 25% | 25.4% | 437/480 (91.0%) | 43 | 1.76x | **107%** | 134% | +0 |
| Partial Repair 50% | 25.4% | 437/480 (91.0%) | 43 | 1.58x | 96% | 120% | +0 |
| Partial Repair 75% | 25.4% | 437/480 (91.0%) | 43 | 1.44x | 87% | 109% | +0 |

## 2. V-Lite Per-Benchmark

| Benchmark | V1 Rerun | V-Lite Rerun | V1 Recov | V-Lite Recov | V1 Speed | V-Lite Speed |
|-----------|:--------:|:------------:|:--------:|:------------:|:--------:|:------------:|
| argparse | 23.8% | 20.0% | 97.5% | 97.5% | 1.30x | 1.37x |
| dict_config | 13.8% | 13.8% | 87.5% | 87.5% | 1.51x | 1.51x |
| openmmlab_config | 28.7% | 27.5% | 95.0% | 95.0% | 1.29x | 1.31x |
| pipeline_stage_config | 26.2% | 18.8% | 97.5% | 97.5% | 1.31x | 1.45x |
| complex_nested_config | 31.2% | 27.5% | 73.8% | 73.8% | 1.24x | 1.30x |
| rich_cli_option_groups | 28.7% | 28.7% | 95.0% | 95.0% | 1.28x | 1.28x |

## 3. Benchmark-Aware Per-Benchmark

| Benchmark | V1 Rerun | BA Rerun | V1 Recov | BA Recov | V1 Speed | BA Speed |
|-----------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| argparse | 23.8% | 23.8% | 97.5% | 97.5% | 1.30x | 1.30x |
| dict_config | 13.8% | 13.8% | 87.5% | 87.5% | 1.51x | 1.51x |
| openmmlab_config | 28.7% | 28.7% | 95.0% | 95.0% | 1.29x | 1.29x |
| pipeline_stage_config | 26.2% | 26.2% | 97.5% | 97.5% | 1.31x | 1.31x |
| complex_nested_config | 31.2% | 15.0% | 73.8% | 62.5% | 1.24x | 1.55x |
| rich_cli_option_groups | 28.7% | 28.7% | 95.0% | 95.0% | 1.28x | 1.28x |

## 4. Partial Repair Speedup Analysis

Partial repair keeps V1 rerun decisions but reduces AR cost to a fraction of full rerun.

| Repair Cost | Speedup | vs FLY | vs V1 Speed |
|:-----------:|:-------:|:------:|:-----------:|
| 25% | 1.76x | **107%** | +34% |
| 50% | 1.58x | 96% | +20% |
| 75% | 1.44x | 87% | +9% |

## 5. Speed/Quality Pareto

| Variant | Speedup | Recoverable | Speed Rank | Quality Rank |
|---------|:-------:|:-----------:|:----------:|:------------:|
| AR | 1.00x | 84.6% | 9 | 7 |
| FLY | 1.64x | 80.8% | 3 | 8 |
| TASD-FG | 2.00x | 72.5% | 1 | 9 |
| TASD-FG-V (V1) | 1.31x | 91.0% | 8 | 1 |
| V-Lite | 1.36x | 91.0% | 6 | 2 |
| Benchmark-Aware | 1.36x | 89.2% | 7 | 6 |
| Partial Repair 25% | 1.76x | 91.0% | 2 | 3 |
| Partial Repair 50% | 1.58x | 91.0% | 4 | 4 |
| Partial Repair 75% | 1.44x | 91.0% | 5 | 5 |

## 6. Conclusions

### Q1: Can V-Lite achieve >= 1.5x speedup with >= 86% recoverable?

**Partial.** V-Lite recoverable=91.0% meets target, but speedup=1.36x is below 1.5x.

### Q2: Does benchmark-aware rerun reduce wasted reruns?

- V1 rerun: 122 samples (25.4%)
- BA rerun: 109 samples (22.7%)
- Saved: 13 reruns
- Speedup gain: 1.36x vs V1 1.31x (+3.7%)
- Recoverable loss: -9 samples

**YES but with tradeoff.** Saves 13 reruns but loses 9 recoverable samples.

### Q3: Is partial repair worth implementing?

- 25% cost: 1.76x — **EXCEEDS FLY**
- 50% cost: 1.58x — 96% of FLY speed
- 75% cost: 1.44x — 87% of FLY speed

**YES.** Partial repair at 25% cost achieves 1.76x >= FLY 1.64x. Worth implementing if actual repair cost is <= 25% of full AR.

## 7. Recommendation

**Recommended: Partial-25%** — speedup=1.76x, recoverable=91.0%
Other meeting criteria: Partial-50%

**TASD-FG-V (V1) remains the best quality option** at 91.0% recoverable. For speed-sensitive scenarios, use TASD-FG (2.00x, 72.5% recoverable) or explore V-Lite/partial-repair combinations.
