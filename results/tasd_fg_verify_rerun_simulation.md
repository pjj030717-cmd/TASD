# TASD-FG-V: Verify-then-AR-rerun Offline Simulation

**480 samples across 6 benchmarks.**

**Baselines**: TASD-FG recoverable=348/480 (72.5%), AR recoverable=406/480 (84.6%)

## 1. Detector Variant Comparison

| Variant | Rerun | Ratio | s=0 | s=1 | s=2 | Recov. | Speedup | TP | FP | FN | FPR | Rescued |
|---------|:-----:|:-----:|:---:|:---:|:---:|:------:|:-------:|:--:|:--:|:--:|:---:|:-------:|
| V1 | 122 | 25.4% | 43 | 178 | 259 | 437/480 (91.0%) | 1.31x | 104 | 18 | 28 | 5.2% | 90 | **PASS** |
| V2 | 34 | 7.1% | 109 | 166 | 205 | 371/480 (77.3%) | 1.73x | 32 | 2 | 100 | 0.6% | 25 | 2/3 criteria |
| V3 | 13 | 2.7% | 121 | 160 | 199 | 359/480 (74.8%) | 1.88x | 13 | 0 | 119 | 0.0% | 11 | 2/3 criteria |

## 2. V1 Per-Benchmark Breakdown

| Benchmark | Rerun | Ratio | s=0 | s=1 | s=2 | Recov. | Rescued | TP | FP | FN | FPR |
|-----------|:-----:|:-----:|:---:|:---:|:---:|:------:|:------:|:--:|:--:|:--:|:---:|
| argparse | 19 | 23.8% | 2 | 23 | 55 | 78/80 (97.5%) | 14 | 15 | 4 | 1 | 6.2% |
| dict_config | 11 | 13.8% | 10 | 32 | 38 | 70/80 (87.5%) | 7 | 9 | 2 | 8 | 3.2% |
| openmmlab_config | 23 | 28.7% | 4 | 17 | 59 | 76/80 (95.0%) | 19 | 21 | 2 | 2 | 3.5% |
| pipeline_stage_config | 21 | 26.2% | 2 | 13 | 65 | 78/80 (97.5%) | 14 | 14 | 7 | 2 | 10.9% |
| complex_nested_config | 25 | 31.2% | 21 | 44 | 15 | 59/80 (73.8%) | 15 | 22 | 3 | 13 | 6.7% |
| rich_cli_option_groups | 23 | 28.7% | 4 | 49 | 27 | 76/80 (95.0%) | 21 | 23 | 0 | 2 | 0.0% |

## 2. V2 Per-Benchmark Breakdown

| Benchmark | Rerun | Ratio | s=0 | s=1 | s=2 | Recov. | Rescued | TP | FP | FN | FPR |
|-----------|:-----:|:-----:|:---:|:---:|:---:|:------:|:------:|:--:|:--:|:--:|:---:|
| argparse | 0 | 0.0% | 16 | 21 | 43 | 64/80 (80.0%) | 0 | 0 | 0 | 16 | 0.0% |
| dict_config | 5 | 6.2% | 14 | 30 | 36 | 66/80 (82.5%) | 3 | 5 | 0 | 12 | 0.0% |
| openmmlab_config | 0 | 0.0% | 23 | 15 | 42 | 57/80 (71.2%) | 0 | 0 | 0 | 23 | 0.0% |
| pipeline_stage_config | 0 | 0.0% | 16 | 16 | 48 | 64/80 (80.0%) | 0 | 0 | 0 | 16 | 0.0% |
| complex_nested_config | 16 | 20.0% | 27 | 38 | 15 | 53/80 (66.2%) | 10 | 14 | 2 | 21 | 4.4% |
| rich_cli_option_groups | 13 | 16.2% | 13 | 46 | 21 | 67/80 (83.8%) | 12 | 13 | 0 | 12 | 0.0% |

## 2. V3 Per-Benchmark Breakdown

| Benchmark | Rerun | Ratio | s=0 | s=1 | s=2 | Recov. | Rescued | TP | FP | FN | FPR |
|-----------|:-----:|:-----:|:---:|:---:|:---:|:------:|:------:|:--:|:--:|:--:|:---:|
| argparse | 0 | 0.0% | 16 | 21 | 43 | 64/80 (80.0%) | 0 | 0 | 0 | 16 | 0.0% |
| dict_config | 1 | 1.2% | 16 | 29 | 35 | 64/80 (80.0%) | 1 | 1 | 0 | 16 | 0.0% |
| openmmlab_config | 0 | 0.0% | 23 | 15 | 42 | 57/80 (71.2%) | 0 | 0 | 0 | 23 | 0.0% |
| pipeline_stage_config | 0 | 0.0% | 16 | 16 | 48 | 64/80 (80.0%) | 0 | 0 | 0 | 16 | 0.0% |
| complex_nested_config | 3 | 3.8% | 34 | 33 | 13 | 46/80 (57.5%) | 1 | 3 | 0 | 32 | 0.0% |
| rich_cli_option_groups | 9 | 11.2% | 16 | 46 | 18 | 64/80 (80.0%) | 9 | 9 | 0 | 16 | 0.0% |

## 3. V2 (Oracle) vs V3 (Real) Gap Analysis

| Metric | V2 (Oracle) | V3 (Real) | Delta |
|--------|:-----------:|:---------:|:-----:|
| Recoverable | 371/480 (77.3%) | 359/480 (74.8%) | -12 |
| Speedup | 1.73x | 1.88x | +0.14x |
| Rerun Ratio | 7.1% | 2.7% | -4.4% |
| Rescued | 25 | 11 | -14 |
| FP Rate | 0.6% | 0.0% | -0.6% |
| TP | 32 | 13 | -19 |
| FN | 100 | 119 | +19 |

**V2-only TP (missed by V3)**: 19 samples that V2 catches but V3 misses. These are samples where structural_f1 < 0.50 is the primary risk signal.

**V2-only rescued**: 14 samples (V2 rescues them but V3 doesn't).

## 4. Judgment Criteria Assessment

| Criterion | V1 | V2 | V3 |
|-----------|:--:|:--:|:--:|
| Recoverable >= 80% | PASS | FAIL | FAIL |
| Speedup >= 1.3x | PASS | PASS | PASS |
| FP Rate <= 30% | PASS | PASS | PASS |

**V1**: 3/3 criteria passed
**V2**: 2/3 criteria passed
**V3**: 2/3 criteria passed

## 5. Conclusions

### Best Real Detector (V3):

- Recoverable: 359/480 (74.8%)
- Speedup: 1.88x
- Rerun ratio: 2.7%
- Rescued: 11 samples
- FP rate: 0.0%
- FN: 119 samples (TASD-FG s=0 not rerun)

**V3 meets 2/3 criteria.**

### Oracle (V2) gap:

- V2 achieves 371/480 recoverable (77.3%) with 1.73x speedup
- Gap to V3: 14 rescued samples, 2.5% recoverable rate
- This gap represents the ceiling of what structural_f1-based detection could achieve

### Recommendation:

- **V3 is deployable** as a real verify-then-rerun pipeline
- V3 recoverable rate is below 80%, need to improve detector sensitivity
- **Per-benchmark tuning may be needed**: some benchmarks trigger more reruns than others
- **TASD-FG-V is worth implementing**: the offline simulation shows the verify-then-rerun strategy can recover quality while maintaining speedup
