# TASD-FG-VR Partial Repair Pilot

**Pilot**: 60 rerun-triggered + 60 clean = 120 samples

## 1. Main Comparison

| Method | Recoverable | Score 2 | Score 1 | Score 0 | Speedup | Repair Triggered | Partial | Full | Avg Cost |
|--------|:----------:|:------:|:------:|:------:|:-------:|:----------------:|:-------:|:----:|:--------:|
| TASD-FG | 67/120 (55.8%) | 60 | 7 | 53 | 1.99x | — | — | — | — |
| TASD-FG-V | 113/120 (94.2%) | 95 | 18 | 7 | 0.99x | 60 | — | 60 | 1.00 |
| **TASD-FG-VR** | 113/120 (94.2%) | 95 | 18 | 7 | 1.53x | 60 | 24 | 36 | 0.91 |

## 2. Repair Mode Breakdown

| Final Source | Count | Pct |
|-------------|:-----:|:---:|
| TASD_FG | 60 | 50.0% |
| PARTIAL_AR_REPAIR | 24 | 20.0% |
| FULL_AR_RERUN | 36 | 30.0% |

## 3. Repair Cost Distribution

- Min: 0.59
- P25: 0.83
- Median: 1.00
- P75: 1.00
- Max: 1.00
- Mean: 0.91
- Full fallback rate: 60.0%

| Cost Range | Count | Pct |
|-----------|:-----:|:---:|
| 0.00-0.25 | 0 | 0.0% |
| 0.25-0.50 | 0 | 0.0% |
| 0.50-0.75 | 10 | 16.7% |
| 0.75-1.00 | 14 | 23.3% |
| 1.00 (full) | 36 | 60.0% |

## 4. Per-Benchmark

| Benchmark | Pilot N | Recoverable | Score 0 | Avg Cost | Partial | Full Fallback | Speedup |
|-----------|:-------:|:----------:|:------:|:--------:|:-------:|:------------:|:-------:|
| argparse | 22 | 21/22 (95.5%) | 1 | 0.85 | 4 | 55.6% | 1.62x |
| dict_config | 15 | 15/15 (100.0%) | 0 | 0.92 | 1 | 80.0% | 1.64x |
| openmmlab_config | 24 | 24/24 (100.0%) | 0 | 0.94 | 5 | 54.5% | 1.59x |
| pipeline_stage_config | 27 | 27/27 (100.0%) | 0 | 0.95 | 3 | 70.0% | 1.66x |
| complex_nested_config | 18 | 13/18 (72.2%) | 5 | 0.93 | 5 | 64.3% | 1.26x |
| rich_cli_option_groups | 14 | 13/14 (92.9%) | 1 | 0.84 | 6 | 45.5% | 1.29x |

## 5. Clean Subset Verification

- Clean samples: 60
- Original score=2: 60
- Score=2 maintained: 60/60 (100.0%)
- False positive repairs: 0
- **No false positives** — clean samples untouched

## 6. Acceptance Criteria

| Criterion | Target | Actual | Status |
|-----------|:------:|:------:|:------:|
| recoverable >= 88% | 88% | 94.2% | **PASS** |
| speedup >= 1.50x | 1.50x | 1.53x | **PASS** |
| avg repair_cost_ratio <= 0.50 | <= 0.50 | 0.91 | **FAIL** |
| full fallback rate <= 30% | <= 30% | 60.0% | **FAIL** |
| clean score=2 >= 90% | >= 90% | 100.0% | **PASS** |

**FAILED criteria**: avg repair_cost_ratio <= 0.50, full fallback rate <= 30%


## 7. Conclusions

- **TASD-FG-VR recoverable**: 113/120 (94.2%)
- **TASD-FG-VR speedup**: 1.53x (vs TASD-FG-V 0.99x)
- **Avg repair cost**: 0.91 (partial: 24, full: 36)
- **Full fallback rate**: 60.0%

**Recommendation**: Average repair cost 0.91 > 0.75. Partial repair may not be worth implementing. Consider combining with V-Lite thresholds.
