# TASD-FG-V Pilot Experiment (120 samples)

**120 samples**: 60 hard (score=0) + 60 clean (score=2)

## 1. Overall Comparison

| Metric | TASD-FG | TASD-FG-V |
|--------|:-------:|:---------:|
| Score 2 | 60 | 89 |
| Score 1 | 0 | 15 |
| Score 0 | 60 | 16 |
| Recoverable | 60/120 (50.0%) | 104/120 (86.7%) |
| Avg TPS | 65.8 | 53.0 |
| Estimated Speedup | 1.0x (baseline) | 1.09x |
| Rerun Ratio | 0.0% | 40.0% |
| Final Source | TASD_FG: 100% | TASD_FG: 72 (60.0%), AR_RERUN: 48 (40.0%) |

## 2. Hard Subset (TASD-FG score=0)

| Metric | Value |
|--------|:-----:|
| Samples | 60 |
| Score=0 before | 60 |
| Score=0 after | 16 |
| Score=0 reduction | 44 (73.3%) |
| Rescued to >=1 | 44 |
| Rescued to 2 | 29 |

## 3. Clean Subset (TASD-FG score=2)

| Metric | Value |
|--------|:-----:|
| Samples | 60 |
| Score=2 before | 60 |
| Score=2 after | 60 |
| Score=2 retention | 100.0% |
| False positive reruns | 0 (0.0%) |

## 4. Rerun Reasons Distribution

| Reason | Count |
|--------|:-----:|
| off_structure | 13 |
| repetition | 11 |
| duplicate_option | 0 |
| bracket_balance | 30 |

## 5. Per-Benchmark Breakdown

| Benchmark | N | Hard | Clean | Rerun | Recoverable | Hard Rescued | Clean FP |
|-----------|:-:|:----:|:-----:|:-----:|:----------:|:------------:|:--------:|
| argparse | 20 | 10 | 10 | 9 (45%) | 18/20 (90%) | 8 | 0 |
| dict_config | 20 | 10 | 10 | 7 (35%) | 17/20 (85%) | 7 | 0 |
| openmmlab_config | 20 | 10 | 10 | 8 (40%) | 17/20 (85%) | 7 | 0 |
| pipeline_stage_config | 20 | 10 | 10 | 9 (45%) | 19/20 (95%) | 9 | 0 |
| complex_nested_config | 20 | 10 | 10 | 7 (35%) | 15/20 (75%) | 5 | 0 |
| rich_cli_option_groups | 20 | 10 | 10 | 8 (40%) | 18/20 (90%) | 8 | 0 |

## 6. Pass/Fail Criteria

| Criterion | Target | Actual | Status |
|-----------|:------:|:------:|:------:|
| hard_score0_reduction_40pct | 40.0 | 73.3 | **PASS** |
| clean_score2_retention_90pct | 90.0 | 100.0 | **PASS** |
| speedup_1.2x | 1.2 | 1.1 | **FAIL** |
| fp_rate_15pct | 15.0 | 0.0 | **PASS** |

**PILOT FAILED** on: speedup_1.2x. Do NOT run full 480.
