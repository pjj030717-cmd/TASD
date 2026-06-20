# TASD-FG-V Pareto Routing Analysis

**Samples**: 480 total, 122 V1-triggered, 358 clean

**Goal**: Find rerun subset that beats FLY on both speed (>=1.64x) and recoverable (>=80.8%)

## 1. Baselines

| Method | Recoverable | Score 0 | Speedup | Avg TPS |
|--------|:----------:|:------:|:-------:|:-------:|
| AR | 406/480 (84.6%) | 74 | 1.00x | 33.2 |
| TASD-FG | 348/480 (72.5%) | 132 | 2.00x | 66.4 |
| FLY | 388/480 (80.8%) | 92 | 1.64x | 54.5 |

## 2. Rerun Gain Analysis

V1 triggered: 122 samples

- Score improved: 99/122 (81.1%)
- Binary gain (0->1+): 90/122 (73.8%)
- Avg gain: 1.28
- Gain distribution: 0=22, 1=41, 2=58

### 2.1 Per-Reason Precision

| Reason | Triggered | Rescued | Precision | Binary Gain | Avg Gain | Improved | Avg AR TPS |
|--------|:--------:|:------:|:---------:|:----------:|:--------:|:--------:|:----------:|
| bracket_balance | 65 | 57 | 87.7% | 57 | 1.42 | 57 | 32.9 |
| repetition | 35 | 20 | 57.1% | 20 | 1.00 | 27 | 33.1 |
| off_structure | 34 | 23 | 67.6% | 23 | 1.32 | 25 | 33.1 |

### 2.2 Per-Reason Benchmark Distribution

**bracket_balance** (65 triggered):
- argparse: 15
- dict_config: 7
- openmmlab_config: 14
- pipeline_stage_config: 5
- complex_nested_config: 12
- rich_cli_option_groups: 12

**repetition** (35 triggered):
- argparse: 3
- dict_config: 5
- openmmlab_config: 2
- pipeline_stage_config: 6
- complex_nested_config: 11
- rich_cli_option_groups: 8

**off_structure** (34 triggered):
- argparse: 1
- openmmlab_config: 7
- pipeline_stage_config: 10
- complex_nested_config: 5
- rich_cli_option_groups: 11

### 2.3 Reason Combinations

| Combination | Triggered | Rescued | Precision |
|------------|:--------:|:------:|:---------:|
| bracket_balance | 62 | 56 | 90.3% |
| repetition | 25 | 10 | 40.0% |
| off_structure | 23 | 14 | 60.9% |
| off_structure + repetition | 9 | 9 | 100.0% |
| bracket_balance + off_structure | 2 | 0 | 0.0% |
| bracket_balance + repetition | 1 | 1 | 100.0% |

## 3. Oracle Top-K Curve

Theoretical upper bound: rerun samples sorted by actual gain.

| K | Rerun Ratio | Recoverable | Score 0 | Speedup | vs FLY Speed | vs FLY Recov |
|---|:----------:|:----------:|:------:|:-------:|:-----------:|:------------:|
| 0 | 0.0% | 348/480 (72.5%) | 132 | 2.00x | PASS | FAIL |
| 10 | 2.1% | 358/480 (74.6%) | 122 | 1.98x | PASS | FAIL |
| 20 | 4.2% | 368/480 (76.7%) | 112 | 1.96x | PASS | FAIL |
| 30 | 6.2% | 378/480 (78.8%) | 102 | 1.94x | PASS | FAIL |
| 40 | 8.3% | 388/480 (80.8%) | 92 | 1.92x | PASS | PASS |
| 41 | 8.5% | 389/480 (81.0%) | 91 | 1.92x | PASS | PASS |
| 42 | 8.8% | 390/480 (81.2%) | 90 | 1.91x | PASS | PASS |
| 43 | 9.0% | 391/480 (81.5%) | 89 | 1.91x | PASS | PASS |
| 44 | 9.2% | 392/480 (81.7%) | 88 | 1.91x | PASS | PASS |
| 45 | 9.4% | 393/480 (81.9%) | 87 | 1.91x | PASS | PASS |
| 46 | 9.6% | 394/480 (82.1%) | 86 | 1.91x | PASS | PASS |
| 47 | 9.8% | 395/480 (82.3%) | 85 | 1.90x | PASS | PASS |
| 48 | 10.0% | 396/480 (82.5%) | 84 | 1.90x | PASS | PASS |
| 49 | 10.2% | 397/480 (82.7%) | 83 | 1.90x | PASS | PASS |
| 50 | 10.4% | 398/480 (82.9%) | 82 | 1.90x | PASS | PASS |
| 51 | 10.6% | 399/480 (83.1%) | 81 | 1.90x | PASS | PASS |
| 52 | 10.8% | 400/480 (83.3%) | 80 | 1.89x | PASS | PASS |
| 53 | 11.0% | 401/480 (83.5%) | 79 | 1.89x | PASS | PASS |
| 54 | 11.2% | 402/480 (83.8%) | 78 | 1.89x | PASS | PASS |
| 55 | 11.5% | 403/480 (84.0%) | 77 | 1.89x | PASS | PASS |
| 56 | 11.7% | 404/480 (84.2%) | 76 | 1.89x | PASS | PASS |
| 57 | 11.9% | 405/480 (84.4%) | 75 | 1.88x | PASS | PASS |
| 58 | 12.1% | 406/480 (84.6%) | 74 | 1.88x | PASS | PASS |
| 59 | 12.3% | 407/480 (84.8%) | 73 | 1.88x | PASS | PASS |
| 60 | 12.5% | 408/480 (85.0%) | 72 | 1.88x | PASS | PASS |
| 61 | 12.7% | 409/480 (85.2%) | 71 | 1.87x | PASS | PASS |
| 62 | 12.9% | 410/480 (85.4%) | 70 | 1.87x | PASS | PASS |
| 63 | 13.1% | 411/480 (85.6%) | 69 | 1.87x | PASS | PASS |
| 64 | 13.3% | 412/480 (85.8%) | 68 | 1.87x | PASS | PASS |
| 65 | 13.5% | 413/480 (86.0%) | 67 | 1.87x | PASS | PASS |
| 66 | 13.8% | 414/480 (86.2%) | 66 | 1.87x | PASS | PASS |
| 67 | 14.0% | 415/480 (86.5%) | 65 | 1.86x | PASS | PASS |
| 68 | 14.2% | 416/480 (86.7%) | 64 | 1.86x | PASS | PASS |
| 69 | 14.4% | 417/480 (86.9%) | 63 | 1.86x | PASS | PASS |
| 70 | 14.6% | 418/480 (87.1%) | 62 | 1.86x | PASS | PASS |
| 71 | 14.8% | 419/480 (87.3%) | 61 | 1.86x | PASS | PASS |
| 72 | 15.0% | 420/480 (87.5%) | 60 | 1.85x | PASS | PASS |
| 73 | 15.2% | 421/480 (87.7%) | 59 | 1.85x | PASS | PASS |
| 74 | 15.4% | 422/480 (87.9%) | 58 | 1.85x | PASS | PASS |
| 75 | 15.6% | 423/480 (88.1%) | 57 | 1.85x | PASS | PASS |
| 76 | 15.8% | 424/480 (88.3%) | 56 | 1.84x | PASS | PASS |
| 77 | 16.0% | 425/480 (88.5%) | 55 | 1.84x | PASS | PASS |
| 78 | 16.2% | 426/480 (88.8%) | 54 | 1.84x | PASS | PASS |
| 79 | 16.5% | 427/480 (89.0%) | 53 | 1.84x | PASS | PASS |
| 80 | 16.7% | 428/480 (89.2%) | 52 | 1.84x | PASS | PASS |
| 81 | 16.9% | 429/480 (89.4%) | 51 | 1.83x | PASS | PASS |
| 82 | 17.1% | 430/480 (89.6%) | 50 | 1.83x | PASS | PASS |
| 83 | 17.3% | 431/480 (89.8%) | 49 | 1.83x | PASS | PASS |
| 84 | 17.5% | 432/480 (90.0%) | 48 | 1.83x | PASS | PASS |
| 85 | 17.7% | 433/480 (90.2%) | 47 | 1.82x | PASS | PASS |
| 86 | 17.9% | 434/480 (90.4%) | 46 | 1.82x | PASS | PASS |
| 87 | 18.1% | 435/480 (90.6%) | 45 | 1.82x | PASS | PASS |
| 88 | 18.3% | 436/480 (90.8%) | 44 | 1.82x | PASS | PASS |
| 89 | 18.5% | 437/480 (91.0%) | 43 | 1.82x | PASS | PASS |
| 90 | 18.8% | 438/480 (91.2%) | 42 | 1.81x | PASS | PASS |
| 91 | 19.0% | 438/480 (91.2%) | 42 | 1.81x | PASS | PASS |
| 92 | 19.2% | 438/480 (91.2%) | 42 | 1.81x | PASS | PASS |
| 93 | 19.4% | 438/480 (91.2%) | 42 | 1.81x | PASS | PASS |
| 94 | 19.6% | 438/480 (91.2%) | 42 | 1.81x | PASS | PASS |
| 95 | 19.8% | 438/480 (91.2%) | 42 | 1.81x | PASS | PASS |
| 96 | 20.0% | 438/480 (91.2%) | 42 | 1.80x | PASS | PASS |
| 97 | 20.2% | 438/480 (91.2%) | 42 | 1.80x | PASS | PASS |
| 98 | 20.4% | 438/480 (91.2%) | 42 | 1.80x | PASS | PASS |
| 99 | 20.6% | 438/480 (91.2%) | 42 | 1.80x | PASS | PASS |
| 100 | 20.8% | 438/480 (91.2%) | 42 | 1.80x | PASS | PASS |
| 101 | 21.0% | 438/480 (91.2%) | 42 | 1.80x | PASS | PASS |
| 102 | 21.2% | 438/480 (91.2%) | 42 | 1.79x | PASS | PASS |
| 103 | 21.5% | 438/480 (91.2%) | 42 | 1.79x | PASS | PASS |
| 104 | 21.7% | 438/480 (91.2%) | 42 | 1.79x | PASS | PASS |
| 105 | 21.9% | 438/480 (91.2%) | 42 | 1.79x | PASS | PASS |
| 106 | 22.1% | 438/480 (91.2%) | 42 | 1.78x | PASS | PASS |
| 107 | 22.3% | 438/480 (91.2%) | 42 | 1.78x | PASS | PASS |
| 108 | 22.5% | 438/480 (91.2%) | 42 | 1.78x | PASS | PASS |
| 109 | 22.7% | 438/480 (91.2%) | 42 | 1.78x | PASS | PASS |
| 110 | 22.9% | 438/480 (91.2%) | 42 | 1.78x | PASS | PASS |
| 111 | 23.1% | 438/480 (91.2%) | 42 | 1.77x | PASS | PASS |
| 112 | 23.3% | 438/480 (91.2%) | 42 | 1.77x | PASS | PASS |
| 113 | 23.5% | 438/480 (91.2%) | 42 | 1.77x | PASS | PASS |
| 114 | 23.8% | 438/480 (91.2%) | 42 | 1.77x | PASS | PASS |
| 115 | 24.0% | 438/480 (91.2%) | 42 | 1.77x | PASS | PASS |
| 116 | 24.2% | 438/480 (91.2%) | 42 | 1.76x | PASS | PASS |
| 117 | 24.4% | 438/480 (91.2%) | 42 | 1.76x | PASS | PASS |
| 118 | 24.6% | 438/480 (91.2%) | 42 | 1.76x | PASS | PASS |
| 119 | 24.8% | 438/480 (91.2%) | 42 | 1.76x | PASS | PASS |
| 120 | 25.0% | 438/480 (91.2%) | 42 | 1.76x | PASS | PASS |
| 121 | 25.2% | 438/480 (91.2%) | 42 | 1.75x | PASS | PASS |
| 122 | 25.4% | 437/480 (91.0%) | 43 | 1.75x | PASS | PASS |

**Oracle Pareto point**: K=40, rerun_ratio=8.3%, speedup=1.92x, recoverable=80.8%

## 4. Rule-Based Routing

| Router | Rerun | Ratio | Recoverable | Score 0 | Speedup | vs FLY Speed | vs FLY Recov | Precision |
|--------|:-----:|:-----:|:----------:|:------:|:-------:|:-----------:|:------------:|:---------:|
| A: bracket_balance only | 65 | 13.5% | 405/480 (84.4%) | 75 | 1.87x | PASS | PASS | 100.0% |
| B: repetition only | 35 | 7.3% | 367/480 (76.5%) | 113 | 1.93x | PASS | FAIL | 100.0% |
| C: off_structure only | 34 | 7.1% | 371/480 (77.3%) | 109 | 1.93x | PASS | FAIL | 100.0% |
| D: bracket + repetition | 99 | 20.6% | 423/480 (88.1%) | 57 | 1.80x | PASS | PASS | 100.0% |
| E: bracket on high-precision BM | 41 | 8.5% | 384/480 (80.0%) | 96 | 1.92x | PASS | FAIL | 100.0% |
| F: severe only (higher thresholds) | 122 | 25.4% | 437/480 (91.0%) | 43 | 1.75x | PASS | PASS | 100.0% |

### 4.1 Per-Benchmark Detail

**A: bracket_balance only**:
| Benchmark | N | Rerun | Recoverable |
|-----------|:--:|:-----:|:----------:|
| argparse | 80 | 15 | 78/80 (97.5%) |
| dict_config | 80 | 7 | 68/80 (85.0%) |
| openmmlab_config | 80 | 14 | 69/80 (86.2%) |
| pipeline_stage_config | 80 | 5 | 69/80 (86.2%) |
| complex_nested_config | 80 | 12 | 54/80 (67.5%) |
| rich_cli_option_groups | 80 | 12 | 67/80 (83.8%) |

**B: repetition only**:
| Benchmark | N | Rerun | Recoverable |
|-----------|:--:|:-----:|:----------:|
| argparse | 80 | 3 | 64/80 (80.0%) |
| dict_config | 80 | 5 | 66/80 (82.5%) |
| openmmlab_config | 80 | 2 | 57/80 (71.2%) |
| pipeline_stage_config | 80 | 6 | 67/80 (83.8%) |
| complex_nested_config | 80 | 11 | 50/80 (62.5%) |
| rich_cli_option_groups | 80 | 8 | 63/80 (78.8%) |

**C: off_structure only**:
| Benchmark | N | Rerun | Recoverable |
|-----------|:--:|:-----:|:----------:|
| argparse | 80 | 1 | 64/80 (80.0%) |
| dict_config | 80 | 0 | 63/80 (78.8%) |
| openmmlab_config | 80 | 7 | 64/80 (80.0%) |
| pipeline_stage_config | 80 | 10 | 70/80 (87.5%) |
| complex_nested_config | 80 | 5 | 46/80 (57.5%) |
| rich_cli_option_groups | 80 | 11 | 64/80 (80.0%) |

**D: bracket + repetition**:
| Benchmark | N | Rerun | Recoverable |
|-----------|:--:|:-----:|:----------:|
| argparse | 80 | 18 | 78/80 (97.5%) |
| dict_config | 80 | 11 | 70/80 (87.5%) |
| openmmlab_config | 80 | 16 | 69/80 (86.2%) |
| pipeline_stage_config | 80 | 11 | 72/80 (90.0%) |
| complex_nested_config | 80 | 23 | 59/80 (73.8%) |
| rich_cli_option_groups | 80 | 20 | 75/80 (93.8%) |

**E: bracket on high-precision BM**:
| Benchmark | N | Rerun | Recoverable |
|-----------|:--:|:-----:|:----------:|
| argparse | 80 | 15 | 78/80 (97.5%) |
| dict_config | 80 | 7 | 68/80 (85.0%) |
| openmmlab_config | 80 | 14 | 69/80 (86.2%) |
| pipeline_stage_config | 80 | 5 | 69/80 (86.2%) |
| complex_nested_config | 80 | 0 | 45/80 (56.2%) |
| rich_cli_option_groups | 80 | 0 | 55/80 (68.8%) |

**F: severe only (higher thresholds)**:
| Benchmark | N | Rerun | Recoverable |
|-----------|:--:|:-----:|:----------:|
| argparse | 80 | 19 | 78/80 (97.5%) |
| dict_config | 80 | 11 | 70/80 (87.5%) |
| openmmlab_config | 80 | 23 | 76/80 (95.0%) |
| pipeline_stage_config | 80 | 21 | 78/80 (97.5%) |
| complex_nested_config | 80 | 25 | 59/80 (73.8%) |
| rich_cli_option_groups | 80 | 23 | 76/80 (95.0%) |

## 5. Key Findings

### 5.1 Answers to Key Questions

**1. Oracle Pareto point exists?** YES — K=40 samples (8.3% rerun) achieves speedup=1.92x and recoverable=80.8%.
   This beats FLY (speedup=1.64x, recoverable=80.8%).

**2. Best rule-based router:** F: severe only (higher thresholds) — rerun=122 (25.4%), speedup=1.75x, recoverable=91.0%.

**3. Most cost-effective risk reason:** bracket_balance (precision=87.7%)

### 5.2 Mode Recommendation

**3 rule-based routers beat FLY on both speed and quality.**

| Router | Speedup | Recoverable | Rerun | vs Oracle |
|--------|:-------:|:----------:|:-----:|:---------:|
| F: severe only (higher thresholds) | 1.75x | 91.0% | 122 | K=40 (gap=82) |
| D: bracket + repetition | 1.80x | 88.1% | 99 | K=40 (gap=59) |
| A: bracket_balance only | 1.87x | 84.4% | 65 | K=40 (gap=25) |

**Recommended: `F: severe only (higher thresholds)`** — best combined speed-quality score.

- Speedup: 1.75x (vs FLY 1.64x)
- Recoverable: 91.0% (vs FLY 80.8%)
- Rerun: 122 samples (25.4% of total)
- Oracle gap: 82 more samples than optimal (K=40)

**Alternative: `A: bracket_balance only`** — most efficient per-rerun.

- Speedup: 1.87x, Recoverable: 84.4%
- Rerun: only 65 samples (13.5%), precision=100.0%
- Reruns only bracket_balance risks, which have 87.7% rescue precision
- This is the closest to oracle (gap=25 vs K=40)

## 6. Data Summary

- Total samples: 480
- V1 triggered: 122 (25.4%)
- FLY: speedup=1.64x, recoverable=80.8%
- TASD-FG: speedup=2.00x, recoverable=72.5%
- TASD-FG-V full: speedup=1.31x, recoverable=91.0%
