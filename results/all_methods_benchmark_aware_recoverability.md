# All Methods Benchmark-Aware Recoverability

## 1. Score Distribution: Unified vs Benchmark-Aware

| Method | U s=0 | U s=1 | U s=2 | BA s=0 | BA s=1 | BA s=2 | U Recov. | BA Recov. | Delta | Improved | Degraded |
|--------|:-----:|:-----:|:-----:|:------:|:------:|:------:|:--------:|:---------:|:-----:|:--------:|:--------:|
| AR | 74 | 155 | 251 | 28 | 160 | 292 | 406/480 (84.6%) | 452/480 (94.2%) | +46 | 58 | 0 |
| GSD | 101 | 141 | 238 | 69 | 138 | 273 | 379/480 (79.0%) | 411/480 (85.6%) | +32 | 40 | 0 |
| N-gram SD | 115 | 209 | 156 | 40 | 214 | 226 | 365/480 (76.0%) | 440/480 (91.7%) | +75 | 89 | 0 |
| FLY | 92 | 102 | 286 | 44 | 94 | 342 | 388/480 (80.8%) | 436/480 (90.8%) | +48 | 65 | 0 |
| TASD | 191 | 33 | 256 | 92 | 79 | 309 | 289/480 (60.2%) | 388/480 (80.8%) | +99 | 99 | 0 |
| TASD-FG | 132 | 156 | 192 | 90 | 157 | 233 | 348/480 (72.5%) | 390/480 (81.2%) | +42 | 52 | 0 |

## 2. Per-Benchmark Recoverable Rate

| Benchmark | Method | U Recov. | BA Recov. | Delta |
|-----------|--------|:--------:|:---------:|:-----:|
| argparse | AR | 73/80 (91.2%) | 80/80 (100.0%) | +7 |
| argparse | GSD | 75/80 (93.8%) | 80/80 (100.0%) | +5 |
| argparse | N-gram SD | 62/80 (77.5%) | 77/80 (96.2%) | +15 |
| argparse | FLY | 76/80 (95.0%) | 79/80 (98.8%) | +3 |
| argparse | TASD | 66/80 (82.5%) | 80/80 (100.0%) | +14 |
| argparse | TASD-FG | 64/80 (80.0%) | 79/80 (98.8%) | +15 |

| dict_config | AR | 51/80 (63.8%) | 52/80 (65.0%) | +1 |
| dict_config | GSD | 36/80 (45.0%) | 36/80 (45.0%) | +0 |
| dict_config | N-gram SD | 49/80 (61.2%) | 51/80 (63.8%) | +2 |
| dict_config | FLY | 39/80 (48.8%) | 40/80 (50.0%) | +1 |
| dict_config | TASD | 19/80 (23.8%) | 19/80 (23.8%) | +0 |
| dict_config | TASD-FG | 63/80 (78.8%) | 63/80 (78.8%) | +0 |

| openmmlab_config | AR | 74/80 (92.5%) | 80/80 (100.0%) | +6 |
| openmmlab_config | GSD | 77/80 (96.2%) | 80/80 (100.0%) | +3 |
| openmmlab_config | N-gram SD | 64/80 (80.0%) | 79/80 (98.8%) | +15 |
| openmmlab_config | FLY | 66/80 (82.5%) | 77/80 (96.2%) | +11 |
| openmmlab_config | TASD | 56/80 (70.0%) | 70/80 (87.5%) | +14 |
| openmmlab_config | TASD-FG | 57/80 (71.2%) | 70/80 (87.5%) | +13 |

| pipeline_stage_config | AR | 80/80 (100.0%) | 80/80 (100.0%) | +0 |
| pipeline_stage_config | GSD | 62/80 (77.5%) | 65/80 (81.2%) | +3 |
| pipeline_stage_config | N-gram SD | 75/80 (93.8%) | 78/80 (97.5%) | +3 |
| pipeline_stage_config | FLY | 80/80 (100.0%) | 80/80 (100.0%) | +0 |
| pipeline_stage_config | TASD | 80/80 (100.0%) | 80/80 (100.0%) | +0 |
| pipeline_stage_config | TASD-FG | 64/80 (80.0%) | 69/80 (86.2%) | +5 |

| complex_nested_config | AR | 51/80 (63.8%) | 80/80 (100.0%) | +29 |
| complex_nested_config | GSD | 56/80 (70.0%) | 76/80 (95.0%) | +20 |
| complex_nested_config | N-gram SD | 55/80 (68.8%) | 80/80 (100.0%) | +25 |
| complex_nested_config | FLY | 48/80 (60.0%) | 80/80 (100.0%) | +32 |
| complex_nested_config | TASD | 7/80 (8.8%) | 70/80 (87.5%) | +63 |
| complex_nested_config | TASD-FG | 45/80 (56.2%) | 50/80 (62.5%) | +5 |

| rich_cli_option_groups | AR | 77/80 (96.2%) | 80/80 (100.0%) | +3 |
| rich_cli_option_groups | GSD | 73/80 (91.2%) | 74/80 (92.5%) | +1 |
| rich_cli_option_groups | N-gram SD | 60/80 (75.0%) | 75/80 (93.8%) | +15 |
| rich_cli_option_groups | FLY | 79/80 (98.8%) | 80/80 (100.0%) | +1 |
| rich_cli_option_groups | TASD | 61/80 (76.2%) | 69/80 (86.2%) | +8 |
| rich_cli_option_groups | TASD-FG | 55/80 (68.8%) | 59/80 (73.8%) | +4 |


## 3. TASD-FG vs Baselines (Benchmark-Aware)

### 3.1 Overall Recoverable Rate

| Method | U Recov. | BA Recov. | BA s=2 | BA s=1 | BA s=0 |
|--------|:--------:|:---------:|:------:|:------:|:------:|
| AR | 406/480 (84.6%) | 452/480 (94.2%) | 292 | 160 | 28 |
| GSD | 379/480 (79.0%) | 411/480 (85.6%) | 273 | 138 | 69 |
| N-gram SD | 365/480 (76.0%) | 440/480 (91.7%) | 226 | 214 | 40 |
| FLY | 388/480 (80.8%) | 436/480 (90.8%) | 342 | 94 | 44 |
| TASD | 289/480 (60.2%) | 388/480 (80.8%) | 309 | 79 | 92 |
| TASD-FG | 348/480 (72.5%) | 390/480 (81.2%) | 233 | 157 | 90 |

### 3.2 Per-Benchmark Recoverable (BA)

| Benchmark | AR | FLY | TASD | TASD-FG | Best |
|-----------|:---:|:---:|:----:|:-------:|:----:|
| argparse | 80/80 (100.0%) | 79/80 (98.8%) | 80/80 (100.0%) | 79/80 (98.8%) | **AR** 80/80 (100.0%) |
| dict_config | 52/80 (65.0%) | 40/80 (50.0%) | 19/80 (23.8%) | 63/80 (78.8%) | **TASD-FG** 63/80 (78.8%) |
| openmmlab_config | 80/80 (100.0%) | 77/80 (96.2%) | 70/80 (87.5%) | 70/80 (87.5%) | **AR** 80/80 (100.0%) |
| pipeline_stage_config | 80/80 (100.0%) | 80/80 (100.0%) | 80/80 (100.0%) | 69/80 (86.2%) | **AR** 80/80 (100.0%) |
| complex_nested_config | 80/80 (100.0%) | 80/80 (100.0%) | 70/80 (87.5%) | 50/80 (62.5%) | **AR** 80/80 (100.0%) |
| rich_cli_option_groups | 80/80 (100.0%) | 80/80 (100.0%) | 69/80 (86.2%) | 59/80 (73.8%) | **AR** 80/80 (100.0%) |

### 3.3 TASD-FG: Samples Rescued by BA Scoring

**52 samples rescued** (from s=0/1 to s=1/2)

**argparse** (15 rescued):
  - argparse_real_001: F1=0.775 bracket=0.000 trunc=0 0→1
  - argparse_real_004: F1=0.678 bracket=0.000 trunc=0 0→1
  - argparse_real_006: F1=0.933 bracket=0.000 trunc=0 0→2
  - argparse_real_010: F1=0.990 bracket=0.000 trunc=0 0→2
  - argparse_real_012: F1=0.852 bracket=0.000 trunc=0 0→2
  - ... and 10 more
**openmmlab_config** (13 rescued):
  - openmmlab_config_real_008: F1=0.992 bracket=0.000 trunc=0 0→2
  - openmmlab_config_real_012: F1=0.992 bracket=0.000 trunc=0 0→2
  - openmmlab_config_real_018: F1=0.992 bracket=0.000 trunc=0 0→2
  - openmmlab_config_real_041: F1=0.992 bracket=0.000 trunc=0 0→2
  - openmmlab_config_real_045: F1=0.992 bracket=0.000 trunc=0 0→2
  - ... and 8 more
**pipeline_stage_config** (5 rescued):
  - pipeline_stage_config_007: F1=0.702 bracket=0.000 trunc=0 0→1
  - pipeline_stage_config_008: F1=0.841 bracket=0.000 trunc=0 0→1
  - pipeline_stage_config_022: F1=1.000 bracket=0.000 trunc=0 0→2
  - pipeline_stage_config_025: F1=1.000 bracket=0.000 trunc=0 0→2
  - pipeline_stage_config_037: F1=0.992 bracket=0.000 trunc=0 0→2
**complex_nested_config** (10 rescued):
  - complex_nested_config_007: F1=0.720 bracket=0.000 trunc=0 0→1
  - complex_nested_config_025: F1=0.831 bracket=0.000 trunc=1 1→2
  - complex_nested_config_031: F1=0.800 bracket=1.000 trunc=0 1→2
  - complex_nested_config_034: F1=0.844 bracket=1.000 trunc=1 1→2
  - complex_nested_config_041: F1=0.825 bracket=1.000 trunc=1 1→2
  - ... and 5 more
**rich_cli_option_groups** (9 rescued):
  - rich_cli_option_groups_007: F1=0.903 bracket=0.000 trunc=0 0→2
  - rich_cli_option_groups_008: F1=0.815 bracket=0.000 trunc=1 1→2
  - rich_cli_option_groups_015: F1=0.837 bracket=1.000 trunc=0 1→2
  - rich_cli_option_groups_016: F1=1.000 bracket=0.000 trunc=0 0→2
  - rich_cli_option_groups_028: F1=0.838 bracket=0.000 trunc=1 1→2
  - ... and 4 more

## 4. Does Benchmark-Aware Scoring Change the Main Conclusion?

### Recoverable ranking (Unified):

1. **AR**: 406/480 (84.6%)
2. **FLY**: 388/480 (80.8%)
3. **GSD**: 379/480 (79.0%)
4. **N-gram SD**: 365/480 (76.0%)
5. **TASD-FG**: 348/480 (72.5%)
6. **TASD**: 289/480 (60.2%)

### Recoverable ranking (Benchmark-Aware):

1. **AR**: 452/480 (94.2%)
2. **N-gram SD**: 440/480 (91.7%)
3. **FLY**: 436/480 (90.8%)
4. **GSD**: 411/480 (85.6%)
5. **TASD-FG**: 390/480 (81.2%)
6. **TASD**: 388/480 (80.8%)

**Ranking DID change.**

- Position 2: FLY → N-gram SD
- Position 3: GSD → FLY
- Position 4: N-gram SD → GSD

### Key Findings:

1. **TASD-FG recoverable**: 348/480 (unified) → 390/480 (BA), delta = +42
2. **TASD-FG vs TASD (base)**: unified gap = +59, BA gap = +2 (preserved)
3. **TASD-FG vs speculative methods (BA)**: AR (gap=-62) < N-gram SD (gap=-50) < FLY (gap=-46) < GSD (gap=-21)
4. **TASD-FG improvement**: 52 samples rescued (TASD: 99, AR: 58, N-gram SD: 89)

### Conclusion:

- **TASD-FG still outperforms TASD (base)** under BA scoring (390 vs 388 recoverable, gap=+2), but the gap narrowed from +59 to +2
- **AR remains #1** under both scoring systems (expected: AR is the quality ceiling)
- **Speculative method ranking changed**: N-gram SD and FLY swapped positions, GSD and N-gram SD swapped. This is because bracket-heavy benchmarks (argparse, openmmlab) differentially penalize methods with higher truncation rates
- **Main conclusion is PRESERVED but weakened**:
  - TASD-FG still improves over TASD (base) by +2 recoverable samples
  - But TASD-FG is behind all other speculative methods and AR
  - The gap to AR (-62) is substantial, primarily due to pipeline_stage and complex_nested benchmarks
- **Recommendation**: Use benchmark-aware scoring for all future quality comparisons. The BA scoring more accurately reflects structural quality by not penalizing truncation-induced bracket mismatches.
