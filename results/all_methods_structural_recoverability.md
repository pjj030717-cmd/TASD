# All Methods Structural Recoverability Comparison

Samples: 6 benchmarks × 80 = 480 per method

## 1. Overall Comparison

| Method | Score 2 | Score 1 | Score 0 | Recoverable Rate | Avg SQ-R | Avg SQ-S | Avg TPS |
|--------|:-------:|:-------:|:-------:|:----------------:|:--------:|:--------:|:-------:|
| AR | 251 (52.3%) | 155 (32.3%) | 74 (15.4%) | 84.6% | 0.8623 | 0.8386 | 33.20 |
| GSD | 238 (49.6%) | 141 (29.4%) | 101 (21.0%) | 79.0% | 0.8117 | 0.8086 | 22.02 |
| N-gram SD | 156 (32.5%) | 209 (43.5%) | 115 (24.0%) | 76.0% | 0.7544 | 0.8053 | 46.88 |
| FLY | 286 (59.6%) | 102 (21.2%) | 92 (19.2%) | 80.8% | 0.8491 | 0.8369 | 54.48 |
| TASD | 256 (53.3%) | 33 (6.9%) | 191 (39.8%) | 60.2% | 0.8294 | 1.0000 | 50.77 |
| TASD-FG | 192 (40.0%) | 156 (32.5%) | 132 (27.5%) | 72.5% | 0.6729 | 0.7802 | 66.40 |

## 2. Per-Benchmark Recoverable Rate

| Benchmark | AR | GSD | N-gram SD | FLY | TASD | TASD-FG |
|-----------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| argparse | 91.2% | 93.8% | 77.5% | 95.0% | 82.5% | 80.0% |
| dict_config | 63.7% | 45.0% | 61.3% | 48.8% | 23.8% | 78.8% |
| openmmlab_config | 92.5% | 96.2% | 80.0% | 82.5% | 70.0% | 71.2% |
| pipeline_stage_config | 100.0% | 77.5% | 93.8% | 100.0% | 100.0% | 80.0% |
| complex_nested_config | 63.7% | 70.0% | 68.8% | 60.0% | 8.8% | 56.2% |
| rich_cli_option_groups | 96.2% | 91.2% | 75.0% | 98.8% | 76.2% | 68.8% |

## 3. Error Tag Distribution

| Method | BRACKET | TRUNC | LOW_F1 | OFF_STRUCT | REPEAT | DUP_OPT |
|--------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| AR | 73 | 296 | 0 | 29 | 0 | 0 |
| GSD | 73 | 336 | 2 | 11 | 45 | 0 |
| N-gram SD | 102 | 347 | 5 | 24 | 10 | 0 |
| FLY | 87 | 304 | 0 | 15 | 5 | 0 |
| TASD | 191 | 0 | 0 | 0 | 0 | 0 |
| TASD-FG | 65 | 375 | 103 | 34 | 35 | 0 |

## 4. Key Interpretation

1. **TASD-FG vs TASD-base**: TASD-FG recoverable rate = 72.5%, TASD-base = 60.2%. TASD-FG improves quality via fallback guard.
2. **FLY trade-off**: Recoverable rate = 80.8%, Avg TPS = 54.48. FLY has speed instability (below-AR cases).
3. **AR as reference**: Recoverable rate = 84.6%, Avg TPS = 33.20. AR is not a perfect quality baseline.
4. **Below-AR vs unrecoverable**: Below-AR cases focus on speed robustness, not quality. Unrecoverable (score=0) samples are structural failures.