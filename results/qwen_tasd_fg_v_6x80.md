# TASD-FG-V Full 480 Experiment (6 benchmarks x 80)

**480 samples across 6 benchmarks.**

## 1. Main Results

| Method | TPS | Speedup | Below-AR | Score 2 | Score 1 | Score 0 | Recoverable | Rerun Ratio |
|--------|:---:|:------:|:--------:|:------:|:------:|:------:|:----------:|:-----------:|
| AR | 33.2 | 1.00x | 0 | 251 | 155 | 74 | 406/480 (84.6%) | — |
| FLY | 54.5 | 1.64x | 99 | 286 | 102 | 92 | 388/480 (80.8%) | — |
| TASD-FG | 66.4 | 2.00x | 3 | 192 | 156 | 132 | 348/480 (72.5%) | — |
| TASD-FG-V | 43.6 | 1.31x | 2 | 259 | 178 | 43 | 437/480 (91.0%) | 25.4% |

> **Note**: TASD-FG-V speedup is wall-time (includes AR rerun overhead). TPS is effective TPS = AR_TPS × wall-time speedup. Below-AR = per-sample TPS < AR TPS (paired comparison). All other methods use output-generation TPS (verifier overhead negligible).

## 2. TASD-FG-V vs TASD-FG

| Metric | TASD-FG | TASD-FG-V | Delta |
|--------|:-------:|:---------:|:-----:|
| Recoverable | 348/480 (72.5%) | 437/480 (91.0%) | +89 |
| Score 0 | 132 | 43 | -89 |
| Speedup | 2.00x | 1.31x | -0.69x |
| Rerun Count | 0 | 122 | — |
| Rerun Ratio | 0% | 25.4% | — |

## 3. TASD-FG-V vs AR

| Metric | AR | TASD-FG-V | Delta |
|--------|:--:|:---------:|:-----:|
| Recoverable | 406/480 (84.6%) | 437/480 (91.0%) | +31 |
| Score 0 | 74 | 43 | -31 |
| Speedup | 1.00x | 1.31x | +0.31x |

**TASD-FG-V recoverable >= AR recoverable** — quality parity achieved
**TASD-FG-V speedup > 1.0x** — still faster than AR

## 4. Per-Benchmark Breakdown

### 4.1 Recoverable Rate

| Benchmark | AR | FLY | TASD-FG | TASD-FG-V |
|-----------|:--:|:---:|:-------:|:---------:|
| argparse | 91.2% | 95.0% | 80.0% | 97.5% |
| dict_config | 63.7% | 48.8% | 78.8% | 87.5% |
| openmmlab_config | 92.5% | 82.5% | 71.2% | 95.0% |
| pipeline_stage_config | 100.0% | 100.0% | 80.0% | 97.5% |
| complex_nested_config | 63.7% | 60.0% | 56.2% | 73.8% |
| rich_cli_option_groups | 96.2% | 98.8% | 68.8% | 95.0% |

### 4.2 TASD-FG-V Detail

| Benchmark | Rerun | Recoverable | Score 0 | Speedup |
|-----------|:-----:|:----------:|:------:|:-------:|
| argparse | 23.8% | 78/80 (97.5%) | 2 | 1.30x |
| dict_config | 13.8% | 70/80 (87.5%) | 10 | 1.51x |
| openmmlab_config | 28.7% | 76/80 (95.0%) | 4 | 1.29x |
| pipeline_stage_config | 26.2% | 78/80 (97.5%) | 2 | 1.31x |
| complex_nested_config | 31.2% | 59/80 (73.8%) | 21 | 1.24x |
| rich_cli_option_groups | 28.7% | 76/80 (95.0%) | 4 | 1.28x |

## 5. Risk Reason Statistics

| Reason | Count | % of Reruns |
|--------|:-----:|:-----------:|
| off_structure | 34 | 27.9% |
| repetition | 35 | 28.7% |
| duplicate_option | 0 | 0.0% |
| bracket_balance | 65 | 53.3% |

### 5.1 Per-Benchmark Risk Reasons

| Benchmark | off_structure | repetition | duplicate_option | bracket_balance | Total Rerun |
|-----------|:------------:|:----------:|:----------------:|:---------------:|:-----------:|
| argparse | 1 | 3 | 0 | 15 | 19 |
| dict_config | 0 | 5 | 0 | 7 | 11 |
| openmmlab_config | 7 | 2 | 0 | 14 | 23 |
| pipeline_stage_config | 10 | 6 | 0 | 5 | 21 |
| complex_nested_config | 5 | 11 | 0 | 12 | 25 |
| rich_cli_option_groups | 11 | 8 | 0 | 12 | 23 |

## 6. Acceptance Criteria

| Criterion | Target | Actual | Status |
|-----------|:------:|:------:|:------:|
| recoverable_rate >= 88% | 88% | 91.0% | **PASS** |
| speedup >= 1.25x | 1.25x | 1.31x | **PASS** |
| score_0 <= AR score_0 | <= 74 | 43 | **PASS** |
| rerun_ratio <= 35% | 35% | 25.4% | **PASS** |

**ALL CRITERIA PASSED.** TASD-FG-V meets quality and speed requirements.

## 7. Conclusions

- **TASD-FG-V recoverable**: 437/480 (91.0%), vs TASD-FG 348/480 (72.5%) (+89)
- **TASD-FG-V speedup**: 1.31x (vs TASD-FG 2.00x)
- **Effective TPS**: 43.6 (AR_TPS × wall-time speedup, includes AR rerun overhead)
- **Rerun ratio**: 25.4% (122 samples)
- **Primary risk signal**: bracket_balance (65 triggers)

**Recommendation**: Include TASD-FG-V in the final paper main table as a quality-aware verify-and-rerun mode.
