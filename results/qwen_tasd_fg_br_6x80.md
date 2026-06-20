# TASD-FG-BR: Bracket-Risk Rerun

**Samples**: 480 total (6 benchmarks x 80)
**Rule**: Rerun AR if `bracket_balance < 0.50` and `is_truncated == 0`

## 1. Main Results

| Method | Speedup | TPS | Below-AR | Score 2 | Score 1 | Score 0 | Recoverable | Rerun |
|--------|:------:|:---:|:--------:|:------:|:------:|:------:|:----------:|:-----:|
| AR | 1.00x | 33.2 | 0 | 251 | 155 | 74 | 406/480 (84.6%) | 0 |
| GSD | 0.66x | 22.0 | 424 | 238 | 141 | 101 | 379/480 (79.0%) | 0 |
| N-gram SD | 1.41x | 46.9 | 231 | 156 | 209 | 115 | 365/480 (76.0%) | 0 |
| FLY | 1.64x | 54.5 | 99 | 286 | 102 | 92 | 388/480 (80.8%) | 0 |
| TASD-FG | 2.00x | 66.4 | 3 | 192 | 156 | 132 | 348/480 (72.5%) | 0 |
| **TASD-FG-BR** | 1.87x | 62.0 | 3 | 227 | 178 | 75 | 405/480 (84.4%) | 65 (13.5%) |
| TASD-FG-V | 1.31x | 43.6 | 2 | 259 | 178 | 43 | 437/480 (91.0%) | 122 (25.4%) |

## 2. Key Findings

- **TASD-FG-BR speedup**: 1.87x (vs FLY 1.64x, TASD-FG 2.00x)
- **TASD-FG-BR recoverable**: 84.4% (vs FLY 80.8%, TASD-FG 72.5%)
- **Rerun ratio**: 65/480 (13.5%) — only bracket-risk samples
- **Below-AR**: 3 (vs FLY 99, TASD-FG 3)
- **Score 0**: 75 (vs FLY 92, TASD-FG 132)

**Conclusion**: TASD-FG-BR beats FLY on both speed and recoverability, with only 13.5% rerun cost.

## 3. Per-Benchmark Results

| Benchmark | N | Rerun | Speedup | Score 2 | Score 1 | Score 0 | Recoverable |
|-----------|:--:|:-----:|:-------:|:------:|:------:|:------:|:----------:|
| argparse | 80 | 15 | 1.78x | 53 | 25 | 2 | 78/80 (97.5%) |
| dict_config | 80 | 7 | 1.89x | 36 | 32 | 12 | 68/80 (85.0%) |
| openmmlab | 80 | 14 | 1.85x | 50 | 19 | 11 | 69/80 (86.2%) |
| pipeline | 80 | 5 | 1.94x | 53 | 16 | 11 | 69/80 (86.2%) |
| complex_nested | 80 | 12 | 1.86x | 15 | 39 | 26 | 54/80 (67.5%) |
| rich_cli | 80 | 12 | 1.88x | 20 | 47 | 13 | 67/80 (83.8%) |

## 4. Per-Benchmark Comparison

| Benchmark | TASD-FG | FLY | TASD-FG-BR | TASD-FG-V |
|-----------|:-------:|:---:|:----------:|:---------:|
| argparse | 80.0% | 95.0% | 97.5% | 97.5% |
| dict_config | 78.8% | 48.8% | 85.0% | 87.5% |
| openmmlab | 71.2% | 82.5% | 86.2% | 95.0% |
| pipeline | 80.0% | 100.0% | 86.2% | 97.5% |
| complex_nested | 56.2% | 60.0% | 67.5% | 73.8% |
| rich_cli | 68.8% | 98.8% | 83.8% | 95.0% |
