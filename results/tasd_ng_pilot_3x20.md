# TASD-NG Pilot Report (3×20)

**Date**: 2026-06-13 16:48
**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Samples**: 3 benchmarks × 20 = 60 total

## Methods

| Method | Description |
|--------|-------------|
| AR | Target autoregressive (greedy) |
| FLY | AMD FLy (k=15, win=6, ngram=4/6) |
| TASD | TASD + Guard-v1.5 (draft_len=16, top_k=3) |
| **TASD-NG** | TASD + n-gram PLD draft channel (ngram_min=2, max=8) |

## TASD-NG Design

Each decoding round:
1. N-gram lookup from full context (min=2, max=8)
2. If match: use matched suffix tokens as draft
3. If no match: fall back to draft model generation
4. Target top-k verification (unchanged TASD)
5. Calibrated structural guard (unchanged TASD)

## Per-Benchmark

### argparse (20 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Details |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|--------|
| **AR** | **1.000x** | 0.8902 | 0.0000 | 0.0000 | 0.7500 | 0 | 1 |  |
| **FLY** | **2.046x** | 0.9126 | 0.0000 | 0.0215 | 0.9500 | 1 | 1 |  |
| **TASD** | **1.045x** | 0.8728 | 0.0000 | 0.0088 | 0.6500 | 9 | 9 |  |
| **TASD_NG** | **0.139x** | 0.0828 | 0.0000 | 0.0000 | 0.9000 | 20 | 20 | ngram_rds=1238, model_rds=2322 |

### openmmlab_config (20 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Details |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|--------|
| **AR** | **1.000x** | 0.9889 | 0.0000 | 0.0000 | 0.6500 | 0 | 0 |  |
| **FLY** | **1.002x** | 1.0000 | 0.0000 | 0.0000 | 0.8500 | 11 | 11 |  |
| **TASD** | **1.678x** | 0.9721 | 0.0622 | 0.0000 | 0.8500 | 0 | 0 |  |
| **TASD_NG** | **0.221x** | 0.1543 | 0.0000 | 0.0000 | 0.9500 | 18 | 18 | ngram_rds=1062, model_rds=2158 |

### pipeline_stage_config (20 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Details |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|--------|
| **AR** | **1.000x** | 0.9141 | 0.0000 | 0.0000 | 0.6500 | 0 | 1 |  |
| **FLY** | **1.161x** | 0.9367 | 0.0000 | 0.0682 | 0.7500 | 11 | 12 |  |
| **TASD** | **1.621x** | 0.9132 | 0.0292 | 0.0136 | 0.9000 | 0 | 1 |  |
| **TASD_NG** | **0.117x** | 0.2023 | 0.0000 | 0.0000 | 0.9000 | 20 | 20 | ngram_rds=1220, model_rds=2340 |

## Overall (60 samples)

| Method | Speedup | SQ | OffStr | Below | Hard |
|--------|:-------:|:--:|:------:|:-----:|:----:|
| **AR** | **1.000x** | 0.9311 | 0.0000 | 0/60 | 2/60 |
| **FLY** | **1.403x** | 0.9498 | 0.0000 | 23/60 | 24/60 |
| **TASD** | **1.448x** | 0.9194 | 0.0305 | 9/60 | 10/60 |
| **TASD_NG** | **0.159x** | 0.1465 | 0.0000 | 58/60 | 58/60 |

## Judgment

- TASD-NG: **0.159x** vs FLY **1.403x** vs TASD **1.448x**
- Below-1.0x: TASD-NG 58/60 vs FLY 23/60 vs TASD 9/60

**TASD-NG does not improve over TASD.** N-gram integration needs refinement.

**Recommendation**: N-gram PLD in TASD draft stage needs structural tuning or different parameters.

## Raw Data

- `results/tasd_ng_pilot_3x20.json`
