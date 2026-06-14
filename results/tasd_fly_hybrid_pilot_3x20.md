# TASD-FLY Hybrid Pilot Report (3×20)

**Date**: 2026-06-13 15:39
**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Samples**: 3 benchmarks × 20 = 60 total

## Methods

| Method | Description |
|--------|-------------|
| AR | Target autoregressive (greedy) |
| Official FLY | AMD FLy SPDGenerate (k=15, win=6, entropy=0.3, ngram=4/6) |
| TASD cal | TASD + Guard-v1.5 calibrated (draft_len=16, blocks=2, top_k=3) |
| **Hybrid** | FLY generates → TASD guard checks → trim + AR fallback if needed |

## Hybrid Mechanism

1. FLY generates complete output using n-gram PLD + model draft
2. TASD calibrated structural guard scans the output
3. If guard triggers: trim to safe token boundary
4. If trimmed: target AR fallback for remaining tokens (max 128 total)

## Per-Benchmark Results

### argparse (AR TPS ~68, 20 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Notes |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|-------|
| **AR** | **1.000x** | 0.8860 | 0.0000 | 0.0000 | 0.7500 | 0 | 1 |  |
| **FLY** | **1.952x** | 0.9126 | 0.0000 | 0.0215 | 0.9500 | 1 | 1 |  |
| **TASD** | **1.013x** | 0.8728 | 0.0000 | 0.0088 | 0.6500 | 9 | 9 |  |
| **Hybrid** | **2.029x** | 0.9094 | 0.0000 | 0.0217 | 1.0000 | 1 | 1 | guard_trig=0/20, fallback=0/20, avg_trim=0.0 |

### openmmlab_config (AR TPS ~43, 20 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Notes |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|-------|
| **AR** | **1.000x** | 0.9876 | 0.0000 | 0.0000 | 0.7000 | 0 | 0 |  |
| **FLY** | **1.031x** | 1.0000 | 0.0000 | 0.0000 | 0.8500 | 9 | 9 |  |
| **TASD** | **1.610x** | 0.9721 | 0.0622 | 0.0000 | 0.8500 | 0 | 0 |  |
| **Hybrid** | **1.027x** | 1.0000 | 0.0000 | 0.0000 | 0.8500 | 11 | 11 | guard_trig=0/20, fallback=0/20, avg_trim=0.0 |

### pipeline_stage_config (AR TPS ~46, 20 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard | Notes |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|-------|
| **AR** | **1.000x** | 0.9144 | 0.0000 | 0.0000 | 0.6000 | 0 | 1 |  |
| **FLY** | **1.155x** | 0.9367 | 0.0000 | 0.0682 | 0.8000 | 11 | 12 |  |
| **TASD** | **1.586x** | 0.9132 | 0.0292 | 0.0136 | 0.9000 | 0 | 1 |  |
| **Hybrid** | **1.156x** | 0.9346 | 0.0000 | 0.0682 | 0.8500 | 11 | 12 | guard_trig=0/20, fallback=0/20, avg_trim=0.0 |

## Overall Comparison (60 samples)

| Method | Speedup | SQ | OffStr | Rep | Trunc | Below | Hard |
|--------|:-------:|:--:|:------:|:---:|:-----:|:-----:|:----:|
| **AR** | **1.000x** | 0.9293 | 0.0000 | 0.0000 | 0.6833 | 0/60 | 2/60 |
| **FLY** | **1.379x** | 0.9498 | 0.0000 | 0.0299 | 0.8667 | 21/60 | 22/60 |
| **TASD** | **1.403x** | 0.9194 | 0.0305 | 0.0075 | 0.8000 | 9/60 | 10/60 |
| **Hybrid** | **1.404x** | 0.9480 | 0.0000 | 0.0300 | 0.9000 | 23/60 | 24/60 |

**Hybrid guard triggers**: 0/60  |  **Hybrid fallbacks**: 0/60

## Judgment

### Against criteria:

1. **Hybrid speedup ≥ 95% of FLY**: Hybrid 1.404x vs FLY 1.379x → PASS (penalty=-1.8%)
2. **Hybrid below-1.0x < FLY below-1.0x**: Hybrid 23/60 vs FLY 21/60 → FAIL
3. **Hybrid off_str ≤ FLY off_str**: Hybrid 0.0000 vs FLY 0.0000 → PASS
4. **On openmmlab/pipeline, Hybrid near or exceeds TASD**: See per-benchmark details

### Overall Assessment

- Hybrid preserves 102% of FLY's speed — **viable**
- Hybrid has 23/21 below-1.0x vs FLY — no reliability gain
- Guard triggers: 0/60 samples (0%) — no structural risks found in FLY output

**Conclusion: NEGATIVE RESULT — Post-hoc guarding is ineffective.**

Guard triggered **0/60 times**. The Hybrid output is effectively identical to FLY.

**Root cause**: FLY's temperature=0.0, window recovery, and entropy rejection already ensure structurally clean output. The TASD guard, designed for in-loop draft pruning, has no effect when applied after FLY generation completes.

**Lesson**: Adding a structural guard after official FLY does not improve results because the guard is never triggered. The useful integration point is inside TASD's draft stage rather than after FLY generation. See `results/tasd_ng_pilot_3x20.md` for the proper approach (N-gram PLD integrated into TASD's draft loop).

## Raw Data

- `results/tasd_fly_hybrid_pilot_3x20.json`
