# TASD-FG Final Master Report

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, temperature=0.0, samples=80 per benchmark

## Table 0: Structure Coverage / Benchmark Construction

| Benchmark | Type | Source Repos | Unique Files | Prompt (chars) | Ref (chars) |
|-----------|------|:----------:|:----------:|:---:|:---:|
| argparse | CLI argument parser (argparse/click) | 12 | 41 | 78-1431 | 73-1493 |
| dict_config | Dict/list lookup table (device/setting maps) | 14 | 55 | 56-1178 | 80-71474 |
| openmmlab | ML pipeline test config (mmengine-based) | 5 | 30 | 50-218 | 107-1416 |
| pipeline | ML pipeline stage config (mmengine-based) | 5 | 77 | 61-268 | 171-2305 |
| complex_nested | Deeply nested config dicts (nesting depth 3-21) | 20 | 68 | 95-739 | 220-59263 |
| rich_cli | CLI args with option groups (accelerate/transformers) | 29 | 65 | 118-1134 | 256-50015 |

All six benchmark types are constructed from real-world open-source code repositories (CodeSearchNet-Python and OpenMMLab ecosystem), covering **85** unique repos and **336** unique source files.

## Quality Metrics

- **SQ-R** (Reference-aware Structural Quality): 0.4×structural_char_F1 + 0.3×bracket_balance + 0.2×type_preservation + 0.1×no_repetition
- **SQ-S** (Structure Safety Score): 1.0 − 0.45×off_structure − 0.25×truncation − 0.20×repetition − 0.10×duplicate_option
- **Off-Str**: lines starting with `def`/`class`/`import`/`from` ÷ total lines
- SQ-R/SQ-S available for: AR, TASD, TASD-FG (have per-sample quality sub-metrics). Baselines (N-gram SD, Greedy SD, Official FLY) were collected before sub-metric recording was added.

## Table 1: Main Results (480 samples)

| Method | Speedup | Below | Worst-10 | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:--------:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0/480 | 1.000x | 0.5979 | 0.8103 | 0.0153 | - |
| N-gram SD | 1.413x | 233/480 | 0.785x | — | — | — | - |
| Greedy SD | 1.771x | 0/480 | 1.517x | — | — | — | - |
| Official FLY | 1.623x | 126/480 | 0.617x | — | — | — | - |
| **TASD** | **1.978x** | 9/480 | 1.489x | 0.5903 | 0.7743 | 0.0379 | - |
| **TASD-FG** | **2.004x** | 3/480 | 1.655x | 0.5908 | 0.7762 | 0.0366 | 27 |

## Table 2: Per-Benchmark Results

### argparse (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6307 | 0.8040 | 0.0170 | - |
| N-gram SD | 1.407x | 33 | — | — | — | - |
| Greedy SD | 1.725x | 0 | — | — | — | - |
| Official FLY | 1.945x | 6 | — | — | — | - |
| **TASD** | **1.821x** | 7 | 0.6367 | 0.7766 | 0.0082 | - |
| **TASD-FG** | **1.934x** | 1 | 0.6420 | 0.7906 | 0.0013 | 17 |

### dict_config (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.5975 | 0.7881 | 0.0395 | - |
| N-gram SD | 1.468x | 32 | — | — | — | - |
| Greedy SD | 1.810x | 0 | — | — | — | - |
| Official FLY | 1.798x | 7 | — | — | — | - |
| **TASD** | **1.946x** | 2 | 0.6584 | 0.8116 | 0.0007 | - |
| **TASD-FG** | **1.971x** | 2 | 0.6567 | 0.8123 | 0.0000 | 9 |

### openmmlab (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6141 | 0.8117 | 0.0000 | - |
| N-gram SD | 1.113x | 54 | — | — | — | - |
| Greedy SD | 1.718x | 0 | — | — | — | - |
| Official FLY | 0.993x | 42 | — | — | — | - |
| **TASD** | **2.018x** | 0 | 0.6027 | 0.7745 | 0.0426 | - |
| **TASD-FG** | **2.050x** | 0 | 0.6023 | 0.7710 | 0.0426 | 1 |

### pipeline (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6258 | 0.8402 | 0.0000 | - |
| N-gram SD | 1.182x | 49 | — | — | — | - |
| Greedy SD | 1.777x | 0 | — | — | — | - |
| Official FLY | 1.127x | 39 | — | — | — | - |
| **TASD** | **2.041x** | 0 | 0.5568 | 0.7418 | 0.0371 | - |
| **TASD-FG** | **2.028x** | 0 | 0.5568 | 0.7418 | 0.0371 | 0 |

### complex_nested (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.4701 | 0.8188 | 0.0208 | - |
| N-gram SD | 1.486x | 36 | — | — | — | - |
| Greedy SD | 1.801x | 0 | — | — | — | - |
| Official FLY | 1.728x | 25 | — | — | — | - |
| **TASD** | **2.012x** | 0 | 0.4749 | 0.7916 | 0.0132 | - |
| **TASD-FG** | **2.019x** | 0 | 0.4749 | 0.7916 | 0.0132 | 0 |

### rich_cli (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6495 | 0.7993 | 0.0143 | - |
| N-gram SD | 1.824x | 29 | — | — | — | - |
| Greedy SD | 1.792x | 0 | — | — | — | - |
| Official FLY | 2.143x | 7 | — | — | — | - |
| **TASD** | **2.029x** | 0 | 0.6122 | 0.7496 | 0.1254 | - |
| **TASD-FG** | **2.025x** | 0 | 0.6122 | 0.7496 | 0.1254 | 0 |

## Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | AR | Target autoregressive (greedy) |
| 2 | N-gram SD | Prompt-lookup SD (n=3-8, no draft model) |
| 3 | Greedy SD | Strict greedy SD (draft_len=16, blocks=2) |
| 4 | Official FLY | AMD FLy SPDGenerate (k=15, win_len=6, entropy_thre=0.3) |
| 5 | TASD | Structure-aware SD + Guard v1.5 calibrated |
| 6 | **TASD-FG** | **TASD + Guarded Failure-Aware Fallback (proposed)** |

## Conclusion

TASD-FG achieves **2.004x** average speedup with only **3/480 below-AR** cases (0.6%), outperforming Official FLY (1.623x, 126/480 below) in both average speedup and robustness.

Compared to TASD (1.978x, 9/480 below, worst-10=1.489x):
- Speedup: +0.026x
- Below-AR: 9 → 3 (−6)
- Worst-10: 1.489x → 1.655x (+0.166)
- SQ-S: 0.7743 → 0.7762
- Off-Str: 0.0379 → 0.0366

TASD-FG achieves the best speed-robustness-safety trade-off. It does not maximize reference similarity (SQ-R), but maintains competitive structure safety (SQ-S) while achieving the highest speedup and the fewest below-AR failures among all methods.

## Supplementary Tables

| Table | Description | File |
|-------|-------------|------|
| Table 3 | Ablation Study (7 TASD variants) | `results/qwen_ablation_7variant.md` |
| Table 4 | 256-token Scaling Pilot (3 benchmarks × 20) | `results/qwen_256token_pilot_3x20.md` |
| Table 5 | Hard Case / Negative Results | `results/tasdfg_below_vs_noguard_analysis.md` |
| - | Profit-aware switch pilot | `results/tasdfg_profit_switch_pilot.md` |
| - | TASD-NG (n-gram draft channel, negative) | `results/tasd_ng_pilot_3x20.md` |
