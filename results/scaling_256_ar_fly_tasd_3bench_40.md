# Scaling Analysis: max_new_tokens=256 (3 benchmarks x 40 samples)

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Settings**: temperature=0.0, max_new_tokens=256, n=40 per benchmark
**Methods**: AR, FLY, TASD

## Speed and Quality

| Benchmark | Method | TPS | Speedup vs AR | Accept | SQ | OffStr | Trunc | AvgTok |
|-----------|--------|-----|---------------|--------|----|--------|-------|--------|
| OpenMMLab-Config | AR | 33.4 | 1.00x | 1.00 | 0.9023 | 0.0000 | 0.0815 | 255.8 |
| OpenMMLab-Config | FLY | 41.9 | 1.25x | MAT=8.89 | 0.9071 | 0.0000 | 0.0634 | 261.6 |
| OpenMMLab-Config | TASD | 58.5 | 1.75x | 0.86 | 0.9039 | 0.0122 | 0.0481 | 256.0 |
| Pipeline-Stage-Config | AR | 33.7 | 1.00x | 1.00 | 0.8995 | 0.0007 | 0.0751 | 254.1 |
| Pipeline-Stage-Config | FLY | 47.9 | 1.42x | MAT=9.29 | 0.9101 | 0.0000 | 0.0621 | 263.9 |
| Pipeline-Stage-Config | TASD | 68.4 | 2.03x | 1.00 | 0.7976 | 0.3346 | 0.0887 | 256.0 |
| Complex-Nested-Config | AR | 33.8 | 1.00x | 1.00 | 0.8065 | 0.0281 | 0.0468 | 252.4 |
| Complex-Nested-Config | FLY | 71.5 | 2.12x | MAT=10.47 | 0.8063 | 0.0313 | 0.0515 | 261.9 |
| Complex-Nested-Config | TASD | 68.1 | 2.02x | 1.00 | 0.8232 | 0.0166 | 0.0590 | 256.0 |

## Summary (3-benchmark average)

| Method | TPS | Speedup vs AR | Accept | SQ | OffStr | Trunc | AvgTok |
|--------|-----|---------------|--------|----|--------|-------|--------|
| AR | 33.6 | 1.00x | 1.00 | 0.8695 | 0.0096 | 0.0678 | 254.1 |
| FLY | 53.8 | 1.60x | MAT=9.55 | 0.8745 | 0.0104 | 0.0590 | 262.4 |
| TASD | 65.0 | 1.93x | 0.95 | 0.8416 | 0.1211 | 0.0653 | 256.0 |

## Speed Decomposition

- **AR baseline**: 33.6 TPS
- **FLY** (k=15, n-gram + window): 53.8 TPS (1.60x)
- **TASD** (b=2x16, k=3): 65.0 TPS (1.93x)
  - TASD vs FLY gap: 11.3 TPS (+20.9%)

## Comparison with 128-token Results (6 benchmarks x 80 samples)

| Metric | 128 tok (6x80) | 256 tok (3x40) | Change |
|--------|----------------|----------------|--------|
| AR TPS | 33.2 | 33.6 | +0.4 |
| FLY TPS | 54.5 | 53.8 | -0.7 |
| TASD TPS | 64.2 | 65.0 | +0.8 |
| FLY speedup | 1.64x | 1.60x | -0.04x |
| TASD speedup | 1.93x | 1.93x | +0.00x |
| TASD SQ | 0.8825 | 0.8416 | -0.0409 |
| TASD OffStr | 0.0267 | 0.1211 | +0.0944 |
| TASD Trunc | 0.0767 | 0.0653 | -0.0114 |

## Key Findings

1. **TASD exceeds 2.0x on 2 of 3 benchmarks at 256 tokens**: Pipeline-Stage-Config (2.03x) and Complex-Nested-Config (2.02x), confirming that longer generation amortizes multi-block verification overhead on structurally rich tasks.
2. **TASD average remains 1.93x**, unchanged from 128-token results. OpenMMLab-Config drops to 1.75x, pulling down the average.
3. **TASD remains faster than FLY on average** (65.0 vs 53.8 TPS, +11.3 TPS), though FLY surpasses TASD on Complex-Nested-Config (71.5 vs 68.1 TPS).
4. **Truncation rate**: TASD truncation at 256 tokens is 0.0653 (slightly lower than 128-token 0.0767), suggesting longer budgets reduce premature termination.
5. **Structural quality**: TASD SQ at 256 tokens is 0.8416 (lower than 128-token 0.8825), driven by Pipeline-Stage-Config off-structure rate of 0.3346. This indicates the structural guard needs refinement for deeply nested pipeline patterns at longer generation.
6. **TASD accept rate**: 0.95, showing stable acceptance at longer generation.

## Paper-Ready Statement

At longer 256-token completions, TASD further improves its speedup on structurally rich benchmarks (Pipeline-Stage-Config 2.03x, Complex-Nested-Config 2.02x), suggesting that its multi-block verification overhead is better amortized for longer structured continuations. However, structural quality degrades on pipeline-style benchmarks at this length, indicating that the guard rules may need adaptation for deeper nesting patterns.