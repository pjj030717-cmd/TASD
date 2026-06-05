# Final Total Experiment Table (Optimized TASD: d16_b2_k3)

**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft)
**Settings**: temperature=0.0, max_new_tokens=128, KV cache enabled
**Sample count**: n=80 for all benchmarks and all methods
**TASD config**: draft_len=16, draft_blocks=2, top_k_accept=3 (optimized per speed search)
**SQ scoring**: `src/evaluator.py` penalty-based structural quality score

*d16_b2_k3 improves TASD speedup from 1.30x-1.53x to 1.44x-1.65x across six benchmarks.*

## Main Table (AR vs Greedy SD vs TASD, all n=80)

| Benchmark | AR TPS | GSD TPS | GSD Spd | TASD(d8) TPS | TASD(d8) Spd | TASD(d16) TPS | TASD(d16) Spd | TASD(d16) Accept | TASD(d16) SQ | TASD(d16) OffStr | TASD(d16) Trunc |
|-----------|--------|---------|---------|-------------|-------------|--------------|--------------|------------------|-------------|-----------------|----------------|
| Real-Python-Argparse | 32.98 | 27.08 | 0.82x | 42.92 | 1.30x | 47.40 | 1.44x | 0.91 | 0.9146 | 0.0000 | 0.0228 |
| Real-Python-DictConfig | 32.67 | 28.36 | 0.87x | 42.62 | 1.30x | 48.10 | 1.47x | 0.92 | 0.8360 | 0.0000 | 0.0939 |
| OpenMMLab-Config | 32.91 | 26.29 | 0.80x | 47.34 | 1.44x | 51.25 | 1.56x | 0.98 | 0.8741 | 0.0031 | 0.1372 |
| Rich-CLI-Option-Groups | 33.14 | 27.39 | 0.83x | 49.12 | 1.48x | 52.88 | 1.60x | 1.00 | 0.8918 | 0.1497 | 0.0513 |
| Complex-Nested-Config | 32.71 | 27.76 | 0.85x | 48.23 | 1.47x | 51.95 | 1.59x | 0.99 | 0.8026 | 0.0179 | 0.0619 |
| Pipeline-Stage-Config | 32.24 | 25.75 | 0.80x | 49.36 | 1.53x | 53.06 | 1.65x | 1.00 | 0.9121 | 0.0000 | 0.0989 |

## SQ Comparison (d16 vs d8)

| Benchmark | TASD(d8) SQ | TASD(d16) SQ | Diff |
|-----------|------------|-------------|------|
| Real-Python-Argparse | 0.9223 | 0.9146 | -0.0077 |
| Real-Python-DictConfig | 0.8310 | 0.8360 | +0.0050 |
| OpenMMLab-Config | 0.8887 | 0.8741 | -0.0146 |
| Rich-CLI-Option-Groups | 0.9074 | 0.8918 | -0.0156 |
| Complex-Nested-Config | 0.7985 | 0.8026 | +0.0041 |
| Pipeline-Stage-Config | 0.9120 | 0.9121 | +0.0001 |

## Key Observations

- TASD(d16_b2_k3) achieves 1.44x-1.65x speedup over AR across 6 benchmarks (up from 1.30x-1.53x with d8_b2_k3)
- Structural quality is stable: d16 SQ within +/-0.02 of d8 on all benchmarks
- d16_b2_k3 improves TASD speedup from 1.30x-1.53x to 1.44x-1.65x across six benchmarks. It also preserves structural quality within +/-0.02 of the conservative d8 setting, likely because longer draft blocks reduce frequent draft-target switching and preserve longer structural spans.
- Original d8_b2_k3 results retained as conservative default / ablation baseline
- Extended benchmarks (Rich-CLI, Complex-Nested, Pipeline-Stage) benefit most from longer drafts
- truncation_rate elevated on benchmarks with long reference blocks (all samples hit max_new_tokens=128)

## Optimized TASD Default

| Parameter | Value |
|-----------|-------|
| draft_len | 16 |
| draft_blocks | 2 |
| top_k_accept | 3 |
| min_token_prob | 1e-4 |
| prefix_budget | 0.2 |
| window_len | 2 |
| enable_guard | True |
| enable_relaxed_accept | True |

*Conservative default (d8_b2_k3) kept for ablation comparisons.*