# Final Total Experiment Table

**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft)
**Settings**: temperature=0.0, max_new_tokens=128, KV cache enabled
**Sample count**: n=80 for all benchmarks and all methods

## Main Table (AR vs Greedy SD vs TASD, all n=80)

| Benchmark | AR TPS | GSD TPS | GSD Spd | GSD Accept | TASD TPS | TASD Spd | TASD Accept | GSD SQ | TASD SQ | GSD OffStr | TASD OffStr | GSD Trunc | TASD Trunc |
|-----------|--------|---------|---------|------------|----------|----------|-------------|--------|---------|------------|-------------|-----------|------------|
| Real-Python-Argparse | 32.98 | 27.08 | 0.82x | 0.84 | 42.92 | 1.30x | 0.92 | 0.8730 | 0.9223 | 0.0716 | 0.0039 | 0.0635 | 0.0178 |
| Real-Python-DictConfig | 32.67 | 28.36 | 0.87x | 0.87 | 42.62 | 1.30x | 0.90 | 0.8332 | 0.8310 | 0.0266 | 0.0006 | 0.0980 | 0.1184 |
| OpenMMLab-Config | 32.91 | 26.29 | 0.80x | 0.82 | 47.34 | 1.44x | 0.97 | 0.8541 | 0.8887 | 0.0063 | 0.0023 | 0.2107 | 0.1250 |
| Rich-CLI-Option-Groups | 33.14 | 27.39 | 0.83x | 0.85 | 49.12 | 1.48x | 1.00 | 0.9159 | 0.9074 | 0.0712 | 0.1218 | 0.0604 | 0.0556 |
| Complex-Nested-Config | 32.71 | 27.76 | 0.85x | 0.85 | 48.23 | 1.47x | 1.00 | 0.7969 | 0.7985 | 0.0194 | 0.0198 | 0.1156 | 0.0590 |
| Pipeline-Stage-Config | 32.24 | 25.75 | 0.80x | 0.82 | 49.36 | 1.53x | 1.00 | 0.9250 | 0.9120 | 0.0000 | 0.0000 | 0.1933 | 0.1272 |

**Key observations:**
- TASD achieves 1.30x-1.53x speedup over AR across all 6 benchmarks
- Greedy SD stays below AR (0.80x-0.87x): strict argmax matching limits draft model contribution
- TASD structural quality scores are within 0.02 of GSD; no systematic degradation
- Extended benchmarks (Rich-CLI, Complex-Nested, Pipeline-Stage) show higher TASD speedups (1.47x-1.53x)
- truncation_rate elevated on OpenMMLab and Pipeline-Stage; these contain longer reference blocks that exceed max_new_tokens=128