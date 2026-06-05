# Draft Model Comparison: 1.5B vs 3B

**Config**: d16_b2_k3, n=80, 14B-AWQ target

## TPS Comparison

| Benchmark | 3B TPS | 1.5B TPS | Delta | 3B SQ | 1.5B SQ | SQ Diff | 3B Accept | 1.5B Accept | 1.5B Low | 1.5B High |
|-----------|--------|----------|-------|-------|---------|---------|-----------|-------------|----------|-----------|
| Real-Python-Argparse | 47.40 | 61.89 | +30.6% | 0.9146 | 0.9010 | -0.0136 | 1.00 | 0.93 | 7 | 71 |
| Real-Python-DictConfig | 48.10 | 60.00 | +24.7% | 0.8360 | 0.8316 | -0.0044 | 1.00 | 0.91 | 10 | 61 |
| OpenMMLab-Config | 51.25 | 63.96 | +24.8% | 0.8741 | 0.8876 | +0.0135 | 1.00 | 0.95 | 6 | 73 |
| Rich-CLI-Option-Groups | 52.88 | 66.06 | +24.9% | 0.8918 | 0.9119 | +0.0201 | 1.00 | 1.00 | 0 | 79 |
| Complex-Nested-Config | 51.95 | 66.35 | +27.7% | 0.8026 | 0.8227 | +0.0201 | 1.00 | 1.00 | 0 | 80 |
| Pipeline-Stage-Config | 53.06 | 66.66 | +25.6% | 0.9121 | 0.9405 | +0.0284 | 1.00 | 1.00 | 0 | 79 |

## Acceptance Distribution

| Benchmark | Mean | Median | P10 | P90 | Low(<0.7) | High(>=0.9) |
|-----------|------|--------|-----|-----|-----------|-------------|
| Real-Python-Argparse | 0.9320 | 1.0000 | 0.7791 | 1.0000 | 7/80 | 71/80 |
| Real-Python-DictConfig | 0.9052 | 1.0000 | 0.6793 | 1.0000 | 10/80 | 61/80 |
| OpenMMLab-Config | 0.9516 | 1.0000 | 0.9922 | 1.0000 | 6/80 | 73/80 |
| Rich-CLI-Option-Groups | 0.9965 | 1.0000 | 1.0000 | 1.0000 | 0/80 | 79/80 |
| Complex-Nested-Config | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 0/80 | 80/80 |
| Pipeline-Stage-Config | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 0/80 | 79/80 |

## Quality Comparison

| Benchmark | 3B OffStr | 1.5B OffStr | 3B Trunc | 1.5B Trunc | 3B Rep | 1.5B Rep | 3B SNP | 1.5B SNP |
|-----------|-----------|-------------|----------|------------|--------|----------|--------|----------|
| Real-Python-Argparse | 0.0000 | 0.0014 | 0.0228 | 0.0269 | --- | 0.0444 | --- | 0.2000 |
| Real-Python-DictConfig | 0.0000 | 0.0000 | 0.0939 | 0.0590 | --- | 0.0600 | --- | 0.8500 |
| OpenMMLab-Config | 0.0031 | 0.0059 | 0.1372 | 0.1134 | --- | 0.0082 | --- | 0.4250 |
| Rich-CLI-Option-Groups | 0.1497 | 0.1141 | 0.0513 | 0.0395 | --- | 0.0261 | --- | 0.1375 |
| Complex-Nested-Config | 0.0179 | 0.0143 | 0.0619 | 0.0727 | --- | 0.0063 | --- | 0.8625 |
| Pipeline-Stage-Config | 0.0000 | 0.0245 | 0.0989 | 0.1489 | --- | 0.0386 | --- | 0.0250 |

## Success Criteria Check (C4: off_structure <= 0.05 or <= 3B+0.03)

| Benchmark | C1 (TPS+8%) | C2 (SQ-0.02) | C3 (Low<=25%) | C4 (OffStr) | Overall |
|-----------|------------|-------------|--------------|-------------|---------|
| Real-Python-Argparse | PASS | PASS | PASS | PASS | PASS |
| Real-Python-DictConfig | PASS | PASS | PASS | PASS | PASS |
| OpenMMLab-Config | PASS | PASS | PASS | PASS | PASS |
| Rich-CLI-Option-Groups | PASS | PASS | PASS | PASS | PASS |
| Complex-Nested-Config | PASS | PASS | PASS | PASS | PASS |
| Pipeline-Stage-Config | PASS | PASS | PASS | PASS | PASS |

**OVERALL: PASS**

1.5B draft passes all criteria. Recommended as optimized speed default.
3B draft retained as conservative/stable draft baseline.