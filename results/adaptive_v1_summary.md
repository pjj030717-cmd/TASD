# Adaptive TASD v1 vs Fixed Baseline

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Config**: adaptive starts from d16_b2_k3, then adjusts per round
**n**: 20 per benchmark

## Main Comparison

| Benchmark | Mode | TPS | Speedup | Accept Mean | Low | SQ | OffStr | Trunc | Repair | Guard |
|-----------|------|-----|---------|-------------|-----|----|--------|-------|--------|-------|
| Real-Python-DictConfig | fixed | 51.4 | 1.57x | 0.81 | 5/20 | 0.8443 | 0.0000 | 0.0445 | 0.6 | 3.6 |
| Real-Python-DictConfig | adaptive | 54.6 | 1.67x | 0.85 | 5/20 | 0.8373 | 0.0000 | 0.0435 | 0.4 | 3.2 |
| OpenMMLab-Config | fixed | 62.8 | 1.91x | 0.93 | 2/20 | 0.8974 | 0.0126 | 0.1554 | 0.3 | 0.8 |
| OpenMMLab-Config | adaptive | 62.3 | 1.89x | 0.93 | 3/20 | 0.8985 | 0.0091 | 0.1548 | 0.6 | 1.1 |
| Pipeline-Stage-Config | fixed | 65.5 | 2.03x | 0.99 | 0/20 | 0.9581 | 0.0303 | 0.1397 | 0.1 | 0.0 |
| Pipeline-Stage-Config | adaptive | 65.2 | 2.02x | 0.99 | 0/20 | 0.9581 | 0.0359 | 0.1231 | 0.1 | 0.0 |

## Delta (Adaptive - Fixed)

| Benchmark | TPS Delta | SQ Delta | OffStr Delta | Low Delta | Chg Count | k5 Rounds | Avg DraftLen | Short Rounds | Long Rounds |
|-----------|-----------|----------|-------------|-----------|-----------|-----------|-------------|-------------|------------|
| Real-Python-DictConfig | +3.2 | -0.0070 | +0.0000 | +0 | 0.7 | 0.0 | 18.9 | 3.0 | 1.4 |
| OpenMMLab-Config | -0.5 | +0.0011 | -0.0035 | +1 | 0.2 | 0.0 | 21.8 | 1.2 | 1.8 |
| Pipeline-Stage-Config | -0.4 | +0.0000 | +0.0056 | +0 | 0.1 | 0.0 | 23.4 | 0.2 | 1.9 |

## Success Criteria

| Criterion | Details |
|-----------|---------|
| Real-Python-DictConfig | C1(TPS+5% or Low↓): PASS (+6.3%, Low 5 vs 5) |
| | C2(SQ-0.02): PASS (-0.0070) |
| | C4(Changes>0): PASS (changes=0.7) |
| OpenMMLab-Config | C1(TPS+5% or Low↓): FAIL (-0.7%, Low 3 vs 2) |
| | C2(SQ-0.02): PASS (+0.0011) |
| | C4(Changes>0): PASS (changes=0.2) |
| Pipeline-Stage-Config | C1(TPS+5% or Low↓): FAIL (-0.5%, Low 0 vs 0) |
| | C2(SQ-0.02): PASS (+0.0000) |
| | C4(Changes>0): PASS (changes=0.1) |

**OVERALL: FAIL**

Adaptive TASD v1 does not pass all criteria. Retained as future work.