# Adaptive TASD v2 vs Fixed Baseline

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Config**: adaptive v2 from d16_b2_k3; draft_len ⇆ {8,16,20}, top_k ⇆ {3,5}
**n**: 20 per benchmark

## Main Comparison (Adaptive v2)

| Benchmark | TPS | Speedup | Accept | Low | SQ | OffStr | Trunc | Chg | k5Rnd | avgDL |
|-----------|-----|---------|--------|-----|----|--------|-------|-----|-------|-------|
| Real-Python-DictConfig | 53.4 | 1.64x | 0.84 | 5/20 | 0.8437 | 0.0000 | 0.0305 | 1.8 | 0.1 | 15.6 |
| OpenMMLab-Config | 61.6 | 1.87x | 0.93 | 3/20 | 0.8985 | 0.0091 | 0.1548 | 1.1 | 0.0 | 16.9 |
| Pipeline-Stage-Config | 62.8 | 1.95x | 0.99 | 0/20 | 0.9609 | 0.0261 | 0.1238 | 1.1 | 0.0 | 17.8 |

## Delta (Adaptive v2 - Fixed)

| Benchmark | TPS Delta | SQ Delta | OffStr Delta | Low Delta | Chg Count | k5 Rounds | Avg DraftLen |
|-----------|-----------|----------|-------------|-----------|-----------|-----------|-------------|
| Real-Python-DictConfig | +2.0 (+3.9%) | -0.0006 | +0.0000 | +0 | 1.8 | 0.1 | 15.6 |
| OpenMMLab-Config | -1.2 (-1.9%) | +0.0011 | -0.0035 | +1 | 1.1 | 0.0 | 16.9 |
| Pipeline-Stage-Config | -2.7 (-4.1%) | +0.0028 | -0.0042 | +0 | 1.1 | 0.0 | 17.8 |

## Success Criteria

| Real-Python-DictConfig | C1(TPS >= +5%): FAIL (+3.9%) |
| | C2(SQ >= -0.02): PASS (-0.0006) |
| | C3(OffStr <= +0.01): PASS (+0.0000) |
| OpenMMLab-Config | C1(TPS >= -1%): FAIL (-1.9%) |
| | C2(SQ >= -0.02): PASS (+0.0011) |
| | C3(OffStr <= +0.01): PASS (-0.0035) |
| Pipeline-Stage-Config | C1(TPS >= -1%): FAIL (-4.1%) |
| | C2(SQ >= -0.02): PASS (+0.0028) |
| | C3(OffStr <= +0.01): PASS (-0.0042) |

**OVERALL: FAIL**

Adaptive TASD v2 does not pass. Retained as future work; main method stays fixed d16_b2_k3.