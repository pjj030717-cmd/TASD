# OpenMMLab-Config 256-token: Comment/String Fallback Validation

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Outliers**: 6 persistent samples (indices: [3, 29, 21, 38, 17, 37])

## 1. Summary

| Metric | TASD | TASD+Fallback | FLY | AR |
|--------|------|---------------|-----|----|
| Mean Speedup | 1.5251x | 1.5316x | 1.4504x | 1.00x |
| Median Speedup | 1.5773x | 1.5658x | 1.2755x | 1.00x |
| Mean Accept Rate | 0.7909 | 0.7923 | - | - |
| Mean Repair Count | 2.17 | 2.67 | - | - |
| Mean SQ | 0.8946 | 0.9118 | - | - |
| Mean Off-Structure | 0.0000 | 0.0000 | - | - |
| Mean Truncation | 0.0789 | 0.0730 | - | - |
| Fallback Triggers | - | 23 | - | - |

## 2. Success Criteria

- Speedup improved: PASS (1.5251x -> 1.5316x)
- SQ not degraded (>0.03): PASS (delta=-0.0172)
- Off-structure not increased: PASS (0.0000 -> 0.0000)
- **All passed**: YES

## 3. Per-Sample Results

| Idx | AR TPS | TASD TPS | TASD Speedup | Fallback TPS | Fallback Speedup | FLY TPS | FLY Speedup | Fallback Triggers | SQ Delta |
|-----|--------|----------|--------------|--------------|------------------|---------|-------------|-------------------|----------|
| 3 | 30.55 | 59.90 | 1.9607x | 62.66 | 2.0511x | 37.66 | 1.2327x | 3 | +0.0000 |
| 29 | 33.67 | 50.55 | 1.5013x | 51.60 | 1.5325x | 39.26 | 1.1660x | 0 | +0.0000 |
| 21 | 33.40 | 34.70 | 1.0389x | 34.64 | 1.0371x | 44.03 | 1.3183x | 5 | +0.0000 |
| 38 | 32.88 | 65.06 | 1.9787x | 64.71 | 1.9681x | 59.16 | 1.7993x | 0 | +0.0000 |
| 17 | 31.96 | 52.84 | 1.6533x | 51.11 | 1.5992x | 68.04 | 2.1289x | 3 | +0.1033 |
| 37 | 32.00 | 32.56 | 1.0175x | 32.06 | 1.0019x | 33.83 | 1.0572x | 12 | +0.0000 |

## 4. Fallback Details

### Sample 3
- Trigger count: 3
- Fallback tokens: 12
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string']
- Avg accept before: 1.0000
- Avg accept after: 1.0000

### Sample 21
- Trigger count: 5
- Fallback tokens: 19
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']
- Avg accept before: 1.0000
- Avg accept after: 1.0000

### Sample 17
- Trigger count: 3
- Fallback tokens: 8
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text']
- Avg accept before: 1.0000
- Avg accept after: 1.0000

### Sample 37
- Trigger count: 12
- Fallback tokens: 35
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']
- Avg accept before: 1.0000
- Avg accept after: 1.0000

## 5. Conclusion

The comment/string fallback **passes all success criteria** on outlier samples.
- Mean speedup improved from 1.5251x to 1.5316x
- SQ delta: -0.0172 (within 0.03 threshold)
- Off-structure rate: 0.0000 -> 0.0000

**Recommendation**: Run on full 40-sample OpenMMLab set to verify no regression on normal samples.