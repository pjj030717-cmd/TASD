# OpenMMLab-Config 256-token: Failure-Aware Fallback Validation (Outliers)

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Outliers**: 6 persistent samples (indices: [3, 29, 21, 38, 17, 37])

## 1. Summary

| Metric | TASD | TASD+Failure-FB | FLY | AR |
|--------|------|-----------------|-----|----|
| Mean Speedup | 1.5334x | 1.5826x | 1.4449x | 1.00x |
| Median Speedup | 1.5880x | 1.6586x | - | 1.00x |
| Mean Accept Rate | 0.7909 | 0.8118 | - | - |
| Mean Repair Count | 2.17 | 1.33 | - | - |
| Mean SQ | 0.8946 | 0.8974 | - | - |
| Mean Off-Structure | 0.0000 | 0.0464 | - | - |
| Mean Truncation | 0.0789 | 0.0474 | - | - |
| Fallback Triggers | - | 7 | - | - |

## 2. Success Criteria

- Speedup improved >= +0.1x: FAIL (1.5334x -> 1.5826x, delta=+0.0492x)
- SQ not degraded >0.03: PASS (delta=+0.0028)
- Off-structure not increased: FAIL (0.0000 -> 0.0464)
- **All passed**: NO

## 3. Per-Sample Results

| Idx | AR TPS | TASD TPS | TASD Speedup | FB TPS | FB Speedup | FLY TPS | FLY Speedup | FB Triggers | SQ Delta |
|-----|--------|----------|--------------|--------|------------|---------|-------------|-------------|----------|
| 3 | 28.17 | 57.93 | 2.0564x | 61.34 | 2.1775x | 35.36 | 1.2552x | 0 | +0.0000 |
| 29 | 31.60 | 48.02 | 1.5196x | 52.58 | 1.6639x | 46.96 | 1.4861x | 1 | +0.0032 |
| 21 | 31.52 | 32.59 | 1.0339x | 31.54 | 1.0006x | 41.68 | 1.3223x | 3 | -0.0642 |
| 38 | 31.01 | 59.84 | 1.9297x | 59.37 | 1.9145x | 55.57 | 1.7920x | 0 | +0.0000 |
| 17 | 30.99 | 51.33 | 1.6563x | 51.24 | 1.6534x | 54.16 | 1.7477x | 0 | +0.0000 |
| 37 | 31.28 | 31.41 | 1.0042x | 33.96 | 1.0857x | 33.34 | 1.0659x | 3 | +0.0779 |

## 4. Fallback Details

### Sample 29
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']
- Avg accept before: 0.2689
- Avg accept after: 1.0000

### Sample 21
- Trigger count: 3
- Fallback tokens: 12
- Reasons: ['low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds']
- Avg accept before: 0.2431
- Avg accept after: 1.0000

### Sample 37
- Trigger count: 3
- Fallback tokens: 12
- Reasons: ['low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds']
- Avg accept before: 0.2257
- Avg accept after: 1.0000

## 5. Conclusion

The failure-aware fallback **does not meet all success criteria** on outlier samples.
- Speedup delta: +0.0492x (need >= +0.1x)
- SQ delta: +0.0028 (need >= -0.03)
- Off-structure: 0.0000 -> 0.0464 (need no increase)

**Recommendation**: Do not proceed to full-40 validation. Document as future work.