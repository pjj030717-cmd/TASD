# OpenMMLab-Config 256-token: Failure-Aware Fallback Full 40-Sample Validation

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct

## 1. Summary

| Metric | TASD | TASD+Failure-FB | Delta |
|--------|------|-----------------|-------|
| Mean Speedup | 1.6522x | 1.7360x | +0.0838x |
| Median Speedup | 1.8788x | 1.8830x | +0.0042x |
| Mean Accept Rate | 0.8572 | 0.9063 | +0.0491 |
| Mean Repair Count | 1.82 | 0.50 | -1.32 |
| Mean SQ | 0.9071 | 0.9022 | -0.0049 |
| Mean Off-Structure | 0.0039 | 0.0145 | +0.0106 |
| FB Wins | - | 24/40 | - |
| TASD Wins | - | 16/40 | - |
| Total Fallback Triggers | - | 25 | - |

## 2. Success Criteria

- Speedup not degraded >0.02x: PASS (delta=+0.0838x)
- FB wins >= 20/40: PASS (24/40 wins, 0 ties)
- Off-structure not increased >0.01: FAIL (0.0039 -> 0.0145)
- **All passed**: NO

## 3. Per-Sample Results

| Idx | AR TPS | TASD TPS | TASD Speedup | FB TPS | FB Speedup | Winner | FB Triggers | SQ Delta |
|-----|--------|----------|--------------|--------|------------|--------|-------------|----------|
| 0 | 27.97 | 27.96 | 0.9996x | 47.57 | 1.7008x | FB | 1 | -0.0522 |
| 1 | 32.82 | 60.89 | 1.8553x | 59.75 | 1.8205x | TASD | 0 | +0.0000 |
| 2 | 32.35 | 30.66 | 0.9478x | 47.55 | 1.4699x | FB | 1 | -0.0522 |
| 3 | 31.38 | 59.96 | 1.9108x | 58.35 | 1.8595x | TASD | 0 | +0.0000 |
| 4 | 32.21 | 60.52 | 1.8789x | 61.45 | 1.9078x | FB | 0 | +0.0000 |
| 5 | 32.44 | 63.27 | 1.9504x | 63.69 | 1.9633x | FB | 0 | +0.0000 |
| 6 | 30.41 | 58.76 | 1.9323x | 64.57 | 2.1233x | FB | 0 | +0.0000 |
| 7 | 33.14 | 17.04 | 0.5142x | 26.53 | 0.8005x | FB | 5 | -0.0670 |
| 8 | 32.94 | 62.27 | 1.8904x | 61.63 | 1.8710x | TASD | 0 | +0.0000 |
| 9 | 32.72 | 48.36 | 1.4780x | 48.13 | 1.4710x | TASD | 1 | +0.0098 |
| 10 | 30.10 | 57.44 | 1.9083x | 57.57 | 1.9126x | FB | 0 | +0.0000 |
| 11 | 31.06 | 16.24 | 0.5229x | 27.10 | 0.8725x | FB | 5 | -0.0670 |
| 12 | 29.90 | 58.88 | 1.9692x | 58.37 | 1.9522x | TASD | 0 | +0.0000 |
| 13 | 30.38 | 49.03 | 1.6139x | 51.40 | 1.6919x | FB | 1 | +0.0098 |
| 14 | 31.88 | 61.52 | 1.9297x | 63.39 | 1.9884x | FB | 0 | +0.0000 |
| 15 | 31.87 | 62.81 | 1.9708x | 58.36 | 1.8312x | TASD | 0 | +0.0000 |
| 16 | 29.91 | 24.07 | 0.8047x | 43.47 | 1.4534x | FB | 1 | +0.0289 |
| 17 | 30.38 | 51.03 | 1.6797x | 50.42 | 1.6596x | TASD | 0 | +0.0000 |
| 18 | 30.04 | 61.02 | 2.0313x | 58.50 | 1.9474x | TASD | 0 | +0.0000 |
| 19 | 31.73 | 64.37 | 2.0287x | 63.25 | 1.9934x | TASD | 0 | +0.0000 |
| 20 | 32.80 | 26.43 | 0.8058x | 46.71 | 1.4241x | FB | 1 | +0.0289 |
| 21 | 32.37 | 34.41 | 1.0630x | 32.40 | 1.0009x | TASD | 3 | -0.0642 |
| 22 | 32.60 | 63.09 | 1.9353x | 63.12 | 1.9362x | FB | 0 | +0.0000 |
| 23 | 32.67 | 63.29 | 1.9373x | 60.78 | 1.8604x | TASD | 0 | +0.0000 |
| 24 | 32.09 | 59.14 | 1.8429x | 58.37 | 1.8189x | TASD | 0 | +0.0000 |
| 25 | 29.83 | 60.23 | 2.0191x | 61.04 | 2.0463x | FB | 0 | +0.0000 |
| 26 | 30.96 | 57.01 | 1.8414x | 62.33 | 2.0132x | FB | 0 | +0.0000 |
| 27 | 32.37 | 61.85 | 1.9107x | 62.01 | 1.9157x | FB | 0 | +0.0000 |
| 28 | 32.28 | 63.40 | 1.9641x | 61.26 | 1.8978x | TASD | 0 | +0.0000 |
| 29 | 32.08 | 47.86 | 1.4919x | 48.97 | 1.5265x | FB | 1 | +0.0032 |
| 30 | 31.62 | 58.74 | 1.8577x | 62.48 | 1.9760x | FB | 0 | +0.0000 |
| 31 | 30.37 | 60.06 | 1.9776x | 61.34 | 2.0198x | FB | 0 | +0.0000 |
| 32 | 32.30 | 60.68 | 1.8786x | 62.14 | 1.9238x | FB | 0 | +0.0000 |
| 33 | 31.03 | 60.49 | 1.9494x | 61.11 | 1.9694x | FB | 0 | +0.0000 |
| 34 | 31.34 | 45.39 | 1.4483x | 45.23 | 1.4432x | TASD | 1 | -0.0259 |
| 35 | 30.03 | 58.51 | 1.9484x | 58.84 | 1.9594x | FB | 0 | +0.0000 |
| 36 | 29.76 | 45.76 | 1.5376x | 45.17 | 1.5178x | TASD | 1 | -0.0259 |
| 37 | 32.57 | 33.55 | 1.0301x | 35.55 | 1.0915x | FB | 3 | +0.0779 |
| 38 | 32.70 | 64.28 | 1.9657x | 62.58 | 1.9138x | TASD | 0 | +0.0000 |
| 39 | 30.87 | 57.58 | 1.8652x | 58.50 | 1.8950x | FB | 0 | +0.0000 |

## 4. Fallback Details (Triggered Samples)

### Sample 0
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 2
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 7
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds']

### Sample 9
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 11
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds']

### Sample 13
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 16
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 20
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 21
- Trigger count: 3
- Fallback tokens: 12
- Reasons: ['low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds']

### Sample 29
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 34
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 36
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['low_accept_2rounds']

### Sample 37
- Trigger count: 3
- Fallback tokens: 12
- Reasons: ['low_accept_2rounds', 'low_accept_2rounds', 'low_accept_2rounds']

## 5. Conclusion

The failure-aware fallback **does not meet all success criteria** on the full 40-sample set.
- Speedup delta: +0.0838x (need >= -0.02x)
- FB wins: 24/40 (need >= 20)
- Off-structure: 0.0039 -> 0.0145 (need <= +0.01)

**Recommendation**: Do not include in main method. Document as future work.