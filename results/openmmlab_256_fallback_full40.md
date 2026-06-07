# OpenMMLab-Config 256-token: Full 40-Sample Fallback Validation

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct

## 1. Summary

| Metric | TASD | TASD+Fallback | Delta |
|--------|------|---------------|-------|
| Mean Speedup | 1.6429x | 1.5475x | -0.0954x |
| Median Speedup | 1.8870x | 1.7349x | -0.1521x |
| Mean Accept Rate | 0.8572 | 0.8639 | +0.0067 |
| Mean Repair Count | 1.82 | 3.60 | +1.78 |
| Mean SQ | 0.9071 | 0.9072 | +0.0001 |
| Mean Off-Structure | 0.0039 | 0.0150 | +0.0111 |
| Fallback Wins | - | 10/40 | - |
| TASD Wins | - | 30/40 | - |
| Total Fallback Triggers | - | 462 | - |

## 2. Per-Sample Results

| Idx | AR TPS | TASD TPS | TASD Speedup | FB TPS | FB Speedup | Winner | FB Triggers | SQ Delta |
|-----|--------|----------|--------------|--------|------------|--------|-------------|----------|
| 0 | 29.01 | 29.58 | 1.0196x | 25.01 | 0.8621x | TASD | 21 | +0.0000 |
| 1 | 32.94 | 63.35 | 1.9232x | 55.15 | 1.6743x | TASD | 13 | +0.0000 |
| 2 | 32.92 | 31.95 | 0.9705x | 25.29 | 0.7682x | TASD | 21 | +0.0000 |
| 3 | 33.41 | 64.61 | 1.9339x | 62.17 | 1.8608x | TASD | 3 | +0.0000 |
| 4 | 32.69 | 63.55 | 1.9440x | 63.46 | 1.9413x | TASD | 0 | +0.0000 |
| 5 | 32.65 | 62.75 | 1.9219x | 57.46 | 1.7599x | TASD | 13 | +0.0000 |
| 6 | 32.81 | 61.45 | 1.8729x | 60.02 | 1.8293x | TASD | 4 | +0.0000 |
| 7 | 32.39 | 16.57 | 0.5116x | 23.76 | 0.7336x | FB | 10 | -0.0305 |
| 8 | 31.97 | 63.13 | 1.9747x | 61.94 | 1.9374x | TASD | 1 | +0.0000 |
| 9 | 32.82 | 49.54 | 1.5094x | 57.51 | 1.7523x | FB | 5 | -0.0124 |
| 10 | 33.62 | 63.68 | 1.8941x | 59.40 | 1.7668x | TASD | 5 | +0.0000 |
| 11 | 33.33 | 16.93 | 0.5080x | 24.06 | 0.7219x | FB | 10 | -0.0305 |
| 12 | 32.56 | 62.57 | 1.9217x | 63.88 | 1.9619x | FB | 1 | +0.0000 |
| 13 | 33.27 | 50.57 | 1.5200x | 55.53 | 1.6691x | FB | 5 | -0.0124 |
| 14 | 32.27 | 60.92 | 1.8878x | 62.73 | 1.9439x | FB | 0 | +0.0000 |
| 15 | 32.15 | 60.96 | 1.8961x | 55.22 | 1.7176x | TASD | 13 | +0.0000 |
| 16 | 31.66 | 25.93 | 0.8190x | 19.14 | 0.6045x | TASD | 50 | -0.0078 |
| 17 | 33.20 | 55.55 | 1.6732x | 52.86 | 1.5922x | TASD | 3 | +0.1033 |
| 18 | 32.69 | 64.74 | 1.9804x | 61.58 | 1.8838x | TASD | 5 | +0.0000 |
| 19 | 32.93 | 62.42 | 1.8955x | 54.90 | 1.6672x | TASD | 13 | +0.0000 |
| 20 | 33.33 | 26.47 | 0.7942x | 19.05 | 0.5716x | TASD | 50 | -0.0078 |
| 21 | 31.71 | 33.59 | 1.0593x | 33.26 | 1.0489x | TASD | 5 | +0.0000 |
| 22 | 32.15 | 60.33 | 1.8765x | 60.10 | 1.8694x | TASD | 5 | +0.0000 |
| 23 | 31.97 | 62.90 | 1.9675x | 57.15 | 1.7876x | TASD | 13 | +0.0000 |
| 24 | 33.13 | 63.09 | 1.9043x | 42.34 | 1.2780x | TASD | 43 | +0.0000 |
| 25 | 33.58 | 62.18 | 1.8517x | 63.29 | 1.8848x | FB | 0 | +0.0000 |
| 26 | 33.40 | 62.51 | 1.8716x | 62.72 | 1.8778x | FB | 0 | +0.0000 |
| 27 | 32.69 | 64.69 | 1.9789x | 57.43 | 1.7568x | TASD | 13 | +0.0000 |
| 28 | 32.92 | 62.09 | 1.8861x | 44.79 | 1.3606x | TASD | 43 | +0.0000 |
| 29 | 33.27 | 49.38 | 1.4842x | 49.52 | 1.4884x | FB | 0 | +0.0000 |
| 30 | 30.91 | 60.33 | 1.9518x | 60.75 | 1.9654x | FB | 1 | +0.0000 |
| 31 | 31.66 | 60.40 | 1.9078x | 53.71 | 1.6965x | TASD | 13 | +0.0000 |
| 32 | 32.38 | 64.68 | 1.9975x | 62.09 | 1.9175x | TASD | 4 | +0.0000 |
| 33 | 32.71 | 63.29 | 1.9349x | 61.21 | 1.8713x | TASD | 5 | +0.0000 |
| 34 | 32.77 | 47.11 | 1.4376x | 46.59 | 1.4217x | TASD | 7 | +0.0000 |
| 35 | 32.95 | 58.98 | 1.7900x | 58.32 | 1.7700x | TASD | 5 | +0.0000 |
| 36 | 32.13 | 49.34 | 1.5356x | 46.26 | 1.4398x | TASD | 7 | +0.0000 |
| 37 | 33.03 | 33.92 | 1.0269x | 31.22 | 0.9452x | TASD | 12 | +0.0000 |
| 38 | 31.67 | 62.39 | 1.9700x | 61.79 | 1.9511x | TASD | 0 | +0.0000 |
| 39 | 31.93 | 61.07 | 1.9126x | 43.07 | 1.3489x | TASD | 40 | +0.0000 |

## 3. Fallback Details (Triggered Samples)

### Sample 0
- Trigger count: 21
- Fallback tokens: 44
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 1
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 2
- Trigger count: 21
- Fallback tokens: 44
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 3
- Trigger count: 3
- Fallback tokens: 12
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string']

### Sample 5
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 6
- Trigger count: 4
- Fallback tokens: 16
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 7
- Trigger count: 10
- Fallback tokens: 36
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 8
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['long_path_string']

### Sample 9
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line']

### Sample 10
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 11
- Trigger count: 10
- Fallback tokens: 36
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 12
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['long_path_string']

### Sample 13
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line']

### Sample 15
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 16
- Trigger count: 50
- Fallback tokens: 70
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 17
- Trigger count: 3
- Fallback tokens: 8
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 18
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 19
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 20
- Trigger count: 50
- Fallback tokens: 70
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 21
- Trigger count: 5
- Fallback tokens: 19
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 22
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 23
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 24
- Trigger count: 43
- Fallback tokens: 172
- Reasons: ['comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line']

### Sample 27
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 28
- Trigger count: 43
- Fallback tokens: 172
- Reasons: ['comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line', 'comment_line']

### Sample 30
- Trigger count: 1
- Fallback tokens: 4
- Reasons: ['long_path_string']

### Sample 31
- Trigger count: 13
- Fallback tokens: 52
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 32
- Trigger count: 4
- Fallback tokens: 16
- Reasons: ['comment_line', 'comment_line', 'comment_line', 'comment_line']

### Sample 33
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 34
- Trigger count: 7
- Fallback tokens: 28
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 35
- Trigger count: 5
- Fallback tokens: 20
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 36
- Trigger count: 7
- Fallback tokens: 28
- Reasons: ['long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string', 'long_path_string']

### Sample 37
- Trigger count: 12
- Fallback tokens: 35
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

### Sample 39
- Trigger count: 40
- Fallback tokens: 160
- Reasons: ['pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text', 'pure_nl_text']

## 4. Conclusion

The fallback **hurts performance** on the full 40-sample set (delta=-0.0954x).

**Recommendation**: Do not include in main method. Document as future work.