# OpenMMLab-Config 256-token: Failure-Aware Fallback Improved Versions

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct

## Variants
- **TASD**: Baseline (no fallback)
- **TASD+FB(v1,4tok)**: AR fallback 4 tokens, unguarded
- **TASD+FB(v2,2tok)**: AR fallback 2 tokens, unguarded
- **TASD+FB(guarded)**: AR fallback 4 tokens, per-token structural guard check

## 1. Summary

| Metric | TASD | FB(v1,4tok) | FB(v2,2tok) | FB(guarded) |
|--------|------|-------------|-------------|-------------|
| Mean Speedup | 1.6516x | 1.7323x | 1.7668x | 1.7462x |
| Median Speedup | 1.8784x | 1.8761x | 1.8763x | 1.8749x |
| Mean Accept Rate | 0.8572 | 0.9063 | 0.9249 | 0.9087 |
| Mean Repair Count | 1.82 | 0.50 | 0.50 | 0.35 |
| Mean SQ | 0.9071 | 0.9022 | 0.9080 | 0.9083 |
| Mean Off-Structure | 0.0039 | 0.0145 | 0.0057 | 0.0000 |
| FB Triggers | - | 25 | 18 | 13 |
| Guard Trims | - | - | - | 0 |

## 2. Success Criteria (vs TASD)

### TASD+FB(v1,4tok)
- Speedup >= TASD+0.04x: PASS (delta=+0.0807x)
- SQ not degraded >0.01: PASS
- Off-structure <= TASD+0.005: FAIL
- Wins >= 20/40: PASS (29/40)
- Repair reduced: PASS
- **All passed**: NO

### TASD+FB(v2,2tok)
- Speedup >= TASD+0.04x: PASS (delta=+0.1152x)
- SQ not degraded >0.01: PASS
- Off-structure <= TASD+0.005: PASS
- Wins >= 20/40: PASS (29/40)
- Repair reduced: PASS
- **All passed**: YES

### TASD+FB(guarded)
- Speedup >= TASD+0.04x: PASS (delta=+0.0946x)
- SQ not degraded >0.01: PASS
- Off-structure <= TASD+0.005: PASS
- Wins >= 20/40: PASS (28/40)
- Repair reduced: PASS
- **All passed**: YES

## 3. Per-Sample Results

| Idx | AR TPS | TASD Speedup | FB(v1) Speedup | FB(v2) Speedup | FB(guarded) Speedup | Best |
|-----|--------|--------------|----------------|----------------|---------------------|------|
| 0 | 31.22 | 0.9603x | 1.5852x | 1.6227x | 1.2072x | v2 |
| 1 | 33.02 | 1.8480x | 1.8783x | 1.7753x | 1.8092x | v1 |
| 2 | 31.98 | 0.9606x | 1.5081x | 1.4931x | 1.0829x | v1 |
| 3 | 31.25 | 1.9517x | 1.9939x | 1.9078x | 1.9094x | v1 |
| 4 | 30.82 | 1.9056x | 1.9225x | 1.9500x | 1.9315x | v2 |
| 5 | 30.68 | 2.0052x | 2.1239x | 1.9919x | 2.0711x | v1 |
| 6 | 32.81 | 1.8348x | 1.8415x | 1.9107x | 1.8016x | v2 |
| 7 | 31.82 | 0.5101x | 0.8014x | 1.2932x | 1.3152x | guarded |
| 8 | 32.30 | 1.9542x | 1.9585x | 1.9467x | 1.9319x | v1 |
| 9 | 32.05 | 1.5994x | 1.6530x | 1.6300x | 1.6340x | v1 |
| 10 | 32.11 | 2.0352x | 2.0125x | 1.9598x | 2.0056x | TASD |
| 11 | 31.24 | 0.5022x | 0.8278x | 1.2436x | 1.3009x | guarded |
| 12 | 31.82 | 1.8715x | 1.8740x | 1.9054x | 1.8865x | v2 |
| 13 | 30.53 | 1.5775x | 1.6214x | 1.6109x | 1.6315x | guarded |
| 14 | 30.85 | 1.9290x | 1.9361x | 1.9887x | 2.0561x | guarded |
| 15 | 30.73 | 1.9092x | 1.9297x | 1.9583x | 2.0748x | guarded |
| 16 | 33.25 | 0.7877x | 1.3459x | 1.4328x | 1.0021x | v2 |
| 17 | 32.01 | 1.6186x | 1.6404x | 1.7341x | 1.6685x | v2 |
| 18 | 32.55 | 1.9392x | 1.9677x | 1.9840x | 2.0080x | guarded |
| 19 | 32.15 | 1.9792x | 2.0348x | 1.9997x | 1.9904x | v1 |
| 20 | 32.75 | 0.8150x | 1.4699x | 1.4821x | 1.0565x | v2 |
| 21 | 33.20 | 0.9702x | 0.9334x | 0.9744x | 1.2904x | guarded |
| 22 | 30.84 | 1.9721x | 1.8395x | 2.0645x | 2.0976x | guarded |
| 23 | 32.73 | 1.9169x | 1.9303x | 1.8744x | 1.9383x | guarded |
| 24 | 32.86 | 1.8615x | 1.9020x | 1.8783x | 1.8408x | v1 |
| 25 | 31.98 | 1.9897x | 2.0047x | 1.9931x | 1.9415x | v1 |
| 26 | 32.28 | 1.9910x | 1.9991x | 1.8894x | 1.9064x | v1 |
| 27 | 32.06 | 1.9376x | 1.8244x | 1.8197x | 1.8734x | TASD |
| 28 | 31.87 | 1.9583x | 1.9470x | 1.8704x | 1.8764x | TASD |
| 29 | 31.08 | 1.6052x | 1.6651x | 1.6133x | 1.6409x | v1 |
| 30 | 30.82 | 1.9390x | 1.9364x | 1.9650x | 2.0818x | guarded |
| 31 | 33.83 | 1.8676x | 1.8345x | 1.8948x | 1.8821x | v2 |
| 32 | 33.37 | 1.8771x | 1.8939x | 1.8648x | 1.8562x | v1 |
| 33 | 31.35 | 1.8963x | 1.8960x | 1.9879x | 1.9994x | guarded |
| 34 | 32.42 | 1.5056x | 1.5194x | 1.5444x | 1.6487x | guarded |
| 35 | 33.54 | 1.8798x | 1.8348x | 1.9216x | 1.9368x | guarded |
| 36 | 33.12 | 1.5009x | 1.5139x | 1.6030x | 1.5320x | v2 |
| 37 | 33.19 | 1.0241x | 1.0572x | 1.3070x | 1.3293x | guarded |
| 38 | 31.18 | 1.9262x | 1.9160x | 1.9949x | 1.9981x | guarded |
| 39 | 33.35 | 1.9520x | 1.9163x | 1.7892x | 1.8021x | TASD |

## 4. Conclusion

**2 variant(s) pass all success criteria.**
- TASD+FB(v2,2tok): speedup ++0.1152x, wins 29/40
- TASD+FB(guarded): speedup ++0.0946x, wins 28/40

**Recommendation**: Include the passing variant(s) as an optional safety extension.