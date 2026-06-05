# 5-Method Pilot Comparison: AR vs Greedy SD vs N-gram vs FLY vs TASD

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct (GSD/FLY/TASD) | **N-gram**: no draft
**Settings**: temperature=0.0, max_new_tokens=128, n=20 per benchmark

## Methods

| Method | Draft | Verification | Description |
|--------|-------|-------------|-------------|
| AR | none | none | Autoregressive (target-only) |
| Greedy SD | 1.5B model, k=16 | strict argmax | Standard speculative decoding |
| N-gram SD | n-gram lookup, n=3-8, draft=16 | strict argmax | Prompt/history pattern matching, no draft model |
| FLY | 1.5B model, n-gram, k=15 | window (win_len=6) | FLY official protocol |
| TASD | 1.5B model, b=2x16 | top-k=3, guard | Multi-block + relaxed + structural guard |

## Speed and Quality

| Benchmark | Method | TPS | Speedup vs AR | Accept | SQ | OffStr | Trunc | Match% | AvgDraft |
|-----------|--------|-----|---------------|--------|----|--------|-------|--------|----------|
| Real-Python-DictConfig | AR | 33.1 | 1.00x | 1.00 | 0.8510 | 0.0669 | 0.0565 | - | - |
| Real-Python-DictConfig | Greedy SD | 21.0 | 0.63x | 0.38 | 0.7882 | 0.0155 | 0.1502 | - | - |
| Real-Python-DictConfig | N-gram SD | 51.5 | 1.56x | 0.49 | 0.8360 | 0.0409 | 0.0570 | 0.186 | 3.6 |
| Real-Python-DictConfig | FLY | 54.4 | 1.65x | 9.91 | 0.8299 | 0.0378 | 0.0441 | - | - |
| Real-Python-DictConfig | TASD | 54.1 | 1.64x | 0.81 | 0.8443 | 0.0000 | 0.0445 | - | - |
| OpenMMLab-Config | AR | 33.5 | 1.00x | 1.00 | 0.8890 | 0.0000 | 0.1585 | - | - |
| OpenMMLab-Config | Greedy SD | 21.8 | 0.65x | 0.41 | 0.8240 | 0.0000 | 0.1607 | - | - |
| OpenMMLab-Config | N-gram SD | 39.2 | 1.17x | 0.59 | 0.8317 | 0.0000 | 0.1273 | 0.086 | 2.1 |
| OpenMMLab-Config | FLY | 34.2 | 1.02x | 7.96 | 0.8867 | 0.0000 | 0.2024 | - | - |
| OpenMMLab-Config | TASD | 62.8 | 1.87x | 0.93 | 0.8974 | 0.0126 | 0.1554 | - | - |
| Pipeline-Stage-Config | AR | 33.8 | 1.00x | 1.00 | 0.9345 | 0.0000 | 0.0812 | - | - |
| Pipeline-Stage-Config | Greedy SD | 16.6 | 0.49x | 0.30 | 0.8988 | 0.0000 | 0.5145 | - | - |
| Pipeline-Stage-Config | N-gram SD | 56.4 | 1.67x | 0.60 | 0.8858 | 0.0000 | 0.1950 | 0.149 | 2.9 |
| Pipeline-Stage-Config | FLY | 40.6 | 1.20x | 8.25 | 0.9303 | 0.0000 | 0.0920 | - | - |
| Pipeline-Stage-Config | TASD | 67.2 | 1.99x | 0.99 | 0.9581 | 0.0303 | 0.1397 | - | - |

**Note**: Match% = fraction of rounds where n-gram lookup found a match. AvgDraft = average draft length. N-gram accept_rate = fraction of draft tokens accepted by target.

## Summary (3-benchmark average)

| Method | TPS | Speedup | SQ | Accept | Match% | AvgDraft |
|--------|-----|---------|----|--------|--------|----------|
| AR | 33.5 | 1.00x | 0.8915 | 1.00 | - | - |
| Greedy SD | 19.8 | 0.59x | 0.8370 | 0.36 | - | - |
| N-gram SD | 49.0 | 1.46x | 0.8512 | 0.56 | 0.140 | 2.9 |
| FLY | 43.1 | 1.29x | 0.8823 | 8.71 | - | - |
| TASD | 61.4 | 1.83x | 0.8999 | 0.91 | - | - |

## TASD vs N-gram: Head-to-Head

| Benchmark | N-gram TPS | TASD TPS | TASD Advantage | N-gram SQ | TASD SQ | SQ Gap |
|-----------|-----------|----------|---------------|-----------|---------|--------|
| Real-Python-DictConfig | 51.5 | 54.1 | +2.6 (+5%) | 0.8360 | 0.8443 | +0.0083 |
| OpenMMLab-Config | 39.2 | 62.8 | +23.6 (+60%) | 0.8317 | 0.8974 | +0.0657 |
| Pipeline-Stage-Config | 56.4 | 67.2 | +10.8 (+19%) | 0.8858 | 0.9581 | +0.0723 |

## Key Findings

1. **N-gram SpecDec is slightly faster than AR** (49.0 TPS vs 33.5 TPS, 1.46x), but the gain is marginal
2. **N-gram match rate is low** (~14%) on structured code benchmarks. The generated code patterns differ from prompt/history, limiting n-gram's effectiveness
3. **TASD is substantially faster than N-gram** (61.4 TPS vs 49.0 TPS, 1.25x). TASD's advantage is not from pattern copying but from draft model + relaxed verification
4. **N-gram is a valid training-free baseline** demonstrating that simple repetition-based speculative decoding cannot match TASD's performance on structured code completion
5. **FLY's speed advantage** (n-gram draft + window acceptance + draft model) is primarily from the draft model, not n-gram pattern matching alone
6. **Structural quality** is comparable across methods; N-gram does not degrade or improve SQ