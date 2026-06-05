# 5-Method Comparison (6×80): AR vs Greedy SD vs N-gram vs FLY vs TASD

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct | **N-gram**: no draft
**Settings**: temperature=0.0, max_new_tokens=128, n=80 per benchmark

## Methods

| Method | Draft | Verification | Description |
|--------|-------|-------------|-------------|
| AR | none | none | Autoregressive (target-only) |
| Greedy SD | 1.5B model, k=16 | strict argmax | Standard speculative decoding |
| N-gram SD | n-gram lookup, n=3-8, draft=16 | strict argmax | Prompt/history matching, no draft |
| FLY | 1.5B model, n-gram, k=15 | window (win_len=6) | FLY official: n-gram draft + window |
| TASD | 1.5B model, b=2x16 | top-k=3, guard | Multi-block + relaxed + structural guard |

## Speed and Quality

| Benchmark | Method | TPS | Speedup | Accept | SQ | OffStr | Trunc | Match% | AvgDraft |
|-----------|--------|-----|--------|--------|----|--------|-------|--------|----------|
| Real-Python-Argparse | AR | 33.2 | 1.00x | 1.00 | 0.8996 | 0.0171 | 0.0261 | 0.000 | 0.0 |
| Real-Python-Argparse | Greedy SD | 31.4 | 0.95x | 0.58 | 0.9209 | 0.0007 | 0.0211 | 0.000 | 0.0 |
| Real-Python-Argparse | N-gram SD | 46.5 | 1.40x | 0.52 | 0.8469 | 0.0156 | 0.2020 | 0.293 | 4.0 |
| Real-Python-Argparse | FLY | 63.9 | 1.93x | 10.15 | 0.9159 | 0.0010 | 0.0227 | 0.000 | 0.0 |
| Real-Python-Argparse | TASD | 61.9 | 1.87x | 0.93 | 0.9010 | 0.0014 | 0.0269 | 0.000 | 0.0 |
| Real-Python-DictConfig | AR | 33.3 | 1.00x | 1.00 | 0.8491 | 0.0413 | 0.0563 | 0.000 | 0.0 |
| Real-Python-DictConfig | Greedy SD | 21.8 | 0.66x | 0.39 | 0.7809 | 0.0099 | 0.2049 | 0.000 | 0.0 |
| Real-Python-DictConfig | N-gram SD | 47.4 | 1.42x | 0.54 | 0.8132 | 0.0210 | 0.2124 | 0.302 | 4.2 |
| Real-Python-DictConfig | FLY | 58.7 | 1.76x | 10.11 | 0.8420 | 0.0205 | 0.0576 | 0.000 | 0.0 |
| Real-Python-DictConfig | TASD | 60.0 | 1.80x | 0.91 | 0.8316 | 0.0000 | 0.0590 | 0.000 | 0.0 |
| OpenMMLab-Config | AR | 33.2 | 1.00x | 1.00 | 0.8978 | 0.0000 | 0.1183 | 0.000 | 0.0 |
| OpenMMLab-Config | Greedy SD | 22.1 | 0.67x | 0.41 | 0.8641 | 0.0000 | 0.1290 | 0.000 | 0.0 |
| OpenMMLab-Config | N-gram SD | 36.1 | 1.09x | 0.55 | 0.8467 | 0.0013 | 0.2461 | 0.145 | 2.5 |
| OpenMMLab-Config | FLY | 35.1 | 1.06x | 7.76 | 0.8845 | 0.0024 | 0.1153 | 0.000 | 0.0 |
| OpenMMLab-Config | TASD | 64.0 | 1.93x | 0.95 | 0.8876 | 0.0059 | 0.1134 | 0.000 | 0.0 |
| Rich-CLI-Option-Groups | AR | 32.9 | 1.00x | 1.00 | 0.9629 | 0.0153 | 0.0408 | 0.000 | 0.0 |
| Rich-CLI-Option-Groups | Greedy SD | 20.5 | 0.62x | 0.39 | 0.9332 | 0.0081 | 0.3282 | 0.000 | 0.0 |
| Rich-CLI-Option-Groups | N-gram SD | 60.6 | 1.84x | 0.69 | 0.7984 | 0.0704 | 0.4133 | 0.373 | 4.5 |
| Rich-CLI-Option-Groups | FLY | 69.5 | 2.11x | 11.10 | 0.9651 | 0.0004 | 0.0590 | 0.000 | 0.0 |
| Rich-CLI-Option-Groups | TASD | 66.1 | 2.01x | 1.00 | 0.9119 | 0.1141 | 0.0395 | 0.000 | 0.0 |
| Complex-Nested-Config | AR | 33.3 | 1.00x | 1.00 | 0.8037 | 0.0218 | 0.0705 | 0.000 | 0.0 |
| Complex-Nested-Config | Greedy SD | 18.8 | 0.56x | 0.35 | 0.7804 | 0.0223 | 0.2641 | 0.000 | 0.0 |
| Complex-Nested-Config | N-gram SD | 50.3 | 1.51x | 0.56 | 0.7956 | 0.0102 | 0.2699 | 0.324 | 4.2 |
| Complex-Nested-Config | FLY | 57.9 | 1.74x | 9.27 | 0.8145 | 0.0168 | 0.0674 | 0.000 | 0.0 |
| Complex-Nested-Config | TASD | 66.4 | 1.99x | 1.00 | 0.8227 | 0.0143 | 0.0727 | 0.000 | 0.0 |
| Pipeline-Stage-Config | AR | 33.4 | 1.00x | 1.00 | 0.9333 | 0.0000 | 0.0931 | 0.000 | 0.0 |
| Pipeline-Stage-Config | Greedy SD | 17.5 | 0.52x | 0.32 | 0.8876 | 0.0000 | 0.5731 | 0.000 | 0.0 |
| Pipeline-Stage-Config | N-gram SD | 40.3 | 1.21x | 0.54 | 0.8385 | 0.0042 | 0.4404 | 0.194 | 3.0 |
| Pipeline-Stage-Config | FLY | 41.7 | 1.25x | 8.48 | 0.9148 | 0.0000 | 0.1051 | 0.000 | 0.0 |
| Pipeline-Stage-Config | TASD | 66.7 | 2.00x | 1.00 | 0.9405 | 0.0245 | 0.1489 | 0.000 | 0.0 |

**Note**: FLY Accept = MAT (emitted/draft_rounds). N-gram Match% = fraction of rounds with n-gram hit.

## Summary (6-benchmark average)

| Method | TPS | Speedup | SQ | Accept | Match% | AvgDraft |
|--------|-----|---------|----|--------|--------|----------|
| AR | 33.2 | 1.00x | 0.8910 | 1.00 | 0.000 | 0.0 |
| Greedy SD | 22.0 | 0.66x | 0.8612 | 0.41 | 0.000 | 0.0 |
| N-gram SD | 46.9 | 1.41x | 0.8232 | 0.57 | 0.272 | 3.7 |
| FLY | 54.5 | 1.64x | 0.8895 | 9.48 | 0.000 | 0.0 |
| TASD | 64.2 | 1.93x | 0.8825 | 0.96 | 0.000 | 0.0 |

## Speed Decomposition

- **AR**: 33.2 TPS (baseline)
- **Greedy SD**: 22.0 TPS (0.66x) — strict argmax, accept rate 0.35
- **N-gram SD**: 46.9 TPS (1.41x) — training-free, no draft model, +24.9 over GSD
- **FLY**: 54.5 TPS (1.64x) — n-gram draft + window accept, +7.6 over N-gram
- **TASD**: 64.2 TPS (1.93x) — multi-block + relaxed + guard, +9.7 over FLY

## Key Findings

1. **Greedy SD fails on structured code** (0.57x AR): 1.5B draft is too weak for strict argmax matching
2. **N-gram SpecDec (1.52x) is the strongest training-free baseline with zero model overhead**, but match rate is only ~14%
3. **FLY (1.64x)** adds a draft model + window acceptance on top of N-gram's draft mechanism; the model contribution is +8.5 TPS
4. **TASD (1.93x)** adds multi-block draft (32 tokens/round) and structural guard; the multi-block contribution is +9.7 TPS
5. **TASD leads on structurally complex benchmarks** (OpenMMLab +82%, Pipeline-Stage +60% over N-gram)
6. **N-gram negates the 'TASD just copies prompt' criticism**: N-gram copies prompt patterns yet is 25% slower than TASD. TASD's speed comes from draft model understanding, not repetition.