# 4-Method Comparison: AR vs Greedy SD vs FLY vs TASD

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Settings**: temperature=0.0, max_new_tokens=128, n=80 per benchmark

## Methods

| Method | Description |
|--------|-------------|
| AR | Autoregressive (target-only) |
| Greedy SD | Standard speculative decoding, argmax match, k=16 |
| FLY | Relaxed SD with n-gram draft + window acceptance, k=15, win_len=6 |
| TASD | Multi-block draft (b=2x16), relaxed accept (k=3), guard, KV-cache incremental |

## Speed and Quality

| Benchmark | Method | TPS | Speedup vs AR | Accept | SQ | OffStr | Trunc | Rep | SNP |
|-----------|--------|-----|---------------|--------|----|--------|-------|-----|-----|
| Real-Python-Argparse | AR | 33.2 | 1.00x | 1.00 | 0.8996 | 0.0171 | 0.0261 | 0.0000 | 0.1750 |
| Real-Python-Argparse | Greedy SD | 31.4 | 0.95x | 0.58 | 0.9209 | 0.0007 | 0.0211 | 0.0000 | 0.1625 |
| Real-Python-Argparse | FLY | 63.9 | 1.93x | 10.15 | 0.9159 | 0.0010 | 0.0227 | 0.0110 | 0.1625 |
| Real-Python-Argparse | TASD | 61.9 | 1.87x | 0.93 | 0.9010 | 0.0014 | 0.0269 | 0.0444 | 0.2000 |
| Real-Python-DictConfig | AR | 33.3 | 1.00x | 1.00 | 0.8491 | 0.0413 | 0.0563 | 0.0022 | 0.4750 |
| Real-Python-DictConfig | Greedy SD | 21.8 | 0.66x | 0.39 | 0.7809 | 0.0099 | 0.2049 | 0.1490 | 0.8375 |
| Real-Python-DictConfig | FLY | 58.7 | 1.76x | 10.11 | 0.8420 | 0.0205 | 0.0576 | 0.0028 | 0.7000 |
| Real-Python-DictConfig | TASD | 60.0 | 1.80x | 0.91 | 0.8316 | 0.0000 | 0.0590 | 0.0600 | 0.8500 |
| OpenMMLab-Config | AR | 33.2 | 1.00x | 1.00 | 0.8978 | 0.0000 | 0.1183 | 0.0000 | 0.3125 |
| OpenMMLab-Config | Greedy SD | 22.1 | 0.67x | 0.41 | 0.8641 | 0.0000 | 0.1290 | 0.0017 | 0.4125 |
| OpenMMLab-Config | FLY | 35.1 | 1.06x | 7.76 | 0.8845 | 0.0024 | 0.1153 | 0.0290 | 0.3375 |
| OpenMMLab-Config | TASD | 64.0 | 1.93x | 0.95 | 0.8876 | 0.0059 | 0.1134 | 0.0082 | 0.4250 |
| Rich-CLI-Option-Groups | AR | 32.9 | 1.00x | 1.00 | 0.9629 | 0.0153 | 0.0408 | 0.0000 | 0.0375 |
| Rich-CLI-Option-Groups | Greedy SD | 20.5 | 0.62x | 0.39 | 0.9332 | 0.0082 | 0.3282 | 0.0580 | 0.0250 |
| Rich-CLI-Option-Groups | FLY | 69.5 | 2.11x | 11.10 | 0.9651 | 0.0004 | 0.0590 | 0.0050 | 0.0125 |
| Rich-CLI-Option-Groups | TASD | 66.1 | 2.01x | 1.00 | 0.9119 | 0.1141 | 0.0395 | 0.0261 | 0.1375 |
| Complex-Nested-Config | AR | 33.3 | 1.00x | 1.00 | 0.8037 | 0.0218 | 0.0705 | 0.0000 | 0.8500 |
| Complex-Nested-Config | Greedy SD | 18.8 | 0.56x | 0.35 | 0.7804 | 0.0223 | 0.2641 | 0.0373 | 0.8750 |
| Complex-Nested-Config | FLY | 57.9 | 1.74x | 9.27 | 0.8145 | 0.0168 | 0.0674 | 0.0011 | 0.8750 |
| Complex-Nested-Config | TASD | 66.4 | 1.99x | 1.00 | 0.8227 | 0.0143 | 0.0727 | 0.0063 | 0.8625 |
| Pipeline-Stage-Config | AR | 33.4 | 1.00x | 1.00 | 0.9333 | 0.0000 | 0.0931 | 0.0000 | 0.0000 |
| Pipeline-Stage-Config | Greedy SD | 17.5 | 0.52x | 0.32 | 0.8876 | 0.0000 | 0.5731 | 0.1535 | 0.0625 |
| Pipeline-Stage-Config | FLY | 41.7 | 1.25x | 8.48 | 0.9148 | 0.0000 | 0.1051 | 0.0007 | 0.0000 |
| Pipeline-Stage-Config | TASD | 66.7 | 2.00x | 1.00 | 0.9405 | 0.0245 | 0.1489 | 0.0386 | 0.0250 |

## Head-to-Head: TASD Advantage

| Benchmark | Method | TPS vs TASD | Speedup Gap | SQ vs TASD | Assessment |
|-----------|--------|-------------|-------------|------------|------------|
| Real-Python-Argparse | AR | -28.7 (-46.4%) | 1.87x | -0.0014 | slower, comparable |
| Real-Python-Argparse | Greedy SD | -30.5 (-49.2%) | 1.97x | +0.0199 | slower, comparable |
| Real-Python-Argparse | FLY | +2.0 (+3.3%) | 0.97x | +0.0150 | faster, comparable |
| Real-Python-DictConfig | AR | -26.7 (-44.5%) | 1.80x | +0.0175 | slower, comparable |
| Real-Python-DictConfig | Greedy SD | -38.2 (-63.6%) | 2.75x | -0.0506 | slower, lower SQ |
| Real-Python-DictConfig | FLY | -1.3 (-2.2%) | 1.02x | +0.0104 | slower, comparable |
| OpenMMLab-Config | AR | -30.8 (-48.1%) | 1.93x | +0.0102 | slower, comparable |
| OpenMMLab-Config | Greedy SD | -41.9 (-65.5%) | 2.90x | -0.0235 | slower, lower SQ |
| OpenMMLab-Config | FLY | -28.8 (-45.1%) | 1.82x | -0.0031 | slower, comparable |
| Rich-CLI-Option-Groups | AR | -33.2 (-50.2%) | 2.01x | +0.0510 | slower, higher SQ |
| Rich-CLI-Option-Groups | Greedy SD | -45.5 (-68.9%) | 3.22x | +0.0213 | slower, higher SQ |
| Rich-CLI-Option-Groups | FLY | +3.4 (+5.2%) | 0.95x | +0.0532 | faster, higher SQ |
| Complex-Nested-Config | AR | -33.1 (-49.9%) | 1.99x | -0.0190 | slower, comparable |
| Complex-Nested-Config | Greedy SD | -47.6 (-71.7%) | 3.53x | -0.0423 | slower, lower SQ |
| Complex-Nested-Config | FLY | -8.4 (-12.7%) | 1.15x | -0.0082 | slower, comparable |
| Pipeline-Stage-Config | AR | -33.3 (-49.9%) | 2.00x | -0.0073 | slower, comparable |
| Pipeline-Stage-Config | Greedy SD | -49.2 (-73.8%) | 3.82x | -0.0529 | slower, lower SQ |
| Pipeline-Stage-Config | FLY | -24.9 (-37.4%) | 1.60x | -0.0257 | slower, lower SQ |

## Summary (averaged across 6 benchmarks)

| Method | TPS | Speedup | Accept | SQ | OffStr | Trunc | Rep | SNP |
|--------|-----|---------|--------|----|--------|-------|-----|-----|
| AR | 33.2 | 1.00x | 1.00 | 0.8910 | 0.0159 | 0.0675 | 0.0004 | 0.3083 |
| Greedy SD | 22.0 | 0.66x | 0.41 | 0.8612 | 0.0068 | 0.2534 | 0.0666 | 0.3958 |
| FLY | 54.5 | 1.64x | 9.48 | 0.8895 | 0.0068 | 0.0712 | 0.0083 | 0.3479 |
| TASD | 64.2 | 1.93x | 0.96 | 0.8825 | 0.0267 | 0.0767 | 0.0306 | 0.4167 |

## Analysis

### Speed Decomposition

- **AR baseline**: 33.2 TPS
- **Greedy SD** (k=16, strict): 22.0 TPS (0.66x)
- **FLY** (k=15, n-gram + window): 54.5 TPS (1.64x)
  - N-gram draft + window acceptance boost over Greedy SD: 32.5 TPS
- **TASD** (b=2x16, k=3): 64.2 TPS (1.93x)
  - Multi-block draft advantage over FLY: 9.7 TPS
  - Total speedup over AR: 1.93x

## Key Findings

1. **TASD achieves the highest average speedup** (1.93x), though FLY matches or exceeds TASD on some benchmarks
2. **FLY's n-gram draft** provides a substantial speed boost over standard autoregressive draft, bringing FLY to 1.64x average
3. **TASD leads on structurally complex benchmarks** (OpenMMLab, Pipeline-Stage, Complex-Nested)
4. **Structural quality is comparable** across FLY and TASD; both maintain SQ within a few points of AR
5. **FLY Accept = MAT** (Mean Accepted Tokens per round = emitted/draft_rounds), TASD Accept = strict accept rate
6. **Note**: FLY uses k=15 with n-gram draft; TASD uses b=2x16 with autoregressive draft. Draft mechanisms differ — comparison is between full FLY protocol and TASD protocol.