# Official FLY Pilot Report (LLaMA-8B)

**Target**: meta-llama/Llama-3.1-8B-Instruct
**Draft**: meta-llama/Llama-3.2-1B-Instruct
**Tokenizer**: Compatible (same vocab=128000)

### FLY Configuration (Official)

| Parameter | Value | Notes |
|-----------|-------|-------|
| enable_fly | True | FLY window recovery |
| win_len | 6 | Recovery pattern window |
| entropy_thre | 0.3 | Post-verify entropy rejection |
| use_ngram | True | Prompt lookup (PLD) |
| max_ngram_size | 4 | N-gram match length |
| num_ngram_pred_tokens | 6 | Draft tokens from n-gram |
| gamma (k) | [8, 16] | Draft tokens per round |
| total_gen_tok | 128 | Max new tokens |

**Source**: FLy/fly/models/FLy.py SPDGenerate class (official AMD implementation)

**Note**: Official FLY from FLy repo. Compared to our previous implementation:
- Always runs n-gram PLD (no entropy gate blocking)
- Has FLY window recovery (converts rejected→accepted)
- Has post-verify entropy rejection (different direction)
- Has n-gram cache fallback (reuses previous n-gram)

## dict_config

AR TPS: 88.5

| Gamma | FLY TPS | Speedup | MAT | AcceptRate | NgramAcc | FLY Rec | Below1.0x |
|-------|---------|---------|-----|------------|----------|---------|----------|
| k=8 | 124.0 | **1.44x** | 6.14 | 1.152 | 1.97 | 0.0892 | 4 |
| k=16 | 124.1 | **1.46x** | 9.76 | 1.576 | 2.05 | 0.3225 | 7 |

## openmmlab_config

AR TPS: 81.7

| Gamma | FLY TPS | Speedup | MAT | AcceptRate | NgramAcc | FLY Rec | Below1.0x |
|-------|---------|---------|-----|------------|----------|---------|----------|
| k=8 | 76.6 | **0.94x** | 4.64 | 1.348 | 1.74 | 0.0471 | 16 |
| k=16 | 69.3 | **0.84x** | 6.51 | 2.054 | 1.83 | 0.1794 | 16 |

## pipeline_stage_config

AR TPS: 84.7

| Gamma | FLY TPS | Speedup | MAT | AcceptRate | NgramAcc | FLY Rec | Below1.0x |
|-------|---------|---------|-----|------------|----------|---------|----------|
| k=8 | 73.8 | **0.86x** | 4.43 | 1.387 | 1.73 | 0.0606 | 17 |
| k=16 | 70.7 | **0.83x** | 6.65 | 2.050 | 1.75 | 0.1741 | 14 |

## Overall Summary

| Gamma | Speedup | MAT | Accept | FLY Rec | Below1.0x |
|-------|---------|-----|--------|---------|----------|
| k=8 | **1.078x** | 5.07 | 1.296 | 0.0656 | 37 |
| k=16 | **1.042x** | 7.64 | 1.893 | 0.2253 | 37 |

**Recommended FLY baseline**: k=8 (speedup=1.078x)

### Classification

- **MLA**: NOT enabled (same as paper - MLA not in FLY code)
- **PLD**: ENABLED, always active (n-gram + cached fallback)
- **FLY recovery**: ENABLED (window=6, converts reject→accept)
- **Entropy rejection**: ENABLED (post-verify, τ=0.3)
- **Label**: Official FLY (no MLA, ngram+model+fly_recovery)
