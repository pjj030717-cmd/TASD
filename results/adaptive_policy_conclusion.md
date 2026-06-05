# Adaptive TASD Policy — Exploratory Results & Conclusion

**Status**: Future Work (not included in main method)

---

## Overview

Two versions of an adaptive draft scheduling policy were tested as exploratory
optimizations on top of the fixed d16_b2_k3 TASD configuration with a 1.5B draft
model. The goal was to dynamically adjust `draft_len` and `top_k_accept` per
decoding round based on rolling acceptance statistics.

---

## Policy Rules Summary

### Adaptive v1

| Parameter | Rule |
|-----------|------|
| `draft_len` | 8 if `roll_accept < 0.75`; 24 if `roll_accept >= 0.95`; else 16 |
| `top_k_accept` | 5 if `(top5 - top3) >= 0.20` AND `roll_accept < 0.90`; else 3 |
| Guard constraint | Shrink to `min(draft_len, 16)` and `top_k=3` on guard trigger |

### Adaptive v2

| Parameter | Rule |
|-----------|------|
| `draft_len` | 8 if `roll_accept < 0.75`; 20 if `roll_accept >= 0.98` AND no repairs AND no guard triggers; else 16 |
| `top_k_accept` | 5 if `roll_accept < 0.90` AND `(top5 - top3) >= 0.08`; else 3 |
| Fallback | Revert to 16 if `accepted/drafted < 0.95` after a `draft_len=20` round |
| Guard constraint | `draft_len <= 16`, `top_k_accept = 3` on guard trigger |

All runs: 3 benchmarks x 20 samples, 1.5B draft, compared against fixed d16_b2_k3 baseline.

---

## Results

### v1 — Adaptive Policy

| Benchmark | Fixed TPS | Adaptive TPS | Delta | SQ Diff | Changes/Round |
|-----------|-----------|-------------|-------|---------|---------------|
| DictConfig | 51.4 | **54.6** | **+6.3%** | -0.007 | 0.7 |
| OpenMMLab | 62.8 | 62.3 | -0.5% | +0.001 | 0.2 |
| Pipeline-Stage | 65.5 | 65.2 | -0.4% | 0.000 | 0.1 |

- DictConfig passes (+6.3% > 5%). OpenMMLab / Pipeline-Stage do not reach >=5% gain.
- `draft_len=24` provides no throughput benefit on already-perfect-accept benchmarks.
- `top_k_accept` never switched to 5 (gap threshold too strict).
- **Overall: FAIL.**

### v2 — Refined Rules

| Benchmark | Fixed TPS | Adaptive TPS | Delta | SQ Diff | Changes/Round |
|-----------|-----------|-------------|-------|---------|---------------|
| DictConfig | 51.4 | 53.4 | +3.9% | -0.001 | 1.8 |
| OpenMMLab | 62.8 | 61.6 | -1.9% | +0.001 | 1.1 |
| Pipeline-Stage | 65.5 | 62.8 | **-4.1%** | +0.003 | 1.1 |

- All benchmarks fail success criteria.
- `draft_len=20` introduces compute overhead on high-accept benchmarks (OpenMMLab/Pipeline).
- `top_k_accept=5` never triggered (60 samples, 0 activations). The `top5 - top3` gap is consistently below 0.08.
- Pipeline-Stage shows the worst regression (-4.1%), confirming that pushing `draft_len` upward on already-saturated acceptance hurts more than helps.
- **Overall: FAIL.**

---

## Root Cause Analysis

1. **draft_len increase is counterproductive on high-accept benchmarks.**  
   OpenMMLab and Pipeline-Stage already achieve ~1.0 accept rate at `draft_len=16`.  
   Increasing to 20/24 only adds verification overhead — extra target model
   forward computation for tokens that would have been generated anyway.

2. **top_k_accept=5 is never useful in practice.**  
   The draft tokens that would benefit from k=5 are precisely those where
   the draft model disagrees with the target argmax (low-accept samples).
   But in those rounds, `roll_accept` drops below 0.75, triggering `draft_len=8`
   which slashes the draft token count — reducing the opportunity for k=5 to matter.
   This creates a self-defeating feedback loop.

3. **The fixed policy is already near-optimal.**  
   `d16_b2_k3` was selected as the highest-performing fixed configuration
   from the speed parameter search. It represents a well-tuned trade-off.
   An adaptive scheduler that starts from this point and adjusts per-round
   can only make marginal improvements when it guesses correctly, but
   the mis-guesses degrade performance.

4. **Structural guard already handles quality.**  
   Even without adaptive scheduling, the structural guard + relaxed acceptance
   mechanism prevents quality degradation on low-accept rounds.
   Adaptive scheduling tries to optimize throughput, but the throughput
   gain is constrained by the draft model's inherent ability to match
   the target distribution.

---

## Conclusion

Adaptive draft scheduling improves DictConfig in one setting (+6.3% in v1)
but does not consistently improve throughput across high-acceptance structured
benchmarks. The optimized fixed policy `d16_b2_k3` is therefore used as the
final TASD configuration, while robust adaptive scheduling is left for future work.

The fixed policy parameters:
- `draft_len = 16`
- `draft_blocks = 2`
- `top_k_accept = 3`
- `min_token_prob = 1e-4`
- `prefix_budget = 0.2`
- `window_len = 2`

---

## Future Directions (not pursued)

- Per-benchmark or per-sample tuning of `draft_len` from offline profiling
- Adaptive draft model switching (1.5B / 3B) based on acceptance history
- Learning-based policy (RL or contextual bandit) that optimizes for
  throughput subject to quality constraints
- `top_k_accept = 5` on a subset of problematic samples after offline diagnosis
  (see `results/low_accept_analysis_1_5b.md`)
