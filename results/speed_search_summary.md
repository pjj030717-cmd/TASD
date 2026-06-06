# TASD Speed Parameter Search

**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft)
**Settings**: max_new_tokens=128, n=20 per benchmark
**Fixed**: enable_guard=True, enable_relaxed_accept=True, prefix_budget=0.2, window_len=2

> **Archive**: This search used the 3B draft model. Current main results use 1.5B draft (see `results/comparison_4method_80.md`).

## Results (ranked by average speedup)

| Sweep | dl | db | k | OpenMMLab TPS | OpenMMLab Spd | DictConfig TPS | DictConfig Spd | Avg Spd | Accept | SQ |
|-------|----|----|---|--------------|---------------|---------------|---------------|---------|--------|----|
| d16_b2_k3 | 16 | 2 | 3 | 51.35 | **1.56x** | 46.35 | 1.42x | **1.49x** | 0.9573 | 0.9625 |
| d8_b3_k3 | 8 | 3 | 3 | 50.10 | **1.52x** | 46.70 | 1.43x | **1.48x** | 0.9681 | 0.9700 |
| d8_b2_k1 | 8 | 2 | 1 | 49.37 | 1.50x | 45.04 | 1.38x | 1.44x | 0.9591 | 0.9625 |
| d8_b2_k5 | 8 | 2 | 5 | 49.26 | 1.50x | 43.78 | 1.34x | 1.42x | 0.9591 | 0.9625 |
| d8_b2_k3 | 8 | 2 | 3 | 47.97 | 1.46x | 44.15 | 1.35x | 1.40x | 0.9605 | 0.9700 |
| d4_b2_k3 | 4 | 2 | 3 | 43.47 | 1.32x | 37.69 | 1.15x | 1.24x | 0.9342 | 0.9925 |
| d8_b1_k3 | 8 | 1 | 3 | 43.31 | 1.32x | 37.69 | 1.15x | 1.23x | 0.9345 | 0.9925 |

## Dimension Analysis

### draft_blocks (fixed dl=8, k=3)

| Benchmark | blk=1 | blk=2 | blk=3 | Gain (1->3) |
|-----------|-------|-------|-------|-------------|
| OpenMMLab | 43.3 TPS | 48.0 TPS | 50.1 TPS | +15.7% |
| DictConfig | 37.7 TPS | 44.2 TPS | 46.7 TPS | +23.9% |

### draft_len (fixed db=2, k=3)

| Benchmark | dl=4 | dl=8 | dl=16 | Gain (4->16) |
|-----------|------|------|-------|--------------|
| OpenMMLab | 43.5 TPS | 48.0 TPS | 51.4 TPS | +18.2% |
| DictConfig | 37.7 TPS | 44.2 TPS | 46.4 TPS | +23.0% |

### top_k_accept (fixed dl=8, db=2)

| Benchmark | k=1 | k=3 | k=5 | Trend |
|-----------|-----|-----|-----|-------|
| OpenMMLab | 49.4 | 48.0 | 49.3 | Flat (accept already 1.0) |
| DictConfig | 45.0 | 44.2 | 43.8 | Slight decline (higher k accepts more bad tokens) |

## Key Findings

1. **draft_len=16 with draft_blocks=2 is the best single config**: 1.49x average speedup. Longer draft sequences exploit the draft model's structural pattern recognition better. On OpenMMLab it reaches 1.56x.

2. **draft_blocks=3 with draft_len=8 is a close second**: 1.48x average. The marginal gain from blk=2 -> blk=3 (+2.5% TPS) is smaller than blk=1 -> blk=2 (+14%). Return diminishes beyond 2 blocks.

3. **draft_len=4 is insufficient**: TPS collapses to ~1.24x on average. Short draft blocks cannot capture enough structural context, leading to more rejections and repairs.

4. **top_k_accept has minimal effect when accept_rate is already near 1.0**: On OpenMMLab, accept_rate=1.0 regardless of k. On DictConfig, k=1 performs similar to k=3/5 — the 3B draft so closely matches the 14B target that even strict argmax acceptance rarely rejects.

5. **OpenMMLab consistently outperforms DictConfig**: The openmmlab config format has more repetitive key=dict() patterns that the 3B draft can reliably reproduce. DictConfig has more varied key-value pairs with different types, causing occasional draft-target mismatch.

## Recommended Config

Parameter search shows that increasing draft_len from 8 to 16 improves speedup by better amortizing target verification overhead, while increasing draft_blocks beyond 2 brings only marginal gains. We therefore use draft_len=16, draft_blocks=2, top_k_accept=3 as the optimized default.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| draft_len | 16 | Best speed in search, largest gain per dimension |
| draft_blocks | 2 | Sweet spot; blk=3 adds only +2% for +50% complexity |
| top_k_accept | 3 | Default; k=1/k=5 make negligible difference |

This config (d16_b2_k3) yields **1.49x average speedup** across 2 benchmarks at n=20. The original default (d8_b2_k3) averages 1.40x — a +0.09x improvement.
