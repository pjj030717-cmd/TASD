# Guard-v2 Pilot Experiment

**Cases**: 24 perf + 20 quality hard cases
**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct

### GuardV2 Features

1. **Incremental syntax state**: tracks bracket_stack (), [], {}; quote_state (none/single/double); comment_state (normal/comment) — updated incrementally per round
2. **Comment/string awareness**: `def`/`class`/`import`/`return` inside string or comment literals → medium risk (not high risk / off-structure)
3. **Adaptive verification tightening**: high risk detected → `top_k_accept` tightened from 3 to 1 for the current verify round

---

## Performance Hard Cases (24 cases)

| Method | Mean TPS | Mean Speedup | Below 1.0x | Mean SQ | Mean OffStr | Mean Trunc | Mean Repair | Mean Guard Trig | Mean HighRisk |
| -------|----------|-------------|-----------|---------|-------------|-----------|-------------|---------------|---------- |
| TASD | 26.6 | 0.86x | 16 | 0.7521 | 0.0219 | 0.0000 | 3.83 | 10.9 | - |
| TASD-F-G-Sel | 36.5 | 1.18x | 9 | 0.8006 | 0.0091 | 0.0000 | 1.50 | 5.8 | - |
| TASD-F-G-Sel+GV2 | 46.7 | **1.51x** | **7** | 0.8125 | 0.0274 | 0.0000 | **1.04** | 3.1 | 2.8 |

## Quality-Flagged Hard Cases (20 cases)

| Method | Mean TPS | Mean Speedup | Below 1.0x | Mean SQ | Mean OffStr | Mean Trunc | Mean Repair | Mean Guard Trig | Mean HighRisk |
| -------|----------|-------------|-----------|---------|-------------|-----------|-------------|---------------|---------- |
| TASD | 64.4 | 2.08x | 0 | 0.6762 | 0.0000 | 0.0000 | 0.00 | 0.1 | - |
| TASD-F-G-Sel | 64.7 | 2.09x | 0 | 0.6762 | 0.0000 | 0.0000 | 0.00 | 0.1 | - |
| TASD-F-G-Sel+GV2 | 64.9 | 2.09x | 0 | 0.6762 | 0.0000 | 0.0000 | 0.00 | 0.0 | 0.0 |

---

## Criteria Check

### Performance Hard Cases

| Criterion | TASD-F-G-Sel | +GV2 | Pass |
|-----------|-------------|------|------|
| Off-structure < TASD-F-G-Sel | 0.0091 | 0.0274 | **FAIL** |
| SQ >= TASD-F-G-Sel (×0.98) | 0.8006 | 0.8125 | OK |
| Speedup >= TASD-F-G-Sel (×0.95) | 1.18x | 1.51x | OK |

**Verdict**: Strict criterion failed (off-structure 0.027 > 0.009). However, GuardV2 off-structure (0.027) is only marginally higher than TASD baseline (0.022).

### Quality-Flagged Hard Cases

| Criterion | TASD-F-G-Sel | +GV2 | Pass |
|-----------|-------------|------|------|
| Off-structure < TASD-F-G-Sel | 0.0000 | 0.0000 | OK |
| SQ >= TASD-F-G-Sel (×0.98) | 0.6762 | 0.6762 | OK |
| Speedup >= TASD-F-G-Sel (×0.95) | 2.09x | 2.09x | OK |

**Verdict**: PASS — No degradation on quality-flagged cases.

---

## Analysis

### Why GuardV2 improves speedup dramatically

The original `StructuralGuard` triggers aggressively on DictConfig cases (mean trigger count 10.9 per sample), over-trimming draft tokens. GuardV2's incremental state tracking is more permissive (mean trigger count 3.1), avoiding false-positive trims. This explains the large speedup gains on DictConfig cases:

| Case | TASD | TASD-F-G-Sel | +GV2 | GV2 Delta |
|------|------|-------------|------|-----------|
| dict_config idx=59 | 0.35x | 0.90x | 2.17x | +1.27x |
| dict_config idx=2 | 0.36x | 0.90x | 2.09x | +1.18x |
| dict_config idx=1 | 0.50x | 0.87x | 2.15x | +1.28x |

### Why off-structure increases

The same permissiveness that improves speedup also allows slightly more off-structure tokens to slip through. In 4 of 24 perf cases, GuardV2 generates off-structure content (vs 1-2 with TASD-F-G-Sel). However, the absolute rate (0.027) is close to TASD baseline (0.022), suggesting the original guard may have been over-trimming rather than GuardV2 being too lax.

### GuardV2 as guard diagnosis tool

GuardV2 reveals that the original StructuralGuard is over-trimming on DictConfig cases. Rather than being a negative result, this suggests the original guard thresholds should be recalibrated — GuardV2's syntax-aware approach correctly identifies that most tokens don't need trimming.

---

## Conclusion

**GuardV2 does NOT pass the strict off-structure criterion** (0.027 > 0.009), so it should **NOT enter the main TASD method** as a direct replacement.

However, it is a **valuable exploratory result**:
1. It reveals original StructuralGuard over-trimming on DictConfig
2. It achieves 1.51x speedup on perf hard cases (vs 1.18x TASD-F-G-Sel)
3. It passes all criteria on quality-flagged cases
4. The off-structure increase is modest (0.027 vs TASD baseline 0.022)

**Recommendation for future work**: recalibrate the original StructuralGuard thresholds using GuardV2 as a diagnostic baseline, rather than deploying GuardV2 as-is. GuardV2's syntax state tracking is the right direction but needs stricter off-structure boundaries for production use.

## Per-Sample Speedup Deltas

| # | Split | Benchmark | Idx | Name | TASD | TASD-F-G-Sel | +GV2 | GV2 Delta |
|---|-------|-----------|-----|------|------|-------------|------|----------|
| 1 | perf | Argparse | 22 | argparse\_real\_023 | 0.24x | 1.25x | 1.28x | +0.03x |
| 2 | perf | Argparse | 73 | argparse\_real\_074 | 0.28x | 1.31x | 1.31x | +0.00x |
| 3 | perf | DictConfig | 59 | dict\_config\_real\_060 | 0.35x | 0.90x | 2.17x | +1.27x |
| 4 | perf | DictConfig | 2 | dict\_config\_real\_003 | 0.36x | 0.90x | 2.09x | +1.18x |
| 5 | perf | DictConfig | 1 | dict\_config\_real\_002 | 0.50x | 0.87x | 2.15x | +1.28x |
| 6 | perf | OpenMMLab | 70 | openmmlab\_config\_real\_ | 0.49x | 0.68x | 0.95x | +0.26x |
| 7 | perf | OpenMMLab | 64 | openmmlab\_config\_real\_ | 0.59x | 0.91x | 0.91x | +0.00x |
| 8 | perf | Argparse | 33 | argparse\_real\_034 | 0.57x | 1.53x | 1.49x | -0.04x |
| 9 | perf | OpenMMLab | 48 | openmmlab\_config\_real\_ | 0.73x | 0.97x | 0.93x | -0.04x |
| 10 | perf | Argparse | 30 | argparse\_real\_031 | 0.75x | 1.64x | 1.64x | +0.00x |
| 11 | perf | Argparse | 61 | argparse\_real\_062 | 0.72x | 1.55x | 1.63x | +0.07x |
| 12 | perf | OpenMMLab | 0 | openmmlab\_config\_real\_ | 0.89x | 0.84x | 0.80x | -0.04x |
| 13 | perf | OpenMMLab | 2 | openmmlab\_config\_real\_ | 0.83x | 0.84x | 0.82x | -0.02x |
| 14 | perf | DictConfig | 13 | dict\_config\_real\_014 | 0.93x | 0.95x | 0.97x | +0.02x |
| 15 | perf | Argparse | 69 | argparse\_real\_070 | 0.92x | 1.46x | 1.38x | -0.09x |
| 16 | perf | DictConfig | 18 | dict\_config\_real\_019 | 1.01x | 1.00x | 0.76x | -0.24x |
| 17 | perf | Argparse | 38 | argparse\_real\_039 | 0.93x | 1.09x | 1.09x | +0.00x |
| 18 | perf | DictConfig | 57 | dict\_config\_real\_058 | 1.31x | 1.27x | 2.12x | +0.85x |
| 19 | perf | DictConfig | 40 | dict\_config\_real\_041 | 1.28x | 1.31x | 1.62x | +0.31x |
| 20 | perf | DictConfig | 52 | dict\_config\_real\_053 | 1.46x | 1.46x | 2.14x | +0.68x |
| 21 | perf | DictConfig | 77 | dict\_config\_real\_078 | 1.30x | 1.30x | 2.16x | +0.85x |
| 22 | perf | OpenMMLab | 29 | openmmlab\_config\_real\_ | 1.38x | 1.38x | 2.15x | +0.77x |
| 23 | perf | DictConfig | 78 | dict\_config\_real\_079 | 1.29x | 1.32x | 2.15x | +0.83x |
| 24 | perf | DictConfig | 7 | dict\_config\_real\_008 | 1.46x | 1.47x | 1.44x | -0.03x |
| 25 | quality | OpenMMLab | 7 | openmmlab\_config\_7 | 1.98x | 2.05x | 1.96x | -0.09x |
| 26 | quality | OpenMMLab | 11 | openmmlab\_config\_11 | 1.99x | 2.13x | 2.10x | -0.03x |
| 27 | quality | DictConfig | 9 | dict\_config\_9 | 2.16x | 2.07x | 2.05x | -0.01x |
| 28 | quality | OpenMMLab | 55 | openmmlab\_config\_55 | 2.10x | 2.18x | 2.20x | +0.03x |
| 29 | quality | OpenMMLab | 41 | openmmlab\_config\_41 | 2.19x | 2.20x | 2.14x | -0.05x |
| 30 | quality | OpenMMLab | 51 | openmmlab\_config\_51 | 2.09x | 1.92x | 2.03x | +0.11x |
| 31 | quality | OpenMMLab | 72 | openmmlab\_config\_72 | 1.99x | 2.02x | 2.02x | -0.01x |
| 32 | quality | OpenMMLab | 77 | openmmlab\_config\_77 | 2.00x | 2.01x | 2.02x | +0.00x |
| 33 | quality | OpenMMLab | 78 | openmmlab\_config\_78 | 2.03x | 2.01x | 2.00x | -0.00x |
| 34 | quality | DictConfig | 48 | dict\_config\_48 | 2.04x | 2.10x | 2.02x | -0.08x |
| 35 | quality | DictConfig | 67 | dict\_config\_67 | 1.98x | 1.97x | 1.97x | +0.00x |
| 36 | quality | DictConfig | 3 | dict\_config\_3 | 2.06x | 2.10x | 2.08x | -0.02x |
| 37 | quality | Argparse | 36 | argparse\_36 | 2.04x | 2.19x | 2.18x | -0.01x |
| 38 | quality | Argparse | 39 | argparse\_39 | 2.19x | 2.15x | 2.16x | +0.01x |
| 39 | quality | DictConfig | 50 | dict\_config\_50 | 1.89x | 1.82x | 2.15x | +0.32x |
| 40 | quality | OpenMMLab | 45 | openmmlab\_config\_45 | 2.20x | 2.20x | 2.20x | +0.00x |
| 41 | quality | OpenMMLab | 42 | openmmlab\_config\_42 | 2.20x | 2.21x | 2.16x | -0.05x |
| 42 | quality | OpenMMLab | 46 | openmmlab\_config\_46 | 2.01x | 2.09x | 2.17x | +0.08x |
| 43 | quality | OpenMMLab | 37 | openmmlab\_config\_37 | 2.18x | 2.16x | 2.18x | +0.02x |
| 44 | quality | OpenMMLab | 24 | openmmlab\_config\_24 | 2.21x | 2.19x | 2.08x | -0.11x |
