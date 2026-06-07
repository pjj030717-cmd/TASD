# TASD-F 128-token No-Regression Sanity Check

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-3B-Instruct
**Settings**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3
**Sample limit**: 20 per benchmark (6 benchmarks, 120 total)

## Purpose
Verify that failure-aware fallback (TASD-F) does NOT harm TASD in the standard
128-token main experiment setting. This is NOT a replacement for the 6x80 main table.

## Methods
- **TASD**: Baseline (no fallback)
- **TASD-F**: TASD + failure-aware fallback (2-token AR fallback, unguarded)

## 1. Overall Summary

| Metric | TASD | TASD-F | Delta |
|--------|------|--------|-------|
| Mean TPS | 47.02 | 47.10 | +0.17% |
| Mean SQ | 0.9850 | 0.9850 | +0.0000 |
| Mean Off-Structure | 0.1167 | 0.1083 | -0.0083 |
| Mean Truncation | 0.7917 | 0.7917 | - |
| Mean Repair | 0.12 | 0.11 | - |
| Total FB Triggers | - | 6 | - |
| Wins (TASD-F > TASD) | - | 63/120 | - |
| Benchmarks Passed | - | 6/6 | - |

## 2. Success Criteria

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| Mean TPS >= TASD - 1% | >= -1.0% | PASS (+0.17%) |
| SQ >= TASD - 0.01 | >= -0.01 | PASS (+0.0000) |
| Off-structure <= TASD + 0.005 | <= +0.005 | PASS (-0.0083) |
| Low FB trigger count | <= 1/sample avg | PASS (6 total) |

## 3. Per-Benchmark Results

| Benchmark | TASD TPS | TASD-F TPS | TPS Delta | SQ Delta | Off Delta | FB Triggers | Wins | Passed |
|-----------|----------|------------|-----------|----------|-----------|-------------|------|--------|
| Real-Python-Argparse | 47.49 | 48.22 | +0.73 (+1.53%) | +0.0000 | +0.0000 | 0 | 11/20 | YES |
| Real-Python-DictConfig | 43.21 | 43.03 | -0.17 (-0.40%) | +0.0000 | -0.0500 | 5 | 11/20 | YES |
| OpenMMLab-Config | 48.06 | 48.13 | +0.07 (+0.14%) | +0.0000 | +0.0000 | 1 | 10/20 | YES |
| Rich-CLI-Option-Groups | 48.16 | 47.73 | -0.43 (-0.89%) | +0.0000 | +0.0000 | 0 | 9/20 | YES |
| Complex-Nested-Config | 46.52 | 46.40 | -0.13 (-0.27%) | +0.0000 | +0.0000 | 0 | 9/20 | YES |
| Pipeline-Stage-Config | 48.68 | 49.11 | +0.43 (+0.89%) | +0.0000 | +0.0000 | 0 | 13/20 | YES |

## 4. Conclusion

**All 6 benchmarks pass the no-regression criteria.**

TASD-F has negligible effect on the 128-token setting, as expected.
The failure-aware fallback rarely triggers because TASD already has high
acceptance rates under 128-token completions.