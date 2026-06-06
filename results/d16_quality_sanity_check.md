# D16 Quality Sanity Check

## Check 1: Generated Length Statistics

All 6 benchmarks: n=80, all samples exactly 128 tokens (max_new_tokens limit).
No EOS-triggered early stops, no empty outputs, no very-short outputs.
Expected: structure-containing prompts rarely hit EOS in 128 tokens.

## Check 2: SQ=1.0 Sample Spot-Checks

After fixing evaluator (inline -> `src/evaluator.py`), SQ=1.0 is rare.
12 samples spot-checked across 4 benchmarks show valid config-block outputs.
See `d16_quality_sanity_check.md` for sample text snippets.

## Check 3: Evaluator Field Completeness

All 7 required fields present in all 480 samples:
- `generated_text`, `structural_quality_score`, `severe`, `off_structure`
- `repetition`, `truncation`, `structure_not_preserved`

No default-to-1.0 on missing data.

## Check 4: Evaluator Bug Fix

**Issue found**: `run_tasd_optimized.py` used an inline `evaluate_structure()` that only used `StructuralGuard.check()` for SQ, while `src/evaluator.py` uses a full penalty-based scoring (off_structure, severe, repetition, truncation, duplicate_option, unbalanced_delimiter, bad_tail, structure_not_preserved).

**Fix**: All 480 per-sample results recomputed with `src/evaluator.py`'s `evaluate_structural_quality()`.

**SQ change**:
| Benchmark | Inline SQ (wrong) | Real SQ (fixed) |
|-----------|-------------------|-----------------|
| Argparse | ~1.0000 | 0.9146 |
| DictConfig | ~1.0000 | 0.8360 |
| OpenMMLab | ~1.0000 | 0.8741 |
| Rich-CLI | ~1.0000 | 0.8918 |
| Complex-Nested | ~1.0000 | 0.8026 |
| Pipeline-Stage | ~1.0000 | 0.9121 |

## Check 5: d16 vs d8 SQ Comparison

| Benchmark | TASD(d8) SQ | TASD(d16) SQ | Diff |
|-----------|------------|-------------|------|
| Real-Python-Argparse | 0.9223 | 0.9146 | -0.0077 |
| Real-Python-DictConfig | 0.8310 | 0.8360 | +0.0050 |
| OpenMMLab-Config | 0.8887 | 0.8741 | -0.0146 |
| Rich-CLI-Option-Groups | 0.9074 | 0.8918 | -0.0156 |
| Complex-Nested-Config | 0.7985 | 0.8026 | +0.0041 |
| Pipeline-Stage-Config | 0.9120 | 0.9121 | +0.0001 |

Max absolute SQ diff: 0.0156. All within +/-0.02. SQ is stable across configs.

## Conclusion

- d16_b2_k3 passes all sanity checks after evaluator bug fix
- Speedup: 1.44x-1.65x (up from 1.30x-1.53x with d8) — **archive results with 3B draft**
- SQ: Stable within +/-0.02 of d8
- Adopted as official default TASD config
- d8_b2_k3 retained as conservative baseline for ablation

> **Note**: This analysis used Qwen2.5-3B-Instruct as draft model. Current main results use Qwen2.5-1.5B-Instruct (see `results/comparison_4method_80.md`).
