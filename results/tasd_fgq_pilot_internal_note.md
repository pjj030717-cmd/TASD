# TASD-FGQ Pilot Results (Internal Note)

**Status**: ❌ FAILED - Do not proceed to full experiment

**Date**: 2026-06-19

## Pilot Configuration

- **No-regression subset**: 60 samples (score=2), 10 per benchmark
- **Hard subset**: 60 samples (score=0), 10 per benchmark
- **Quality Guard mechanisms**:
  - Repetition-aware trim (4-gram, window=64)
  - Off-structure strict mode (trigger=2, window=3, rounds=2)
  - Low-progress repair (threshold=1, patience=2, tokens=2)

## Results

### No-Regression Pilot (score=2)
- **Score=2 retention**: 100% (13/13) ✅
- **Speed loss**: 5.9% ⚠️ (threshold: ≤5%)
- **Quality guard triggers**: 0 (rep_trim, strict, repair all inactive)

### Hard Subset Pilot (score=0)
- **TPS loss**: 11.9% ❌ (threshold: ≤10%)
- **rep_trim_count**: 0 ❌ (core mechanism not triggered)
- **Score=0 reduction**: Not observed

## Analysis

1. **Repetition detection ineffective**: 4-gram repetition threshold too conservative, or score=0 samples fail due to other reasons (not repetition)
2. **Speed overhead unacceptable**: 11.9% loss even without mechanism triggers suggests overhead from quality guard monitoring
3. **No quality improvement**: Since mechanisms didn't trigger, no improvement in score=0 samples

## Conclusion

**TASD-FGQ is abandoned as a quality improvement direction.**

The core issue is that score=0 samples in TASD-FG are not caused by repetition patterns that can be detected and trimmed. The quality guard mechanisms (rep-trim, strict-mode, low-progress-repair) either:
- Don't trigger (rep_trim_count=0)
- Add overhead without benefit (11.9% speed loss)

**Decision**: Keep TASD-FG as the final method. Do not merge FGQ into final_master_report.

## Paper Note (Supplementary)

> A quality-guarded variant (TASD-FGQ) was tested on hard/no-regression subsets but did not improve automatic recoverability under acceptable speed overhead. The repetition-aware trim mechanism showed zero triggers, suggesting that structural failures in TASD-FG are not primarily caused by token-level repetition patterns.
