# Guard Calibration Ablation (LLaMA-8B DictConfig, 20 samples)

## Summary

Guard-v1.5 calibration downgrades repetition and bracket-balance rules from hard-trim to warning, keeping only genuine `def`/`class`/`import` off-structure detection as hard trim on DictConfig.

## Results

| Metric | Uncalibrated | Calibrated | Delta |
|--------|:-----------:|:----------:|:-----:|
| Mean Speedup | 1.138x | **1.285x** | +0.147x |
| Mean Accept Rate | 0.9135 | **0.9867** | +0.0732 |
| Mean SQ | 0.6229 | 0.6242 | +0.0013 |
| Mean Off-Str Rate | 0.0000 | 0.0000 | +0.0000 |
| Mean Trim/Round | 1.6 | **0.1** | -1.5 |
| Mean Guard Triggers | 1.6 | 0.6 | -1.0 |
| Below-1.0x Count | 5 | **2** | -3 |

## Top Improvements

| Sample | Uncalibrated | Calibrated | Delta |
|--------|:-----------:|:----------:|:-----:|
| dict_config_real_003 | 0.306x | 1.264x | +0.958x |
| dict_config_real_005 | 0.788x | 1.149x | +0.361x |
| dict_config_real_014 | 1.235x | 1.505x | +0.270x |
| dict_config_real_001 | 1.413x | 1.675x | +0.262x |
| dict_config_real_010 | 1.131x | 1.307x | +0.176x |

## Calibrated Guard Breakdown (20 samples)

| Signal | Count | Action |
|--------|:-----:|--------|
| Hard trim (`def`/`class`) | 2 | Trim draft |
| Repetition | 3 | Warning only |
| Bracket imbalance | 7 | Warning only (depth/gap not severe) |
| Import in dict | 0 | n/a |

## Conclusion

- **Speed**: Calibrated guard removes 93.75% of false positive trims (1.6 → 0.1 per round), boosting DictConfig speedup from 1.14x → 1.28x
- **Quality**: SQ unchanged (+0.0013), off-structure rate remains 0.0000
- **Accept rate**: +0.073 (draft tokens that were incorrectly trimmed by repetition/balanced-bracket rules are now kept and verified)
- **Only 2 genuine hard trims** remain (both on `dict_config_real_019` where draft generated a stray `def bytes_to_human_readable`)
- **Recommendation**: Set `guard_calibrated=True` as the default TASD configuration
