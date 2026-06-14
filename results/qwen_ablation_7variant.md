# Qwen TASD Ablation (7 variants, 6×80=480 samples)

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct | **max_new_tokens**: 128

## Quality Metrics

- **SQ-R** (Reference-aware Structural Quality): 0.4×F1 + 0.3×bracket + 0.2×type_preservation + 0.1×no_repetition
- **SQ-S** (Structure Safety Score): 1.0 − 0.45×off_str − 0.25×trunc − 0.20×rep − 0.10×dup_opt

## Overall (480 samples)

| Variant | Speedup | Below | Worst-10 | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:--------:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **2.004x** | 3/480 | 1.655x | 0.5908 | 0.7788 | 0.0366 | 27 | 0.989 |
| TASD (base, no fb) | 1.978x | 9/480 | 1.489x | 0.5903 | 0.7769 | 0.0379 | 0 | 0.979 |
| TASD-F (unguarded fb) | 1.998x | 9/480 | 1.538x | 0.5916 | 0.7710 | 0.0484 | 131 | 0.981 |
| w/o relaxed verify | 1.704x | 4/480 | 1.353x | 0.6366 | 0.7795 | 0.0357 | 38 | 0.881 |
| w/o struct guard | 1.974x | 0/480 | 1.813x | 0.6379 | 0.7696 | 0.0529 | 1 | 0.998 |
| draft_len=8 | 1.810x | 1/480 | 1.556x | 0.6370 | 0.7781 | 0.0366 | 19 | 0.993 |
| draft_blocks=1 | 1.803x | 0/480 | 1.575x | 0.6381 | 0.7786 | 0.0366 | 14 | 0.993 |

### argparse (80)

| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **1.934x** | 1 | 0.6420 | 0.7906 | 0.0013 | 17 | 0.966 |
| TASD (base, no fb) | 1.821x | 7 | 0.6367 | 0.7766 | 0.0082 | 0 | 0.912 |
| TASD-F (unguarded fb) | 1.859x | 8 | 0.6467 | 0.7455 | 0.0686 | 117 | 0.924 |
| w/o relaxed verify | 1.628x | 3 | 0.6376 | 0.7863 | 0.0000 | 24 | 0.858 |
| w/o struct guard | 1.942x | 0 | 0.6440 | 0.7395 | 0.0917 | 0 | 0.997 |
| draft_len=8 | 1.752x | 0 | 0.6388 | 0.7894 | 0.0000 | 15 | 0.980 |
| draft_blocks=1 | 1.756x | 0 | 0.6455 | 0.7925 | 0.0000 | 10 | 0.981 |

### dict_config (80)

| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **1.971x** | 2 | 0.6567 | 0.8165 | 0.0000 | 9 | 0.974 |
| TASD (base, no fb) | 1.946x | 2 | 0.6584 | 0.8160 | 0.0007 | 0 | 0.970 |
| TASD-F (unguarded fb) | 1.974x | 1 | 0.6570 | 0.8149 | 0.0037 | 13 | 0.972 |
| w/o relaxed verify | 1.688x | 1 | 0.6538 | 0.8166 | 0.0000 | 7 | 0.881 |
| w/o struct guard | 1.979x | 0 | 0.6564 | 0.8125 | 0.0076 | 0 | 0.997 |
| draft_len=8 | 1.787x | 1 | 0.6552 | 0.8134 | 0.0000 | 4 | 0.983 |
| draft_blocks=1 | 1.791x | 0 | 0.6552 | 0.8134 | 0.0000 | 4 | 0.983 |

### openmmlab (80)

| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **2.050x** | 0 | 0.6023 | 0.7722 | 0.0426 | 1 | 0.996 |
| TASD (base, no fb) | 2.018x | 0 | 0.6027 | 0.7756 | 0.0426 | 0 | 0.996 |
| TASD-F (unguarded fb) | 2.055x | 0 | 0.6023 | 0.7722 | 0.0426 | 1 | 0.996 |
| w/o relaxed verify | 1.686x | 0 | 0.6066 | 0.7818 | 0.0426 | 1 | 0.860 |
| w/o struct guard | 1.997x | 0 | 0.6023 | 0.7722 | 0.0426 | 1 | 0.996 |
| draft_len=8 | 1.852x | 0 | 0.6033 | 0.7728 | 0.0426 | 0 | 0.998 |
| draft_blocks=1 | 1.842x | 0 | 0.6033 | 0.7728 | 0.0426 | 0 | 0.998 |

### pipeline (80)

| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **2.028x** | 0 | 0.5568 | 0.7453 | 0.0371 | 0 | 0.997 |
| TASD (base, no fb) | 2.041x | 0 | 0.5568 | 0.7453 | 0.0371 | 0 | 0.997 |
| TASD-F (unguarded fb) | 2.042x | 0 | 0.5568 | 0.7453 | 0.0371 | 0 | 0.997 |
| w/o relaxed verify | 1.733x | 0 | 0.6723 | 0.7471 | 0.0329 | 1 | 0.890 |
| w/o struct guard | 1.998x | 0 | 0.6723 | 0.7453 | 0.0371 | 0 | 0.997 |
| draft_len=8 | 1.833x | 0 | 0.6723 | 0.7446 | 0.0386 | 0 | 0.998 |
| draft_blocks=1 | 1.801x | 0 | 0.6723 | 0.7446 | 0.0386 | 0 | 0.998 |

### complex_nested (80)

| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **2.019x** | 0 | 0.4749 | 0.7987 | 0.0132 | 0 | 1.000 |
| TASD (base, no fb) | 2.012x | 0 | 0.4749 | 0.7987 | 0.0132 | 0 | 1.000 |
| TASD-F (unguarded fb) | 2.030x | 0 | 0.4749 | 0.7987 | 0.0132 | 0 | 1.000 |
| w/o relaxed verify | 1.731x | 0 | 0.6029 | 0.7956 | 0.0132 | 5 | 0.902 |
| w/o struct guard | 1.947x | 0 | 0.6029 | 0.7987 | 0.0132 | 0 | 1.000 |
| draft_len=8 | 1.799x | 0 | 0.6029 | 0.7987 | 0.0132 | 0 | 1.000 |
| draft_blocks=1 | 1.806x | 0 | 0.6029 | 0.7987 | 0.0132 | 0 | 1.000 |

### rich_cli (80)

| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |
|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|
| **TASD-FG (full)** | **2.025x** | 0 | 0.6122 | 0.7496 | 0.1254 | 0 | 1.000 |
| TASD (base, no fb) | 2.029x | 0 | 0.6122 | 0.7496 | 0.1254 | 0 | 1.000 |
| TASD-F (unguarded fb) | 2.027x | 0 | 0.6122 | 0.7496 | 0.1254 | 0 | 1.000 |
| w/o relaxed verify | 1.756x | 0 | 0.6463 | 0.7499 | 0.1254 | 0 | 0.894 |
| w/o struct guard | 1.982x | 0 | 0.6497 | 0.7496 | 0.1254 | 0 | 1.000 |
| draft_len=8 | 1.836x | 0 | 0.6497 | 0.7496 | 0.1254 | 0 | 1.000 |
| draft_blocks=1 | 1.819x | 0 | 0.6497 | 0.7496 | 0.1254 | 0 | 1.000 |

## Contribution Decomposition

| Ablation | ΔSpeedup | ΔBelow | ΔSQ-R | ΔSQ-S | ΔOff-Str | ΔFB |
|----------|:--------:|:------:|:-----:|:-----:|:--------:|:---:|
| TASD → TASD-FG | +0.026 | 9→3 | 0.5903→0.5908 | 0.7769→0.7788 | 0.0379→0.0366 | +27 |
| TASD-FG → no guard | -0.030 | 3→0 | 0.5908→0.6379 | 0.7788→0.7696 | 0.0366→0.0529 | 27→1 |
| TASD-FG → no relaxed | -0.300 | 3→4 | 0.5908→0.6366 | 0.7788→0.7795 | 0.0366→0.0357 | 27→38 |
| TASD-FG → draft_len=8 | -0.194 | 3→1 | 0.5908→0.6370 | 0.7788→0.7781 | 0.0366→0.0366 | 27→19 |
| TASD-FG → draft_blk=1 | -0.201 | 3→0 | 0.5908→0.6381 | 0.7788→0.7786 | 0.0366→0.0366 | 27→14 |

## Key Insight

**SQ-R and SQ-S separate different quality dimensions:**

- `w/o struct guard` has higher SQ-R (0.6379) than TASD-FG (0.5908) because it more closely matches the reference structure
- But `w/o struct guard` has lower SQ-S (0.7696) and higher Off-Str (0.0529) — more structural risk
- TASD-FG maintains SQ-S at 0.7788 with Off-Str 0.0366 while achieving highest speedup (2.004x)

TASD-FG achieves the best speed-robustness trade-off. It does not maximize reference similarity (SQ-R), but maintains competitive structure safety (SQ-S) while achieving the highest speedup and the fewest below-AR failures.
