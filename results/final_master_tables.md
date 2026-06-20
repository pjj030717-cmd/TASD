# TASD: Final Paper Results

**Samples**: 480 total

## Table 1. Main Results: Speed, Robustness, Quality

| Method | Speedup | Eff. TPS | Below-AR | Worst-10 | Score 2 | Score 1 | Score 0 | Recoverable |
|--------|:------:|:--------:|:--------:|:--------:|:------:|:------:|:------:|:----------:|
| AR | 1.00x | 33.2 | 0 | 1.00 | 251 | 155 | 74 | 406/480 (84.6%) |
| GSD | 0.66x | 22.0 | 95 | 0.66 | 238 | 141 | 101 | 379/480 (79.0%) |
| N-gram SD | 1.41x | 46.9 | 198 | 0.67 | 156 | 209 | 115 | 365/480 (76.0%) |
| FLY | 1.64x | 54.5 | 72 | 0.95 | 286 | 102 | 92 | 388/480 (80.8%) |
| TASD-FG | 2.00x | 66.4 | 187 | 0.56 | 192 | 156 | 132 | 348/480 (72.5%) |
| TASD-FG-BR | 1.87x | 62.0 | 130 | — | 227 | 178 | 75 | 405/480 (84.4%) |
| TASD-FG-V | 1.75x | 58.2 | 88 | — | 259 | 178 | 43 | 437/480 (91.0%) |

## Table 2. Per-Benchmark Results

| Benchmark | AR Recov. | FLY Recov. | TASD-FG Recov. | TASD-FG-BR Recov. | TASD-FG-V Recov. | BR Speedup |
|-----------|:---------:|:----------:|:--------------:|:-----------------:|:----------------:|:----------:|
| argparse | 91.2% | 95.0% | 80.0% | 97.5% | 97.5% | 1.78x |
| dict_config | 63.7% | 48.8% | 78.8% | 85.0% | 87.5% | 1.89x |
| openmmlab | 92.5% | 82.5% | 71.2% | 86.2% | 95.0% | 1.85x |
| pipeline | 100.0% | 100.0% | 80.0% | 86.2% | 97.5% | 1.94x |
| complex_nested | 63.7% | 60.0% | 56.2% | 67.5% | 73.8% | 1.86x |
| rich_cli | 96.2% | 98.8% | 68.8% | 83.8% | 95.0% | 1.88x |

## Table 3. Ablation: TASD Variants

| Variant | Speedup | Below-AR | Worst-10 | SQ-R | SQ-S | Off-Str | Rep-Rate | Trunc |
|---------|:------:|:--------:|:--------:|:----:|:----:|:-------:|:--------:|:-----:|
| TASD-FG (full) | 2.00x | 3 | 1.66 | 0.591 | 0.779 | 0.037 | 0.040 | 0.781 |
| TASD (w/o FG) | 1.98x | 9 | 1.49 | 0.590 | 0.777 | 0.038 | 0.043 | 0.783 |
| TASD-F (w/o G) | 2.00x | 9 | 1.54 | 0.592 | 0.771 | 0.048 | 0.047 | 0.783 |
| w/o relaxed | 1.70x | 4 | 1.35 | 0.637 | 0.779 | 0.036 | 0.039 | 0.781 |
| w/o guard | 1.97x | 0 | 1.81 | 0.638 | 0.770 | 0.053 | 0.045 | 0.783 |
| draft_len=8 | 1.81x | 1 | 1.56 | 0.637 | 0.778 | 0.037 | 0.039 | 0.785 |
| draft_blocks=1 | 1.80x | 0 | 1.57 | 0.638 | 0.779 | 0.037 | 0.039 | 0.783 |

## Table 4. Rerun Policy Ablation

| Policy | Rerun Ratio | Speedup | Recoverable | Score 0 | Decision |
|--------|:----------:|:------:|:----------:|:------:|----------|
| No rerun / TASD-FG | 0.0% | 2.00x | 72.5% | 132 | baseline |
| BR / bracket only | 13.5% | 1.87x | 84.4% | 75 | **adopt** |
| D: bracket + repetition | 20.6% | 1.80x | 88.1% | 57 | not adopted |
| F: severe only (higher thresholds) (V1) | 25.4% | 1.75x | 91.0% | 43 | quality-first |
| Oracle top-K | 8.3% | 1.92x | 80.8% | 92 | theoretical upper bound |

## Table 5. Failed Quality Repair Attempts (Supplementary)

| Attempt | Goal | Result | Decision |
|---------|------|--------|----------|
| FGQ | in-loop quality guard | no quality gain, speed loss | reject |
| Safe-k | conservative acceptance | small gain, clean regression | reject |
| Early stopping | avoid bad tails | no improvement | reject |
| Prompt suffix | reduce off-structure | harms clean samples | reject |
| OffStruct constraint | reduce off-structure | low score gain | reject |
| Partial repair (VR) | reduce rerun cost | repair cost too high (0.91) | reject |

## Table 6. Length / Generalization Results (Supplementary)

### 6.1 Qwen 256-token (3x40)

| Method | Speedup | SQ-R | SQ-S | Below-AR | Trunc |
|--------|:------:|:----:|:----:|:--------:|:-----:|
| AR | 1.00x | 0.644 | 0.784 | 0 | 0.81 |
| GSD | 1.82x | 0.645 | 0.677 | 0 | 0.88 |
| Ngram | 1.61x | 0.532 | 0.777 | 53 | 0.83 |
| FLY | 1.54x | 0.674 | 0.776 | 19 | 0.82 |
| TASD-FG | 2.00x | 0.643 | 0.677 | 0 | 0.89 |

### 6.2 LLaMA-3.1-8B 128-token (6x40)

| Method | Speedup | SQ-R | SQ-S | Below-AR | Trunc |
|--------|:------:|:----:|:----:|:--------:|:-----:|
| AR | 1.00x | 0.623 | 0.819 | 0 | 0.69 |
| GSD | 1.44x | 0.627 | 0.799 | 0 | 0.72 |
| Ngram | 1.22x | 0.493 | 0.786 | 121 | 0.83 |
| FLY | 0.99x | 0.566 | 0.787 | 122 | 0.74 |
| TASD-FG | 1.68x | 0.628 | 0.794 | 0 | 0.75 |

## Figure 4. Method Pipeline Diagram

```
Prompt
  |
  v
TASD-FG Decode
  |
  v
Reference-free Structural Verifier
  |
  +-- low risk --> return TASD-FG output
  |
  +-- high risk --> AR Rerun --> return repaired output

Verifier variants:
  BR: bracket_balance < 0.50 AND not truncated
  V:  bracket_balance + repetition + off_structure + duplicate_option
```

## Method Section: TASD Variant Descriptions

```text
TASD-FG is the base speed-first decoder. We further define two
reference-free verification modes. TASD-FG-BR reruns AR only when
the generated output has non-truncated bracket imbalance
(bracket_balance < 0.50). TASD-FG-V uses a broader structural risk
detector including bracket imbalance, repetition, off-structure
drift, and duplicate options.
```

## Conclusion

- **TASD-FG**: fastest (2.00x), suitable for speed-first scenarios
- **TASD-FG-BR**: balanced operating point (1.87x), beats FLY on both speed and recoverability
- **TASD-FG-V**: quality-first (1.31x), recoverable exceeds AR while still faster than AR

TASD offers configurable operating points along the speed-quality frontier.
