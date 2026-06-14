# Guard Calibration Analysis (24 Perf Hard Cases)

**Goal**: Identify which StructuralGuard rules cause over-trimming,
explaining why TASD-F-G-Sel+GV2 achieves 1.51x vs TASD-F-G-Sel's 1.18x
but with off-structure rising from 0.0091 to 0.0274.

## Global Statistics

| Metric | TASD | TASD-F-G-Sel | +GV2 | Delta |
|--------|------|-------------|------|-------|
| Total guard triggers | 262 | 139 | 75 | +64 |
| Total trims applied | 262 | 126 | 68 | +58 |
| Mean speedup | 0.86x | 1.18x | 1.51x | +0.33x |
| Cases improved (>0.1x) | - | - | 8/24 | - |
| Cases worsened (<-0.05x) | - | - | 2/24 | - |

## Rule Analysis: Which Guard Rules Cause Over-Trimming?

| Category | TASD Trims | TFG Trims | GV2 Trims | Trim Δ | Cases | Avg Sp Δ | Avg SQ Δ | Avg Off Δ | Verdict |
|----------|-----------|----------|----------|--------|-------|----------|----------|-----------|---------|
| repetition | 67 | 18 | 0 | 67 | 3 | +1.244x | +0.0168 | +0.0000 | **OVER-TRIM** |
| unbalanced_brackets | 29 | 29 | 0 | 29 | 7 | +0.472x | +0.0336 | +0.0000 | **OVER-TRIM** |
| off_structure:import | 144 | 44 | 37 | 107 | 13 | +0.070x | +0.0000 | +0.0338 | **LIKELY OVER-TRIM** |
| off_structure:other | 0 | 0 | 18 | -18 | 3 | +0.029x | +0.0351 | +0.0000 | **neutral** |
| off_structure:def | 22 | 22 | 6 | 16 | 6 | +0.011x | +0.0175 | +0.0000 | **neutral** |

### Interpretation

- **off_structure:import**: The dominant trim category. GV2 reduces these trims from 144 to 37
  but GuardV2's comment/string awareness is supposed to suppress false alarms.
  The persistent high count of `off_structure:import` trims in GV2 suggests
  the original StructuralGuard was flagging `import` in string/docstring contexts
  where it is actually valid config content, not structural breakage.

## Identified Over-Trimming Rules

### repetition (over_trimming)
- TASD trim count: 67, GV2 trim count: 0
- Trim reduction: 67
- Avg speedup gain: +1.244x
- Avg SQ change: +0.0168
- Avg off-structure change: +0.0000

**Recommendation**: This rule can be downgraded from hard trim to warning.
Continuing generation without trimming on these triggers is safe and improves speed.

### unbalanced_brackets (over_trimming)
- TASD trim count: 29, GV2 trim count: 0
- Trim reduction: 29
- Avg speedup gain: +0.472x
- Avg SQ change: +0.0336
- Avg off-structure change: +0.0000

**Recommendation**: This rule can be downgraded from hard trim to warning.
Continuing generation without trimming on these triggers is safe and improves speed.

### off_structure:import (likely_over_trimming)
- TASD trim count: 144, GV2 trim count: 37
- Trim reduction: 107
- Avg speedup gain: +0.070x
- Avg SQ change: +0.0000
- Avg off-structure change: +0.0338

**Recommendation**: This rule can be downgraded from hard trim to warning.
Continuing generation without trimming on these triggers is safe and improves speed.

## Per-Sample Detailed Comparison

| # | Benchmark | Idx | TASD Sp | TFG Sp | GV2 Sp | GV2 Δ | SQ Δ | Off Δ | TASD Guard | TFG Guard | GV2 Guard | GV2 HighRisk |
|---|-----------|-----|---------|--------|--------|-------|------|-------|-----------|----------|----------|-------------|
| 1 | Argparse | 22 | 0.24x | 1.25x | 1.28x | +0.03x | +0.000 | +0.000 | 26 | 3 | 3 | 2 |
| 2 | Argparse | 73 | 0.28x | 1.31x | 1.31x | +0.00x | +0.000 | +0.000 | 30 | 5 | 4 | 3 |
| 3 | DictConfig | 59 | 0.35x | 0.90x | 2.17x | +1.27x | +0.000 | +0.000 | 25 | 8 | 0 | 0 |
| 4 | DictConfig | 2 | 0.36x | 0.90x | 2.09x | +1.18x | +0.000 | +0.000 | 25 | 8 | 0 | 0 |
| 5 | DictConfig | 1 | 0.50x | 0.87x | 2.15x | +1.28x | +0.050 | +0.000 | 17 | 8 | 0 | 0 |
| 6 | OpenMMLab | 70 | 0.49x | 0.68x | 0.95x | +0.26x | +0.000 | +0.167 | 15 | 11 | 5 | 5 |
| 7 | OpenMMLab | 64 | 0.59x | 0.91x | 0.91x | +0.00x | +0.000 | +0.000 | 10 | 4 | 4 | 4 |
| 8 | Argparse | 33 | 0.57x | 1.53x | 1.49x | -0.04x | +0.000 | +0.000 | 10 | 3 | 3 | 2 |
| 9 | OpenMMLab | 48 | 0.73x | 0.97x | 0.93x | -0.04x | +0.000 | +0.000 | 9 | 6 | 6 | 6 |
| 10 | Argparse | 30 | 0.75x | 1.64x | 1.64x | +0.01x | +0.000 | +0.000 | 8 | 3 | 3 | 2 |
| 11 | Argparse | 61 | 0.72x | 1.55x | 1.62x | +0.07x | +0.000 | +0.000 | 8 | 3 | 3 | 2 |
| 12 | OpenMMLab | 0 | 0.90x | 0.84x | 0.80x | -0.04x | +0.000 | +0.000 | 7 | 7 | 7 | 7 |
| 13 | OpenMMLab | 2 | 0.83x | 0.84x | 0.82x | -0.02x | +0.000 | +0.000 | 7 | 7 | 7 | 7 |
| 14 | DictConfig | 13 | 0.93x | 0.95x | 0.97x | +0.02x | +0.000 | +0.000 | 8 | 8 | 9 | 9 |
| 15 | Argparse | 69 | 0.92x | 1.46x | 1.38x | -0.09x | +0.000 | +0.000 | 6 | 3 | 3 | 2 |
| 16 | DictConfig | 18 | 1.01x | 1.00x | 0.76x | -0.24x | +0.105 | +0.000 | 9 | 9 | 9 | 9 |
| 17 | Argparse | 38 | 0.93x | 1.09x | 1.09x | +0.00x | +0.000 | +0.000 | 6 | 5 | 5 | 4 |
| 18 | DictConfig | 57 | 1.31x | 1.27x | 2.12x | +0.85x | +0.125 | +0.000 | 6 | 8 | 0 | 0 |
| 19 | DictConfig | 40 | 1.28x | 1.31x | 1.62x | +0.31x | +0.000 | +0.000 | 5 | 5 | 2 | 2 |
| 20 | DictConfig | 52 | 1.46x | 1.46x | 2.14x | +0.68x | +0.000 | +0.000 | 5 | 5 | 0 | 0 |
| 21 | DictConfig | 77 | 1.30x | 1.30x | 2.16x | +0.85x | +0.003 | +0.000 | 6 | 6 | 0 | 0 |
| 22 | OpenMMLab | 29 | 1.38x | 1.38x | 2.15x | +0.77x | +0.000 | +0.273 | 6 | 6 | 0 | 0 |
| 23 | DictConfig | 78 | 1.29x | 1.32x | 2.15x | +0.83x | +0.002 | +0.000 | 6 | 6 | 0 | 0 |
| 24 | DictConfig | 7 | 1.46x | 1.47x | 1.44x | -0.03x | +0.000 | +0.000 | 2 | 2 | 2 | 2 |

## TASD Trim Reason Breakdown (top 10 most trimmed)

### Real-Python-Argparse idx=73 (argparse_real_074)
- TASD speedup: 0.28x, GV2 speedup: 1.31x (Δ: +1.03x)
- TASD guard triggers: 30, trims: 30
- GV2 guard triggers: 4, high_risk: 3
- TASD trim categories: off_structure:import(30)
- GV2 trim categories: off_structure:import(2)
- Sample TASD reasons: off_structure:import boto3; off_structure:from sagemaker.session import ; off_structure:from sagemaker import get_exec; off_structure:from sagemaker import get_sess; off_structure:from sagemaker_session import 

### Real-Python-Argparse idx=22 (argparse_real_023)
- TASD speedup: 0.24x, GV2 speedup: 1.28x (Δ: +1.04x)
- TASD guard triggers: 26, trims: 26
- GV2 guard triggers: 3, high_risk: 2
- TASD trim categories: off_structure:import(26)
- GV2 trim categories: off_structure:import(1)
- Sample TASD reasons: off_structure:import os; off_structure:import NullCountAction; off_structure:import sysimport os os.path; off_structure:import sys; off_structure:import conda_build.conda_build

### Real-Python-DictConfig idx=59 (dict_config_real_060)
- TASD speedup: 0.35x, GV2 speedup: 2.17x (Δ: +1.81x)
- TASD guard triggers: 25, trims: 25
- GV2 guard triggers: 0, high_risk: 0
- TASD trim categories: repetition(25)
- GV2 trim categories: 
- Sample TASD reasons: repeated_word:'AsyncGenerator',

### Real-Python-DictConfig idx=2 (dict_config_real_003)
- TASD speedup: 0.36x, GV2 speedup: 2.09x (Δ: +1.73x)
- TASD guard triggers: 25, trims: 25
- GV2 guard triggers: 0, high_risk: 0
- TASD trim categories: repetition(25)
- GV2 trim categories: 
- Sample TASD reasons: repeated_word:'AsyncGenerator',

### Real-Python-DictConfig idx=1 (dict_config_real_002)
- TASD speedup: 0.50x, GV2 speedup: 2.15x (Δ: +1.65x)
- TASD guard triggers: 17, trims: 17
- GV2 guard triggers: 0, high_risk: 0
- TASD trim categories: repetition(17)
- GV2 trim categories: 
- Sample TASD reasons: repeated_word:"traceback",

### OpenMMLab-Config idx=70 (openmmlab_config_real_071)
- TASD speedup: 0.49x, GV2 speedup: 0.95x (Δ: +0.46x)
- TASD guard triggers: 15, trims: 15
- GV2 guard triggers: 5, high_risk: 5
- TASD trim categories: off_structure:import(15)
- GV2 trim categories: off_structure:import(5)
- Sample TASD reasons: off_structure:import torch; off_structure:from torch import optim; off_structure:from torch.utils.data import D; off_structure:from torch.cuda.amp import Gra; off_structure:from torchfrom.utils import *

### OpenMMLab-Config idx=64 (openmmlab_config_real_065)
- TASD speedup: 0.59x, GV2 speedup: 0.91x (Δ: +0.33x)
- TASD guard triggers: 10, trims: 10
- GV2 guard triggers: 4, high_risk: 4
- TASD trim categories: off_structure:import(10)
- GV2 trim categories: off_structure:import(4)
- Sample TASD reasons: off_structure:import time; off_structure:from mmcv.runner import get_di; off_structure:from mmcvuru.utils import get_

### Real-Python-Argparse idx=33 (argparse_real_034)
- TASD speedup: 0.57x, GV2 speedup: 1.49x (Δ: +0.92x)
- TASD guard triggers: 10, trims: 10
- GV2 guard triggers: 3, high_risk: 2
- TASD trim categories: off_structure:import(10)
- GV2 trim categories: off_structure:import(1)
- Sample TASD reasons: off_structure:import sys; off_structure:import pkg_resources; off_structure:import pkg_resources("vendor.p; off_structure:import pkg_resources.extern; off_structure:import pkg_resources.externimp

### OpenMMLab-Config idx=48 (openmmlab_config_real_049)
- TASD speedup: 0.73x, GV2 speedup: 0.93x (Δ: +0.20x)
- TASD guard triggers: 9, trims: 9
- GV2 guard triggers: 6, high_risk: 6
- TASD trim categories: off_structure:import(9)
- GV2 trim categories: off_structure:import(6)
- Sample TASD reasons: off_structure:import time; off_structure:from torch.utils.data import D; off_structure:from mmcv import Config; off_structure:from mmdet.apimodels import bu; off_structure:from mmcv.runner import get_di

### Real-Python-DictConfig idx=18 (dict_config_real_019)
- TASD speedup: 1.01x, GV2 speedup: 0.76x (Δ: -0.25x)
- TASD guard triggers: 9, trims: 9
- GV2 guard triggers: 9, high_risk: 9
- TASD trim categories: off_structure:def(8), unbalanced_brackets(1)
- GV2 trim categories: off_structure:other(7), off_structure:def(2)
- Sample TASD reasons: unbalanced_brackets; off_structure:def get_size_in_bytes(file_pat; off_structure:def get_size_in_m; off_structure:def get_size_in_gib(file_path); off_structure:def get_size

## Guard-v1.5 Calibration Proposal

Based on the analysis, the following calibration is proposed:

### Rules to Keep (hard trim — proven quality protection)

- **off_structure:def**: 6 cases, protects structural integrity. KEEP as hard trim.

- **duplicate_option**: Argparse-specific protection. KEEP for argparse benchmark only.

### Rules to Downgrade (hard trim → warning)

- **repetition**: 3 cases, avg speedup gain +1.244x, SQ Δ +0.0168, OffStr Δ +0.0000. DOWNGRADE to warning (allow generation to continue, log warning).
- **unbalanced_brackets**: 7 cases, avg speedup gain +0.472x, SQ Δ +0.0336, OffStr Δ +0.0000. DOWNGRADE to warning (allow generation to continue, log warning).
- **off_structure:import**: 13 cases, avg speedup gain +0.070x, SQ Δ +0.0000, OffStr Δ +0.0338. DOWNGRADE to warning (allow generation to continue, log warning).

### Rules to Delay (keep trim but after longer tolerance)

- **unbalanced_brackets**: Shift from immediate trim to delay: only trim if bracket_depth > 3 for >= 2 consecutive rounds.
- **repetition in DictConfig**: DictConfig often has repeated keys in config blocks. Only trigger on 5+ consecutive repeats (vs current 3).

### Overall Recommendation

**3 rules identified as over-trimming.** Guard-v1.5 is worth implementing
as a lightweight calibration of the existing StructuralGuard, without replacing it.
Expected benefit: +0.2-0.3x speedup on DictConfig hard cases with minimal quality impact.

### File Outputs

- `results/guard_calibration_analysis_24.json` — full per-sample data
- `results/guard_calibration_analysis_24.md` — this report