# TASD-FG Automatic Structural Recoverability Score

## Scoring Criteria

Each sample is scored 0/1/2 based on deterministic structural metrics. Truncation is reported separately and is not used as a standalone failure criterion, because fixed-budget generation naturally truncates long references.

| Score | Name | Criteria |
|------:|------|----------|
| **2** | Structurally clean | structural_f1 >= 0.85, repetition_rate < 0.08, off_structure < 0.02, bracket balanced (if not truncated) |
| **1** | Recoverable | No severe off-structure/repetition/low-F1; main structure still identifiable |
| **0** | Unrecoverable | Severe repetition (>=0.40), severe off-structure (>=0.20), very low F1 (<0.30), or multiple severe errors |

**Error tags** (reported separately, do not affect score):
- BRACKET: bracket/quote/colon structural symbol errors
- OFF_STRUCT: drifted to unrelated structure (def/class/import)
- REPEAT: obvious repetition patterns
- TRUNC: output truncated (reported separately, not a failure criterion)
- DRIFT: structure present but content deviates from task

## Overall Results

| Score | Count | Percentage |
|:-----:|:-----:|:----------:|
| 2 | 192 | 40.0% |
| 1 | 156 | 32.5% |
| 0 | 132 | 27.5% |

**Truncation (reported separately):** 375/480 (78.1%) samples are truncated due to fixed 128-token budget. Truncation does not affect the recoverability score.

## Per-Benchmark Results

| Benchmark | Score 2 | Score 1 | Score 0 | Usable (1+2) |
|-----------|:-------:|:-------:|:-------:|:------------:|
| argparse | 43 | 21 | 16 | 80.0% |
| complex_nested_config | 13 | 32 | 35 | 56.2% |
| dict_config | 35 | 28 | 17 | 78.8% |
| openmmlab_config | 42 | 15 | 23 | 71.2% |
| pipeline_stage_config | 48 | 16 | 16 | 80.0% |
| rich_cli_option_groups | 11 | 44 | 25 | 68.8% |

## Below-AR Analysis

- Below-AR samples (3): score 2=1, score 1=2, score 0=0
- Non-below-AR samples (477): score 2=191, score 1=154, score 0=132

## Error Tag Distribution

| Tag | Count | Percentage |
|-----|:-----:|:----------:|
| BRACKET | 350 | 72.9% |
| OFF_STRUCT | 34 | 7.1% |
| REPEAT | 47 | 9.8% |
| TRUNC | 375 | 78.1% |
| DRIFT | 97 | 20.2% |

## Score 0 Sample Details (first 10)

### argparse_real_001 (argparse)
- Speedup: 1.742x
- Composite SQ: 0.6102
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.775, bb=0.000, trunc=0.0

### argparse_real_004 (argparse)
- Speedup: 1.995x
- Composite SQ: 0.5498
- Error tags: BRACKET
- Metrics: rep=0.071, off=0.000, f1=0.678, bb=0.000, trunc=0.0

### argparse_real_006 (argparse)
- Speedup: 2.092x
- Composite SQ: 0.6733
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.933, bb=0.000, trunc=0.0

### argparse_real_010 (argparse)
- Speedup: 2.113x
- Composite SQ: 0.6959
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.990, bb=0.000, trunc=0.0

### argparse_real_012 (argparse)
- Speedup: 2.052x
- Composite SQ: 0.6406
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.852, bb=0.000, trunc=0.0

### argparse_real_018 (argparse)
- Speedup: 2.098x
- Composite SQ: 0.6512
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.878, bb=0.000, trunc=0.0

### argparse_real_019 (argparse)
- Speedup: 1.587x
- Composite SQ: 0.6860
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.965, bb=0.000, trunc=0.0

### argparse_real_029 (argparse)
- Speedup: 1.795x
- Composite SQ: 0.6960
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.990, bb=0.000, trunc=0.0

### argparse_real_030 (argparse)
- Speedup: 1.514x
- Composite SQ: 0.6500
- Error tags: TRUNC, DRIFT
- Metrics: rep=0.000, off=0.000, f1=0.125, bb=1.000, trunc=1.0

### argparse_real_031 (argparse)
- Speedup: 1.179x
- Composite SQ: 0.4268
- Error tags: BRACKET
- Metrics: rep=0.000, off=0.000, f1=0.817, bb=0.000, trunc=0.0

