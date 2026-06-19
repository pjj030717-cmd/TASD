# TASD-FGQ No-Regression Pilot Report

## Summary
- **Samples**: 60 score=2 samples
- **Score=2 retention rate**: 100.0% (13/13)
- **TASD-FG avg speedup**: 0.475x
- **TASD-FGQ avg speedup**: 0.447x
- **Speed loss**: 5.9%

## Acceptance Criteria
- [ ] Score=2 retention >= 90%: PASS
- [ ] Speed loss <= 5%: FAIL

## Score Distribution

| Score | TASD-FG | TASD-FGQ |
|:-----:|:-------:|:--------:|
| 2 | 13 | 13 |
| 1 | 2 | 2 |
| 0 | 45 | 45 |

## Quality Guard Triggers
- rep_trim_count: 0
- strict_rounds: 0
- low_progress_repairs: 0
- total_trimmed_tokens: 0

## Per-Benchmark Results

| Benchmark | N | FG speedup | FGQ speedup | Speed loss | Score=2 retained |
|-----------|:-:|:----------:|:-----------:|:----------:|:----------------:|
| argparse | 10 | 0.471x | 0.485x | -2.9% | 0/0 |
| complex_nested_config | 10 | 0.476x | 0.418x | 12.2% | 1/1 |
| dict_config | 10 | 0.459x | 0.401x | 12.5% | 7/7 |
| openmmlab_config | 10 | 0.472x | 0.428x | 9.4% | 0/0 |
| pipeline_stage_config | 10 | 0.492x | 0.473x | 4.0% | 0/0 |
| rich_cli_option_groups | 10 | 0.478x | 0.475x | 0.7% | 5/5 |

## Conclusion
TASD-FGQ fails no-regression criteria. Do NOT proceed.
