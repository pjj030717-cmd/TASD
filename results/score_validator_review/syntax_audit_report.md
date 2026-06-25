# Independent Syntax Audit Report

## Summary

- **Total items**: 90
- **Applicable for AST parsing** (prompt is complete Python unit): 1 (1.1%)
- **Not applicable** (prompt is file fragment/partial): 89 (98.9%)
- **Full text parses successfully**: 2 (2.2%)
- **Items with text anomalies**: 7 (7.8%)

## Tail Trim Results (applicable samples only)

- `not_applicable`: 89
- `trim_parse_fail`: 1
  (Based on 1 applicable samples)

## Text Anomaly Types

- `ngram_repetition_6`: 5
- `ngram_repetition_8`: 1
- `markdown_fence`: 1
- `role_marker`: 1
- `low_python_structure`: 1

## Per-Benchmark Breakdown

| Benchmark | Total | Applicable | Parse OK | Trim Fail | Anomalies |
|-----------|------:|-----------:|---------:|----------:|----------:|
| argparse | 10 | 0 | 0 | 0 | 1 |
| complex_nested_config | 20 | 0 | 0 | 0 | 2 |
| dict_config | 19 | 0 | 0 | 0 | 1 |
| openmmlab_config | 12 | 0 | 0 | 0 | 0 |
| pipeline_stage_config | 13 | 0 | 2 | 0 | 0 |
| rich_cli_option_groups | 16 | 1 | 0 | 1 | 3 |

## Per-Score Breakdown (Diagnostic Only)

### By Round-1 Human Score

| Score | Total | Applicable | Parse OK | Anomalies |
|-------|------:|-----------:|---------:|----------:|
| 0 | 1 | 0 | 0 | 0 |
| 1 | 79 | 1 | 0 | 6 |
| 2 | 10 | 0 | 2 | 1 |

### By Automatic Score

| Score | Total | Applicable | Parse OK | Anomalies |
|-------|------:|-----------:|---------:|----------:|
| 0 | 30 | 1 | 0 | 4 |
| 1 | 30 | 0 | 0 | 1 |
| 2 | 30 | 0 | 2 | 2 |

## Important Caveats

1. **AST parse is NOT ground truth for human scoring.** Many prompts are file fragments
   (partial code blocks, REPL snippets) and `not_applicable` is expected and acceptable.
2. A `not_applicable` sample may still be a perfectly valid score-2 continuation if the
   truncated file fragment is internally consistent.
3. Tail trim only deletes from the END of the continuation; it does not fix mid-text errors.
4. This audit is a diagnostic tool to contextualize intra-rater disagreement patterns,
   not to replace human judgment.
