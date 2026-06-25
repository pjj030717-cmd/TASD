# Score Validator — Retest Annotation Guideline

## Purpose

You have already scored 90 items. This is a **30-item retest** drawn from the same set.
The goal is to measure **intra-rater reliability** — how consistent your scoring is
across two independent passes.

## Timing Requirements

- **Minimum interval: 7 days** after completing the first 90-item round.
- Do NOT review your round-1 scores during this interval.
- Do NOT discuss specific samples with anyone.
- Score independently as if seeing these items for the first time.

## Scoring Rules (identical to round 1)

### Structural Recoverability Score (0/1/2)

| Score | Name | Criteria |
|------:|------|----------|
| **2** | Directly usable | Clean continuation. No structural edits needed. |
| **1** | Locally recoverable | 1-2 local edits recover it (trim tail, fix bracket, remove dup). |
| **0** | Unrecoverable | >2 edits, major rewrite, or chaotic output. |

Evaluate the raw output as-is. Token limit truncation does NOT automatically deduct score.

### Completion Status (REQUIRED)

- `complete`: Ends at natural boundary
- `tail_cutoff`: Clear continuous incomplete tail at end
- `severe_incomplete`: Major structures unfinished; no valid prefix

### Issue Tags (at least one required)

- `bracket_or_delimiter`, `indentation`, `repetition`, `duplicate_field`
- `off_structure`, `wrong_content`, `other`, `none` (excludes others)

## Process

1. Open `annotator_A_retest30.html`
2. Score all 30 items independently
3. Click "Save Progress" periodically
4. Export JSON when done → `annotations_A_retest30.json`

Do NOT look up your previous round-1 scores.
