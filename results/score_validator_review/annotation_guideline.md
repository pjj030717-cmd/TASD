# Score Validator — Human Annotation Guideline

## Purpose

Verify whether the **automatic structural recoverability score** (0/1/2) agrees with independent human judgment. This is NOT a method comparison. Do NOT try to guess which method produced the output.

## Scoring Standard

Evaluate the **raw** continuation as-is. Do NOT ignore truncation errors.

| Score | Name | Criteria |
|------:|------|----------|
| **2** | Directly usable | Clean continuation. No structural or syntax errors. No edits needed. Incomplete comments that do NOT affect code structure are acceptable. |
| **1** | Locally recoverable | Main content is reasonable. 1-2 local edits recover it: delete a truncated tail, fix one bracket, remove one duplicate field, fix one indentation error. |
| **0** | Unrecoverable | Needs >2 independent edits or a full rewrite. Severe repetition, off-topic content, wrong structure, or chaotic output. |

Key rules:
- Token limit truncation does NOT automatically deduct score.
- But if truncation causes a real syntax error, score by repair cost.
- A "one-edit" fix means a single contiguous change (e.g., trim one tail, or add one closing bracket).
- Score what you see — do not guess what the model "meant to write."

## Completion Status (REQUIRED per item)

| Status | Meaning |
|--------|---------|
| `complete` | Ends at natural boundary (complete statement, code block, or file). |
| `tail_cutoff` | Clear continuous incomplete tail at end. A meaningful prefix remains. |
| `severe_incomplete` | Major structures unfinished. No valid prefix can be identified. |

Completion status does NOT directly determine score. A `tail_cutoff` sample can still be score 2 if the prefix is perfect and no repair is needed.

## Issue Tags (Select at least one per item)

- `bracket_or_delimiter`: Unbalanced or missing brackets, parentheses, or delimiters
- `indentation`: Wrong indentation
- `repetition`: Repeated lines or blocks
- `duplicate_field`: Same field/option appears twice
- `off_structure`: Shifted to a different structure type (e.g., config became Python function)
- `wrong_content`: Content is factually or contextually wrong
- `other`: Other issues not covered above
- `none`: No issues found (mutually exclusive with all other tags)

If you select `none`, you cannot select any other tag.

## Process

1. Read the **6 calibration examples** together (calibration_examples.html).
2. **Do NOT discuss any specific item from the 90-set during independent annotation.**
3. Open your assigned HTML file.
4. For each of the 90 items:
   - Read the prompt and continuation
   - Select a score (0/1/2)
   - Select a completion status
   - Select at least one issue tag
   - Optionally add notes
5. Click "Save Progress" periodically.
6. When done, click "Export JSON" and send the file back.

## Important

- You are NOT comparing AR vs FLY vs TASD-BR.
- You are NOT guessing which method produced the output.
- You are judging structural quality and completeness.
- Score by repair difficulty, not by token count.
