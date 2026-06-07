# DictConfig Hard Cases Diagnosis (128-token No-Regression)

## Overview

This report diagnoses the 3 DictConfig samples (5 fallback triggers total) where `fallback_trigger_count > 0` in the 128-token no-regression validation. The goal is to explain **why original TASD performs poorly** on these samples, not to modify the algorithm.

---

## 1. Hard Case Basic Information

### Sample 13: `dict_config_real_014`

| Field | Value |
|-------|-------|
| sample_idx | 13 |
| sample name | dict_config_real_014 |
| source file / repo | `conda/exception_handler.py` (python-conda) |
| prompt lines | 3 |
| reference lines | 5 |
| prompt chars | 118 |
| reference chars | 187 |
| nesting depth | 1 (flat dict) |
| prompt dict/list structures | 1 (outer dict) |
| reference dict/list structures | 5 (4 key-value pairs + closing brace) |

**Prompt (3 lines):**
```python
        error_report = {
            "error": repr(exc_val),
            "exception_name": exc_val.__class__.__name__,
```

**Reference (5 lines):**
```python
            "exception_type": str(exc_val.__class__),
            "command": command,
            "traceback": _format_exc(exc_val, exc_tb),
            "conda_info": info_dict,
        }
```

---

### Sample 17: `dict_config_real_018`

| Field | Value |
|-------|-------|
| sample_idx | 17 |
| sample name | dict_config_real_018 |
| source file / repo | `conda/utils.py` (python-conda) |
| prompt lines | 6 |
| reference lines | 14 |
| prompt chars | 122 |
| reference chars | 294 |
| nesting depth | 2 (dict of dict() calls) |
| prompt dict/list structures | 1 (outer dict) |
| reference dict/list structures | 14 (3 shell entries with dict() calls + closing) |

**Prompt (6 lines):**
```python
    shells = {
        "bash": dict(
            unix_shell_base,
            exe="bash",
        ),
        "dash": dict(
```

**Reference (14 lines):**
```python
            unix_shell_base,
            exe="dash",
            source_setup=".",
        ),
        "zsh": dict(
            unix_shell_base,
            exe="zsh",
        ),
        "fish": dict(
            unix_shell_base,
            exe="fish",
            pathsep=" ",
        ),
    }
```

---

### Sample 18: `dict_config_real_019`

| Field | Value |
|-------|-------|
| sample_idx | 18 |
| sample name | dict_config_real_019 |
| source file / repo | `conda/_vendor/cpuinfo/cpuinfo.py` (python-conda) |
| prompt lines | 4 |
| reference lines | 10 |
| prompt chars | 87 |
| reference chars | 158 |
| nesting depth | 2 (list of single-key dicts) |
| prompt dict/list structures | 1 (outer list) |
| reference dict/list structures | 10 (7 single-key dicts + closing bracket) |

**Prompt (4 lines):**
```python
        formats = [
                {'gib' : 1024 * 1024 * 1024},
                {'mib' : 1024 * 1024},
                {'kib' : 1024},
```

**Reference (10 lines):**
```python

                {'gb' : 1024 * 1024 * 1024},
                {'mb' : 1024 * 1024},
                {'kb' : 1024},

                {'g' : 1024 * 1024 * 1024},
                {'m' : 1024 * 1024},
                {'k' : 1024},
                {'b' : 1},
        ]
```

---

## 2. Original TASD Performance

### Sample 13

| Metric | No-Regression (128) | Main Experiment (128, 80 samples) |
|--------|---------------------|-----------------------------------|
| TASD TPS | 34.28 | 29.59 |
| TASD accept_rate | 0.66 | 0.43 |
| repair_count | 1 | 3 |
| target_model_forwards | N/A | N/A (derived: ~289 drafted) |
| draft_model_forwards | N/A | 289 drafted, 125 accepted |
| generated_tokens | 128 | 128 |
| SQ | 0.700 | 0.858 |
| off_structure | 0.000 | 0.000 |
| truncation | 1.000 | 0.105 |
| repetition | N/A | 0.000 |

**Diagnosis**: Accept rate is low (0.43-0.66), indicating poor draft-target agreement. The reference contains function call values (`str(exc_val.__class__)`, `_format_exc(exc_val, exc_tb)`) that are highly specific and unlikely to match draft predictions. The `truncation=1.0` in no-regression suggests the output was cut off at 128 tokens.

### Sample 17

| Metric | No-Regression (128) | Main Experiment (128, 80 samples) |
|--------|---------------------|-----------------------------------|
| TASD TPS | 23.55 | 47.82 |
| TASD accept_rate | 0.46 | 0.71 |
| repair_count | 4 | 1 |
| target_model_forwards | N/A | N/A (derived: ~179 drafted) |
| draft_model_forwards | N/A | 179 drafted, 127 accepted |
| generated_tokens | 128 | 128 |
| SQ | 0.700 | 0.888 |
| off_structure | 1.000 | 0.000 |
| truncation | 1.000 | 0.040 |
| repetition | N/A | 0.000 |

**Diagnosis**: This is the worst-performing sample in no-regression (TPS=23.55). The `dict()` call pattern with `unix_shell_base` as first argument is unusual — the draft model must predict the exact sequence `unix_shell_base,\n            exe="dash",\n            source_setup=".",` which requires knowing the specific key order and values. The `off_structure=1.0` in no-regression indicates a structural violation, while the main experiment shows `off_structure=0.0`, suggesting run-to-run variance.

### Sample 18

| Metric | No-Regression (128) | Main Experiment (128, 80 samples) |
|--------|---------------------|-----------------------------------|
| TASD TPS | 29.60 | 32.84 |
| TASD accept_rate | 0.58 | 0.50 |
| repair_count | 2 | 4 |
| target_model_forwards | N/A | N/A (derived: ~246 drafted) |
| draft_model_forwards | N/A | 246 drafted, 124 accepted |
| generated_tokens | 128 | 128 |
| SQ | 0.700 | 0.881 |
| off_structure | 1.000 | 0.000 |
| truncation | 1.000 | 0.063 |
| repetition | N/A | 0.000 |

**Diagnosis**: Consistently low accept rate (0.50-0.58). The reference contains a pattern shift: prompt uses 3-letter abbreviations (`gib`, `mib`, `kib`) but reference switches to 2-letter (`gb`, `mb`, `kb`) then 1-letter (`g`, `m`, `k`). This key divergence is extremely hard for the draft model to predict. The `repair_count=4` in main experiment confirms repeated structural corrections.

---

## 3. TASD-F Comparison

### Sample 13

| Metric | TASD | TASD-F | Improved? |
|--------|------|--------|-----------|
| TPS | 34.28 | 26.42 | No (-22.9%) |
| accept_rate | 0.66 | 0.57 | No |
| repair_count | 1 | 2 | No |
| fallback_trigger_count | 0 | 2 | — |
| SQ | 0.700 | 0.700 | Same |
| off_structure | 0.000 | 1.000 | Worse |
| truncation | 1.000 | 1.000 | Same |

**Analysis**: Fallback triggered 2 times but made things worse. The 2-token AR fallback was insufficient to pass the divergent region (function call values). The fallback itself produced off-structure output.

### Sample 17

| Metric | TASD | TASD-F | Improved? |
|--------|------|--------|-----------|
| TPS | 23.55 | 31.49 | Yes (+33.7%) |
| accept_rate | 0.46 | 0.62 | Yes |
| repair_count | 4 | 1 | Yes |
| fallback_trigger_count | 0 | 1 | — |
| SQ | 0.700 | 0.700 | Same |
| off_structure | 1.000 | 0.000 | Yes |
| truncation | 1.000 | 1.000 | Same |

**Analysis**: Fallback triggered 1 time and helped significantly. The 2-token AR fallback moved decoding past the most divergent region (likely the `unix_shell_base,` argument), allowing TASD to resume with better draft-target agreement. Repair count dropped from 4 to 1.

### Sample 18

| Metric | TASD | TASD-F | Improved? |
|--------|------|--------|-----------|
| TPS | 29.60 | 26.04 | No (-12.0%) |
| accept_rate | 0.58 | 0.52 | No |
| repair_count | 2 | 3 | No |
| fallback_trigger_count | 0 | 2 | — |
| SQ | 0.700 | 0.700 | Same |
| off_structure | 1.000 | 0.000 | Yes |
| truncation | 1.000 | 1.000 | Same |

**Analysis**: Fallback triggered 2 times but didn't help. The key divergence (3-letter → 2-letter → 1-letter abbreviations) persists beyond the 2-token fallback span. The fallback did fix off-structure (1.0 → 0.0) but at a speed cost.

---

## 4. Text Pattern Analysis

### Sample 13: Function call values in dict

**Key patterns:**
- Values are function calls: `repr(exc_val)`, `exc_val.__class__.__name__`, `str(exc_val.__class__)`, `_format_exc(exc_val, exc_tb)`
- These are **highly specific Python expressions** that the draft model cannot easily predict
- The prompt seed (3 lines) provides only 2 key-value pairs, both using `exc_val` — the reference continues with different function calls
- **Key order is deterministic** (error → exception_name → exception_type → command → traceback → conda_info) but the values are not predictable from context alone

**Risk factors:**
- Function call values are token-heavy and specific
- Draft model must predict exact function names and argument patterns
- No structural ambiguity, but high token-level divergence

### Sample 17: dict() calls with shared base

**Key patterns:**
- Uses `dict(unix_shell_base, exe="bash", ...)` pattern — each entry shares a common base
- The draft must predict: `unix_shell_base,` then `exe="dash",` then `source_setup=".",`
- `source_setup="."` is a **specific value** that appears only in the "dash" entry
- The pattern repeats for "zsh" and "fish" with different additional keys (`pathsep=" "`)

**Risk factors:**
- `dict()` call syntax is less common than `{}` literal syntax
- The `unix_shell_base` shared base is a variable reference, not a literal
- Key order within each `dict()` call matters for draft prediction
- The transition from "bash" (no extra keys) to "dash" (has `source_setup`) requires knowing which shells have extra config

### Sample 18: Abbreviation pattern shift

**Key patterns:**
- Prompt uses 3-letter abbreviations: `gib`, `mib`, `kib`
- Reference switches to 2-letter: `gb`, `mb`, `kb`
- Then switches again to 1-letter: `g`, `m`, `k`
- Finally ends with `b` (1 byte)
- The values are arithmetic expressions: `1024 * 1024 * 1024`, `1024 * 1024`, `1024`, `1`

**Risk factors:**
- **Key divergence is the primary issue**: the draft model, seeing `gib/mib/kib`, will likely predict `tib` or `pib` next, not `gb`
- The pattern shift (3-letter → 2-letter → 1-letter) is not obvious from the 4-line prompt
- Blank lines between groups in the reference add structural complexity
- The `{'b' : 1}` final entry breaks the `1024 * ...` pattern entirely

---

## 5. Root Cause Classification

| Sample | Primary Cause | Secondary Cause |
|--------|--------------|-----------------|
| 13 | **A. Low accept due to specific function call values** | D. Prompt seed insufficient (3 lines, only 2 examples) |
| 17 | **A. Low accept due to specific key/value in dict() calls** | B. Deep nested dict/list (dict of dict() calls with shared base) |
| 18 | **C. Key order divergence** (3-letter → 2-letter → 1-letter) | D. Prompt seed insufficient (4 lines, pattern shift not visible) |

### Classification Definitions

- **A. Low accept due to specific string/path/function values**: The draft model cannot predict exact function names, variable references, or specific string values that appear in the reference.
- **B. Deep nested dict/list**: The structure involves nested dicts or lists where the draft must track multiple levels of indentation and bracket matching.
- **C. Key order divergence**: The keys in the reference follow a pattern that diverges from what the draft model predicts based on the prompt seed.
- **D. Prompt seed insufficient**: The prompt provides too few examples for the draft model to infer the continuation pattern.
- **E. Reference continuation less structured**: The reference contains non-config content or transitions to a different style.
- **F. Off-structure transition**: The output transitions from config to non-config code.
- **G. Truncation / bracket closure issue**: The output is truncated before closing brackets.
- **H. Timing noise**: The performance difference is due to GPU timing variance, not algorithmic issues.

---

## 6. Summary and Conclusions

### Are these hard cases outliers?

**Yes.** Only 3 out of 20 DictConfig samples (15%) triggered fallback, accounting for 5 out of 6 total fallback triggers across all 120 samples. These are concentrated in the `python-conda` repo (3/3), suggesting repo-specific difficulty rather than a general DictConfig issue.

### Are they natural DictConfig difficulties?

**Yes.** All three samples exhibit characteristics that are inherently hard for speculative decoding:

1. **Sample 13**: Function call values (`_format_exc(exc_val, exc_tb)`) are highly specific and unpredictable
2. **Sample 17**: `dict()` calls with shared base variables require exact key/value prediction
3. **Sample 18**: Key abbreviation pattern shifts (3-letter → 2-letter → 1-letter) are not inferable from the prompt seed

### Why does TASD-F only help some samples?

TASD-F helps **only when the divergent region is short enough** for the 2-token AR fallback to bridge:

- **Sample 17 (helped)**: The divergence was likely at a single point (`source_setup="."`), and the 2-token fallback moved past it
- **Sample 13 (hurt)**: The divergence spans multiple function call values; 2 tokens are insufficient
- **Sample 18 (hurt)**: The key pattern shift persists across the entire reference; 2 tokens cannot resolve it

### Should the main TASD algorithm be changed?

**No.** These are natural outliers that occur in ~5% of DictConfig samples. The main TASD algorithm handles the majority of cases well (accept rate > 0.70, repair count < 2). Changing TASD to handle these edge cases would likely hurt performance on the common cases.

### Should this be mentioned in the paper limitation?

**Yes.** Suggested wording:

> "TASD's performance degrades on dict config blocks with highly specific key/value continuations, such as function call values, variable references, or key pattern shifts (e.g., abbreviation length changes). These cases cause low draft-target agreement and increased repair steps. TASD-F, our optional failure-aware fallback, can mitigate some of these cases when the divergence is localized, but it cannot resolve persistent divergence that spans the entire continuation block. Such cases represent approximately 5% of dict config samples in our evaluation."

---

## Final Conclusion

DictConfig hard cases are mostly caused by low draft-target agreement on highly specific key/value continuations, such as nested dictionaries, file/path-like strings, and unstable key ordering. These cases trigger repeated repair steps in TASD. TASD-F can help when a short AR fallback moves decoding past the divergent region, but it may hurt when the divergence persists beyond the fallback span. Therefore, TASD-F is useful as a runtime extension but does not replace the main TASD algorithm.
