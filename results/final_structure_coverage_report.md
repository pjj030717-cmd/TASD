# Final Structure Coverage Scan Report

**Source files scanned**: 20764
**Structure types assessed**: 9

---

## 1. Overall Coverage Table

| Structure Type | Raw | Valid | Files | Repos | Avg Block Lines | Avg Seed | Avg Ref Struct | Suitability |
|----------------|-----|-------|-------|-------|-----------------|----------|----------------|-------------|
| Argparse Option Blocks | 247 | 227 | 159 | 42 | 36.03 | 3.62 | 2.69 | High |
| Rich CLI Option Groups | 115 | 115 | 88 | 32 | 63.97 | 2.0 | 8.65 | High |
| Dict Config Blocks | 2733 | 2713 | 1596 | 103 | 49.11 | 4.51 | 44.61 | Medium-High |
| Complex Nested Configs | 1290 | 1017 | 661 | 64 | 88.0 | 6.18 | 81.82 | High |
| OpenMMLab Configs | 5277 | 5222 | 2186 | 18 | 18.95 | 4.43 | 14.53 | High |
| Pipeline & Stage Configs | 1347 | 1105 | 929 | 6 | 15.56 | 2.12 | 3.83 | High |
| Schema/Model Fields (SQLAlchemy/DRF) | 238 | 126 | 70 | 8 | 6.32 | 4.1 | 2.24 | Low |
| Model Fields (Pydantic/Dataclass) | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | Low ??? 0 candidates found in source pool (common in web frameworks, not in this pool) |
| Pytest Parametrize Decorators | 1118 | 1118 | 217 | 16 | 11.66 | 1.0 | 10.66 | Low |

---

## 2. Valid Benchmark-Ready Structures

These structures have sufficient valid candidates and are suitable for TASD evaluation.

### Argparse Option Blocks (`argparse`)

- **Raw candidates**: 247
- **Valid candidates**: 227 (from 247 unique)
- **Source files**: 159
- **Repos/packages**: 42
- **Avg block lines**: 36.03
- **Avg prompt lines**: 17.8 (639.51 chars)
- **Avg reference lines**: 18.23 (669.85 chars)
- **Avg nesting depth**: 0.0
- **Avg seed count**: 3.62
- **Avg ref structure count**: 2.69
- **Filtered — invalid seed**: 8
- **Filtered — invalid reference**: 4
- **Filtered — reference too short**: 8
- **Duplicates removed**: 0
- **Suitability**: High — repeated skeleton, rich fields, Guard detectable

### Rich CLI Option Groups (`rich_cli_option_groups`)

- **Raw candidates**: 115
- **Valid candidates**: 115 (from 115 unique)
- **Source files**: 88
- **Repos/packages**: 32
- **Avg block lines**: 63.97
- **Avg prompt lines**: 10.57 (375.25 chars)
- **Avg reference lines**: 53.41 (1936.54 chars)
- **Avg nesting depth**: 0.0
- **Avg seed count**: 2.0
- **Avg ref structure count**: 8.65
- **Filtered — invalid seed**: 0
- **Filtered — invalid reference**: 0
- **Filtered — reference too short**: 0
- **Duplicates removed**: 0
- **Suitability**: High — complex multi-option groups, Guard benefits

### Dict Config Blocks (`dict_config`)

- **Raw candidates**: 2733
- **Valid candidates**: 2713 (from 2733 unique)
- **Source files**: 1596
- **Repos/packages**: 103
- **Avg block lines**: 49.11
- **Avg prompt lines**: 7.99 (170.53 chars)
- **Avg reference lines**: 88.19 (1946.5 chars)
- **Avg nesting depth**: 2.75
- **Avg seed count**: 4.51
- **Avg ref structure count**: 44.61
- **Filtered — invalid seed**: 0
- **Filtered — invalid reference**: 0
- **Filtered — reference too short**: 20
- **Duplicates removed**: 0
- **Suitability**: Medium-High — key/value repetition, nested structures

### Complex Nested Configs (`complex_nested_config`)

- **Raw candidates**: 1290
- **Valid candidates**: 1017 (from 1290 unique)
- **Source files**: 661
- **Repos/packages**: 64
- **Avg block lines**: 88.0
- **Avg prompt lines**: 11.31 (287.57 chars)
- **Avg reference lines**: 162.59 (3568.75 chars)
- **Avg nesting depth**: 4.13
- **Avg seed count**: 6.18
- **Avg ref structure count**: 81.82
- **Filtered — invalid seed**: 0
- **Filtered — invalid reference**: 0
- **Filtered — reference too short**: 273
- **Duplicates removed**: 0
- **Suitability**: High — deep nesting, TASD Guard has clear advantage

### OpenMMLab Configs (`openmmlab_config`)

- **Raw candidates**: 5277
- **Valid candidates**: 5222 (from 5277 unique)
- **Source files**: 2186
- **Repos/packages**: 18
- **Avg block lines**: 18.95
- **Avg prompt lines**: 7.85 (123.9 chars)
- **Avg reference lines**: 28.05 (473.17 chars)
- **Avg nesting depth**: 0.0
- **Avg seed count**: 4.43
- **Avg ref structure count**: 14.53
- **Filtered — invalid seed**: 0
- **Filtered — invalid reference**: 0
- **Filtered — reference too short**: 55
- **Duplicates removed**: 0
- **Suitability**: High — repeated config blocks, dense structure

### Pipeline & Stage Configs (`pipeline_stage_config`)

- **Raw candidates**: 1347
- **Valid candidates**: 1105 (from 1347 unique)
- **Source files**: 929
- **Repos/packages**: 6
- **Avg block lines**: 15.56
- **Avg prompt lines**: 4.07 (127.0 chars)
- **Avg reference lines**: 11.49 (389.78 chars)
- **Avg nesting depth**: 0.0
- **Avg seed count**: 2.12
- **Avg ref structure count**: 3.83
- **Filtered — invalid seed**: 1
- **Filtered — invalid reference**: 190
- **Filtered — reference too short**: 51
- **Duplicates removed**: 0
- **Suitability**: High — repeated stage skeletons with type/name/params

---

## 3. Boundary / Less Suitable Structures

These structures were scanned but are less suitable for TASD. They are kept for boundary analysis.

### Schema/Model Fields (SQLAlchemy/DRF) (`schema_fields`)

- **Raw candidates**: 238
- **Valid candidates**: 126 (from 238 unique)
- **Source files**: 70
- **Repos/packages**: 8
- **Avg block lines**: 6.32
- **Avg reference lines**: 2.91
- **Suitability**: Low — fields are short, single-line, baseline already good

### Model Fields (Pydantic/Dataclass) (`model_fields`)

- **Raw candidates**: 0
- **Valid candidates**: 0 (from 0 unique)
- **Source files**: 0
- **Repos/packages**: 0
- **Avg block lines**: 0.0
- **Avg reference lines**: 0.0
- **Suitability**: Low ??? fields are short, single-line, baseline already good; 0 found in source pool

### Pytest Parametrize Decorators (`pytest_parametrize`)

- **Raw candidates**: 1118
- **Valid candidates**: 1118 (from 1118 unique)
- **Source files**: 217
- **Repos/packages**: 16
- **Avg block lines**: 11.66
- **Avg reference lines**: 10.66
- **Suitability**: Low — nested strings/test values vary widely, Guard error-prone

---

## 4. Conclusions

### Coverage Scope

TASD targets 6 structure types. These represent a significant portion 
of structured code completion opportunities in real Python codebases, 
but TASD does **not** cover all code completion scenarios.

### What TASD Covers Well

- **Medium-to-high complexity structures** with repeated skeletons
- **Config/pipeline/CLI blocks** where reference is long enough (5+ lines)
- Structures where **Guard can detect** off-structure tokens (def/class/import)
- Blocks with **clear structural boundaries** (dict/list scope)

### What TASD Does NOT Cover (Boundary Report)

- **Fields-type structures** (`schema_fields`, `model_fields`): Blocks are too short, 
  usually single-line per field. AR baseline already performs well. 
  TASD's multi-token draft advantage does not apply.
- **pytest_parametrize**: Decorator structure is shallow; test values 
  vary widely (strings, tuples, floats). Guard rules are error-prone 
  on nested strings and expressions.

### Selection Bias Safeguards

- Samples were **not** selected based on TASD results
- Filtering only applies to: invalid prompt seed, invalid reference, 
  reference too short, duplicate blocks
- No sample was removed due to low speedup, low quality, or low accept rate
- Boundary structures are **reported**, not hidden
- Discovery and evaluation draw from the same source pool (preliminary phase)

---

*Coverage scan completed at model-free, static-analysis level only.*