# Benchmark Selection Protocol

**Version**: 1.0
**Date**: 2026-06-04
**Status**: Preliminary Benchmark Expansion

---

## 1. Principle: Rules-First, Not Results-First

This phase of benchmark expansion follows a strict **rules-first** pipeline:

1. **Define** candidate structure types and selection rules
2. **Extract** candidates from source code using automated regex-based extraction
3. **Filter** candidates using predefined, structure-agnostic validity rules
4. **Sample** candidates using a fixed random seed
5. **Run** AR / Greedy SD / TASD on selected samples
6. **Report** results without cherry-picking or post-hoc exclusion

**Prohibited**:
- Removing samples because TASD speedup is low
- Removing samples because structural quality is low
- Adding samples to improve aggregate metrics
- Replacing "bad" samples with "good" samples
- Any form of post-hoc sample manipulation based on experimental results

---

## 2. Candidate Structure Priority

Three structure types are targeted for expansion, in fixed priority order:

| Priority | Structure Type | Rationale |
|----------|---------------|-----------|
| 1 | `pipeline_stage_config` | Common in ML/data pipelines; repeated stage skeletons with type/name/params |
| 2 | `complex_nested_config` | Deeply nested dict/list configs; baseline prone to bracket/closure errors |
| 3 | `rich_cli_option_groups` | Multi-option CLI definitions with choices/default/type fields |

### 2.1 Structure Extraction Patterns

**pipeline_stage_config**:
```
train_pipeline = [
test_pipeline = [
pipeline = [
transforms = [
stages = [
processors = [
dict(type=
{"type":
{"name":
Compose([
```

**complex_nested_config**:
```
config = {
CONFIG = {
DEFAULTS = {
settings = {
params = {
dict(
list of dicts (2+ nesting levels)
```

**rich_cli_option_groups**:
```
parser.add_argument(
group.add_argument(
subparser.add_argument(
click.option(
typer.Option(
add_option(
```
Prefer samples with: `choices=`, `default=`, `type=`, `action=`, `help=`, `required=`, `nargs=`, `metavar=`

---

## 3. Fixed Filtering Rules

### 3.1 Inclusion Criteria

| Rule | Description |
|------|-------------|
| Real source | Samples must come from real code files (site-packages, OpenMMLab repos, GitHub repos) |
| Structure seed | Prompt must contain at least 1-2 recognizable instances of the target structure |
| Continuation reference | Reference must contain 2+ instances of the same structure type |
| Minimum block size | Entire block must meet minimum line requirements per structure type |

### 3.2 Exclusion Criteria (only these are permitted)

| Rule | Description | Tracked In |
|------|-------------|------------|
| Invalid prompt seed | prompt contains < min required structure instances | `invalid_prompt_seed_count` |
| Invalid reference | reference contains < min required structure instances | `invalid_reference_count` |
| Reference too short | reference_lines < min threshold per structure type | `reference_too_short_count` |
| Non-target structure | block does not match any target structure pattern | excluded before counting |
| Duplicate block | same (source_file, block_start_line) already selected | `duplicate_removed_count` |
| Format invalid | cannot read file, encoding error | counted in notes |

### 3.3 Explicit Prohibition

- **NO** exclusion based on TASD speedup
- **NO** exclusion based on structural quality score
- **NO** exclusion based on accept rate
- **NO** exclusion based on any experimental outcome
- **NO** manual replacement of any sample

---

## 4. Fixed Sampling Procedure

### 4.1 Candidate Pool Construction

1. Walk all Python files in `SOURCE_DIRS`
2. For each file, extract candidate blocks using structure-specific regex patterns
3. Each candidate identified by `(source_file, block_start_line)`
4. Deduplicate: if same block appears via multiple patterns, keep first occurrence
5. Apply filtering rules (Section 3)
6. Sort valid candidates by `source_file` then `block_start_line`

### 4.2 Sampling

```
random_seed = 20260604
```

- **Pilot phase**: sample `min(20, valid_candidate_count)` from valid candidates
- **Expansion phase**: if pilot passes criteria, sample `min(80, valid_candidate_count)` using same seed
- If `valid_candidate_count < target`, use all valid candidates
- If `valid_candidate_count < 20`, mark benchmark as "insufficient candidates"

### 4.3 Pilot Pass Criteria

| Metric | Threshold |
|--------|-----------|
| TASD speedup | > 1.15x over AR |
| TASD speedup | > Greedy SD speedup |
| TASD structural_quality_score | >= Greedy SD score |
| TASD off_structure_rate | <= Greedy SD off_structure_rate |
| TASD accept_rate | >= 0.85 |
| guard_trigger_count | not anomalously high (> 3x GSD accept_rate complement) |

---

## 5. Fixed Report Fields

### 5.1 Per-Benchmark Summary

```json
{
  "sample_count": 20,
  "raw_candidate_count": N,
  "valid_candidate_count": N,
  "selected_sample_count": 20,
  "source_file_count": N,
  "repo_count": N,
  "source_counts": {},
  "repo_counts": {},
  "avg_prompt_lines": N,
  "avg_reference_lines": N,
  "avg_prompt_chars": N,
  "avg_reference_chars": N,
  "min_prompt_chars": N,
  "max_prompt_chars": N,
  "min_reference_chars": N,
  "max_reference_chars": N,
  "invalid_prompt_seed_count": N,
  "invalid_reference_count": N,
  "reference_too_short_count": N,
  "duplicate_removed_count": N,
  "selection_seed": 20260604,
  "notes": []
}
```

### 5.2 Sample Record

```json
{
  "name": "pipeline_stage_config_001",
  "source": "Pipeline-Stage-Config",
  "structure_type": "pipeline_stage_config",
  "prompt": "...",
  "reference": "...",
  "metadata": {
    "source_repo": "...",
    "source_file": "...",
    "block_start_line": N,
    "block_end_line": N,
    "prompt_lines": N,
    "reference_lines": N,
    "prompt_chars": N,
    "reference_chars": N,
    "seed_count": N,
    "reference_structure_count": N,
    "nesting_depth": N,
    "augmented_from_same_block": false
  }
}
```

---

## 6. Failed Structure Types (Screening Boundary)

The following structure types were screened during discovery but did not produce sufficient valid candidates.
They are **reported**, not hidden, to characterize the applicability boundary of TASD.

| Structure Type | Status | Reason |
|----------------|--------|--------|
| `pytest_parametrize` | failed | Too few distinguishable blocks per file; most are single-line decorators |
| `schema_fields` | failed | SQLAlchemy/DRF model fields lack consistent multi-instance grouping; structure too sparse |
| `SQLAlchemy model_fields` | failed | Column definitions interspersed with other code; insufficient contiguous blocks |

These failures indicate that TASD benefits most from **contiguous, multi-instance repeated structures** — not isolated single-instance or scattered patterns.

---

## 7. Discovery / Evaluation Separation

### 7.1 Discovery Sources

Used to observe the spectrum of candidate structures and estimate availability:

- `/root/miniconda3/lib/python3.12/site-packages/*.py` — installed Python packages
- `/root/autodl-tmp/benchmark_sources/openmmlab*/**/*.py` — OpenMMLab config directories

### 7.2 Evaluation Sources

The same source pool is used for final sampling due to current data constraints.
This is noted as a **preliminary benchmark expansion**.

In future work:
- Discovery could use broader sources (PyPI, GitHub crawl)
- Evaluation could use held-out sources (e.g., specific repos not in discovery)
- The current setup transparently acknowledges this limitation

### 7.3 Current Limitation

This is a **preliminary benchmark expansion**. Discovery and evaluation draw from the same source pool. The selection protocol (fixed rules, fixed seed, no post-hoc exclusion) is the primary safeguard against selection bias. Future work should separate discovery and evaluation source pools.

---

## 8. Benchmark File Naming Convention

| Phase | File Pattern |
|-------|-------------|
| Pilot (20 samples) | `data/{structure_type}_20.jsonl` |
| Pilot summary | `data/{structure_type}_20_summary.json` |
| Expansion (80 samples) | `data/{structure_type}_80.jsonl` |
| Expansion summary | `data/{structure_type}_80_summary.json` |

---

## 9. Execution Order

1. ✅ Create `benchmark_selection_protocol.md` (this document)
2. Build pipeline_stage_config 20 samples (pilot)
3. Run AR / Greedy SD / TASD on pilot
4. Evaluate pilot against pass criteria
5. If passed: expand to 80 samples
6. Repeat for complex_nested_config
7. Repeat for rich_cli_option_groups
8. Compile benchmark_screening_summary with all results and failure reports
