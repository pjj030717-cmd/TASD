## Total Experiment Table

All benchmarks: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft), temperature=0, max_new_tokens=128, KV cache enabled.

| Benchmark | AR TPS | GSD TPS | GSD Spd | GSD Accept | TASD TPS | TASD Spd | TASD Accept | GSD SQ | TASD SQ | GSD OffStr | TASD OffStr | GSD Trunc | TASD Trunc |
|-----------|--------|---------|---------|------------|----------|----------|-------------|--------|---------|------------|-------------|-----------|------------|
| Real-Python-Argparse | 32.98 | 26.24 | 0.83x | 0.83 | 42.92 | 1.30x | 0.92 | 0.8733 | 0.9223 | 0.0149 | 0.0039 | 0.0522 | 0.0178 |
| Real-Python-DictConfig | 32.67 | 26.98 | 0.87x | 0.87 | 42.62 | 1.30x | 0.90 | 0.7993 | 0.8310 | 0.0287 | 0.0006 | 0.1420 | 0.1184 |
| OpenMMLab-Config | 32.91 | 25.54 | 0.81x | 0.81 | 47.34 | 1.44x | 0.97 | 0.8393 | 0.8887 | 0.0111 | 0.0023 | 0.2071 | 0.1250 |
| Rich-CLI-Option-Groups* | 33.14 | 27.39 | 0.83x | 0.85 | 49.12 | 1.48x | 1.00 | 0.9159 | 0.9074 | 0.0712 | 0.1218 | 0.0604 | 0.0556 |
| Complex-Nested-Config* | 32.71 | 27.76 | 0.85x | 0.85 | 48.23 | 1.47x | 1.00 | 0.7969 | 0.7985 | 0.0194 | 0.0198 | 0.1156 | 0.0590 |
| Pipeline-Stage-Config* | 32.24 | 25.75 | 0.80x | 0.82 | 49.36 | 1.53x | 1.00 | 0.9250 | 0.9120 | 0.0000 | 0.0000 | 0.1933 | 0.1272 |

*Extended benchmarks added in Phase 2.*

**Notes:**
- Original benchmark GSD data from 20-sample pilot (AR/TASD at n=80). Extended benchmarks at n=80 for all methods.
- Greedy SD uses strict argmax matching (not full rejection sampling). TASD uses top-k/prob relaxed acceptance with structural guard.
- All timings use `torch.cuda.synchronize()`, exclude model loading, and count TPS by actual generated tokens.

---

## Benchmark Coverage

### Structure Categories

| Structure Type | Raw | Valid | Files | Packages | Avg Lines | Suitability |
|----------------|-----|-------|-------|----------|-----------|-------------|
| argparse | 247 | 227 | 159 | 42 | 36 | High |
| rich_cli_option_groups | 115 | 115 | 88 | 32 | 64 | High |
| dict_config | 2733 | 2713 | 1596 | 103 | 49 | Medium-High |
| complex_nested_config | 1290 | 1017 | 661 | 64 | 88 | High |
| openmmlab_config | 5277 | 5222 | 2186 | 18 | 19 | High |
| pipeline_stage_config | 1347 | 1105 | 929 | 6 | 16 | High |

6/9 structure categories provide benchmark-ready candidates under our extraction rules. These cover config, pipeline, CLI, and nested structure types across 2,000+ source files and 100+ packages/repos.

### Boundary Structures

| Structure Type | Raw | Valid | Files | Packages | Avg Lines | Note |
|----------------|-----|-------|-------|----------|-----------|------|
| schema_fields | 238 | 126 | 70 | 8 | 6 | short, single-line fields |
| model_fields | 0 | 0 | 0 | 0 | 0 | not found in scanned pool |
| pytest_parametrize | 1118 | 1118 | 217 | 16 | 12 | nested strings, guard error-prone |

### Source Pool Limitations

- `model_fields` (Pydantic/dataclass) returned 0 candidates. This structure is common in web frameworks (FastAPI, Flask) and typed data layers but is not well represented in the scanned source pool (site-packages + OpenMMLab configs). This is a source-pool limitation, not a statement that the structure does not exist in real code.
- `schema_fields` (SQLAlchemy/DRF) produces only 238 raw candidates with avg block lines of 6. Such short blocks are dominated by single-line field definitions where AR baseline already performs well; TASD's multi-token draft advantage does not apply.
- `pytest_parametrize` has 1,118 raw candidates but the decorator yields shallow, string-heavy blocks where guard rules on nested strings and expressions are error-prone.
- These three boundary structures are reported, not hidden, to characterize the applicability boundary of TASD.

### Selection Bias Safeguards

- All samples extracted by automated rules before any model run
- Fixed random seed (20260604) for sampling
- Filtering only for: invalid prompt seed, invalid reference, reference too short, duplicate blocks
- No post-hoc exclusion based on speedup, quality, or accept rate
- Boundary structures included in report, not omitted
- Discovery and evaluation from same source pool (preliminary phase)
