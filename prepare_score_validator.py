#!/usr/bin/env python3
"""
Score Validator Experiment: verify automatic structural recoverability score (0/1/2)
against independent human judgment.

Builds 90-item stratified sample, generates blind annotation HTMLs for annotators
A and B, plus 6 calibration examples, guideline, and public manifest.

Uses official scoring from all_methods_structural_recoverability.json.
TASD-BR uses official BR rerun policy from src/br_rerun_policy.py.

Output:
  results/score_validator_review/private/annotator_A.html
  results/score_validator_review/private/annotator_B.html
  results/score_validator_review/private/blind_mapping_private.json
  results/score_validator_review/private/calibration_examples.html
  results/score_validator_review/annotation_guideline.md
  results/score_validator_review/public_manifest.json
"""

import json, os, sys, hashlib, secrets, random, re
from collections import Counter, defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────
OUT_DIR = "results/score_validator_review"
PRIVATE_DIR = f"{OUT_DIR}/private"
SAMPLING_SEED = 20260625

os.makedirs(PRIVATE_DIR, exist_ok=True)

BENCHMARKS = [
    ("argparse",                "data/codesearchnet_argparse_blocks_80.jsonl"),
    ("dict_config",             "data/codesearchnet_dict_config_blocks_80.jsonl"),
    ("openmmlab_config",        "data/ml_config_blocks_openmmlab_80.jsonl"),
    ("pipeline_stage_config",   "data/pipeline_stage_config_80.jsonl"),
    ("complex_nested_config",   "data/complex_nested_config_80.jsonl"),
    ("rich_cli_option_groups",  "data/rich_cli_option_groups_80.jsonl"),
]

BENCHMARK_TO_STYPE = {
    "argparse": "argparse",
    "dict_config": "dict_config",
    "openmmlab_config": "openmmlab_config",
    "pipeline_stage_config": "pipeline_stage_config",
    "complex_nested_config": "complex_nested_config",
    "rich_cli_option_groups": "rich_cli",
}

# ─── Load blinding secret (or create) ─────────────────────────────────────
SECRET_FILE = f"{PRIVATE_DIR}/blinding_secret.txt"
if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE) as f:
        SECRET = f.read().strip()
else:
    SECRET = secrets.token_hex(32)
    with open(SECRET_FILE, "w") as f:
        f.write(SECRET)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Load all source data (prompts + generated texts)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading source data")
print("=" * 60)

# 1a. Prompts from JSONL files
prompts_data = {}
for bname, fpath in BENCHMARKS:
    with open(fpath) as f:
        samples = [json.loads(line.strip()) for line in f.readlines()]
    prompts_data[bname] = samples
    print(f"  Prompts: {bname} → {len(samples)} samples")

# 1b. AR texts and prompts from tasd_fg_6x80 (480 AR quality data)
ar_texts = {}
with open("results/qwen_tasd_fg_6x80.json") as f:
    ar_quality = json.load(f)
for bname, _ in BENCHMARKS:
    ar_texts[bname] = {}
    for s in ar_quality["per_sample"][bname]["AR"]:
        ar_texts[bname][s["name"]] = s["text"]
    print(f"  AR texts: {bname} → {len(ar_texts[bname])}")

# 1c. TASD texts from per-benchmark files
TS_MAP = {
    "argparse": "argparse", "dict_config": "dict_config",
    "openmmlab_config": "openmmlab", "pipeline_stage_config": "pipeline_stage_config",
    "complex_nested_config": "complex_nested_config", "rich_cli_option_groups": "rich_cli_option_groups",
}
tasd_texts = {}
for bname, _ in BENCHMARKS:
    tasd_file = f"results/tasd_{TS_MAP[bname]}_d16b2k3_80.json"
    with open(tasd_file) as f:
        tdata = json.load(f)
    tasd_texts[bname] = {}
    for s in tdata["per_sample"]:
        # Map by sample_idx
        tasd_texts[bname][s["sample_idx"]] = s["generated_text"]
    print(f"  TASD texts: {bname} → {len(tasd_texts[bname])}")

# 1d. FLY texts (from cache if available)
FLY_CACHE_PATH = f"{PRIVATE_DIR}/fly_texts_all_480.json"
fly_texts_cache = {}
if os.path.exists(FLY_CACHE_PATH):
    with open(FLY_CACHE_PATH) as f:
        raw_cache = json.load(f)
    # Cache format: {"bname/name": {"text": ..., ...}}
    for key, entry in raw_cache.items():
        if isinstance(entry, dict):
            fly_texts_cache[key] = entry.get("text", "")
        else:
            fly_texts_cache[key] = str(entry)
    print(f"  FLY texts: {len(fly_texts_cache)} from cache")
else:
    print(f"  FLY texts: NOT YET GENERATED. Run generate_fly_texts_score_validator.py first.")
    print(f"  Using placeholders for now.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Load automatic scores from final master table source
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading automatic scores")
print("=" * 60)

with open("results/all_methods_structural_recoverability.json") as f:
    auto_scores = json.load(f)

# Build score lookups
ar_scores = {}   # (bname, sample_idx) → dict
fly_scores = {}  # (bname, sample_idx) → dict
fg_scores = {}   # (bname, name) → dict

for s in auto_scores["AR"]:
    ar_scores[(s["benchmark"], s["sample_idx"])] = s

for s in auto_scores["FLY"]:
    fly_scores[(s["benchmark"], s["sample_idx"])] = s

for s in auto_scores["TASD-FG"]:
    # TASD-FG uses 'name' not 'sample_idx'
    sname = s["name"]  # e.g., "argparse_real_001"
    fg_scores[(s["benchmark"], sname)] = s

print(f"  AR scores: {len(ar_scores)}")
print(f"  FLY scores: {len(fly_scores)}")
print(f"  TASD-FG scores: {len(fg_scores)}")

# Sample name → sample_idx mapping (for TASD-FG which uses names)
name_to_idx = {}
for bname, samples in prompts_data.items():
    for i, s in enumerate(samples):
        name_to_idx[(bname, s["name"])] = i

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Build candidate pool (AR + FLY + TASD-BR, each 480)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building candidate pool (1440 total)")
print("=" * 60)

from src.br_rerun_policy import is_br_rerun

candidates = []  # dicts per record

for bname, samples in prompts_data.items():
    for si, sample in enumerate(samples):
        name = sample["name"]
        prompt = sample["prompt"]
        stype = BENCHMARK_TO_STYPE[bname]

        # ── AR ──
        ar_text = ar_texts[bname].get(name, "")
        ar_sc = ar_scores.get((bname, si), {})
        if ar_text:
            candidates.append({
                "method": "AR", "benchmark": bname, "sample_id": si,
                "sample_name": name, "prompt": prompt, "continuation": ar_text,
                "structure_type": stype,
                "automatic_score": ar_sc.get("score", 2),
                "automatic_error_tags": ar_sc.get("error_tags", []),
                "generated_tokens": ar_sc.get("structural_f1", 0),  # proxy
                "is_truncated": ar_sc.get("is_truncated", 0),
                "source_file": "qwen_tasd_fg_6x80.json",
            })

        # ── FLY ──
        fly_sc = fly_scores.get((bname, si), {})
        fly_text = fly_texts_cache.get(f"{bname}/{name}", "")
        if not fly_text:
            fly_text = f"[FLY text for {name} — run generate_fly_texts_score_validator.py]"
        candidates.append({
            "method": "FLY", "benchmark": bname, "sample_id": si,
            "sample_name": name, "prompt": prompt, "continuation": fly_text,
            "structure_type": stype,
            "automatic_score": fly_sc.get("score", 2),
            "automatic_error_tags": fly_sc.get("error_tags", []),
            "generated_tokens": 128,
            "is_truncated": fly_sc.get("is_truncated", 0),
            "source_file": "fly_*_80.json",
        })

        # ── TASD-BR ──
        # Get TASD-FG metrics for BR policy decision
        tasd_text = tasd_texts[bname].get(si, "")
        fg_sc_for_br = fg_scores.get((bname, name), {})
        rerun = is_br_rerun(fg_sc_for_br)
        
        if rerun:
            br_text = ar_text
            br_score = ar_sc.get("score", 2)
            br_tags = ar_sc.get("error_tags", [])
        else:
            br_text = tasd_text
            # Score from fg_scores (which stores TASD-FG scoring)
            br_score = fg_sc_for_br.get("score", 2)
            br_tags = fg_sc_for_br.get("error_tags", [])

        candidates.append({
            "method": "TASD-BR", "benchmark": bname, "sample_id": si,
            "sample_name": name, "prompt": prompt, "continuation": br_text,
            "structure_type": stype,
            "automatic_score": br_score,
            "automatic_error_tags": br_tags,
            "generated_tokens": 128,
            "is_truncated": fg_sc_for_br.get("is_truncated", 0),
            "br_rerun": rerun,
            "source_file": "tasd_*_d16b2k3_80.json",
        })

print(f"  Total candidates: {len(candidates)}")
per_method = Counter(c["method"] for c in candidates)
for m, n in per_method.items():
    print(f"    {m}: {n}")

# Score distribution per method
for m in ["AR", "FLY", "TASD-BR"]:
    sc = Counter(c["automatic_score"] for c in candidates if c["method"] == m)
    print(f"  {m} scores: 0={sc.get(0,0)} 1={sc.get(1,0)} 2={sc.get(2,0)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Stratified sampling — 30 per auto score level (total 90)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Stratified sampling (30/30/30)")
print("=" * 60)

random.seed(SAMPLING_SEED)

# Record full pool proportions for later weighting
pool_counts = Counter(c["automatic_score"] for c in candidates)
print(f"  Full pool: 0={pool_counts[0]} 1={pool_counts[1]} 2={pool_counts[2]}")
print(f"  Full pool proportions: 0={pool_counts[0]/len(candidates):.1%} "
      f"1={pool_counts[1]/len(candidates):.1%} 2={pool_counts[2]/len(candidates):.1%}")

def stratified_sample(candidates, target_per_score=30):
    """Sample 30 per auto score, balancing methods and benchmarks."""
    random.seed(SAMPLING_SEED)
    sampled = []
    used_prompts = set()  # track (benchmark, prompt) to avoid duplication

    for score in [2, 1, 0]:
        pool = [c for c in candidates if c["automatic_score"] == score]
        # Shuffle with seed
        rng = random.Random(SAMPLING_SEED + score * 100)
        rng.shuffle(pool)

        # Stratify by method first
        by_method = defaultdict(list)
        for c in pool:
            by_method[c["method"]].append(c)

        for m in ["AR", "FLY", "TASD-BR"]:
            rng.shuffle(by_method[m])

        target_per_method = 10  # ~10 per method per score level
        round_robin = []
        max_len = max(len(by_method[m]) for m in ["AR", "FLY", "TASD-BR"])
        for i in range(max_len):
            for m in ["AR", "FLY", "TASD-BR"]:
                if i < len(by_method[m]):
                    round_robin.append(by_method[m][i])

        # Pick from round-robin, respecting same-prompt constraint
        per_method_count = defaultdict(int)
        for c in round_robin:
            if len([x for x in sampled if x["automatic_score"] == score]) >= target_per_score:
                break
            if per_method_count[c["method"]] >= target_per_method + 2:  # slight tolerance
                continue
            prompt_key = (c["benchmark"], c["prompt"])
            if prompt_key in used_prompts:
                continue
            sampled.append(c)
            used_prompts.add(prompt_key)
            per_method_count[c["method"]] += 1

    return sampled

sampled = stratified_sample(candidates)

# If we didn't get exactly 90, try again with relaxed constraints
if len(sampled) < 90:
    print(f"  WARNING: Only got {len(sampled)} — relaxing prompt constraint")
    random.seed(SAMPLING_SEED)
    used_prompts = set(c["prompt"] for c in sampled)
    remaining_pool = [c for c in candidates if c not in sampled]
    needed = 90 - len(sampled)
    rng = random.Random(SAMPLING_SEED + 999)
    rng.shuffle(remaining_pool)
    for c in remaining_pool:
        if len(sampled) >= 90:
            break
        prompt_key = (c["benchmark"], c["prompt"])
        if prompt_key in used_prompts:
            continue
        sampled.append(c)
        used_prompts.add(prompt_key)

print(f"  Sampled: {len(sampled)} items")

# Report distribution
sc_dist = Counter(c["automatic_score"] for c in sampled)
print(f"  By auto score: 0={sc_dist[0]} 1={sc_dist[1]} 2={sc_dist[2]}")

method_dist = Counter(c["method"] for c in sampled)
print(f"  By method: {dict(method_dist)}")

bench_dist = Counter(c["benchmark"] for c in sampled)
print(f"  By benchmark: {dict(bench_dist)}")

# Cross-tabulation
print("\n  Cross-tab (benchmark × method):")
for bname, _ in BENCHMARKS:
    row = f"    {bname}:"
    for m in ["AR", "FLY", "TASD-BR"]:
        n = sum(1 for c in sampled if c["benchmark"] == bname and c["method"] == m)
        row += f" {m}={n}"
    print(row)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Generate blind IDs (SVR-XXXXXXXX format)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Generating SVR- blind IDs")
print("=" * 60)

secret_bytes = hashlib.sha256(SECRET.encode()).digest()
blind_rng = random.Random(secret_bytes)
ids_set = set()
while len(ids_set) < len(sampled):
    r = blind_rng.randint(0, 0xFFFFFFFF)
    bid = f"SVR-{r:08X}"
    ids_set.add(bid)
blind_ids = sorted(ids_set)
blind_rng.shuffle(blind_ids)

for item, bid in zip(sampled, blind_ids):
    item["blind_id"] = bid

print(f"  Generated {len(blind_ids)} blind IDs (SVR-XXXXXXXX)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Create private mapping
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Saving private mapping")
print("=" * 60)

mapping = []
for item in sampled:
    mapping.append({
        "blind_id": item["blind_id"],
        "method": item["method"],
        "benchmark": item["benchmark"],
        "sample_id": item["sample_id"],
        "sample_name": item["sample_name"],
        "automatic_score": item["automatic_score"],
        "automatic_error_tags": item["automatic_error_tags"],
        "source_file": item["source_file"],
    })

with open(f"{PRIVATE_DIR}/blind_mapping_private.json", "w") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
print(f"  Saved {len(mapping)} entries to blind_mapping_private.json")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Generate calibration examples (6 items, fixed)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: 6 Calibration examples")
print("=" * 60)

calibration_examples = [
    {
        "id": "CAL-01",
        "title": "Complete and Usable — Score 2",
        "prompt": "    parser.add_argument('--verbose', action='store_true',\n                       help='Enable verbose output')",
        "continuation": "\n    parser.add_argument('--quiet', action='store_true',\n                       help='Suppress output')\n    parser.add_argument('--debug', action='store_true',\n                       help='Enable debug mode')",
        "score": 2,
        "completion": "complete",
        "explanation": "Clean continuation. All args complete with correct indentation and balanced parentheses. No edits needed.",
    },
    {
        "id": "CAL-02",
        "title": "Half-line Truncation, Recoverable — Score 1",
        "prompt": "train_pipeline = [\n    dict(type='LoadImageFromFile'),\n    dict(type='LoadAnnotations', with_bbox=True),",
        "continuation": "\n    dict(type='Resize', scale=(1333, 800), keep_ratio=True),\n    dict(type='RandomFlip', prob=0.5),\n    dict(type='Normalize', mean=[123.675, 116.28, 103.53],",
        "score": 1,
        "completion": "tail_cutoff",
        "explanation": "The Normalize dict is cut mid-line. Deleting this incomplete tail yields a valid pipeline. One local edit to remove the trailing fragment. Score 1 (not 2) because one edit is required.",
    },
    {
        "id": "CAL-03",
        "title": "Incomplete Comment, Code Still Valid — Score 2",
        "prompt": "    def __init__(self, config):\n        self.config = config\n        self.logger = logging.getLogger(__name__)",
        "continuation": "\n        # Initialize all components from config\n        # TODO: add validation for missing fields\n        self._init_backend()\n        self._init_cache()\n        self._ready = True",
        "score": 2,
        "completion": "tail_cutoff",
        "explanation": "The TODO comment is incomplete but the code lines (init_backend, init_cache) are syntactically valid. The incomplete comment does not affect code structure. Score 2 — the comment is non-structural.",
    },
    {
        "id": "CAL-04",
        "title": "Single Duplicate Field — Score 1",
        "prompt": "model = dict(\n    type='FasterRCNN',\n    backbone=dict(\n        type='ResNet',\n        depth=50,",
        "continuation": "\n        num_stages=4,\n        out_indices=(0, 1, 2, 3),\n        frozen_stages=1,\n        num_stages=4,\n        norm_cfg=dict(type='BN'),\n    ),\n    neck=dict(\n        type='FPN',",
        "score": 1,
        "completion": "tail_cutoff",
        "explanation": "num_stages=4 appears twice. One duplicate field to remove. The rest of the structure is clean. Score 1 — one local fix recovers it.",
    },
    {
        "id": "CAL-05",
        "title": "Severe Repetition — Score 0",
        "prompt": "    tokens = {\n        'root': [\n            (r'\\b(abstract|assert)\\b', Keyword),\n            (r'\\b(break|case)\\b', Keyword),",
        "continuation": "\n            (r'\\b(break|case)\\b', Keyword),\n            (r'\\b(break|case)\\b', Keyword),\n            (r'\\b(break|case)\\b', Keyword),\n            (r'\\b(break|case)\\b', Keyword),\n            (r'\\b(break|case)\\b', Keyword),",
        "score": 0,
        "completion": "tail_cutoff",
        "explanation": "The same keyword line repeats 6+ times. This is severe repetition — main content is unusable without rewriting. The repetition starts before any meaningful tail. Score 0.",
    },
    {
        "id": "CAL-06",
        "title": "Multi-layer Structure Collapse — Score 0",
        "prompt": "    skeleton_info = {\n        0: dict(link=('nose', 'neck'), id=0),\n        1: dict(link=('neck', 'left_shoulder'), id=1),",
        "continuation": "  # broken continuation\n    def random_function():\n        import os\n        print('hello')\n        return None\n\n    class SomeClass:\n        pass",
        "score": 0,
        "completion": "severe_incomplete",
        "explanation": "The output has completely abandoned the nested dict structure and switched to def/class/import — off-structure with wrong content. The original structure cannot be recovered. Score 0.",
    },
]

# Calibration HTML will be generated below after function definitions

# ══════════════════════════════════════════════════════════════════════════
# STEP 7b: Define HTML generation functions
# ══════════════════════════════════════════════════════════════════════════

def generate_annotator_html(items, annotator_label):
    """Generate self-contained offline HTML for score validator annotation."""
    ls_key = f"svr_blind_review_{annotator_label}"
    items_json = json.dumps(items, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Score Validator — Annotator {annotator_label}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1b5e20; color: #fff; padding: 16px 24px; position: sticky; top: 0; z-index: 100; display: flex; justify-content: space-between; align-items: center; }}
.header h1 {{ font-size: 18px; }}
.progress {{ font-size: 14px; opacity: 0.9; }}
.container {{ max-width: 960px; margin: 24px auto; padding: 0 16px; }}
.card {{ background: #fff; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card h2 {{ font-size: 16px; color: #1b5e20; margin-bottom: 8px; }}
.meta {{ font-size: 12px; color: #888; margin-bottom: 16px; }}
.prompt-box, .text-box {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; margin: 8px 0 16px; overflow-x: auto; max-height: 400px; overflow-y: auto; }}
.prompt-box pre, .text-box pre {{ font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all; margin: 0; font-family: "SF Mono", "Menlo", "Monaco", monospace; }}
.missing-warning {{ color: #c62828; font-weight: bold; padding: 8px; background: #ffebee; border-radius: 4px; }}
.section-title {{ font-size: 15px; font-weight: 600; color: #1b5e20; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #1b5e20; padding-bottom: 4px; }}
.score-group, .status-group, .tag-group {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.score-group label, .status-group label, .tag-group label {{ display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; cursor: pointer; font-size: 13px; background: #fff; transition: all 0.15s; user-select: none; }}
.score-group label:hover, .status-group label:hover, .tag-group label:hover {{ border-color: #1b5e20; background: #e8f5e9; }}
.score-group label.sel, .status-group label.sel, .tag-group label.tsel {{ border-color: #1b5e20; background: #c8e6c9; font-weight: 600; }}
.score-group label input, .status-group label input, .tag-group label input {{ margin: 0; }}
.notes, .trimpos {{ width: 100%; min-height: 60px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; font-family: inherit; resize: vertical; }}
.completion-indicator {{ padding: 10px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-top: 12px; }}
.completion-indicator.done {{ background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }}
.completion-indicator.partial {{ background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }}
.actions {{ display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }}
.btn-save, .btn-export, .btn-prev, .btn-next, .btn-jump {{ padding: 8px 18px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; }}
.btn-save {{ background: #1b5e20; color: #fff; }}
.btn-export {{ background: #ff8f00; color: #fff; }}
.btn-prev, .btn-next {{ background: #e0e0e0; color: #333; }}
.btn-jump {{ background: #90caf9; color: #000; }}
.btn-save:hover, .btn-export:hover, .btn-prev:hover, .btn-next:hover, .btn-jump:hover {{ opacity: 0.85; }}
button:disabled {{ opacity: 0.4; cursor: default; }}
input[type="number"] {{ padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }}
.footer {{ text-align: center; padding: 24px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Score Validator — Annotator {annotator_label} <span style="opacity:0.7;font-size:13px;">(90 items)</span></h1>
    <div class="progress" id="progress">Initializing...</div>
</div>
<div class="container" id="app"></div>
<div class="footer">Score Validator Experiment — Human Blind Review</div>

<script>
const STORAGE_KEY = '{ls_key}';
const ITEMS = {items_json};
const TOTAL = ITEMS.length;
let annotations = {{}};
let currentIdx = 0;
const saved = localStorage.getItem(STORAGE_KEY);
if (saved) {{ try {{ annotations = JSON.parse(saved); }} catch(e) {{}} }}

function saveState() {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
}}

function updateProgress() {{
    const done = Object.keys(annotations).filter(function(k) {{
        var a = annotations[k];
        return a.human_score !== undefined && a.completion_status !== undefined
               && a.issue_tags !== undefined && a.issue_tags.length > 0;
    }}).length;
    document.getElementById('progress').textContent =
        'Completed: ' + done + ' / ' + TOTAL + ' (' + Math.round(done/TOTAL*100) + '%)';
}}

function render(idx) {{
    currentIdx = idx;
    var item = ITEMS[idx];
    var ann = annotations[item.blind_id] || {{}};
    var hs = ann.human_score;
    var cs = ann.completion_status;

    var h = '';
    h += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap;">';
    h += '<button class="btn-prev" onclick="nav(-1)"' + (idx===0?' disabled':'') + '>Prev</button>';
    h += '<span style="font-size:13px;color:#666;">Item ' + (idx+1) + ' / ' + TOTAL + '</span>';
    h += '<button class="btn-next" onclick="nav(1)"' + (idx===TOTAL-1?' disabled':'') + '>Next</button>';
    h += '<input type="number" id="jumpIn" min="1" max="' + TOTAL + '" value="' + (idx+1) + '" style="width:60px;padding:4px;">';
    h += '<button class="btn-jump" onclick="go()">Go</button>';
    h += '</div>';

    h += '<div class="card">';
    h += '<h2>' + item.blind_id + '</h2>';
    h += '<div class="meta">Item ' + (idx+1) + ' of 90</div>';

    h += '<strong style="font-size:13px;color:#555;">Prompt / Context:</strong>';
    h += '<div class="prompt-box"><pre>' + esc(item.prompt) + '</pre></div>';

    h += '<strong style="font-size:13px;color:#555;">Generated Continuation:</strong>';
    if (!item.continuation) {{
        h += '<div class="missing-warning">Text not available</div>';
    }} else {{
        h += '<div class="text-box"><pre>' + esc(item.continuation) + '</pre></div>';
    }}

    // ---- Section 1: Human Score (0/1/2) ----
    h += '<div class="section-title">1. Structural Recoverability Score</div>';
    h += '<div style="font-size:12px;color:#666;margin-bottom:6px;">Evaluate the raw output as-is. Do NOT ignore truncation errors. Score based on repair cost.</div>';
    h += '<div class="score-group">';
    var scores = [
        ['2','2 — Directly usable: clean, complete, no edits needed'],
        ['1','1 — Locally recoverable: 1-2 local edits (trim tail, fix bracket, remove dup)'],
        ['0','0 — Unrecoverable: need >2 edits, major rewrite, or chaotic']
    ];
    for (var si = 0; si < scores.length; si++) {{
        var v = scores[si][0], lbl = scores[si][1];
        var sel = hs === parseInt(v) ? ' sel' : '';
        var chk = hs === parseInt(v) ? 'checked' : '';
        h += '<label class="' + sel + '" onclick="setHumanScore(\\'' + item.blind_id + '\\',' + v + ');update(' + idx + ')">';
        h += '<input type="radio" name="hs_' + item.blind_id + '" value="' + v + '" ' + chk + '> ' + lbl;
        h += '</label>';
    }}
    h += '</div>';

    // ---- Section 2: Completion Status ----
    h += '<div class="section-title">2. Completion Status (REQUIRED)</div>';
    h += '<div style="font-size:12px;color:#666;margin-bottom:6px;">Judge the raw output as-is. Token limit truncation does NOT automatically deduct score.</div>';
    h += '<div class="status-group">';
    var statuses = [
        ['complete','Complete — ends at natural boundary'],
        ['tail_cutoff','Tail Cutoff — clear continuous incomplete tail'],
        ['severe_incomplete','Severe Incomplete — major structures unfinished; no valid prefix']
    ];
    for (var si = 0; si < statuses.length; si++) {{
        var v = statuses[si][0], lbl = statuses[si][1];
        var sel = cs === v ? ' sel' : '';
        var chk = cs === v ? 'checked' : '';
        h += '<label class="' + sel + '" onclick="setCompletion(\\'' + item.blind_id + '\\',\\'' + v + '\\');update(' + idx + ')">';
        h += '<input type="radio" name="cs_' + item.blind_id + '" value="' + v + '" ' + chk + '> ' + lbl;
        h += '</label>';
    }}
    h += '</div>';

    // ---- Section 3: Issue Tags (at least one required) ----
    h += '<div class="section-title">3. Issue Tags (select at least one; "none" excludes others)</div>';
    h += '<div class="tag-group">';
    var tags = ['bracket_or_delimiter','indentation','repetition','duplicate_field',
                'off_structure','wrong_content','other','none'];
    var tl = {{bracket_or_delimiter:'Bracket/Delimiter',indentation:'Indentation',
              repetition:'Repetition',duplicate_field:'Duplicate Field',
              off_structure:'Off-Structure',wrong_content:'Wrong Content',
              other:'Other',none:'None'}};
    for (var ti = 0; ti < tags.length; ti++) {{
        var t = tags[ti];
        var itags = ann.issue_tags || [];
        var chk = itags.indexOf(t) >= 0 ? 'checked' : '';
        var cls = itags.indexOf(t) >= 0 ? ' tsel' : '';
        h += '<label class="' + cls + '" onclick="toggleTag(\\'' + item.blind_id + '\\',\\'' + t + '\\');update(' + idx + ')">';
        h += '<input type="checkbox" ' + chk + '> ' + (tl[t]||t);
        h += '</label>';
    }}
    h += '</div>';

    // ---- Section 4: Notes ----
    h += '<div class="section-title">4. Notes (optional)</div>';
    h += '<textarea class="notes" id="n_' + item.blind_id + '" onchange="setNotes(\\'' + item.blind_id + '\\',this.value)">' + (ann.notes||'') + '</textarea>';

    // ---- Completion indicator ----
    if (hs !== undefined && cs !== undefined && ann.issue_tags !== undefined && ann.issue_tags.length > 0) {{
        h += '<div class="completion-indicator done">Fully scored: Score ' + hs + ', Completion: ' + cs + '</div>';
    }} else {{
        h += '<div class="completion-indicator partial">Incomplete: score, completion, and at least one issue tag are required</div>';
    }}

    h += '</div>';

    h += '<div class="actions">';
    h += '<button class="btn-save" onclick="saveState();alert(&quot;Saved!&quot;)">Save Progress</button>';
    h += '<button class="btn-export" onclick="exportJSON()">Export JSON</button>';
    h += '</div>';

    document.getElementById('app').innerHTML = h;
    updateProgress();
    window.scrollTo(0, 0);
}}

function update(idx) {{ render(idx); }}
function nav(d) {{ var ni = currentIdx + d; if (ni >= 0 && ni < TOTAL) render(ni); }}
function go() {{ var v = parseInt(document.getElementById('jumpIn').value) - 1; if (v >= 0 && v < TOTAL) render(v); }}

function setHumanScore(bid, s) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    annotations[bid].human_score = s;
    saveState();
}}
function setCompletion(bid, s) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    annotations[bid].completion_status = s;
    saveState();
}}
function toggleTag(bid, t) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    if (!annotations[bid].issue_tags) annotations[bid].issue_tags = [];
    if (t === 'none') {{
        annotations[bid].issue_tags = ['none'];
    }} else {{
        var i = annotations[bid].issue_tags.indexOf('none');
        if (i >= 0) annotations[bid].issue_tags.splice(i, 1);
        var j = annotations[bid].issue_tags.indexOf(t);
        if (j >= 0) annotations[bid].issue_tags.splice(j, 1);
        else annotations[bid].issue_tags.push(t);
    }}
    saveState();
}}
function setNotes(bid, n) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    annotations[bid].notes = n;
    saveState();
}}
function exportJSON() {{
    var r = ITEMS.map(function(it) {{
        var a = annotations[it.blind_id] || {{}};
        return {{
            blind_id: it.blind_id,
            human_score: a.human_score,
            completion_status: a.completion_status,
            issue_tags: a.issue_tags || [],
            notes: a.notes || ''
        }};
    }});
    var b = new Blob([JSON.stringify(r, null, 2)], {{type: 'application/json'}});
    var u = URL.createObjectURL(b);
    var a = document.createElement('a');
    a.href = u; a.download = 'annotations_{annotator_label}_v2.json'; a.click();
    URL.revokeObjectURL(u);
}}
function esc(s) {{
    if (!s) return '';
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}}
render(0);
</script>
</body>
</html>"""
    return html


def generate_calibration_html(examples):
    """Generate self-contained HTML for the 6 calibration examples."""
    ls_key = "svr_calibration"
    items_json = json.dumps(examples, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Score Validator — Calibration Examples</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #e65100; color: #fff; padding: 16px 24px; position: sticky; top: 0; z-index: 100; }}
.header h1 {{ font-size: 18px; }}
.container {{ max-width: 960px; margin: 24px auto; padding: 0 16px; }}
.card {{ background: #fff; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #e65100; }}
.card h2 {{ font-size: 15px; color: #e65100; margin-bottom: 4px; }}
.card .meta {{ font-size: 12px; color: #888; margin-bottom: 12px; }}
.prompt-box, .text-box {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; margin: 8px 0 12px; overflow-x: auto; max-height: 300px; overflow-y: auto; }}
.prompt-box pre, .text-box pre {{ font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all; margin: 0; font-family: "SF Mono", "Menlo", "Monaco", monospace; }}
.result {{ display: flex; gap: 12px; margin-top: 12px; font-size: 13px; }}
.result .badge {{ padding: 4px 10px; border-radius: 4px; font-weight: 600; }}
.score-2 {{ background: #c8e6c9; color: #1b5e20; }}
.score-1 {{ background: #fff3e0; color: #e65100; }}
.score-0 {{ background: #ffcdd2; color: #b71c1c; }}
.explanation {{ font-size: 13px; color: #555; margin-top: 8px; line-height: 1.6; background: #fafafa; padding: 10px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Score Validator — Calibration Examples (6 items)</h1>
    <div style="font-size:13px;opacity:0.85;margin-top:4px;">Read together. Do NOT discuss specific items from the 90-set.</div>
</div>
<div class="container" id="app"></div>

<script>
const ITEMS = {items_json};

function esc(s) {{
    if (!s) return '';
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}}

function renderAll() {{
    var h = '';
    for (var i = 0; i < ITEMS.length; i++) {{
        var item = ITEMS[i];
        var scoreClass = item.score === 2 ? 'score-2' : (item.score === 1 ? 'score-1' : 'score-0');
        h += '<div class="card">';
        h += '<h2>' + item.id + ': ' + item.title + '</h2>';
        h += '<div class="meta">Completion: ' + item.completion + '</div>';

        h += '<strong style="font-size:13px;color:#555;">Prompt:</strong>';
        h += '<div class="prompt-box"><pre>' + esc(item.prompt) + '</pre></div>';

        h += '<strong style="font-size:13px;color:#555;">Continuation:</strong>';
        h += '<div class="text-box"><pre>' + esc(item.continuation) + '</pre></div>';

        h += '<div class="result">';
        h += '<span>Score:</span><span class="badge ' + scoreClass + '">' + item.score + '</span>';
        h += '<span>| Completion:</span><span>' + item.completion + '</span>';
        h += '</div>';

        h += '<div class="explanation">' + esc(item.explanation) + '</div>';
        h += '</div>';
    }}
    document.getElementById('app').innerHTML = h;
}}
renderAll();
</script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════
# STEP 8b: Generate calibration HTML
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8b: Generating calibration examples HTML")
print("=" * 60)
cal_html = generate_calibration_html(calibration_examples)
cal_path = f"{PRIVATE_DIR}/calibration_examples.html"
with open(cal_path, "w") as f:
    f.write(cal_html)
print(f"  -> {cal_path} ({len(cal_html)} bytes)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8c: Generate annotator HTMLs (A and B)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Generating annotator HTMLs")
print("=" * 60)
# Strip sensitive fields for HTML (only keep what annotator should see)
def strip_for_html(items):
    """Keep only blind_id, benchmark, prompt, continuation."""
    return [{"blind_id": i["blind_id"], "benchmark": i["benchmark"],
             "prompt": i["prompt"], "continuation": i["continuation"]}
            for i in items]

items_for_a = strip_for_html(list(sampled))
items_for_b = strip_for_html(list(sampled))
random.shuffle(items_for_a)
random.shuffle(items_for_b)

html_a = generate_annotator_html(items_for_a, "A")
html_b = generate_annotator_html(items_for_b, "B")

a_path = f"{PRIVATE_DIR}/annotator_A.html"
b_path = f"{PRIVATE_DIR}/annotator_B.html"
with open(a_path, "w") as f:
    f.write(html_a)
with open(b_path, "w") as f:
    f.write(html_b)

print(f"  -> {a_path} ({len(html_a)} bytes)")
print(f"  -> {b_path} ({len(html_b)} bytes)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: Generate annotation guideline
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Annotation guideline")
print("=" * 60)

guideline_md = """# Score Validator — Human Annotation Guideline

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
"""

with open(f"{OUT_DIR}/annotation_guideline.md", "w") as f:
    f.write(guideline_md)
print(f"  Saved {OUT_DIR}/annotation_guideline.md")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: Public manifest (aggregate only, no texts or mappings)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: Public manifest")
print("=" * 60)

manifest = {
    "experiment": "score_validator_review",
    "description": "Validate automatic structural recoverability score against human judgment",
    "sampling_seed": SAMPLING_SEED,
    "total_items": len(sampled),
    "per_auto_score": {str(k): v for k, v in sc_dist.items()},
    "per_method": dict(method_dist),
    "per_benchmark": dict(bench_dist),
    "blind_id_format": "SVR-XXXXXXXX",
    "annotators": ["A", "B"],
    "calibration_examples_count": 6,
    "date_generated": __import__('datetime').datetime.now().isoformat(),
    "pool_distribution": {
        "total_candidates": len(candidates),
        "auto_score_0": pool_counts[0],
        "auto_score_1": pool_counts[1],
        "auto_score_2": pool_counts[2],
        "pool_proportions": {
            "0": round(pool_counts[0]/len(candidates), 4),
            "1": round(pool_counts[1]/len(candidates), 4),
            "2": round(pool_counts[2]/len(candidates), 4),
        }
    },
    "automatic_score_source": "results/all_methods_structural_recoverability.json",
    "br_rerun_policy": "src/br_rerun_policy.py::is_br_rerun()",
}

with open(f"{OUT_DIR}/public_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"  Saved {OUT_DIR}/public_manifest.json")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: Integrity checks
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 11: Integrity checks")
print("=" * 60)

checks_passed = 0
checks_total = 0

def check(desc, condition):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  PASS {desc}")
    else:
        print(f"  FAIL {desc}")

check("1. 90 items total", len(sampled) == 90)
check(f"2. Score 0 = 30", sc_dist[0] == 30)
check(f"2. Score 1 = 30", sc_dist[1] == 30)
check(f"2. Score 2 = 30", sc_dist[2] == 30)

# Check no duplicate prompts
prompts = [(c["benchmark"], c["prompt"]) for c in sampled]
check("3. No duplicate prompts", len(prompts) == len(set(prompts)))

# Method balance
for m in ["AR", "FLY", "TASD-BR"]:
    n = method_dist.get(m, 0)
    check(f"4. {m} count: {n} (target ~10 per score level)", 25 <= n <= 35)

# Benchmark coverage
for bname, _ in BENCHMARKS:
    n = bench_dist.get(bname, 0)
    check(f"5. Benchmark {bname}: {n} items", n > 0)

# Texts non-empty
empty_texts = sum(1 for c in sampled if not c["continuation"].strip())
check(f"6. All continuations non-empty (empty: {empty_texts})", empty_texts == 0)

# A/B have same blind IDs
a_ids = set(i["blind_id"] for i in items_for_a)
b_ids = set(i["blind_id"] for i in items_for_b)
check(f"7. A/B share same 90 blind IDs", a_ids == b_ids and len(a_ids) == 90)

# A/B different orders
a_order = [i["blind_id"] for i in items_for_a]
b_order = [i["blind_id"] for i in items_for_b]
check("8. A/B different orders", a_order != b_order)

# HTML doesn't leak method or auto score
for label, html_str in [("A", html_a), ("B", html_b)]:
    check(f"9. HTML {label}: no method leak", "TASD-BR" not in html_str and '"FLY"' not in html_str and '"AR"' not in html_str)
    check(f"9. HTML {label}: no auto score leak", '"automatic_score"' not in html_str)
    check(f"9. HTML {label}: no sample_idx leak", '"sample_idx"' not in html_str)

# Frontend validation: none tag exclusion
check("10. none tag excludes others (JS logic present)", "annotations[bid].issue_tags = ['none']" in html_a)

# Required fields
check("11. human_score required", "human_score" in html_a and "human_score" in html_b)
check("11. completion_status required", "completion_status" in html_a)
check("11. issue_tags required", "issue_tags.length > 0" in html_a)

# localStorage key
check("12. localStorage A key", "svr_blind_review_A" in html_a)
check("12. localStorage B key", "svr_blind_review_B" in html_b)

print(f"\n  PASSED: {checks_passed}/{checks_total}")

# ══════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
print(f"\nPrivate files (DO NOT COMMIT):")
for root, dirs, files in os.walk(PRIVATE_DIR):
    for fn in files:
        fp = os.path.join(root, fn)
        print(f"  {fp} ({os.path.getsize(fp)} bytes)")

print(f"\nPublic files:")
for fn in ["annotation_guideline.md", "public_manifest.json"]:
    fp = f"{OUT_DIR}/{fn}"
    if os.path.exists(fp):
        print(f"  {fp} ({os.path.getsize(fp)} bytes)")
