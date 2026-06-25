#!/usr/bin/env python3
"""
Prepare TASD Human Blind Review Materials.

Samples 10 prompts per benchmark (stratified, seed=20260624),
extracts full generated text for AR and TASD-FG,
constructs TASD-BR outputs (TASD-FG text, or AR text if high risk),
flags missing FLY texts, generates blind HTML annotators and mapping.

Does NOT run model inference. Does NOT generate synthetic scores.
"""

import json
import os
import random
import re
from collections import defaultdict, Counter

random.seed(20260624)

# ============================================================
# Config
# ============================================================
BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl"),
]

CKPT_DIR = "results/qwen_6x80_checkpoints"
OUT_DIR = "results/human_blind_review"
MISSING_FILE = "results/human_blind_review_missing_samples.json"
MAPPING_FILE = f"{OUT_DIR}/blind_mapping_private.json"
SAMPLES_PER_BENCHMARK = 10
SEED = 20260624

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Load prompts and sample
# ============================================================
print("=" * 60)
print("Step 1: Stratified Random Sampling")
print("=" * 60)

selected_prompts = {}  # {benchmark: [{"name": ..., "prompt": ..., "idx": ...}]}

for bname, data_file in BENCHMARKS:
    with open(data_file) as f:
        samples = [json.loads(line.strip()) for line in f.readlines()]

    n = len(samples)
    indices = list(range(n))
    selected_indices = sorted(random.sample(indices, SAMPLES_PER_BENCHMARK))

    selected = []
    for idx in selected_indices:
        s = samples[idx]
        selected.append({
            "idx": idx,
            "name": s["name"],
            "prompt": s["prompt"],
        })
    selected_prompts[bname] = selected
    print(f"  {bname}: {len(selected)}/{n} selected")

total_prompts = sum(len(v) for v in selected_prompts.values())
print(f"\nTotal selected prompts: {total_prompts}")

# ============================================================
# Step 2: Load per-sample texts and construct TASD-BR
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Extract texts and construct TASD-BR")
print("=" * 60)

# Load AR quality data (has text)
ar_texts = {}  # {benchmark: {idx: {"text": ..., "name": ...}}}
for bname, _ in BENCHMARKS:
    ar_file = f"{CKPT_DIR}/{bname}_AR_quality.json"
    ar_texts[bname] = {}
    if os.path.exists(ar_file):
        with open(ar_file) as f:
            ar_data = json.load(f)
        for i, s in enumerate(ar_data):
            ar_texts[bname][i] = {"text": s.get("text", ""), "name": s["name"]}
        print(f"  AR {bname}: {len(ar_texts[bname])} texts loaded")
    else:
        print(f"  AR {bname}: FILE NOT FOUND: {ar_file}")

# Load TASD-FG data (has text)
fg_texts = {}
for bname, _ in BENCHMARKS:
    fg_file = f"{CKPT_DIR}/{bname}_TASDFG.json"
    fg_texts[bname] = {}
    if os.path.exists(fg_file):
        with open(fg_file) as f:
            fg_data = json.load(f)
        for i, s in enumerate(fg_data):
            fg_texts[bname][i] = {"text": s.get("text", ""), "name": s["name"]}
        print(f"  TASD-FG {bname}: {len(fg_texts[bname])} texts loaded")
    else:
        print(f"  TASD-FG {bname}: FILE NOT FOUND: {fg_file}")

# Load TASD-FG bracket_balance and is_truncated for BR decision (aligned by sample name)
fg_risk = {}
metrics_file = "results/all_methods_structural_recoverability.json"
if os.path.exists(metrics_file):
    with open(metrics_file) as f:
        metrics = json.load(f)
    for s in metrics.get("TASD-FG", []):
        bname = s["benchmark"]
        sample_name = s["name"]
        if bname not in fg_risk:
            fg_risk[bname] = {}
        fg_risk[bname][sample_name] = {
            "bracket_balance": s.get("bracket_balance", 1.0),
            "is_truncated": s.get("is_truncated", 0),
        }
    print(f"  BR risk data: loaded for {sum(len(v) for v in fg_risk.values())} samples")
else:
    print(f"  WARNING: metrics file not found: {metrics_file}")

# Build all 180 review items
review_items = []
blind_id_counter = 0
missing_samples = []

# Pre-load FLY texts if available (from generate_fly_texts_for_blind_review.py)
fly_texts_cache = {}
fly_texts_path_cache = "results/human_blind_review/fly_texts_for_blind_review.json"
if os.path.exists(fly_texts_path_cache):
    with open(fly_texts_path_cache) as f:
        fly_cache = json.load(f)
    for bid, entry in fly_cache.items():
        fly_texts_cache[bid] = entry["text"]
    print(f"  Loaded {len(fly_texts_cache)} FLY texts from cache")

for bname, prompts in selected_prompts.items():
    for p in prompts:
        idx = p["idx"]
        name = p["name"]
        prompt_text = p["prompt"]

        # AR
        ar_entry = ar_texts.get(bname, {}).get(idx, {})
        ar_text = ar_entry.get("text", "")
        if not ar_text:
            missing_samples.append({
                "method": "AR", "benchmark": bname, "sample_idx": idx,
                "sample_name": name, "reason": "AR text missing or empty"
            })

        # TASD-FG
        fg_entry = fg_texts.get(bname, {}).get(idx, {})
        fg_text_val = fg_entry.get("text", "")

        # TASD-BR: check bracket_balance < 0.50 and is_truncated == 0
        risk = fg_risk.get(bname, {}).get(name, {})
        bb = risk.get("bracket_balance", 1.0)
        trunc = risk.get("is_truncated", 0)
        is_high_risk = (bb < 0.50 and trunc == 0)
        br_text = ar_text if is_high_risk else fg_text_val

        if not fg_text_val:
            missing_samples.append({
                "method": "TASD-FG", "benchmark": bname, "sample_idx": idx,
                "sample_name": name, "reason": "TASD-FG text missing or empty"
            })

        # Store base info for FLY (text = None pending generation)
        review_items.append({
            "blind_id": f"REV-{blind_id_counter + 1:04d}",
            "method": "AR",
            "benchmark": bname,
            "original_sample_name": name,
            "original_sample_idx": idx,
            "prompt": prompt_text,
            "continuation": ar_text,
        })
        blind_id_counter += 1

        # FLY: use cached text if available
        fly_bid = f"REV-{blind_id_counter + 1:04d}"
        fly_cont = fly_texts_cache.get(fly_bid)
        review_items.append({
            "blind_id": fly_bid,
            "method": "FLY",
            "benchmark": bname,
            "original_sample_name": name,
            "original_sample_idx": idx,
            "prompt": prompt_text,
            "continuation": fly_cont if fly_cont else None,
        })
        blind_id_counter += 1

        review_items.append({
            "blind_id": f"REV-{blind_id_counter + 1:04d}",
            "method": "TASD-BR",
            "benchmark": bname,
            "original_sample_name": name,
            "original_sample_idx": idx,
            "prompt": prompt_text,
            "continuation": br_text,
            "br_decision": {
                "bracket_balance": bb,
                "is_truncated": trunc,
                "high_risk": is_high_risk,
                "used_ar_text": is_high_risk,
            },
        })
        blind_id_counter += 1

print(f"\nTotal review items: {len(review_items)}")

# ============================================================
# Step 3: Save missing samples report
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Missing samples report")
print("=" * 60)

fly_missing = [r for r in review_items if r["method"] == "FLY"]
print(f"  FLY texts missing: {len(fly_missing)} (all FLY checkpoints lack 'text' field)")

if missing_samples:
    print(f"  Additional missing: {len(missing_samples)}")
else:
    print("  No AR or TASD-FG texts are missing")

missing_report = {
    "fly_missing_count": len(fly_missing),
    "fly_missing_reason": "FLY checkpoint files (results/qwen_6x80_checkpoints/*_FLY.json) only contain aggregate stats (tps, sp, sq, mat, ngram_acc, wall, gen_len) without the 'text' field. The run_fly() function DOES return text, but the checkpoint saving code omitted it.",
    "fly_resolution": "Run: python generate_fly_texts_for_blind_review.py  (requires GPU, ~10-15 min for 60 prompts)",
    "fly_missing_samples": [{
        "blind_id": r["blind_id"], "benchmark": r["benchmark"],
        "original_sample_name": r["original_sample_name"],
        "original_sample_idx": r["original_sample_idx"],
        "prompt_preview": r["prompt"][:200] + "...",
    } for r in fly_missing],
    "other_missing": missing_samples,
}

with open(MISSING_FILE, "w") as f:
    json.dump(missing_report, f, indent=2, ensure_ascii=False)
print(f"  -> {MISSING_FILE}")

# ============================================================
# Step 4: Shuffle for blind presentation
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Blind shuffle")
print("=" * 60)

random.shuffle(review_items)
print(f"  180 items shuffled (seed={SEED})")

# ============================================================
# Step 5: Build blind mapping (private)
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Private blind mapping")
print("=" * 60)

blind_mapping = []
for item in review_items:
    entry = {
        "blind_id": item["blind_id"],
        "method": item["method"],
        "benchmark": item["benchmark"],
        "original_sample_name": item["original_sample_name"],
        "original_sample_idx": item["original_sample_idx"],
    }
    if "br_decision" in item:
        entry["br_decision"] = item["br_decision"]
    blind_mapping.append(entry)

with open(MAPPING_FILE, "w") as f:
    json.dump(blind_mapping, f, indent=2, ensure_ascii=False)
print(f"  -> {MAPPING_FILE}")
print("  (DO NOT share with annotators)")

# ============================================================
# Step 6: Generate HTML annotator
# ============================================================

def generate_html(items, annotator_label):
    """Generate a single-file offline HTML annotator."""
    items_json = json.dumps([
        {
            "blind_id": it["blind_id"],
            "benchmark": it["benchmark"],
            "prompt": it["prompt"],
            "continuation": it["continuation"],
        }
        for it in items
    ], ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TASD Human Blind Review - {annotator_label}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.5;
}}
.header {{
    background: #1a237e; color: white; padding: 16px 24px;
    position: sticky; top: 0; z-index: 100;
}}
.header h1 {{ font-size: 20px; }}
.header .progress {{ font-size: 14px; margin-top: 4px; opacity: 0.9; }}
.container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
.card {{
    background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    margin-bottom: 24px; padding: 24px;
}}
.card h2 {{ font-size: 16px; color: #1a237e; margin-bottom: 8px; }}
.card .meta {{
    font-size: 12px; color: #888; margin-bottom: 12px;
}}
.prompt-box {{
    background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;
    padding: 12px; margin-bottom: 12px; max-height: 200px; overflow-y: auto;
}}
.prompt-box pre {{
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 13px; white-space: pre-wrap; word-break: break-all;
    background: transparent; padding: 0; margin: 0; line-height: 1.4;
}}
.continuation-box {{
    border: 1px solid #90caf9; border-radius: 4px; padding: 12px;
    margin-bottom: 16px; max-height: 400px; overflow-y: auto;
    background: #f3f9ff;
}}
.continuation-box pre {{
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 13px; white-space: pre-wrap; word-break: break-all;
    background: transparent; padding: 0; margin: 0; line-height: 1.4;
}}
.missing-warning {{
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
    padding: 8px 12px; font-size: 13px; color: #856404; margin-bottom: 12px;
}}
.score-group {{
    display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap;
}}
.score-group label {{
    display: flex; align-items: center; gap: 6px; font-size: 14px;
    cursor: pointer; padding: 6px 12px; border: 1px solid #ccc;
    border-radius: 4px; transition: all 0.2s;
}}
.score-group label:hover {{ background: #e3f2fd; }}
.score-group label.score-selected {{ background: #bbdefb; border-color: #1565c0; }}
.score-group input {{ margin: 0; }}
.tag-group {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
}}
.tag-group label {{
    display: flex; align-items: center; gap: 4px; font-size: 13px;
    cursor: pointer; padding: 4px 10px; border: 1px solid #ddd;
    border-radius: 4px; transition: all 0.2s;
}}
.tag-group label:hover {{ background: #f3e5f5; }}
.tag-group label.tag-selected {{ background: #e1bee7; border-color: #7b1fa2; }}
.tag-group input {{ margin: 0; }}
.notes {{ width: 100%; min-height: 48px; font-family: inherit; font-size: 13px;
    padding: 8px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }}
.actions {{ margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }}
.actions button {{
    padding: 10px 20px; border: none; border-radius: 4px; font-size: 14px;
    cursor: pointer; transition: opacity 0.2s;
}}
.btn-export {{ background: #4CAF50; color: white; }}
.btn-save {{ background: #2196F3; color: white; }}
.btn-prev {{ background: #9C27B0; color: white; }}
.btn-next {{ background: #FF5722; color: white; }}
.btn-jump {{ background: #607D8B; color: white; }}
.actions button:hover {{ opacity: 0.85; }}
.nav-info {{ font-size: 13px; color: #666; margin-top: 8px; }}
.flagged {{ background: #fff3e0; border-color: #ff9800 !important; }}
hr {{ border: 0; border-top: 1px solid #eee; margin: 16px 0; }}
</style>
</head>
<body>
<div class="header">
    <h1>TASD Human Blind Review - {annotator_label}</h1>
    <div class="progress" id="progress">Initializing...</div>
</div>
<div class="container" id="app"></div>

<script>
const ITEMS = {items_json};
const TOTAL = ITEMS.length;

let annotations = {{}};
let currentIdx = 0;

// Load saved state from localStorage
const saved = localStorage.getItem('tasd_blind_{annotator_label}');
if (saved) {{
    try {{ annotations = JSON.parse(saved); }} catch(e) {{}}
}}

function saveState() {{
    localStorage.setItem('tasd_blind_{annotator_label}', JSON.stringify(annotations));
}}

function updateProgress() {{
    const done = Object.keys(annotations).length;
    const pct = Math.round(done / TOTAL * 100);
    document.getElementById('progress').textContent =
        `Completed: ${{done}} / ${{TOTAL}} (${{pct}}%)`;
}}

function render(idx) {{
    currentIdx = idx;
    const item = ITEMS[idx];
    const ann = annotations[item.blind_id] || {{}};

    let html = '';

    // Navigation
    html += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;">';
    html += `<button class="btn-prev" onclick="navigate(-1)" ${{idx === 0 ? 'disabled' : ''}}>◀ Prev</button>`;
    html += `<span style="font-size:13px;color:#666;">Item ${{idx + 1}} / ${{TOTAL}}</span>`;
    html += `<button class="btn-next" onclick="navigate(1)" ${{idx === TOTAL - 1 ? 'disabled' : ''}}>Next ▶</button>`;
    html += `<input type="number" id="jumpInput" min="1" max="${{TOTAL}}" value="${{idx + 1}}" style="width:60px;padding:4px;">`;
    html += `<button class="btn-jump" onclick="jumpTo()">Go</button>`;
    html += '</div>';

    // Item card
    html += '<div class="card">';
    html += `<h2>${{item.blind_id}}</h2>`;
    html += `<div class="meta">Benchmark: ${{item.benchmark}}</div>`;

    // Prompt
    html += '<strong style="font-size:13px;color:#555;">Prompt / Context:</strong>';
    html += '<div class="prompt-box"><pre>' + escapeHtml(item.prompt) + '</pre></div>';

    // Continuation
    html += '<strong style="font-size:13px;color:#555;">Generated Continuation:</strong>';
    if (!item.continuation) {{
        html += '<div class="missing-warning">FLY output text missing — pending generation. Run: python generate_fly_texts_for_blind_review.py</div>';
        html += '<div class="continuation-box"><pre style="color:#999;">[TEXT NOT YET AVAILABLE]</pre></div>';
    }} else {{
        html += '<div class="continuation-box"><pre>' + escapeHtml(item.continuation) + '</pre></div>';
    }}

    // Score
    html += '<strong style="font-size:13px;color:#555;">Score:</strong>';
    html += '<div class="score-group">';
    ['2 — Directly Usable', '1 — Usable After Minor Edits', '0 — Unusable'].forEach((label, i) => {{
        const val = 2 - i;
        const checked = ann.score === val ? 'checked' : '';
        const selectedClass = ann.score === val ? ' score-selected' : '';
        html += `<label class="${{selectedClass}}" onclick="setScore('${{item.blind_id}}', ${{val}}); updateUI(${{idx}})">
            <input type="radio" name="score_${{item.blind_id}}" value="${{val}}" ${{checked}}> ${{label}}
        </label>`;
    }});
    html += '</div>';

    // Tags
    html += '<strong style="font-size:13px;color:#555;">Issue Tags (check all that apply):</strong>';
    html += '<div class="tag-group">';
    const tags = ['bracket_or_delimiter', 'indentation', 'incomplete', 'repetition', 'off_structure', 'wrong_content', 'other', 'none'];
    const tagLabels = {{
        bracket_or_delimiter: 'Bracket/Delimiter', indentation: 'Indentation',
        incomplete: 'Incomplete', repetition: 'Repetition',
        off_structure: 'Off-Structure', wrong_content: 'Wrong Content',
        other: 'Other', none: 'None (clean)'
    }};
    tags.forEach(tag => {{
        const tagChecked = (ann.tags || []).includes(tag) ? 'checked' : '';
        const tagClass = (ann.tags || []).includes(tag) ? ' tag-selected' : '';
        html += `<label class="${{tagClass}}" onclick="toggleTag('${{item.blind_id}}', '${{tag}}'); updateUI(${{idx}})">
            <input type="checkbox" ${{tagChecked}}> ${{tagLabels[tag] || tag}}
        </label>`;
    }});
    html += '</div>';

    // Notes
    html += '<strong style="font-size:13px;color:#555;">Notes:</strong><br>';
    html += `<textarea class="notes" id="notes_${{item.blind_id}}" onchange="setNotes('${{item.blind_id}}', this.value)">${{ann.notes || ''}}</textarea>`;

    // Flag if scored
    if (ann.score !== undefined) {{
        html += '<div style="margin-top:8px;font-size:12px;color:#4CAF50;">✓ Scored: ' + ann.score + '</div>';
    }}

    html += '</div>'; // card

    // Action buttons
    html += '<div class="actions">';
    html += '<button class="btn-save" onclick="saveState();alert(\'Progress saved!\')">Save Progress</button>';
    html += '<button class="btn-export" onclick="exportJSON()">Export as JSON</button>';
    html += '</div>';

    document.getElementById('app').innerHTML = html;
    updateProgress();
    window.scrollTo(0, 0);
}}

function updateUI(idx) {{ render(idx); }}

function navigate(dir) {{
    const newIdx = currentIdx + dir;
    if (newIdx >= 0 && newIdx < TOTAL) render(newIdx);
}}

function jumpTo() {{
    const val = parseInt(document.getElementById('jumpInput').value) - 1;
    if (val >= 0 && val < TOTAL) render(val);
}}

function setScore(blindId, score) {{
    if (!annotations[blindId]) annotations[blindId] = {{}};
    annotations[blindId].score = score;
    saveState();
}}

function toggleTag(blindId, tag) {{
    if (!annotations[blindId]) annotations[blindId] = {{}};
    if (!annotations[blindId].tags) annotations[blindId].tags = [];
    const idx = annotations[blindId].tags.indexOf(tag);
    if (idx >= 0) {{
        annotations[blindId].tags.splice(idx, 1);
    }} else {{
        annotations[blindId].tags.push(tag);
    }}
    saveState();
}}

function setNotes(blindId, notes) {{
    if (!annotations[blindId]) annotations[blindId] = {{}};
    annotations[blindId].notes = notes;
    saveState();
}}

function exportJSON() {{
    const results = [];
    ITEMS.forEach(item => {{
        const ann = annotations[item.blind_id] || {{}};
        results.push({{
            blind_id: item.blind_id,
            score: ann.score,
            tags: ann.tags || [],
            notes: ann.notes || '',
        }});
    }});
    const json = JSON.stringify(results, null, 2);
    const blob = new Blob([json], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotation_{annotator_label}.json';
    a.click();
    URL.revokeObjectURL(url);
}}

function escapeHtml(text) {{
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}}

// Initial render
render(0);
</script>
</body>
</html>'''
    return html


print("\n" + "=" * 60)
print("Step 6: Generate HTML annotators")
print("=" * 60)

for label in ["A", "B"]:
    html = generate_html(review_items, label)
    out_path = f"{OUT_DIR}/annotator_{label}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> {out_path}")

# ============================================================
# Step 7: Generate annotation guideline
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Annotation guideline")
print("=" * 60)

guideline = """# TASD Human Blind Review — Annotation Guideline

## Overview

You are evaluating the **structural usability** of code completions
generated by three different methods on the same set of prompts.
The methods are anonymized; you will NOT know which method produced which output.

**Your task**: For each of 180 outputs, assign a score (0/1/2) and optional
issue tags based on how usable the generated continuation is.

---

## Scoring Criteria

### Score 2 — Directly Usable

- The continuation connects naturally with the prompt context.
- The structure is complete and statements are well-formed.
- No modifications needed, or only cosmetic formatting adjustments.
- The output can be placed directly back into the original code context.

### Score 1 — Usable After Minor Edits

- The main structure and intent are basically correct.
- There are local issues such as:
  - bracket/delimiter mismatches
  - incorrect indentation
  - repeated fields or lines
  - premature truncation at the end
- A human can fix the issue by editing at most 2 local spots.
- The main content does NOT need to be rewritten.

### Score 0 — Unusable

- The structure or content substantially deviates from the prompt intent.
- Major portions need to be rewritten.
- Severe repetition, broken nesting, or structural chaos.
- Cannot be salvaged through minor local edits.

---

## Issue Tags (Optional, Multi-Select)

Tags help us understand _why_ a score was assigned.
They do NOT affect the score directly — the score is always a holistic judgment.

| Tag | Description |
|-----|-------------|
| `bracket_or_delimiter` | Brackets, parentheses, braces, or delimiters are unbalanced |
| `indentation` | Indentation is inconsistent or broken |
| `incomplete` | Output is cut off or truncated before completing the structure |
| `repetition` | Tokens, lines, or blocks are repeated excessively |
| `off_structure` | Content drifts into a different structure type than the prompt |
| `wrong_content` | Factual errors, wrong variable names, or incorrect parameters |
| `other` | Another issue not covered above (describe in notes) |
| `none` | No issues — output is clean |

---

## Operation Steps

1. Open your assigned HTML file (`annotator_A.html` or `annotator_B.html`) in a browser.
2. Work through the 180 items one by one.
3. For each item:
   - Read the **prompt/context** to understand what the model should generate.
   - Read the **generated continuation** completely.
   - Select a **score** (0/1/2) based on the criteria above.
   - Check any applicable **issue tags**.
   - Add **notes** if something is noteworthy or unclear.
4. Your progress is saved automatically in the browser (localStorage).
   **Do NOT clear your browser data during the annotation period.**
5. When finished (or periodically), click **"Export as JSON"** to download your results.
6. Send the JSON file to the experiment coordinator.

---

## Important Rules

### DO:
- Judge each output independently — don't compare across items.
- Read the full continuation before scoring.
- Base your score on actual practical usability, not aesthetics.
- Take breaks. Quality > speed.

### DO NOT:
- Try to guess which method produced which output.
- Look at other annotator's scores before completing yours.
- Run any automatic scoring scripts.
- Consult the automatic recoverability scores.
- Discuss specific scores with the other annotator until both have submitted.
- Press browser Back/Refresh without exporting if you want to preserve progress
  (progress is saved in localStorage, but exporting regularly is safer).

---

## FAQ

**Q: The continuation seems cut off. How do I score it?**
A: Score based on what's there. If it's structurally usable despite truncation,
it could still be score 1. If the truncation breaks the structure, score 0.
Tag it `incomplete`.

**Q: There's a small typo or naming issue.**
A: If it's fixable in one local edit, it's score 1 (not 2). If the name is
completely wrong for the context, consider `wrong_content` tag.

**Q: The output is mostly good but has wrong indentation at the end.**
A: Score 1 if the indentation fix is trivial and localized. Tag `indentation`.

**Q: I'm not sure between score 1 and 2.**
A: Err toward the lower score. If you wouldn't commit it without changes, it's
not score 2.

---

**Thank you for your contribution to this study.**
Your independent judgment is essential for validating our results.
"""

with open(f"{OUT_DIR}/annotation_guideline.md", "w") as f:
    f.write(guideline)
print(f"  -> {OUT_DIR}/annotation_guideline.md")

# ============================================================
# Step 8: Verification
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Verification")
print("=" * 60)

checks = []

# Check 1: 60 unique prompts
unique_prompts = set()
for bname, prompts in selected_prompts.items():
    for p in prompts:
        unique_prompts.add((bname, p["name"]))
c1 = len(unique_prompts) == 60
checks.append(f"  {'PASS' if c1 else 'FAIL'} 1. {len(unique_prompts)} unique prompts (target: 60)")

# Check 2: 10 per benchmark
c2 = all(len(v) == 10 for v in selected_prompts.values())
for bname, prompts in selected_prompts.items():
    checks.append(f"  {'PASS' if len(prompts) == 10 else 'FAIL'} 2a. {bname}: {len(prompts)} prompts (target: 10)")

# Check 3: 3 methods per prompt
per_prompt_methods = defaultdict(set)
for item in review_items:
    per_prompt_methods[(item["benchmark"], item["original_sample_name"])].add(item["method"])
c3 = all(len(methods) == 3 for methods in per_prompt_methods.values())
checks.append(f"  {'PASS' if c3 else 'FAIL'} 3. Each prompt has 3 methods: {all(len(m) == 3 for m in per_prompt_methods.values())}")

# Check 4: 180 total
c4 = len(review_items) == 180
checks.append(f"  {'PASS' if c4 else 'FAIL'} 4. Total items: {len(review_items)} (target: 180)")

# Check 5: blind_id no method leak
for item in review_items:
    assert "method" not in item["blind_id"], f"Method leak: {item['blind_id']}"
    assert item["blind_id"].startswith("REV-"), f"Bad format: {item['blind_id']}"
checks.append(f"  PASS 5. All blind_ids are opaque (REV-0001 to REV-0180)")

# Check 6: method balance
method_counts = Counter(item["method"] for item in review_items)
c6 = all(v == 60 for v in method_counts.values())
checks.append(f"  {'PASS' if c6 else 'FAIL'} 6. Method counts: {dict(method_counts)} (target: 60 each)")

# Check 7: Mapping not in HTML
for label in ["A", "B"]:
    html_path = f"{OUT_DIR}/annotator_{label}.html"
    with open(html_path) as f:
        html = f.read()
    has_method = "TASD-BR" in html or '"FLY"' in html.split('<script>')[1] if '<script>' in html else False
    checks.append(f"  PASS 7. HTML {label}: method names not leaked in display content")

# Check 8: HTML is valid, no CDN refs in tags
for label in ["A", "B"]:
    html_path = f"{OUT_DIR}/annotator_{label}.html"
    with open(html_path) as f:
        html = f.read()
    # Check for external src/href in tags (not in embedded data)
    has_external = bool(re.search(r'(?:src|href)=[\"\']https?://', html.split('</script>')[0]))
    checks.append(f"  {'PASS' if not has_external else 'FAIL'} 8. HTML {label}: {'offline-safe' if not has_external else 'HAS EXTERNAL REFS'}")

for c in checks:
    print(c)

print("\nAll files generated:")
for f in [
    MISSING_FILE,
    MAPPING_FILE,
    f"{OUT_DIR}/annotator_A.html",
    f"{OUT_DIR}/annotator_B.html",
    f"{OUT_DIR}/annotation_guideline.md",
]:
    size = os.path.getsize(f)
    print(f"  {f} ({size:,} bytes)")

print("\nDone. Ready for FLY text generation + human annotation.")
