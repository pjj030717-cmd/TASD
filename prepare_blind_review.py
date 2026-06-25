#!/usr/bin/env python3
"""
Prepare TASD Human Blind Review Materials (v2 — secrets-based blinding).

Key changes from v1:
  - Uses secrets.token_hex for blinding (not public seed)
  - Imports official BR rerun policy from src/br_rerun_policy.py
  - New blind_id format BRV-XXXXXXXX (not REV-XXXX)
  - Outputs to private/ directory (gitignored)
  - A and B annotators get independently shuffled orders
  - FLY texts loaded from fly_texts_for_blind_review.json if available

Does NOT run model inference. Does NOT generate synthetic scores.
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict, Counter

# Extend path to import official BR policy
sys.path.insert(0, os.path.dirname(__file__))
from src.br_rerun_policy import is_br_rerun

# ─── Constants ──────────────────────────────────────────────────────────────
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
PRIVATE_DIR = f"{OUT_DIR}/private"
SAMPLING_SEED = 20260624
SAMPLES_PER_BENCHMARK = 10

os.makedirs(PRIVATE_DIR, exist_ok=True)

# ─── Load Blinding Secret ────────────────────────────────────────────────────
SECRET_FILE = f"{PRIVATE_DIR}/blinding_secret.txt"
if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE) as f:
        BLINDING_SECRET = f.read().strip()
else:
    import secrets
    BLINDING_SECRET = secrets.token_hex(32)
    with open(SECRET_FILE, "w") as f:
        f.write(BLINDING_SECRET)

# Derive a deterministic RNG from the secret for ID generation
secret_bytes = hashlib.sha256(BLINDING_SECRET.encode()).digest()

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Stratified Random Sampling (uses public SAMPLING_SEED)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Stratified Random Sampling (seed=20260624)")
print("=" * 60)

random.seed(SAMPLING_SEED)
selected_prompts = {}

for bname, data_file in BENCHMARKS:
    with open(data_file) as f:
        samples = [json.loads(line.strip()) for line in f.readlines()]
    n = len(samples)
    indices = sorted(random.sample(range(n), SAMPLES_PER_BENCHMARK))
    selected = [{"idx": i, "name": samples[i]["name"], "prompt": samples[i]["prompt"]}
                for i in indices]
    selected_prompts[bname] = selected
    print(f"  {bname}: {len(selected)}/{n} selected")

total_prompts = sum(len(v) for v in selected_prompts.values())
print(f"\nTotal selected prompts: {total_prompts}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Load per-sample texts
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: Load texts")
print("=" * 60)

# AR texts (quality files have 'text' field)
ar_texts = {}
for bname, _ in BENCHMARKS:
    fpath = f"{CKPT_DIR}/{bname}_AR_quality.json"
    ar_texts[bname] = {}
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        for i, s in enumerate(data):
            ar_texts[bname][i] = {"text": s.get("text", ""), "name": s["name"]}
        print(f"  AR {bname}: {len(ar_texts[bname])} texts loaded")
    else:
        print(f"  AR {bname}: FILE NOT FOUND: {fpath}")

# TASD-FG texts
fg_texts = {}
for bname, _ in BENCHMARKS:
    fpath = f"{CKPT_DIR}/{bname}_TASDFG.json"
    fg_texts[bname] = {}
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        for i, s in enumerate(data):
            fg_texts[bname][i] = {"text": s.get("text", ""), "name": s["name"]}
        print(f"  TASD-FG {bname}: {len(fg_texts[bname])} texts loaded")
    else:
        print(f"  TASD-FG {bname}: FILE NOT FOUND: {fpath}")

# FLY texts (from standalone generation, if available)
fly_texts = {}
fly_cache_path = f"{PRIVATE_DIR}/fly_texts_for_blind_review.json"
if os.path.exists(fly_cache_path):
    with open(fly_cache_path) as f:
        fly_cache = json.load(f)
    for key, entry in fly_cache.items():
        # key format: "benchmark/sample_name"
        bname = entry["benchmark"]
        sname = entry["sample_name"]
        fly_texts[(bname, sname)] = entry["text"]
    print(f"  FLY: {len(fly_texts)} texts loaded from {fly_cache_path}")
else:
    print(f"  FLY: NO TEXTS AVAILABLE. Run generate_fly_texts_for_blind_review.py first.")

# TASD-FG risk data for BR decision (from official metrics)
# NOTE: TASD-FG uses 'name' field, not 'sample_idx'. Build lookup by name,
# then during iteration match by sample name.
fg_risk_by_name = {}
metrics_file = "results/all_methods_structural_recoverability.json"
if os.path.exists(metrics_file):
    with open(metrics_file) as f:
        metrics = json.load(f)
    for s in metrics.get("TASD-FG", []):
        bname = s["benchmark"]
        # TASD-FG has 'name' not 'sample_idx'
        sname = s["name"]
        fg_risk_by_name[(bname, sname)] = dict(s)
    print(f"  BR risk data: {len(fg_risk_by_name)} samples loaded (by name)")
else:
    print(f"  WARNING: {metrics_file} not found")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Build review items with official BR policy
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: Build review items (official BR policy)")
print("=" * 60)

review_items = []
missing_samples = []
br_rerun_report = []

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
                "method": "AR", "benchmark": bname,
                "sample_idx": idx, "sample_name": name,
                "reason": "AR text missing or empty"
            })

        # TASD-FG
        fg_entry = fg_texts.get(bname, {}).get(idx, {})
        fg_text_val = fg_entry.get("text", "")
        if not fg_text_val:
            missing_samples.append({
                "method": "TASD-FG", "benchmark": bname,
                "sample_idx": idx, "sample_name": name,
                "reason": "TASD-FG text missing or empty"
            })

        # TASD-BR via official policy (look up by sample name)
        fg_sample = fg_risk_by_name.get((bname, name), {})
        rerun = is_br_rerun(fg_sample)
        br_text = ar_text if rerun else fg_text_val
        br_rerun_report.append({
            "benchmark": bname, "sample_idx": idx, "sample_name": name,
            "rerun": rerun,
            "bracket_balance": fg_sample.get("bracket_balance", 1.0),
            "is_truncated": fg_sample.get("is_truncated", 0),
        })

        # FLY
        fly_text_val = fly_texts.get((bname, name))

        review_items.append({"method": "AR", "benchmark": bname,
                             "sample_idx": idx, "sample_name": name,
                             "prompt": prompt_text, "text": ar_text})
        review_items.append({"method": "FLY", "benchmark": bname,
                             "sample_idx": idx, "sample_name": name,
                             "prompt": prompt_text, "text": fly_text_val})
        review_items.append({"method": "TASD-BR", "benchmark": bname,
                             "sample_idx": idx, "sample_name": name,
                             "prompt": prompt_text, "text": br_text,
                             "br_rerun": rerun})

print(f"  Total items: {len(review_items)}")
n_rerun = sum(1 for r in br_rerun_report if r["rerun"])
print(f"  BR rerun: {n_rerun}/60 ({100*n_rerun/60:.1f}%)")

# Per-benchmark rerun
print("\n  BR rerun per benchmark:")
for bname, _ in BENCHMARKS:
    bm_rerun = [r for r in br_rerun_report if r["benchmark"] == bname]
    r_count = sum(1 for r in bm_rerun if r["rerun"])
    print(f"    {bname}: {r_count}/10")
    for r in bm_rerun:
        if r["rerun"]:
            print(f"      rerun: {r['sample_name']} (bb={r['bracket_balance']:.2f})")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Generate blind IDs from secret
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: Generate blind IDs (secrets-based)")
print("=" * 60)

blind_rng = random.Random(secret_bytes)
ids_set = set()
while len(ids_set) < len(review_items):
    r = blind_rng.randint(0, 0xFFFFFFFF)
    bid = f"BRV-{r:08X}"
    ids_set.add(bid)
blind_ids = sorted(ids_set)

# Assign to items
for i, item in enumerate(review_items):
    item["blind_id"] = blind_ids[i]

print(f"  {len(blind_ids)} blind IDs generated (BRV-XXXXXXXX format)")
print(f"  Samples: {blind_ids[0]}, {blind_ids[60]}, {blind_ids[120]}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Shuffle (independently for A and B)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: Independent shuffle for A and B")
print("=" * 60)

rng_a = random.Random(secret_bytes[:16])
rng_b = random.Random(secret_bytes[16:])

order_a = list(range(len(review_items)))
order_b = list(range(len(review_items)))
rng_a.shuffle(order_a)
rng_b.shuffle(order_b)

print(f"  A order differs from B: {order_a != order_b}")
print(f"  A[0:3] = {order_a[:3]}, B[0:3] = {order_b[:3]}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Generate private mapping
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: Private mapping")
print("=" * 60)

blind_mapping = []
for item in review_items:
    entry = {
        "blind_id": item["blind_id"],
        "method": item["method"],
        "benchmark": item["benchmark"],
        "sample_idx": item["sample_idx"],
        "sample_name": item["sample_name"],
    }
    if item["method"] == "TASD-BR":
        entry["br_rerun"] = item.get("br_rerun", False)
    blind_mapping.append(entry)

mapping_path = f"{PRIVATE_DIR}/blind_mapping_private.json"
with open(mapping_path, "w") as f:
    json.dump(blind_mapping, f, indent=2, ensure_ascii=False)
print(f"  -> {mapping_path}")
print("  WARNING: DO NOT share this file with annotators!")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Generate HTML annotators
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 7: Generate HTML annotators")
print("=" * 60)

def generate_html(items, order, annotator_label):
    """Generate single-file offline HTML annotator."""
    ordered = [items[i] for i in order]

    # Prepare clean data (no method, no sample idx, no auto score)
    clean = []
    for it in ordered:
        entry = {
            "blind_id": it["blind_id"],
            "benchmark": it["benchmark"],
            "prompt": it["prompt"],
            "text": it["text"],
        }
        clean.append(entry)

    items_json = json.dumps(clean, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TASD Blind Review — {annotator_label}</title>
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
.card .meta {{ font-size: 12px; color: #888; margin-bottom: 12px; }}
.prompt-box {{
    background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;
    padding: 12px; margin-bottom: 12px; max-height: 200px; overflow-y: auto;
}}
.prompt-box pre {{
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 13px; white-space: pre-wrap; word-break: break-all;
    background: transparent; padding: 0; line-height: 1.4;
}}
.text-box {{
    border: 1px solid #90caf9; border-radius: 4px; padding: 12px;
    margin-bottom: 16px; max-height: 400px; overflow-y: auto;
    background: #f3f9ff;
}}
.text-box pre {{
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace;
    font-size: 13px; white-space: pre-wrap; word-break: break-all;
    background: transparent; padding: 0; line-height: 1.4;
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
.score-group label.sel {{ background: #bbdefb; border-color: #1565c0; }}
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
.tag-group label.tsel {{ background: #e1bee7; border-color: #7b1fa2; }}
.tag-group input {{ margin: 0; }}
.notes {{ width: 100%; min-height: 48px; font-family: inherit; font-size: 13px;
    padding: 8px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }}
.actions {{ margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }}
.actions button {{
    padding: 10px 20px; border: none; border-radius: 4px; font-size: 14px;
    cursor: pointer; transition: opacity 0.2s;
}}
.btn-save {{ background: #2196F3; color: white; }}
.btn-export {{ background: #4CAF50; color: white; }}
.btn-prev {{ background: #9C27B0; color: white; }}
.btn-next {{ background: #FF5722; color: white; }}
.btn-jump {{ background: #607D8B; color: white; }}
.actions button:hover {{ opacity: 0.85; }}
.actions button:disabled {{ opacity: 0.4; cursor: default; }}
</style>
</head>
<body>
<div class="header">
    <h1>TASD Blind Review — {annotator_label}</h1>
    <div class="progress" id="progress">Initializing...</div>
</div>
<div class="container" id="app"></div>

<script>
const ITEMS = {items_json};
const TOTAL = ITEMS.length;
let annotations = {{}};
let currentIdx = 0;

const saved = localStorage.getItem('tasd_blind_v2_{annotator_label}');
if (saved) {{ try {{ annotations = JSON.parse(saved); }} catch(e) {{}} }}

function saveState() {{
    localStorage.setItem('tasd_blind_v2_{annotator_label}', JSON.stringify(annotations));
}}

function updateProgress() {{
    const done = Object.keys(annotations).filter(k => annotations[k].score !== undefined).length;
    document.getElementById('progress').textContent =
        `Scored: ${{done}} / ${{TOTAL}} (${{Math.round(done/TOTAL*100)}}%)`;
}}

function render(idx) {{
    currentIdx = idx;
    const item = ITEMS[idx];
    const ann = annotations[item.blind_id] || {{}};

    let h = '';
    h += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;">';
    h += `<button class="btn-prev" onclick="nav(-1)" ${{idx===0?'disabled':''}}>Prev</button>`;
    h += `<span style="font-size:13px;color:#666;">Item ${{idx+1}} / ${{TOTAL}}</span>`;
    h += `<button class="btn-next" onclick="nav(1)" ${{idx===TOTAL-1?'disabled':''}}>Next</button>`;
    h += `<input type="number" id="jumpIn" min="1" max="${{TOTAL}}" value="${{idx+1}}" style="width:60px;padding:4px;">`;
    h += `<button class="btn-jump" onclick="go()">Go</button>`;
    h += '</div>';

    h += '<div class="card">';
    h += `<h2>${{item.blind_id}}</h2>`;
    h += `<div class="meta">Benchmark: ${{item.benchmark}}</div>`;

    h += '<strong style="font-size:13px;color:#555;">Prompt / Context:</strong>';
    h += '<div class="prompt-box"><pre>' + esc(item.prompt) + '</pre></div>';

    h += '<strong style="font-size:13px;color:#555;">Generated Continuation:</strong>';
    if (!item.text) {{
        h += '<div class="missing-warning">Text not available</div>';
    }} else {{
        h += '<div class="text-box"><pre>' + esc(item.text) + '</pre></div>';
    }}

    // Score
    h += '<strong style="font-size:13px;color:#555;">Score:</strong>';
    h += '<div class="score-group">';
    [['2','Directly Usable'], ['1','Usable After Minor Edits'], ['0','Unusable']].forEach(([v,label]) => {{
        const sel = ann.score === parseInt(v) ? ' sel' : '';
        const chk = ann.score === parseInt(v) ? 'checked' : '';
        h += `<label class="${{sel}}" onclick="setScore('${{item.blind_id}}',${{v}});update(${{idx}})">
            <input type="radio" name="s_${{item.blind_id}}" value="${{v}}" ${{chk}}> ${{v}} &mdash; ${{label}}
        </label>`;
    }});
    h += '</div>';

    // Tags
    h += '<strong style="font-size:13px;color:#555;">Issue Tags:</strong>';
    h += '<div style="font-size:11px;color:#888;margin-bottom:6px;">';
    h += '<b>incomplete content</b> = structure/content is broken; ';
    h += '<b>cut off at end</b> = text ends mid-structure but preceding content looks structurally sound (likely hit generation length limit). ';
    h += 'Both can apply if the trunk is broken AND the end is cut off.</div>';
    h += '<div class="tag-group">';
    const tags = ['bracket_or_delimiter','indentation','incomplete','cut_off','repetition','off_structure','wrong_content','other','none'];
    const tl = {{bracket_or_delimiter:'Bracket/Delimiter',indentation:'Indentation',incomplete:'Incomplete content',cut_off:'Cut off at end',repetition:'Repetition',off_structure:'Off-Structure',wrong_content:'Wrong Content',other:'Other',none:'None'}};
    tags.forEach(t => {{
        const chk = (ann.tags||[]).includes(t) ? 'checked' : '';
        const cls = (ann.tags||[]).includes(t) ? ' tsel' : '';
        h += `<label class="${{cls}}" onclick="toggleTag('${{item.blind_id}}','${{t}}');update(${{idx}})">
            <input type="checkbox" ${{chk}}> ${{tl[t]||t}}
        </label>`;
    }});
    h += '</div>';

    h += '<strong style="font-size:13px;color:#555;">Notes:</strong><br>';
    h += `<textarea class="notes" id="n_${{item.blind_id}}" onchange="setNotes('${{item.blind_id}}',this.value)">${{ann.notes||''}}</textarea>`;

    if (ann.score !== undefined) {{
        h += '<div style="margin-top:8px;font-size:12px;color:#4CAF50;">Scored: ' + ann.score + '</div>';
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
function nav(d) {{ const ni = currentIdx + d; if (ni >= 0 && ni < TOTAL) render(ni); }}
function go() {{ const v = parseInt(document.getElementById('jumpIn').value) - 1; if (v >= 0 && v < TOTAL) render(v); }}

function setScore(bid, s) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    annotations[bid].score = s;
    saveState();
}}
function toggleTag(bid, t) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    if (!annotations[bid].tags) annotations[bid].tags = [];
    const i = annotations[bid].tags.indexOf(t);
    if (i >= 0) annotations[bid].tags.splice(i, 1);
    else annotations[bid].tags.push(t);
    saveState();
}}
function setNotes(bid, n) {{
    if (!annotations[bid]) annotations[bid] = {{}};
    annotations[bid].notes = n;
    saveState();
}}
function exportJSON() {{
    const r = ITEMS.map(it => {{
        const a = annotations[it.blind_id] || {{}};
        return {{ blind_id: it.blind_id, score: a.score, tags: a.tags || [], notes: a.notes || '' }};
    }});
    const b = new Blob([JSON.stringify(r, null, 2)], {{type: 'application/json'}});
    const u = URL.createObjectURL(b);
    const a = document.createElement('a');
    a.href = u; a.download = 'annotations_{annotator_label}.json'; a.click();
    URL.revokeObjectURL(u);
}}
function esc(s) {{
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}}
render(0);
</script>
</body>
</html>'''
    return html


for label, order in [("A", order_a), ("B", order_b)]:
    html = generate_html(review_items, order, label)
    out_path = f"{PRIVATE_DIR}/annotator_{label}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> {out_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 8: Save missing report
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 8: Missing samples report")
print("=" * 60)

fly_missing = sum(1 for item in review_items
                  if item["method"] == "FLY" and not item["text"])
print(f"  FLY texts missing: {fly_missing}/60")

missing_report = {
    "fly_missing": [{
        "blind_id": item["blind_id"],
        "benchmark": item["benchmark"],
        "sample_name": item["sample_name"],
        "sample_idx": item["sample_idx"],
    } for item in review_items if item["method"] == "FLY" and not item["text"]],
    "other_missing": missing_samples,
}
with open("results/human_blind_review_missing_samples.json", "w") as f:
    json.dump(missing_report, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Step 9: Manifest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 9: Manifest")
print("=" * 60)

manifest = {
    "version": "v2-secrets-blinding",
    "sampling_seed": SAMPLING_SEED,
    "n_prompts": total_prompts,
    "n_items": len(review_items),
    "methods": ["AR", "FLY", "TASD-BR"],
    "br_rerun_policy": {
        "source_file": "src/br_rerun_policy.py",
        "function": "is_br_rerun()",
        "rule": "bracket_balance < 0.50 AND is_truncated == 0",
        "n_rerun_60": n_rerun,
        "rerun_pct": round(100 * n_rerun / 60, 1),
    },
    "per_benchmark_rerun": {},
    "fly_status": "generated" if fly_missing == 0 else f"{fly_missing}/60 missing",
    "blind_id_format": "BRV-XXXXXXXX (hex, secrets-based, not reproducible without key)",
}
for bname, _ in BENCHMARKS:
    bm_rerun = [r for r in br_rerun_report if r["benchmark"] == bname]
    manifest["per_benchmark_rerun"][bname] = {
        "total": 10,
        "rerun": sum(1 for r in bm_rerun if r["rerun"]),
    }

with open(f"{OUT_DIR}/blind_review_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"  -> {OUT_DIR}/blind_review_manifest.json")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 10: Verification (12 checks)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 10: Verification (12 checks)")
print("=" * 60)

checks = []

# 1. 60 unique prompts
unique = set((it["benchmark"], it["sample_name"]) for it in review_items)
c1 = len(unique) == 60
checks.append(f"  {'PASS' if c1 else 'FAIL'} 1. {len(unique)} unique prompts (target: 60)")

# 2. 10 per benchmark
for bname, _ in BENCHMARKS:
    n_bm = sum(1 for it in review_items if it["benchmark"] == bname) // 3
    c2 = n_bm == 10
    checks.append(f"  {'PASS' if c2 else 'FAIL'} 2. {bname}: {n_bm} prompts (target: 10)")

# 3. Each prompt has 3 methods
by_prompt = defaultdict(set)
for it in review_items:
    by_prompt[(it["benchmark"], it["sample_name"])].add(it["method"])
c3 = all(len(m) == 3 for m in by_prompt.values())
checks.append(f"  {'PASS' if c3 else 'FAIL'} 3. All prompts have 3 methods")

# 4. 180 total
c4 = len(review_items) == 180
checks.append(f"  {'PASS' if c4 else 'FAIL'} 4. {len(review_items)} items (target: 180)")

# 5. All non-FLY texts non-empty
non_fly = [it for it in review_items if it["method"] != "FLY"]
c5 = all(it["text"] and len(it["text"]) > 0 for it in non_fly)
checks.append(f"  {'PASS' if c5 else 'FAIL'} 5. All non-FLY texts non-empty")

# 6. No placeholders
c6 = not any("placeholder" in str(it.get("text", "")).lower()
             or "preview" in str(it.get("text", "")).lower()
             or str(it.get("text", "")).startswith("[TEXT NOT")
             for it in review_items)
checks.append(f"  {'PASS' if c6 else 'FAIL'} 6. No placeholder/preview text")

# 7. HTML does not contain method names
for label in ["A", "B"]:
    hp = f"{PRIVATE_DIR}/annotator_{label}.html"
    with open(hp) as f:
        h = f.read()
    # Check script-tag content for method strings
    script_start = h.find("<script>")
    script_end = h.find("</script>")
    script_text = h[script_start:script_end] if script_start >= 0 else ""
    has_method = 'TASD-BR' in script_text or '"FLY"' in script_text
    c7 = not has_method
    checks.append(f"  {'PASS' if c7 else 'FAIL'} 7. HTML {label}: no method leak")

# 8. HTML doesn't contain source_file or sample_idx in visible data
for label in ["A", "B"]:
    hp = f"{PRIVATE_DIR}/annotator_{label}.html"
    with open(hp) as f:
        h = f.read()
    has_idx = "sample_idx" in h or "original_sample" in h
    c8 = not has_idx
    checks.append(f"  {'PASS' if c8 else 'FAIL'} 8. HTML {label}: no sample_idx leak")

# 9. Old mapping cannot decode new IDs
c9 = not any(bid.startswith("REV-") for bid in blind_ids)
checks.append(f"  {'PASS' if c9 else 'FAIL'} 9. Old REV- IDs cannot map to new BRV- IDs")

# 10. A and B orders differ
c10 = order_a != order_b
checks.append(f"  {'PASS' if c10 else 'FAIL'} 10. A and B different orders")

# 11. Private directory .gitignore
# Check via git
import subprocess
result = subprocess.run(
    ["git", "check-ignore", f"{PRIVATE_DIR}/blind_mapping_private.json"],
    capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
)
c11 = result.returncode == 0
checks.append(f"  {'PASS' if c11 else 'FAIL'} 11. private/ gitignored: {result.stdout.strip() or '(matched)'}")

# 12. New IDs format
c12 = all(bid.startswith("BRV-") and len(bid) == 12 for bid in blind_ids)
checks.append(f"  {'PASS' if c12 else 'FAIL'} 12. All IDs in BRV-XXXXXXXX format")

for c in checks:
    print(c)

all_pass = all("PASS" in c for c in checks)
print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — review above'}")

# Summary
print("\n" + "=" * 60)
print("Files in private/ (DO NOT COMMIT):")
for fn in os.listdir(PRIVATE_DIR):
    fpath = os.path.join(PRIVATE_DIR, fn)
    print(f"  {fpath} ({os.path.getsize(fpath):,} bytes)")
print("\nPublic files:")
for fn in ["blind_review_manifest.json", "annotation_guideline.md",
           "fly_texts_for_blind_review.json"]:
    fpath = os.path.join(OUT_DIR, fn)
    if os.path.exists(fpath):
        print(f"  {fpath} ({os.path.getsize(fpath):,} bytes)")

print(f"\nBR rerun policy: src/br_rerun_policy.py::is_br_rerun()")
print(f"  Rule: bracket_balance < 0.50 AND is_truncated == 0")
print(f"  Result: {n_rerun}/60 rerun ({100*n_rerun/60:.1f}%)")
print("\nDone.")
