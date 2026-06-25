#!/usr/bin/env python3
"""
Intra-rater reliability workflow for single-annotator score validator.

Steps:
 1. Archive round-1 annotations
 2. Stratified 30-item retest sampling (seed=20260702)
 3. Re-blind with new secret → RET-XXXXXXXX IDs
 4. Generate retest HTML (clean, no round-1 leakage)
 5. Generate retest annotation guideline
 6. Verify integrity

Output:
  results/score_validator_review/private/annotations_A_round1.json
  results/score_validator_review/private/retest_mapping_private.json
  results/score_validator_review/private/annotator_A_retest30.html
  results/score_validator_review/annotation_guideline_retest.md
"""

import json, os, sys, hashlib, secrets, random, re
from collections import Counter, defaultdict

PRIVATE_DIR = "results/score_validator_review/private"
os.makedirs(PRIVATE_DIR, exist_ok=True)

RETEST_SEED = 20260702

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Archive round-1 annotations
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Archive round-1 annotations")
print("=" * 60)

src = f"{PRIVATE_DIR}/annotations_A_v2 (1).json"
dst = f"{PRIVATE_DIR}/annotations_A_round1.json"

with open(src) as f:
    round1 = json.load(f)

# Verify integrity
assert len(round1) == 90, f"Expected 90, got {len(round1)}"
ids = [a["blind_id"] for a in round1]
assert len(ids) == len(set(ids)), "Duplicate blind_ids found"

complete = sum(1 for a in round1 if a.get("human_score") is not None
               and a.get("completion_status") is not None
               and a.get("issue_tags") and len(a["issue_tags"]) > 0)
assert complete == 90, f"Only {complete}/90 complete"

with open(dst, "w") as f:
    json.dump(round1, f, indent=2, ensure_ascii=False)
print(f"  Saved {len(round1)} items -> {dst}")
print(f"  Integrity: 90 items, no duplicates, all complete - OK")

# Build round1 lookup by original SVR blind_id
r1_by_svr = {a["blind_id"]: a for a in round1}

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Stratified retest sampling (30 items)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Stratified retest sampling (seed=20260702)")
print("=" * 60)

random.seed(RETEST_SEED)

# Load mapping for benchmark info
with open(f"{PRIVATE_DIR}/blind_mapping_private.json") as f:
    mapping = json.load(f)
m_by_svr = {m["blind_id"]: m for m in mapping}

# Load original items text (prompt + continuation)
with open(f"{PRIVATE_DIR}/annotator_A.html") as f:
    html = f.read()
match = re.search(r'const ITEMS = (\[.*?\]);', html, re.DOTALL)
items_by_svr = {}
if match:
    for item in json.loads(match.group(1)):
        items_by_svr[item["blind_id"]] = item

# Group candidates by round1 score
by_score = defaultdict(list)
for a in round1:
    svr = a["blind_id"]
    if svr in items_by_svr:
        by_score[a["human_score"]].append(svr)

for s in [2, 1, 0]:
    random.shuffle(by_score[s])

print(f"  Round1 score distribution: 2={len(by_score[2])} 1={len(by_score[1])} 0={len(by_score[0])}")

# Sampling strategy:
# - Must include all 1 score-0 items
# - Include all 10 score-2 items (since only 10, take all)
# - Remaining from score-1: 30 - 1 - 10 = 19
retest_svr_ids = []

# MUST include the score-0
retest_svr_ids.extend(by_score[0])  # 1 item

# Take all score-2
retest_svr_ids.extend(by_score[2])  # 10 items

# Need 19 more from score-1
random.shuffle(by_score[1])

# Try to cover benchmarks
retest_benchmarks = set()
for svr in retest_svr_ids:
    m = m_by_svr.get(svr, {})
    retest_benchmarks.add(m.get("benchmark", "?"))

# Fill remaining from score-1, preferring benchmark coverage
remaining = 30 - len(retest_svr_ids)
score1_pool = list(by_score[1])
# Prioritize items from uncovered benchmarks
uncovered_first = []
covered_first = []
for svr in score1_pool:
    b = m_by_svr.get(svr, {}).get("benchmark", "?")
    if b not in retest_benchmarks:
        uncovered_first.append(svr)
    else:
        covered_first.append(svr)
random.shuffle(uncovered_first)
random.shuffle(covered_first)
ordered_pool = uncovered_first + covered_first

for svr in ordered_pool:
    if svr in retest_svr_ids:
        continue
    if len(retest_svr_ids) >= 30:
        break
    retest_svr_ids.append(svr)
    b = m_by_svr.get(svr, {}).get("benchmark", "?")
    retest_benchmarks.add(b)

print(f"  Retest items: {len(retest_svr_ids)}")
r1_sc = Counter(r1_by_svr[svr]["human_score"] for svr in retest_svr_ids)
r1_bm = Counter(m_by_svr.get(svr, {}).get("benchmark", "?") for svr in retest_svr_ids)
print(f"  Round1 scores in retest: {dict(r1_sc)}")
print(f"  Benchmarks in retest: {dict(r1_bm)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Re-blind with new secret
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Re-blind with new secret → RET-XXXXXXXX")
print("=" * 60)

new_secret = secrets.token_hex(32)
new_secret_bytes = hashlib.sha256(new_secret.encode()).digest()
retest_rng = random.Random(new_secret_bytes)

retest_ids_set = set()
while len(retest_ids_set) < 30:
    r = retest_rng.randint(0, 0xFFFFFFFF)
    rid = f"RET-{r:08X}"
    retest_ids_set.add(rid)

retest_ids = sorted(retest_ids_set)
retest_rng.shuffle(retest_ids)

# Create mapping: retest_blind_id → original_blind_id + benchmark
retest_mapping = []
for svr, rid in zip(retest_svr_ids, retest_ids):
    m = m_by_svr.get(svr, {})
    retest_mapping.append({
        "retest_blind_id": rid,
        "original_blind_id": svr,
        "benchmark": m.get("benchmark", "?")
    })

retest_mapping_path = f"{PRIVATE_DIR}/retest_mapping_private.json"
with open(retest_mapping_path, "w") as f:
    json.dump(retest_mapping, f, indent=2, ensure_ascii=False)
print(f"  Saved {retest_mapping_path}")

# Save new secret (separate file to avoid contamination)
secret_path = f"{PRIVATE_DIR}/retest_blinding_secret.txt"
with open(secret_path, "w") as f:
    f.write(new_secret)

# Verify RET IDs are distinct from SVR
svr_ids_set = set(a["blind_id"] for a in round1)
ret_only = [rid for _, rid in zip(retest_svr_ids, retest_ids)]
assert not any(r in svr_ids_set or r.startswith("SVR-") for r in ret_only), "ID collision!"
print(f"  RET IDs verified: all distinct from SVR IDs")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Generate retest HTML
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Generate retest HTML")
print("=" * 60)

# Build clean items with only retest_blind_id, prompt, continuation
retest_items = []
for rid, svr in zip(retest_ids, retest_svr_ids):
    item = items_by_svr.get(svr, {})
    retest_items.append({
        "retest_blind_id": rid,
        "prompt": item.get("prompt", ""),
        "continuation": item.get("continuation", ""),
    })

# Shuffle for presentation
random.shuffle(retest_items)

STORAGE_KEY = "svr_retest_A"

items_json = json.dumps(retest_items, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Score Validator — Retest (Annotator A, 30 items)</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #6a1b9a; color: #fff; padding: 16px 24px; position: sticky; top: 0; z-index: 100; display: flex; justify-content: space-between; align-items: center; }}
.header h1 {{ font-size: 18px; }}
.progress {{ font-size: 14px; opacity: 0.9; }}
.container {{ max-width: 960px; margin: 24px auto; padding: 0 16px; }}
.card {{ background: #fff; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card h2 {{ font-size: 16px; color: #6a1b9a; margin-bottom: 8px; }}
.meta {{ font-size: 12px; color: #888; margin-bottom: 16px; }}
.prompt-box, .text-box {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; margin: 8px 0 16px; overflow-x: auto; max-height: 400px; overflow-y: auto; }}
.prompt-box pre, .text-box pre {{ font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all; margin: 0; font-family: "SF Mono", "Menlo", "Monaco", monospace; }}
.missing-warning {{ color: #c62828; font-weight: bold; padding: 8px; background: #ffebee; border-radius: 4px; }}
.section-title {{ font-size: 15px; font-weight: 600; color: #6a1b9a; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #6a1b9a; padding-bottom: 4px; }}
.score-group, .status-group, .tag-group {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.score-group label, .status-group label, .tag-group label {{ display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; cursor: pointer; font-size: 13px; background: #fff; transition: all 0.15s; user-select: none; }}
.score-group label:hover, .status-group label:hover, .tag-group label:hover {{ border-color: #6a1b9a; background: #f3e5f5; }}
.score-group label.sel, .status-group label.sel, .tag-group label.tsel {{ border-color: #6a1b9a; background: #e1bee7; font-weight: 600; }}
.score-group label input, .status-group label input, .tag-group label input {{ margin: 0; }}
.notes {{ width: 100%; min-height: 60px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; font-family: inherit; resize: vertical; }}
.completion-indicator {{ padding: 10px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-top: 12px; }}
.completion-indicator.done {{ background: #f3e5f5; color: #6a1b9a; border: 1px solid #ce93d8; }}
.completion-indicator.partial {{ background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }}
.actions {{ display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }}
.btn-save, .btn-export, .btn-prev, .btn-next, .btn-jump {{ padding: 8px 18px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; }}
.btn-save {{ background: #6a1b9a; color: #fff; }}
.btn-export {{ background: #ff8f00; color: #fff; }}
.btn-prev, .btn-next {{ background: #e0e0e0; color: #333; }}
.btn-jump {{ background: #ce93d8; color: #000; }}
.btn-save:hover, .btn-export:hover, .btn-prev:hover, .btn-next:hover, .btn-jump:hover {{ opacity: 0.85; }}
button:disabled {{ opacity: 0.4; cursor: default; }}
input[type="number"] {{ padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }}
.retest-banner {{ background: #f3e5f5; border: 1px solid #ce93d8; border-radius: 6px; padding: 10px 16px; margin-bottom: 20px; font-size: 13px; color: #4a148c; }}
.footer {{ text-align: center; padding: 24px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Score Validator — Retest <span style="opacity:0.7;font-size:13px;">(Annotator A, 30 items)</span></h1>
    <div class="progress" id="progress">Initializing...</div>
</div>
<div class="container">
<div class="retest-banner">
    <strong>ReteSt instructions:</strong> You have previously scored 90 items. This is a 30-item retest subset.
    Do NOT look up your previous scores. Score independently as if seeing these for the first time.
    Use the same scoring rules: 0/1/2 for recoverability, completion status, and at least one issue tag.
</div>
<div id="app"></div>
</div>
<div class="footer">Intra-Rater Reliability — Retest Phase</div>

<script>
const STORAGE_KEY = '{STORAGE_KEY}';
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
    var ann = annotations[item.retest_blind_id] || {{}};
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
    h += '<h2>' + item.retest_blind_id + '</h2>';
    h += '<div class="meta">Item ' + (idx+1) + ' of 30</div>';

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
    h += '<div style="font-size:12px;color:#666;margin-bottom:6px;">Evaluate the raw output as-is. Do NOT ignore truncation. Score by repair cost.</div>';
    h += '<div class="score-group">';
    var scores = [
        ['2','2 — Directly usable: clean, complete, no edits needed'],
        ['1','1 — Locally recoverable: 1-2 local edits fix it'],
        ['0','0 — Unrecoverable: >2 edits needed, major rewrite, chaotic']
    ];
    for (var si = 0; si < scores.length; si++) {{
        var v = scores[si][0], lbl = scores[si][1];
        var sel = hs === parseInt(v) ? ' sel' : '';
        var chk = hs === parseInt(v) ? 'checked' : '';
        h += '<label class="' + sel + '" onclick="setHumanScore(\\'' + item.retest_blind_id + '\\',' + v + ');update(' + idx + ')">';
        h += '<input type="radio" name="hs_' + item.retest_blind_id + '" value="' + v + '" ' + chk + '> ' + lbl;
        h += '</label>';
    }}
    h += '</div>';

    // ---- Section 2: Completion Status ----
    h += '<div class="section-title">2. Completion Status (REQUIRED)</div>';
    h += '<div class="status-group">';
    var statuses = [
        ['complete','Complete — ends at natural boundary'],
        ['tail_cutoff','Tail Cutoff — clear continuous incomplete tail'],
        ['severe_incomplete','Severe Incomplete — major structures unfinished']
    ];
    for (var si = 0; si < statuses.length; si++) {{
        var v = statuses[si][0], lbl = statuses[si][1];
        var sel = cs === v ? ' sel' : '';
        var chk = cs === v ? 'checked' : '';
        h += '<label class="' + sel + '" onclick="setCompletion(\\'' + item.retest_blind_id + '\\',\\'' + v + '\\');update(' + idx + ')">';
        h += '<input type="radio" name="cs_' + item.retest_blind_id + '" value="' + v + '" ' + chk + '> ' + lbl;
        h += '</label>';
    }}
    h += '</div>';

    // ---- Section 3: Issue Tags ----
    h += '<div class="section-title">3. Issue Tags (at least one; "none" excludes others)</div>';
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
        h += '<label class="' + cls + '" onclick="toggleTag(\\'' + item.retest_blind_id + '\\',\\'' + t + '\\');update(' + idx + ')">';
        h += '<input type="checkbox" ' + chk + '> ' + (tl[t]||t);
        h += '</label>';
    }}
    h += '</div>';

    // ---- Section 4: Notes ----
    h += '<div class="section-title">4. Notes (optional)</div>';
    h += '<textarea class="notes" id="n_' + item.retest_blind_id + '" onchange="setNotes(\\'' + item.retest_blind_id + '\\',this.value)">' + (ann.notes||'') + '</textarea>';

    // ---- Completion indicator ----
    if (hs !== undefined && cs !== undefined && ann.issue_tags !== undefined && ann.issue_tags.length > 0) {{
        h += '<div class="completion-indicator done">Fully scored: Score ' + hs + ', Completion: ' + cs + '</div>';
    }} else {{
        h += '<div class="completion-indicator partial">Incomplete: score, completion, and at least one issue tag required</div>';
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
        var a = annotations[it.retest_blind_id] || {{}};
        return {{
            retest_blind_id: it.retest_blind_id,
            human_score: a.human_score,
            completion_status: a.completion_status,
            issue_tags: a.issue_tags || [],
            notes: a.notes || ''
        }};
    }});
    var b = new Blob([JSON.stringify(r, null, 2)], {{type: 'application/json'}});
    var u = URL.createObjectURL(b);
    var a = document.createElement('a');
    a.href = u; a.download = 'annotations_A_retest30.json'; a.click();
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

html_path = f"{PRIVATE_DIR}/annotator_A_retest30.html"
with open(html_path, "w") as f:
    f.write(html_content)
print(f"  -> {html_path} ({len(html_content)} bytes)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Retest annotation guideline
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Retest annotation guideline")
print("=" * 60)

guideline = """# Score Validator — Retest Annotation Guideline

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
"""

with open("results/score_validator_review/annotation_guideline_retest.md", "w") as f:
    f.write(guideline)
print(f"  Saved annotation_guideline_retest.md")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Retest manifest (aggregate only)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Retest manifest")
print("=" * 60)

manifest = {
    "experiment": "intra_rater_reliability_retest",
    "description": "30-item stratified retest from original 90-item score validator",
    "retest_sampling_seed": RETEST_SEED,
    "total_retest_items": 30,
    "original_round1_items": 90,
    "retest_score_distribution_from_round1": {str(k): v for k, v in dict(r1_sc).items()},
    "retest_benchmark_distribution": {str(k): v for k, v in dict(r1_bm).items()},
    "retest_blind_id_format": "RET-XXXXXXXX",
    "localStorage_key": STORAGE_KEY,
    "date_generated": __import__('datetime').datetime.now().isoformat(),
}

with open("results/score_validator_review/retest_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"  Saved retest_manifest.json")

# ══════════════════════════════════════════════════════════════════════════
# INTEGRITY CHECKS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("INTEGRITY CHECKS")
print("=" * 60)
passed = 0
total = 0

def check(desc, cond):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print(f"  PASS {desc}")
    else:
        print(f"  FAIL {desc}")

check("1. Round1 exactly 90 items", len(round1) == 90)
check("2. Retest exactly 30 items", len(retest_items) == 30)
check("3. No RET ID matches any SVR ID", not any(r.startswith("SVR-") for r in retest_ids))
check("3. RET IDs all in RET-XXXXXXXX format", all(r.startswith("RET-") and len(r)==12 for r in retest_ids))
check("4. 30 retest items are unique", len(set(it["retest_blind_id"] for it in retest_items)) == 30)
check("5. Contains the only round1 score-0 item", r1_sc.get(0, 0) >= 1)
check("6. Contains >=8 round1 score-2 items", r1_sc.get(2, 0) >= 8)
check("7. Covers all 6 benchmarks", len(r1_bm) == 6)
check("8. HTML: no SVR- ID leak", "SVR-" not in html_content.split("const ITEMS")[1] if "const ITEMS" in html_content else True)
check("8. HTML: no 'round1' text", "round1" not in html_content.lower())
check("9. localStorage key is new: svr_retest_A", STORAGE_KEY == "svr_retest_A")
check("10. Retest mapping saved", os.path.exists(retest_mapping_path))
check("10. Round1 archive saved", os.path.exists(dst))

print(f"\n  PASSED: {passed}/{total}")

print("\nDone!")
