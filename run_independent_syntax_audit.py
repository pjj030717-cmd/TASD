#!/usr/bin/env python3
"""
Independent syntax audit for the 90-item score validator corpus.

Checks:
 1. raw_parse: ast.parse(prompt + continuation), noting if prompt itself
    is not a complete Python unit (not_applicable vs syntax_error)
 2. tail_trim_parse: delete 1/2/3/5/10 lines from end, find minimal trim
 3. Text anomalies: repetition, NL explanations, markdown fences, non-code

Output:
  results/score_validator_review/syntax_audit.json
  results/score_validator_review/syntax_audit_report.md

NOTE: AST audit is NOT human ground truth. Many prompts are file fragments.
not_applicable samples must be reported separately, not as failures.
"""

import ast
import json
import os
import re
from collections import Counter

PRIVATE_DIR = "results/score_validator_review/private"
OUT_JSON = "results/score_validator_review/syntax_audit.json"
OUT_REPORT = "results/score_validator_review/syntax_audit_report.md"

TRIM_LINES = [1, 2, 3, 5, 10]

# ── Load data ──

# 1. HTML items (prompt + continuation)
with open(f"{PRIVATE_DIR}/annotator_A.html") as f:
    html = f.read()
match = re.search(r'const ITEMS = (\[.*?\]);', html, re.DOTALL)
items = json.loads(match.group(1))

# 2. Original mapping for metadata
with open(f"{PRIVATE_DIR}/blind_mapping_private.json") as f:
    mapping = json.load(f)
m_by_id = {m["blind_id"]: m for m in mapping}

# 3. Round1 annotations for cross-ref (without contaminating retest)
r1_path = f"{PRIVATE_DIR}/annotations_A_round1.json"
r1_by_id = {}
if os.path.exists(r1_path):
    with open(r1_path) as f:
        for a in json.load(f):
            r1_by_id[a["blind_id"]] = a


# ── Helpers ──

def try_parse(code_text):
    """Try ast.parse. Returns (success, error_msg_or_None)."""
    try:
        ast.parse(code_text)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset})"
    except Exception as e:
        return False, f"Error: {str(e)[:120]}"


def is_complete_unit(code_text):
    """Check if a code string is a syntactically complete Python unit."""
    try:
        ast.parse(code_text)
        return True
    except SyntaxError as e:
        # "unexpected EOF" means incomplete, not necessarily malformed
        if "unexpected EOF" in str(e) or "EOF while" in str(e):
            return False
        # SyntaxError in middle means there's a real error
        return False
    except Exception:
        return False


def trim_tail_lines(code_text, n):
    """Remove last n lines from code_text."""
    lines = code_text.split('\n')
    if n >= len(lines):
        return ''
    return '\n'.join(lines[:len(lines) - n])


def check_tail_trim(prompt, continuation):
    """Try tail-trimming continuation only, finding minimal lines to remove."""
    full = prompt + continuation
    # Parse prompt alone to check
    prompt_ok, _ = try_parse(prompt)

    # First try full text
    full_ok, full_err = try_parse(full)
    if full_ok:
        return {
            "full_parses": True,
            "trim_lines_required": 0,
            "trim_parse_result": "already_parses"
        }

    # Try trimming from the END of continuation ONLY
    cont_lines = continuation.split('\n')
    for n in TRIM_LINES:
        if n >= len(cont_lines):
            continue
        trimmed_cont = '\n'.join(cont_lines[:len(cont_lines) - n])
        test_text = prompt + trimmed_cont
        ok, _ = try_parse(test_text)
        if ok:
            return {
                "full_parses": False,
                "trim_lines_required": n,
                "trim_parse_result": f"parses_after_trim_{n}_lines"
            }

    # Try 10 lines too (allowed in spec)
    for n in TRIM_LINES:
        if n >= len(cont_lines):
            continue
    # Max 10 didn't work
    max_n = min(10, len(cont_lines) - 1)
    if max_n > 0:
        trimmed_cont = '\n'.join(cont_lines[:len(cont_lines) - max_n])
        ok, _ = try_parse(prompt + trimmed_cont)
        if ok:
            return {
                "full_parses": False,
                "trim_lines_required": max_n,
                "trim_parse_result": f"parses_after_trim_{max_n}_lines"
            }

    return {
        "full_parses": False,
        "trim_lines_required": None,
        "trim_parse_result": "trim_parse_fail"
    }


def check_text_anomalies(prompt, continuation):
    """Check for text-level anomalies."""
    anomalies = []
    full = prompt + continuation
    lines = full.split('\n')

    # 1. Repeated consecutive lines (>=3 identical in a row)
    rep_runs = []
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j].strip() == lines[i].strip() and lines[i].strip():
            j += 1
        run_len = j - i
        if run_len >= 3:
            rep_runs.append({"line": i + 1, "count": run_len, "text": lines[i].strip()[:80]})
        i = j

    if rep_runs:
        anomalies.append({
            "type": "line_repetition",
            "detail": f"{len(rep_runs)} runs of >=3 repeated identical lines",
            "runs": rep_runs[:5]
        })

    # 2. High n-gram repetition (6-gram, 8-gram overlap)
    words = full.split()
    if len(words) >= 12:
        for n in [6, 8]:
            ngrams = {}
            for i in range(len(words) - n + 1):
                ng = ' '.join(words[i:i+n])
                ngrams[ng] = ngrams.get(ng, 0) + 1
            high_reps = [(ng, c) for ng, c in ngrams.items() if c >= 3]
            if high_reps:
                anomalies.append({
                    "type": f"ngram_repetition_{n}",
                    "detail": f"{len(high_reps)} {n}-grams appear >=3 times",
                    "examples": [ng[:60] for ng, _ in high_reps[:3]]
                })

    # 3. Natural language / explanations
    nl_patterns = [
        (r'I apologize', 'apology phrase'),
        (r'Here.*(?:is|are).*(?:code|example|implementation|solution)', 'explanation intro'),
        (r'Let me explain', 'explanation intro'),
        (r'Note that', 'explanatory note'),
        (r'This (?:code|function|script|program|implementation)', 'descriptive text'),
    ]
    for pattern, label in nl_patterns:
        if re.search(pattern, continuation, re.IGNORECASE):
            anomalies.append({"type": "natural_language", "detail": f"Found '{label}' in continuation"})
            break  # one is enough

    # 4. Markdown code fences in continuation
    if re.search(r'```', continuation):
        anomalies.append({"type": "markdown_fence", "detail": "Markdown code fence found in continuation"})

    # 5. Assistant/User role markers
    if re.search(r'\b(Assistant|User|Human|System):', continuation):
        anomalies.append({"type": "role_marker", "detail": "Assistant/User role markers found"})

    # 6. Heavily non-Python (very few structural characters)
    structural_ratio = sum(1 for c in continuation if c in '()[]{}:=,.')
    total_chars = max(len(continuation), 1)
    if structural_ratio / total_chars < 0.01 and total_chars > 100:
        anomalies.append({"type": "low_python_structure", "detail": f"Only {structural_ratio} structural chars out of {total_chars}"})

    return anomalies


# ── Run audit ──

print(f"Auditing {len(items)} items...")
audit_results = []

for item in items:
    bid = item["blind_id"]
    meta = m_by_id.get(bid, {})
    prompt = item["prompt"]
    cont = item["continuation"]

    # 1. raw_parse
    full_text = prompt + cont
    full_ok, full_err = try_parse(full_text)

    # Check if prompt alone parses
    prompt_ok, _ = try_parse(prompt)
    # Check if prompt alone is a complete unit
    prompt_complete = is_complete_unit(prompt)

    if not prompt_complete:
        parse_applicability = "not_applicable"
        parse_note = "Prompt is not a complete Python unit (fragment, partial block)"
    elif full_ok:
        parse_applicability = "applicable_parses"
        parse_note = None
    else:
        parse_applicability = "applicable_error"
        parse_note = full_err

    # 2. tail_trim_parse
    # Only meaningful if prompt IS a complete unit
    if prompt_complete:
        trim_result = check_tail_trim(prompt, cont)
    else:
        trim_result = {
            "full_parses": None,
            "trim_lines_required": None,
            "trim_parse_result": "not_applicable",
            "note": "Prompt is not a complete Python unit"
        }

    # 3. text anomalies
    anomalies = check_text_anomalies(prompt, cont)

    # Cross-reference with round1 human score (diagnostic, not truth)
    r1_hs = r1_by_id.get(bid, {}).get("human_score")

    entry = {
        "blind_id": bid,
        "benchmark": meta.get("benchmark", "?"),
        "method": meta.get("method", "?"),
        "auto_score": meta.get("automatic_score", "?"),
        "round1_human_score": r1_hs,
        "continuation_len": len(cont),
        "raw_parse": {
            "prompt_parses": prompt_ok,
            "prompt_is_complete_unit": prompt_complete,
            "full_text_parses": full_ok,
            "full_text_error": full_err if not full_ok else None,
            "applicability": parse_applicability,
            "note": parse_note
        },
        "tail_trim_parse": trim_result,
        "text_anomalies": anomalies
    }
    audit_results.append(entry)

# ── Save JSON ──
with open(OUT_JSON, "w") as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)
print(f"Saved -> {OUT_JSON}")

# ── Build report ──
total = len(audit_results)
applicable = sum(1 for a in audit_results if a["raw_parse"]["applicability"] != "not_applicable")
not_applicable = total - applicable

parse_ok = sum(1 for a in audit_results if a["raw_parse"]["full_text_parses"])
parse_fail = total - parse_ok

trim_counts = Counter()
for a in audit_results:
    t = a["tail_trim_parse"].get("trim_parse_result", "unknown")
    trim_counts[t] += 1
    if t == "trim_parse_fail":
        # Check if prompt is applicable
        pass

# Anomaly statistics
has_anomalies = sum(1 for a in audit_results if len(a["text_anomalies"]) > 0)
anomaly_types = Counter()
for a in audit_results:
    for anom in a["text_anomalies"]:
        anomaly_types[anom["type"]] += 1

# By benchmark
bm_stats = {}
for a in audit_results:
    b = a["benchmark"]
    if b not in bm_stats:
        bm_stats[b] = {"total": 0, "applicable": 0, "parse_ok": 0, "trim_fail": 0, "has_anomalies": 0}
    bm_stats[b]["total"] += 1
    if a["raw_parse"]["applicability"] != "not_applicable":
        bm_stats[b]["applicable"] += 1
    if a["raw_parse"]["full_text_parses"]:
        bm_stats[b]["parse_ok"] += 1
    if a["tail_trim_parse"]["trim_parse_result"] == "trim_parse_fail":
        bm_stats[b]["trim_fail"] += 1
    if len(a["text_anomalies"]) > 0:
        bm_stats[b]["has_anomalies"] += 1

# By round1 human score
hs_stats = {}
for a in audit_results:
    hs = a.get("round1_human_score")
    if hs is None:
        continue
    if hs not in hs_stats:
        hs_stats[hs] = {"total": 0, "applicable": 0, "parse_ok": 0, "has_anomalies": 0}
    hs_stats[hs]["total"] += 1
    if a["raw_parse"]["applicability"] != "not_applicable":
        hs_stats[hs]["applicable"] += 1
    if a["raw_parse"]["full_text_parses"]:
        hs_stats[hs]["parse_ok"] += 1
    if len(a["text_anomalies"]) > 0:
        hs_stats[hs]["has_anomalies"] += 1

# By auto score
auto_stats = {}
for a in audit_results:
    as_ = a.get("auto_score", "?")
    if as_ not in auto_stats:
        auto_stats[as_] = {"total": 0, "applicable": 0, "parse_ok": 0, "has_anomalies": 0}
    auto_stats[as_]["total"] += 1
    if a["raw_parse"]["applicability"] != "not_applicable":
        auto_stats[as_]["applicable"] += 1
    if a["raw_parse"]["full_text_parses"]:
        auto_stats[as_]["parse_ok"] += 1
    if len(a["text_anomalies"]) > 0:
        auto_stats[as_]["has_anomalies"] += 1

# ── Generate report ──
lines = []
lines.append("# Independent Syntax Audit Report")
lines.append("")
lines.append("## Summary")
lines.append("")
lines.append(f"- **Total items**: {total}")
lines.append(f"- **Applicable for AST parsing** (prompt is complete Python unit): {applicable} ({applicable/total*100:.1f}%)")
lines.append(f"- **Not applicable** (prompt is file fragment/partial): {not_applicable} ({not_applicable/total*100:.1f}%)")
lines.append(f"- **Full text parses successfully**: {parse_ok} ({parse_ok/total*100:.1f}%)")
lines.append(f"- **Items with text anomalies**: {has_anomalies} ({has_anomalies/total*100:.1f}%)")
lines.append("")

lines.append("## Tail Trim Results (applicable samples only)")
lines.append("")
trim_applicable = sum(1 for a in audit_results if a["raw_parse"]["applicability"] != "not_applicable")
for result_type, count in sorted(trim_counts.items(), key=lambda x: -x[1]):
    lines.append(f"- `{result_type}`: {count}")
lines.append(f"  (Based on {trim_applicable} applicable samples)")
lines.append("")

if anomaly_types:
    lines.append("## Text Anomaly Types")
    lines.append("")
    for atype, count in sorted(anomaly_types.items(), key=lambda x: -x[1]):
        lines.append(f"- `{atype}`: {count}")
    lines.append("")

lines.append("## Per-Benchmark Breakdown")
lines.append("")
lines.append("| Benchmark | Total | Applicable | Parse OK | Trim Fail | Anomalies |")
lines.append("|-----------|------:|-----------:|---------:|----------:|----------:|")
for b in sorted(bm_stats.keys()):
    s = bm_stats[b]
    lines.append(f"| {b} | {s['total']} | {s['applicable']} | {s['parse_ok']} | {s['trim_fail']} | {s['has_anomalies']} |")
lines.append("")

lines.append("## Per-Score Breakdown (Diagnostic Only)")
lines.append("")
lines.append("### By Round-1 Human Score")
lines.append("")
lines.append("| Score | Total | Applicable | Parse OK | Anomalies |")
lines.append("|-------|------:|-----------:|---------:|----------:|")
for hs in sorted(hs_stats.keys()):
    s = hs_stats[hs]
    lines.append(f"| {hs} | {s['total']} | {s['applicable']} | {s['parse_ok']} | {s['has_anomalies']} |")
lines.append("")

lines.append("### By Automatic Score")
lines.append("")
lines.append("| Score | Total | Applicable | Parse OK | Anomalies |")
lines.append("|-------|------:|-----------:|---------:|----------:|")
for as_ in sorted(auto_stats.keys()):
    s = auto_stats[as_]
    lines.append(f"| {as_} | {s['total']} | {s['applicable']} | {s['parse_ok']} | {s['has_anomalies']} |")
lines.append("")

lines.append("## Important Caveats")
lines.append("")
lines.append("1. **AST parse is NOT ground truth for human scoring.** Many prompts are file fragments")
lines.append("   (partial code blocks, REPL snippets) and `not_applicable` is expected and acceptable.")
lines.append("2. A `not_applicable` sample may still be a perfectly valid score-2 continuation if the")
lines.append("   truncated file fragment is internally consistent.")
lines.append("3. Tail trim only deletes from the END of the continuation; it does not fix mid-text errors.")
lines.append("4. This audit is a diagnostic tool to contextualize intra-rater disagreement patterns,")
lines.append("   not to replace human judgment.")
lines.append("")

report = '\n'.join(lines)
with open(OUT_REPORT, "w") as f:
    f.write(report)
print(f"Saved -> {OUT_REPORT}")

# Print one-line summary
print(f"\nApplicable samples for AST: {applicable}/{total} ({applicable/total*100:.1f}%)")
print(f"Parse OK: {parse_ok}, Trim fails (applicable): {sum(1 for a in audit_results if a['tail_trim_parse']['trim_parse_result'] == 'trim_parse_fail' and a['raw_parse']['applicability'] != 'not_applicable')}")
print(f"Anomalies: {has_anomalies}/{total}")
