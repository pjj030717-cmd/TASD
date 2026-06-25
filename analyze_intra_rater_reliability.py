#!/usr/bin/env python3
"""
Intra-rater reliability analysis for single-annotator score validator.

Reads:
  - annotations_A_round1.json    (90 items)
  - annotations_A_retest30.json  (30 items)
  - retest_mapping_private.json  (RET → SVR mapping)

Outputs:
  results/score_validator_review/intra_rater_report.md
  results/score_validator_review/intra_rater_statistics.json

Usage:
  python analyze_intra_rater_reliability.py
  python analyze_intra_rater_reliability.py --dummy   (test with synthetic data)
"""

import json
import os
import sys
import math
from collections import Counter, defaultdict

PRIVATE_DIR = "results/score_validator_review/private"

# ── Helpers ──

def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion."""
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    lo = max(0, center - margin)
    hi = min(1, center + margin)
    return (lo, hi)


def quadratic_weighted_kappa(a_scores, b_scores, n_classes=3):
    """Quadratic weighted Cohen's kappa with standard error."""
    if len(a_scores) != len(b_scores):
        raise ValueError("Length mismatch")
    N = len(a_scores)
    if N < 2:
        return {'kappa': None, 'se': None, 'ci95': None, 'error': 'n<2'}

    # Build confusion matrix
    cm = [[0] * n_classes for _ in range(n_classes)]
    for a, b in zip(a_scores, b_scores):
        cm[a][b] += 1

    # Weight matrix (quadratic)
    weights = [[0.0] * n_classes for _ in range(n_classes)]
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i][j] = ((i - j) / (n_classes - 1)) ** 2

    # Observed agreement
    po = sum(cm[i][i] for i in range(n_classes)) / N

    # Expected
    row_sums = [sum(cm[i]) for i in range(n_classes)]
    col_sums = [sum(cm[i][j] for i in range(n_classes)) for j in range(n_classes)]
    pe = 0
    for i in range(n_classes):
        for j in range(n_classes):
            pe += weights[i][j] * row_sums[i] * col_sums[j] / N
    pe /= N

    po_w = 0
    for i in range(n_classes):
        for j in range(n_classes):
            po_w += weights[i][j] * cm[i][j]
    po_w /= N

    if pe >= 1:
        return {'kappa': None, 'se': None, 'ci95': None, 'error': 'pe>=1'}

    kappa = 1 - po_w / pe

    # SE
    if N > 1:
        sum_w2 = 0
        for i in range(n_classes):
            for j in range(n_classes):
                w_ij = weights[i][j]
                w_bar_i = sum(weights[i][k] * col_sums[k] for k in range(n_classes)) / N
                w_bar_j = sum(weights[k][j] * row_sums[k] for k in range(n_classes)) / N
                w_bar = sum(weights[r][c] * row_sums[r] * col_sums[c] for r in range(n_classes) for c in range(n_classes)) / (N * N)
                sum_w2 += cm[i][j] * ((w_ij - w_bar_i - w_bar_j + w_bar) ** 2)
        var = sum_w2 / (N * pe * pe)
        se = math.sqrt(var / N)
    else:
        se = None

    ci95 = None
    if se is not None:
        ci95 = (kappa - 1.96 * se, kappa + 1.96 * se)

    return {
        'kappa': kappa,
        'se': se,
        'ci95': ci95,
        'po': po,
        'pe': pe,
        'n': N,
        'confusion_matrix': cm
    }


def generate_dummy_data():
    """Generate synthetic data for testing."""
    import random
    random.seed(42)
    r1 = []
    for i in range(90):
        sc = random.choices([0, 1, 2], weights=[0.01, 0.88, 0.11])[0]
        r1.append({
            "blind_id": f"SVR-DUMMY{i:04X}",
            "human_score": sc,
            "completion_status": random.choice(["complete", "tail_cutoff", "severe_incomplete"]),
            "issue_tags": ["none"] if random.random() < 0.9 else ["bracket_or_delimiter"]
        })
    mapping = []
    retest = []
    for i in range(30):
        orig = r1[i]
        sc = orig["human_score"]
        # 80% stay same, 20% flip (mostly to adjacent)
        if random.random() < 0.8:
            sc2 = sc
        else:
            sc2 = (sc + random.choice([-1, 1])) % 3
        mapping.append({
            "retest_blind_id": f"RET-DUMMY{i:04X}",
            "original_blind_id": orig["blind_id"],
            "benchmark": f"bench_{i % 6}"
        })
        retest.append({
            "retest_blind_id": f"RET-DUMMY{i:04X}",
            "human_score": sc2,
            "completion_status": random.choice(["complete", "tail_cutoff", "severe_incomplete"]),
            "issue_tags": ["none"] if random.random() < 0.9 else ["bracket_or_delimiter"]
        })
    return r1, retest, mapping


# ── Analysis ──

def analyze(r1_path, retest_path, mapping_path, syntax_audit_path=None):
    with open(r1_path) as f:
        r1 = json.load(f)
    with open(retest_path) as f:
        retest = json.load(f)
    with open(mapping_path) as f:
        mapping = json.load(f)

    # Load syntax audit if available
    syntax = {}
    if syntax_audit_path and os.path.exists(syntax_audit_path):
        with open(syntax_audit_path) as f:
            syntax_list = json.load(f)
        syntax = {a["blind_id"]: a for a in syntax_list}

    r1_by_id = {a["blind_id"]: a for a in r1}
    retest_by_retid = {a["retest_blind_id"]: a for a in retest}
    ret_to_orig = {m["retest_blind_id"]: m["original_blind_id"] for m in mapping}

    N = len(mapping)

    # ── 1. Completion rate ──
    complete = 0
    for m in mapping:
        ra = retest_by_retid.get(m["retest_blind_id"], {})
        if (ra.get("human_score") is not None and
            ra.get("completion_status") is not None and
            ra.get("issue_tags") and len(ra["issue_tags"]) > 0):
            complete += 1

    # ── 2. Score exact agreement ──
    score_pairs = []
    status_pairs = []
    for m in mapping:
        rid = m["retest_blind_id"]
        oid = m["original_blind_id"]
        oa = r1_by_id.get(oid, {})
        ra = retest_by_retid.get(rid, {})
        if oa and ra:
            score_pairs.append((oa["human_score"], ra["human_score"]))
            status_pairs.append((oa.get("completion_status"), ra.get("completion_status")))

    score_agree = sum(1 for a, b in score_pairs if a == b)

    # ── 3. Quadratic weighted kappa ──
    a_scores = [a for a, b in score_pairs]
    b_scores = [b for a, b in score_pairs]
    kappa_result = quadratic_weighted_kappa(a_scores, b_scores)

    # Confusion matrix
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for a, b in score_pairs:
        cm[a][b] += 1

    # ── 5. Completion status agreement ──
    status_agree = sum(1 for a, b in status_pairs if a == b)
    status_values = ["complete", "tail_cutoff", "severe_incomplete"]
    status_to_idx = {v: i for i, v in enumerate(status_values)}
    a_status_idx = [status_to_idx.get(a, 0) for a, b in status_pairs]
    b_status_idx = [status_to_idx.get(b, 0) for a, b in status_pairs]
    status_kappa = quadratic_weighted_kappa(a_status_idx, b_status_idx)

    # ── 6. Disagreement list ──
    disagreements = []
    for m in mapping:
        rid = m["retest_blind_id"]
        oid = m["original_blind_id"]
        oa = r1_by_id.get(oid, {})
        ra = retest_by_retid.get(rid, {})
        if (oa.get("human_score") != ra.get("human_score") or
            oa.get("completion_status") != ra.get("completion_status")):
            disagreements.append({
                "original_blind_id": oid,
                "retest_blind_id": rid,
                "benchmark": m.get("benchmark", "?"),
                "round1_score": oa.get("human_score"),
                "retest_score": ra.get("human_score"),
                "round1_completion": oa.get("completion_status"),
                "retest_completion": ra.get("completion_status"),
            })

    # ── 7. Direction bias ──
    r1_higher = sum(1 for a, b in score_pairs if a > b)
    retest_higher = sum(1 for a, b in score_pairs if a < b)

    # ── 8. Per-benchmark agreement ──
    bm_agree = defaultdict(lambda: {"total": 0, "agree": 0})
    for i, m in enumerate(mapping):
        b = m.get("benchmark", "?")
        bm_agree[b]["total"] += 1
        if i < len(score_pairs) and score_pairs[i][0] == score_pairs[i][1]:
            bm_agree[b]["agree"] += 1

    # ── 9. Cross-ref with syntax audit ──
    syntax_cross = []
    if syntax:
        for m in mapping:
            oid = m["original_blind_id"]
            rid = m["retest_blind_id"]
            if oid in syntax and rid in retest_by_retid:
                sa = syntax[oid]
                ra = retest_by_retid[rid]
                oa = r1_by_id.get(oid, {})
                syntax_cross.append({
                    "original_blind_id": oid,
                    "round1_score": oa.get("human_score"),
                    "retest_score": ra.get("human_score"),
                    "syntax_applicable": sa["raw_parse"]["applicability"] != "not_applicable",
                    "syntax_parses": sa["raw_parse"]["full_text_parses"],
                    "trim_lines": sa["tail_trim_parse"].get("trim_lines_required"),
                    "anomalies": len(sa["text_anomalies"]),
                    "agree": oa.get("human_score") == ra.get("human_score"),
                })

    # ── Build report ──

    ci_agree = wilson_ci(score_agree / N, N)
    ci_status = wilson_ci(status_agree / N, N)

    report = []
    report.append("# Intra-Rater Reliability Report")
    report.append("")
    report.append("## 1. Completion Rate")
    report.append("")
    report.append(f"- Retest completed: {complete}/{N} ({complete/N*100:.1f}%)")
    report.append(f"- Round 1 total: {len(r1)}")
    report.append("")

    report.append("## 2. Score Exact Agreement (0/1/2)")
    report.append("")
    report.append(f"- Exact agreement: {score_agree}/{N} ({score_agree/N*100:.1f}%)")
    report.append(f"- Wilson 95% CI: [{ci_agree[0]:.3f}, {ci_agree[1]:.3f}]")
    report.append("")

    report.append("## 3. Quadratic Weighted Cohen's Kappa")
    report.append("")
    if kappa_result['kappa'] is not None:
        k_level = "Good" if kappa_result['kappa'] >= 0.70 else ("Moderate" if kappa_result['kappa'] >= 0.50 else "Poor")
        report.append(f"- Weighted kappa: **{kappa_result['kappa']:.4f}** ({k_level})")
        if kappa_result['ci95']:
            report.append(f"- 95% CI: [{kappa_result['ci95'][0]:.4f}, {kappa_result['ci95'][1]:.4f}]")
        if kappa_result['se'] is not None:
            report.append(f"- SE: {kappa_result['se']:.4f}")
    else:
        report.append(f"- Could not compute: {kappa_result.get('error', 'unknown')}")
    report.append("")

    report.append("## 4. Score Confusion Matrix (Round1 rows, Retest cols)")
    report.append("")
    report.append("| R1\\Retest | 0 | 1 | 2 | Total |")
    report.append("|-----------|---|---|---|-------|")
    for i in range(3):
        row_total = sum(cm[i])
        report.append(f"| {i} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} | {row_total} |")
    report.append("")

    report.append("## 5. Completion Status Agreement")
    report.append("")
    report.append(f"- Exact agreement: {status_agree}/{N} ({status_agree/N*100:.1f}%)")
    report.append(f"- Wilson 95% CI: [{ci_status[0]:.3f}, {ci_status[1]:.3f}]")
    if status_kappa['kappa'] is not None:
        report.append(f"- Weighted kappa: {status_kappa['kappa']:.4f}")
    report.append("")

    report.append("## 6. Disagreements")
    report.append("")
    report.append(f"- Total disagreements: {len(disagreements)}/{N}")
    report.append("")
    if disagreements:
        report.append("| Original ID | Retest ID | R1 Score | Re Score | R1 Status | Re Status |")
        report.append("|-------------|-----------|:--------:|:--------:|-----------|-----------|")
        for d in disagreements[:20]:
            report.append(f"| {d['original_blind_id']} | {d['retest_blind_id']} | {d['round1_score']} | {d['retest_score']} | {d['round1_completion']} | {d['retest_completion']} |")
        if len(disagreements) > 20:
            report.append(f"... and {len(disagreements) - 20} more")
    report.append("")

    report.append("## 7. Direction Bias")
    report.append("")
    report.append(f"- R1 score > Retest score: {r1_higher}/{N}")
    report.append(f"- R1 score < Retest score: {retest_higher}/{N}")
    report.append(f"- Same: {score_agree}/{N}")
    report.append("")

    report.append("## 8. Per-Benchmark Agreement")
    report.append("")
    report.append("| Benchmark | Total | Agree | % |")
    report.append("|-----------|------:|------:|--:|")
    for b in sorted(bm_agree.keys()):
        s = bm_agree[b]
        pct = s["agree"] / s["total"] * 100 if s["total"] > 0 else 0
        report.append(f"| {b} | {s['total']} | {s['agree']} | {pct:.0f}% |")
    report.append("")

    if syntax_cross:
        report.append("## 9. Syntax Audit Cross-Reference (Diagnostic)")
        report.append("")
        agree_syntax = sum(1 for sc in syntax_cross if sc["agree"])
        disagree_syntax = len(syntax_cross) - agree_syntax
        report.append(f"- Agreements: {agree_syntax}, Disagreements: {disagree_syntax}")
        report.append("")
        # Split
        report.append("| Group | N | Syntax Parse OK | Mean Anomalies | Trim Needed |")
        report.append("|-------|---|:---------------:|:--------------:|:-----------:|")
        for label, group in [("Agree", [sc for sc in syntax_cross if sc["agree"]]),
                              ("Disagree", [sc for sc in syntax_cross if not sc["agree"]])]:
            n_g = len(group)
            if n_g == 0:
                continue
            p_ok = sum(1 for sc in group if sc["syntax_parses"]) / n_g
            avg_anom = sum(sc["anomalies"] for sc in group) / n_g
            trim_need = sum(1 for sc in group if sc["trim_lines"] is not None and sc["trim_lines"] > 0) / n_g
            report.append(f"| {label} | {n_g} | {p_ok:.2f} | {avg_anom:.2f} | {trim_need:.2f} |")
        report.append("")
        report.append("**Note**: Syntax audit is diagnostic only, not human ground truth.")
        report.append("")

    report.append("## Interpretation Guide")
    report.append("")
    if kappa_result['kappa'] is not None and kappa_result['kappa'] >= 0.70:
        report.append("**Conclusion: STABLE.** Single-annotator scoring shows good consistency.")
        report.append("Can serve as limited human validation for the paper.")
    elif kappa_result['kappa'] is not None and kappa_result['kappa'] >= 0.50:
        report.append("**Conclusion: MODERATE.** Acceptable but should be noted in limitations.")
    else:
        report.append("**Conclusion: UNSTABLE.** Scoring scale is not reliable enough to support main conclusions.")
        report.append("Re-calibration of score boundaries may be needed.")
    report.append("")

    report_text = '\n'.join(report)

    return {
        "report": report_text,
        "stats": {
            "n_round1": len(r1),
            "n_retest": N,
            "complete": complete,
            "score_agreement": score_agree,
            "score_agreement_pct": score_agree / N if N > 0 else 0,
            "kappa": kappa_result['kappa'],
            "kappa_ci95": kappa_result['ci95'],
            "kappa_se": kappa_result['se'],
            "status_agreement": status_agree,
            "status_agreement_pct": status_agree / N if N > 0 else 0,
            "status_kappa": status_kappa['kappa'],
            "r1_higher": r1_higher,
            "retest_higher": retest_higher,
            "n_disagreements": len(disagreements),
            "confusion_matrix": cm,
            "per_benchmark": {b: dict(s) for b, s in bm_agree.items()},
        },
        "disagreements": disagreements,
    }


# ── Main ──

if __name__ == "__main__":
    if "--dummy" in sys.argv:
        print("Generating dummy data for testing...")
        r1, retest, mapping = generate_dummy_data()
        os.makedirs(PRIVATE_DIR, exist_ok=True)
        with open(f"{PRIVATE_DIR}/annotations_A_round1.json", "w") as f:
            json.dump(r1, f, indent=2)
        with open(f"{PRIVATE_DIR}/annotations_A_retest30.json", "w") as f:
            json.dump(retest, f, indent=2)
        with open(f"{PRIVATE_DIR}/retest_mapping_private.json", "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  Dummy data saved: round1={len(r1)}, retest={len(retest)}, mapping={len(mapping)}")

    r1_path = f"{PRIVATE_DIR}/annotations_A_round1.json"
    retest_path = f"{PRIVATE_DIR}/annotations_A_retest30.json"
    mapping_path = f"{PRIVATE_DIR}/retest_mapping_private.json"
    syntax_path = "results/score_validator_review/syntax_audit.json"

    if not all(os.path.exists(p) for p in [r1_path, retest_path, mapping_path]):
        print("ERROR: Missing input files")
        print(f"  round1: {'OK' if os.path.exists(r1_path) else 'MISSING'}")
        print(f"  retest: {'OK' if os.path.exists(retest_path) else 'MISSING'}")
        print(f"  mapping: {'OK' if os.path.exists(mapping_path) else 'MISSING'}")
        sys.exit(1)

    result = analyze(r1_path, retest_path, mapping_path, syntax_path)

    report_path = "results/score_validator_review/intra_rater_report.md"
    stats_path = "results/score_validator_review/intra_rater_statistics.json"

    with open(report_path, "w") as f:
        f.write(result["report"])
    with open(stats_path, "w") as f:
        json.dump(result["stats"], f, indent=2, ensure_ascii=False)

    print(f"Report -> {report_path}")
    print(f"Stats  -> {stats_path}")
    print(f"\nKey metrics:")
    print(f"  Score agreement: {result['stats']['score_agreement']}/{result['stats']['n_retest']} ({result['stats']['score_agreement_pct']*100:.1f}%)")
    if result['stats']['kappa'] is not None:
        print(f"  Weighted kappa: {result['stats']['kappa']:.4f}")
    print(f"  Disagreements: {result['stats']['n_disagreements']}")
