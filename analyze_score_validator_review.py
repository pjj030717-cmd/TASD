#!/usr/bin/env python3
"""
Analyze Score Validator human review results.

Compares human scores (0/1/2) against automatic scores from the final master table.
Outputs: agreement metrics, confusion matrix, kappa, per-method/benchmark diagnostics,
and disagreement adjudication CSV.

Usage:
  python analyze_score_validator_review.py A.json B.json [--adjudicated adjudicated.csv]
  python analyze_score_validator_review.py --dummy  (generate synthetic data and test)
"""

import json, os, sys, csv, random
from collections import Counter, defaultdict
import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# Wilson 95% CI
# ══════════════════════════════════════════════════════════════════════════

def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportion p with n trials."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0, center - margin), min(1, center + margin))


# ══════════════════════════════════════════════════════════════════════════
# Cohen's weighted kappa
# ══════════════════════════════════════════════════════════════════════════

def weighted_cohens_kappa(a_scores, b_scores, n_categories=3):
    """Compute Cohen's weighted kappa with linear weights."""
    n = len(a_scores)
    if n == 0:
        return 0.0, 0.0

    # Observed matrix
    observed = np.zeros((n_categories, n_categories))
    for a, b in zip(a_scores, b_scores):
        if 0 <= a < n_categories and 0 <= b < n_categories:
            observed[int(a)][int(b)] += 1

    # Expected matrix
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    # Linear weights
    weights = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i][j] = abs(i - j) / (n_categories - 1)

    po = np.sum(weights * observed) / n
    pe = np.sum(weights * expected) / n

    if pe == 1.0:
        return 1.0, 0.0

    kappa = 1 - (po / pe) if pe > 0 else 0.0

    # Standard error (approximate)
    total = observed.sum()
    se = np.sqrt(po * (1 - po) / (total * (1 - pe)**2)) if pe < 1 and total > 0 else 0.0

    return kappa, se


# ══════════════════════════════════════════════════════════════════════════
# Macro precision/recall/F1
# ══════════════════════════════════════════════════════════════════════════

def macro_metrics(cm, n_classes=3):
    """Compute macro-averaged precision, recall, F1 from confusion matrix."""
    precisions = []
    recalls = []
    for i in range(n_classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    return macro_p, macro_r, macro_f1


# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════

def load_mapping():
    mapping_path = "results/score_validator_review/private/blind_mapping_private.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    return {m["blind_id"]: m for m in mapping}


def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {a["blind_id"]: a for a in data}


# ══════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════

def analyze(a_path, b_path, adjudicated_csv=None):
    """Full analysis pipeline."""
    mapping = load_mapping()
    ann_a = load_annotations(a_path)
    ann_b = load_annotations(b_path)

    common_ids = set(ann_a.keys()) & set(ann_b.keys()) & set(mapping.keys())
    print(f"Common blind IDs: {len(common_ids)}")

    # ── 1. Completion rates ──
    a_done = sum(1 for bid in ann_a if ann_a[bid].get("human_score") is not None
                 and ann_a[bid].get("completion_status") is not None
                 and ann_a[bid].get("issue_tags") and len(ann_a[bid]["issue_tags"]) > 0)
    b_done = sum(1 for bid in ann_b if ann_b[bid].get("human_score") is not None
                 and ann_b[bid].get("completion_status") is not None
                 and ann_b[bid].get("issue_tags") and len(ann_b[bid]["issue_tags"]) > 0)
    print(f"\n1. Completion rates:")
    print(f"  A: {a_done}/{len(ann_a)} ({100*a_done/len(ann_a):.1f}%)")
    print(f"  B: {b_done}/{len(ann_b)} ({100*b_done/len(ann_b):.1f}%)")

    # Build adjudicated scores
    adjudicated = {}
    if adjudicated_csv:
        with open(adjudicated_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                adjudicated[row["blind_id"]] = int(row["adjudicated_score"])

    # ── 2. Human score raw agreement ──
    a_common = []
    b_common = []
    for bid in sorted(common_ids):
        hs_a = ann_a[bid].get("human_score")
        hs_b = ann_b[bid].get("human_score")
        if hs_a is not None and hs_b is not None:
            a_common.append(hs_a)
            b_common.append(hs_b)

    raw_agree = sum(1 for a, b in zip(a_common, b_common) if a == b)
    print(f"\n2. Human score raw agreement: {raw_agree}/{len(a_common)} ({100*raw_agree/len(a_common):.1f}%)")

    # ── 3. Human score weighted kappa ──
    kappa, kappa_se = weighted_cohens_kappa(a_common, b_common)
    print(f"3. Human score weighted kappa: {kappa:.4f} (SE={kappa_se:.4f})")

    # ── 4. Completion status agreement ──
    cs_map = {"complete": 0, "tail_cutoff": 1, "severe_incomplete": 2}
    cs_a_common = []
    cs_b_common = []
    for bid in sorted(common_ids):
        cs_a = ann_a[bid].get("completion_status")
        cs_b = ann_b[bid].get("completion_status")
        if cs_a in cs_map and cs_b in cs_map:
            cs_a_common.append(cs_map[cs_a])
            cs_b_common.append(cs_map[cs_b])

    cs_raw = sum(1 for a, b in zip(cs_a_common, cs_b_common) if a == b)
    cs_kappa, cs_kappa_se = weighted_cohens_kappa(cs_a_common, cs_b_common)
    print(f"\n4. Completion Status agreement:")
    print(f"  Raw: {cs_raw}/{len(cs_a_common)} ({100*cs_raw/len(cs_a_common):.1f}%)")
    print(f"  Weighted kappa: {cs_kappa:.4f} (SE={cs_kappa_se:.4f})")

    # ── 5. Disagreement adjudication ──
    disagreements = []
    for bid in sorted(common_ids):
        hs_a = ann_a[bid].get("human_score")
        hs_b = ann_b[bid].get("human_score")
        if hs_a is not None and hs_b is not None and hs_a != hs_b:
            m = mapping[bid]
            disagreements.append({
                "blind_id": bid,
                "annotator_A_score": hs_a,
                "annotator_B_score": hs_b,
                "automatic_score": m["automatic_score"],
                "method": m["method"],
                "benchmark": m["benchmark"],
                "a_notes": ann_a[bid].get("notes", ""),
                "b_notes": ann_b[bid].get("notes", ""),
            })

    out_dir = "results/score_validator_review"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/disagreements_for_adjudication.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=disagreements[0].keys() if disagreements else ["blind_id"])
        writer.writeheader()
        writer.writerows(disagreements)
    print(f"\n5. Disagreements: {len(disagreements)} -> {csv_path}")

    # ── 6-9. Auto vs Human confusion matrix ──
    auto_scores = []
    human_scores = []
    for b in range(len(common_ids)):
        bid = sorted(common_ids)[b]
        m = mapping[bid]
        if bid in adjudicated:
            hs = adjudicated[bid]
        else:
            hs_a = ann_a[bid].get("human_score")
            hs_b = ann_b[bid].get("human_score")
            if hs_a is not None and hs_b is not None and hs_a == hs_b:
                hs = hs_a
            else:
                continue  # skip disagreements unless adjudicated
        auto_scores.append(m["automatic_score"])
        human_scores.append(hs)

    n_eval = len(auto_scores)
    print(f"\n6. Evaluating on {n_eval} items (agreed or adjudicated)")

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for a, h in zip(auto_scores, human_scores):
        cm[h][a] += 1  # rows=human, cols=auto

    print("   Confusion Matrix (rows=Human, cols=Auto):")
    print("            Auto=0  Auto=1  Auto=2")
    for i in range(3):
        print(f"   Human={i}:  {cm[i][0]:7d} {cm[i][1]:7d} {cm[i][2]:7d}")

    # ── 7. Exact agreement ──
    exact = sum(1 for a, h in zip(auto_scores, human_scores) if a == h)
    print(f"\n7. Exact agreement: {exact}/{n_eval} ({100*exact/n_eval:.1f}%)")
    lo, hi = wilson_ci(exact/n_eval, n_eval)
    print(f"   95% CI: [{lo:.1%}, {hi:.1%}]")

    # ── 8. Weighted kappa (auto vs human) ──
    auk, auk_se = weighted_cohens_kappa(auto_scores, human_scores)
    print(f"8. Auto-Human weighted kappa: {auk:.4f} (SE={auk_se:.4f})")

    # ── 9. Macro P/R/F1 ──
    mp, mr, mf1 = macro_metrics(cm)
    print(f"9. Macro Precision={mp:.4f} Recall={mr:.4f} F1={mf1:.4f}")

    # ── 10. Binary: recoverable (>=1) ──
    auto_rec = [1 if s >= 1 else 0 for s in auto_scores]
    human_rec = [1 if s >= 1 else 0 for s in human_scores]
    tp = sum(1 for a, h in zip(auto_rec, human_rec) if a == 1 and h == 1)
    fp = sum(1 for a, h in zip(auto_rec, human_rec) if a == 1 and h == 0)
    tn = sum(1 for a, h in zip(auto_rec, human_rec) if a == 0 and h == 0)
    fn = sum(1 for a, h in zip(auto_rec, human_rec) if a == 0 and h == 1)

    b_prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    b_rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    b_spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    b_f1 = 2*b_prec*b_rec/(b_prec+b_rec) if (b_prec+b_rec) > 0 else 0

    print(f"\n10. Binary recoverable (score >= 1):")
    print(f"    Auto recov={sum(auto_rec)} Human recov={sum(human_rec)}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"    Precision={b_prec:.4f} Recall={b_rec:.4f} Specificity={b_spec:.4f} F1={b_f1:.4f}")
    lo, hi = wilson_ci(b_rec, tp+fn)
    print(f"    Sensitivity 95% CI: [{lo:.1%}, {hi:.1%}]")

    # ── 11. Score 0 detection ──
    auto_is0 = [1 if s == 0 else 0 for s in auto_scores]
    human_is0 = [1 if s == 0 else 0 for s in human_scores]
    tp0 = sum(1 for a, h in zip(auto_is0, human_is0) if a == 1 and h == 1)
    fp0 = sum(1 for a, h in zip(auto_is0, human_is0) if a == 1 and h == 0)
    tn0 = sum(1 for a, h in zip(auto_is0, human_is0) if a == 0 and h == 0)
    fn0 = sum(1 for a, h in zip(auto_is0, human_is0) if a == 0 and h == 1)

    p0 = tp0/(tp0+fp0) if (tp0+fp0) > 0 else 0
    r0 = tp0/(tp0+fn0) if (tp0+fn0) > 0 else 0
    print(f"\n11. Auto Score=0 detection of human-unrecoverable:")
    print(f"    TP={tp0} FP={fp0} TN={tn0} FN={fn0}")
    print(f"    Precision={p0:.4f} Recall={r0:.4f}")

    # ── 12. Per-benchmark and per-method diagnostics ──
    print(f"\n12. Per-benchmark diagnostics (for diagnosis only, NOT quality ranking):")
    print(f"    {'Benchmark':<25} {'N':>4} {'Exact':>6} {'Exact%':>7} {'RecovP':>7} {'RecovR':>7} {'F1':>6}")
    print(f"    {'-'*70}")
    for bname in sorted(set(m["benchmark"] for m in mapping.values())):
        bids_b = [bid for bid in sorted(common_ids) if mapping[bid]["benchmark"] == bname]
        if len(bids_b) < 3:
            continue
        auto_b = [mapping[bid]["automatic_score"] for bid in bids_b]
        hum_b = []
        for bid in bids_b:
            if bid in adjudicated:
                hum_b.append(adjudicated[bid])
            else:
                ha = ann_a[bid].get("human_score")
                hb = ann_b[bid].get("human_score")
                if ha is not None and hb is not None and ha == hb:
                    hum_b.append(ha)
        if len(hum_b) < 3:
            continue
        exact_b = sum(1 for a, h in zip(auto_b[:len(hum_b)], hum_b) if a == h)
        # recoverable per benchmark
        auto_rb = [1 if s >= 1 else 0 for s in auto_b[:len(hum_b)]]
        hum_rb = [1 if s >= 1 else 0 for s in hum_b]
        tp_b = sum(1 for a, h in zip(auto_rb, hum_rb) if a == 1 and h == 1)
        fp_b = sum(1 for a, h in zip(auto_rb, hum_rb) if a == 1 and h == 0)
        fn_b = sum(1 for a, h in zip(auto_rb, hum_rb) if a == 0 and h == 1)
        p_b = tp_b/(tp_b+fp_b) if (tp_b+fp_b)>0 else 0
        r_b = tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else 0
        f1_b = 2*p_b*r_b/(p_b+r_b) if (p_b+r_b)>0 else 0
        print(f"    {bname:<25} {len(hum_b):>4} {exact_b:>6} {100*exact_b/len(hum_b):>6.1f}% {p_b:>6.3f} {r_b:>6.3f} {f1_b:>5.3f}")

    print(f"\n    {'Method':<15} {'N':>4} {'Exact':>6} {'Exact%':>7} {'RecovP':>7} {'RecovR':>7} {'F1':>6}")
    print(f"    {'-'*60}")
    for mname in ["AR", "FLY", "TASD-BR"]:
        bids_m = [bid for bid in sorted(common_ids) if mapping[bid]["method"] == mname]
        if len(bids_m) < 3:
            continue
        auto_m = [mapping[bid]["automatic_score"] for bid in bids_m]
        hum_m = []
        for bid in bids_m:
            if bid in adjudicated:
                hum_m.append(adjudicated[bid])
            else:
                ha = ann_a[bid].get("human_score")
                hb = ann_b[bid].get("human_score")
                if ha is not None and hb is not None and ha == hb:
                    hum_m.append(ha)
        if len(hum_m) < 3:
            continue
        exact_m = sum(1 for a, h in zip(auto_m[:len(hum_m)], hum_m) if a == h)
        auto_rm = [1 if s >= 1 else 0 for s in auto_m[:len(hum_m)]]
        hum_rm = [1 if s >= 1 else 0 for s in hum_m]
        tp_m = sum(1 for a, h in zip(auto_rm, hum_rm) if a == 1 and h == 1)
        fp_m = sum(1 for a, h in zip(auto_rm, hum_rm) if a == 1 and h == 0)
        fn_m = sum(1 for a, h in zip(auto_rm, hum_rm) if a == 0 and h == 1)
        p_m = tp_m/(tp_m+fp_m) if (tp_m+fp_m)>0 else 0
        r_m = tp_m/(tp_m+fn_m) if (tp_m+fn_m)>0 else 0
        f1_m = 2*p_m*r_m/(p_m+r_m) if (p_m+r_m)>0 else 0
        print(f"    {mname:<15} {len(hum_m):>4} {exact_m:>6} {100*exact_m/len(hum_m):>6.1f}% {p_m:>6.3f} {r_m:>6.3f} {f1_m:>5.3f}")

    # ── 13. Weighted estimates using pool proportions ──
    manifest_path = "results/score_validator_review/public_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            man = json.load(f)
        pool_props = man.get("pool_distribution", {}).get("pool_proportions", {})
        if pool_props:
            w0 = float(pool_props.get("0", 1/3))
            w1 = float(pool_props.get("1", 1/3))
            w2 = float(pool_props.get("2", 1/3))
            weights = {0: w0, 1: w1, 2: w2}
            # Per-class accuracy weighted
            per_class_correct = {0: 0, 1: 0, 2: 0}
            per_class_total = {0: 0, 1: 0, 2: 0}
            for a, h in zip(auto_scores, human_scores):
                per_class_total[h] += 1
                if a == h:
                    per_class_correct[h] += 1
            weighted_acc = 0.0
            for c in [0, 1, 2]:
                if per_class_total[c] > 0:
                    weighted_acc += weights[c] * per_class_correct[c] / per_class_total[c]
            print(f"\n13. Inverse-prevalence weighted accuracy: {weighted_acc:.4f}")
            print(f"    (Pool proportions: 0={w0:.1%} 1={w1:.1%} 2={w2:.1%})")
            print(f"    NOTE: This is a weighted estimate, NOT the raw sample accuracy.")

    # Save statistics
    stats = {
        "n_items": len(common_ids),
        "n_evaluated": n_eval,
        "annotator_a_completed": a_done,
        "annotator_b_completed": b_done,
        "human_agreement_raw": raw_agree / len(a_common) if a_common else 0,
        "human_cohens_kappa": kappa,
        "completion_status_raw_agree": cs_raw / len(cs_a_common) if cs_a_common else 0,
        "completion_status_kappa": cs_kappa,
        "disagreement_count": len(disagreements),
        "exact_agreement": exact / n_eval if n_eval else 0,
        "auto_human_weighted_kappa": auk,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "binary_precision": b_prec,
        "binary_recall": b_rec,
        "binary_specificity": b_spec,
        "binary_f1": b_f1,
        "score0_precision": p0,
        "score0_recall": r0,
        "confusion_matrix": cm.tolist(),
    }
    stats_path = f"{out_dir}/statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {stats_path}")

    return stats


# ══════════════════════════════════════════════════════════════════════════
# Dummy data for testing
# ══════════════════════════════════════════════════════════════════════════

def generate_dummy_annotations():
    """Generate synthetic V2 annotations for testing the analysis pipeline."""
    mapping = load_mapping()
    blind_ids = sorted(mapping.keys())
    random.seed(20260625)

    # Synthesize with controlled agreement patterns
    ann_a = []
    ann_b = []
    for bid in blind_ids:
        m = mapping[bid]
        auto_score = m["automatic_score"]

        # A: 70% agree with auto, 20% off by 1, 10% off by 2
        r = random.random()
        if r < 0.70:
            hs_a = auto_score
        elif r < 0.90:
            hs_a = max(0, min(2, auto_score + random.choice([-1, 1])))
        else:
            hs_a = (auto_score + 2) % 3

        # B: similar pattern but different noise
        r = random.random()
        if r < 0.65:
            hs_b = auto_score
        elif r < 0.88:
            hs_b = max(0, min(2, auto_score + random.choice([-1, 1])))
        else:
            hs_b = (auto_score + 1) % 3

        cs_opts = ["complete", "tail_cutoff", "severe_incomplete"]
        cs_a = random.choice(cs_opts)
        cs_b = random.choice(cs_opts)
        if random.random() < 0.6:
            cs_b = cs_a  # moderate agreement

        tags_pool = ["bracket_or_delimiter", "indentation", "repetition", "duplicate_field",
                     "off_structure", "wrong_content", "other", "none"]
        tags_a = [random.choice(tags_pool)]
        tags_b = [random.choice(tags_pool)]

        ann_a.append({"blind_id": bid, "human_score": hs_a, "completion_status": cs_a,
                      "issue_tags": tags_a, "notes": ""})
        ann_b.append({"blind_id": bid, "human_score": hs_b, "completion_status": cs_b,
                      "issue_tags": tags_b, "notes": ""})

    out_dir = "results/score_validator_review"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/annotator_A.json", "w") as f:
        json.dump(ann_a, f, indent=2, ensure_ascii=False)
    with open(f"{out_dir}/annotator_B.json", "w") as f:
        json.dump(ann_b, f, indent=2, ensure_ascii=False)
    print(f"Dummy annotations generated: {out_dir}/annotator_A.json, {out_dir}/annotator_B.json")
    return f"{out_dir}/annotator_A.json", f"{out_dir}/annotator_B.json"


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--dummy" in sys.argv:
        a_path, b_path = generate_dummy_annotations()
        analyze(a_path, b_path)
    elif len(sys.argv) >= 3:
        a_path = sys.argv[1]
        b_path = sys.argv[2]
        adj_csv = None
        for i, arg in enumerate(sys.argv):
            if arg == "--adjudicated" and i + 1 < len(sys.argv):
                adj_csv = sys.argv[i+1]
        analyze(a_path, b_path, adj_csv)
    else:
        print(__doc__)
