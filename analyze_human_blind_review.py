#!/usr/bin/env python3
"""
TASD Human Blind Review — Statistical Analysis.

Reads:
  - annotator_A.json / annotator_B.json (exported from HTML)
  - blind_mapping_private.json (method ground truth)

Outputs:
  - results/human_blind_review/human_review_report.md
  - results/human_blind_review/human_review_statistics.json
  - results/human_blind_review/disagreements_for_adjudication.csv

Rules:
  - Does NOT auto-average or auto-adjudicate disagreements.
  - Outputs disagreement list for third-party adjudication.
  - Final method comparison uses adjudicated scores only.
"""

import json
import os
import sys
import csv
import random
from collections import Counter, defaultdict

import numpy as np

OUT_DIR = "results/human_blind_review"
MAPPING_FILE = f"{OUT_DIR}/blind_mapping_private.json"

# ============================================================
# Helper: Load annotations
# ============================================================

def load_annotations(json_path):
    """Load annotator JSON export."""
    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found")
        return None
    with open(json_path) as f:
        data = json.load(f)
    # Convert to dict keyed by blind_id
    result = {}
    for item in data:
        result[item["blind_id"]] = {
            "score": item.get("score"),
            "tags": item.get("tags", []),
            "notes": item.get("notes", ""),
        }
    return result


def load_annotations_v2(json_path):
    """Load V2 annotator JSON export with dual-dimension fields."""
    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found")
        return None
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    for item in data:
        result[item["blind_id"]] = {
            "prefix_score": item.get("prefix_score"),
            "completion_status": item.get("completion_status"),
            "issue_tags": item.get("issue_tags", []),
            "notes": item.get("notes", ""),
            "trim_position": item.get("trim_position", ""),
        }
    return result


def load_private_mapping():
    """Load ground truth method labels for each blind_id.
    Supports both v1 (original_sample_name, original_sample_idx) and
    v2 (sample_name, sample_idx) formats."""
    if not os.path.exists(MAPPING_FILE):
        print(f"  ERROR: {MAPPING_FILE} not found")
        return None
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    result = {}
    for item in mapping:
        # Normalize keys
        entry = {
            "method": item["method"],
            "benchmark": item["benchmark"],
            "sample_name": item.get("sample_name", item.get("original_sample_name", "")),
            "sample_idx": item.get("sample_idx", item.get("original_sample_idx", -1)),
        }
        if "br_rerun" in item:
            entry["br_rerun"] = item["br_rerun"]
        result[item["blind_id"]] = entry
    return result


# ============================================================
# Cohen's Weighted Kappa
# ============================================================

def cohen_weighted_kappa(rater1, rater2):
    """Cohen's weighted kappa using quadratic weights for ordinal 0/1/2."""
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have same length")

    n = len(rater1)
    # Observed matrix
    obs = np.zeros((3, 3))
    for a, b in zip(rater1, rater2):
        if a is not None and b is not None:
            obs[int(a), int(b)] += 1

    obs_sum = obs.sum()
    if obs_sum == 0:
        return 0.0

    # Expected matrix
    row_sum = obs.sum(axis=1)
    col_sum = obs.sum(axis=0)
    exp = np.outer(row_sum, col_sum) / obs_sum

    # Quadratic weights: w_ij = ((i - j) / 2)^2
    weights = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            weights[i, j] = ((i - j) / 2.0) ** 2

    num = (weights * obs).sum()
    den = (weights * exp).sum()

    if den == 0:
        return 1.0

    kappa = 1 - num / den
    return kappa


# ============================================================
# McNemar test
# ============================================================

def mcnemar_test(pairs_a, pairs_b):
    """
    McNemar test for paired binary data.
    pairs_a and pairs_b are lists of (blind_id, binary_score) aligned by prompt.
    Returns chi2 statistic and p-value.
    """
    # Build contingency table: counts of (a=0,b=0), (a=0,b=1), (a=1,b=0), (a=1,b=1)
    b = 0  # both 0
    c = 0  # a=0, b=1
    d = 0  # a=1, b=0
    e = 0  # both 1

    for (_, sa), (_, sb) in zip(pairs_a, pairs_b):
        if sa == 0 and sb == 0:
            b += 1
        elif sa == 0 and sb == 1:
            c += 1
        elif sa == 1 and sb == 0:
            d += 1
        elif sa == 1 and sb == 1:
            e += 1

    if c + d == 0:
        return None, 1.0  # No discordant pairs

    # Yates correction
    chi2_val = (abs(c - d) - 1) ** 2 / (c + d) if c + d > 0 else 0

    # p-value from chi-squared distribution with 1 df
    # Using approximation: p = exp(-chi2/2) * sqrt(2/(pi*chi2)) for large chi2
    # Or simply use the survival function approximation
    import math
    if chi2_val <= 0:
        p_val = 1.0
    elif chi2_val > 50:
        p_val = 0.0
    else:
        # Gamma regularized lower: P(a,x) where a=df/2=0.5, x=chi2/2
        # Simplified: use complementary error function
        z = math.sqrt(chi2_val)
        p_val = math.erfc(z / math.sqrt(2))
    return chi2_val, p_val


# ============================================================
# Main
# ============================================================

def main(annotator_a_path=None, annotator_b_path=None,
         adjudicated_path=None):
    """
    Run full analysis.

    Args:
        annotator_a_path: path to annotator A's JSON export
        annotator_b_path: path to annotator B's JSON export
        adjudicated_path: optional path to adjudicated scores CSV
                          (with columns: blind_id, adjudicated_score)
    """
    print("=" * 60)
    print("TASD Human Blind Review — Statistical Analysis")
    print("=" * 60)

    # Default paths
    if annotator_a_path is None:
        annotator_a_path = f"{OUT_DIR}/annotator_A.json"
    if annotator_b_path is None:
        annotator_b_path = f"{OUT_DIR}/annotator_B.json"

    # Load
    print("\n1. Loading data...")
    ann_a = load_annotations(annotator_a_path)
    ann_b = load_annotations(annotator_b_path)
    mapping = load_private_mapping()

    if ann_a is None or ann_b is None:
        print("\nERROR: Both annotator files required.")
        print("Expected files:")
        print(f"  {annotator_a_path}")
        print(f"  {annotator_b_path}")
        print("\nTo test with dummy data, run: python analyze_human_blind_review.py --dummy")
        sys.exit(1)

    if mapping is None:
        print("\nERROR: Private mapping required.")
        sys.exit(1)

    print(f"  Annotator A: {len(ann_a)} items")
    print(f"  Annotator B: {len(ann_b)} items")
    print(f"  Mapping: {len(mapping)} items")

    # ============================================================
    # 1. Completion rate
    # ============================================================
    print("\n2. Completion rates...")
    total = len(mapping)
    done_a = sum(1 for bid in mapping if bid in ann_a and ann_a[bid]["score"] is not None)
    done_b = sum(1 for bid in mapping if bid in ann_b and ann_b[bid]["score"] is not None)
    print(f"  Annotator A: {done_a}/{total} ({100*done_a/total:.1f}%)")
    print(f"  Annotator B: {done_b}/{total} ({100*done_b/total:.1f}%)")

    # ============================================================
    # 2-4: Agreement
    # ============================================================
    print("\n3. Inter-annotator agreement...")
    common = []
    for bid in mapping:
        if bid in ann_a and bid in ann_b:
            sa = ann_a[bid]["score"]
            sb = ann_b[bid]["score"]
            if sa is not None and sb is not None:
                common.append((bid, sa, sb))

    n_common = len(common)
    if n_common == 0:
        print("  WARNING: No common scored items. Run annotations first.")
        return

    raw_agree = sum(1 for _, sa, sb in common if sa == sb)
    raw_rate = raw_agree / n_common
    print(f"  Common items: {n_common}")
    print(f"  Raw agreement: {raw_agree}/{n_common} ({100*raw_rate:.1f}%)")

    # Cohen's weighted kappa
    a_scores = [sa for _, sa, _ in common]
    b_scores = [sb for _, _, sb in common]
    kappa = cohen_weighted_kappa(a_scores, b_scores)
    print(f"  Cohen's weighted kappa: {kappa:.4f}")

    # Score distributions
    print("\n4. Score distributions...")
    for label, ann in [("Annotator A", ann_a), ("Annotator B", ann_b)]:
        dist = Counter()
        for bid in mapping:
            if bid in ann and ann[bid]["score"] is not None:
                dist[ann[bid]["score"]] += 1
        print(f"  {label}: 0={dist[0]}, 1={dist[1]}, 2={dist[2]}")

    # ============================================================
    # 5. Disagreements
    # ============================================================
    print("\n5. Disagreement list...")
    disagreements = []
    for bid, sa, sb in common:
        if sa != sb:
            disagreements.append({
                "blind_id": bid,
                "annotator_A_score": int(sa),
                "annotator_B_score": int(sb),
                "method": mapping[bid]["method"],
                "benchmark": mapping[bid]["benchmark"],
                "sample_name": mapping[bid]["sample_name"],
            })
    print(f"  Disagreements: {len(disagreements)}/{n_common}")

    # Save disagreements for adjudication
    import csv
    disp_path = f"{OUT_DIR}/disagreements_for_adjudication.csv"
    with open(disp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blind_id", "annotator_A_score", "annotator_B_score",
            "method", "benchmark", "sample_name",
            "adjudicated_score"
        ])
        writer.writeheader()
        for d in disagreements:
            d_copy = dict(d)
            d_copy["adjudicated_score"] = ""
            writer.writerow(d_copy)
    print(f"  -> {disp_path}")

    # ============================================================
    # Load adjudicated scores if available
    # ============================================================
    adjudicated = {}
    if adjudicated_path and os.path.exists(adjudicated_path):
        with open(adjudicated_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bid = row["blind_id"]
                adj = row.get("adjudicated_score", "").strip()
                if adj != "":
                    adjudicated[bid] = int(adj)
        print(f"  Adjudicated scores loaded: {len(adjudicated)} items")

    # ============================================================
    # 6. Per-method analysis
    # ============================================================
    print("\n6. Per-method human usability...")

    # Use annotator A if no adjudication available
    def get_final_score(bid):
        if bid in adjudicated:
            return adjudicated[bid]
        # If no adjudication, use annotator A (or flag)
        if bid in ann_a and ann_a[bid]["score"] is not None:
            return ann_a[bid]["score"]
        return None

    method_scores = defaultdict(list)
    for bid, info in mapping.items():
        s = get_final_score(bid)
        if s is not None:
            method_scores[info["method"]].append((bid, s))

    print(f"  Using {'adjudicated' if adjudicated else 'Annotator A'} scores")
    print()
    print("  | Method | Score 2 (%) | Score 1 (%) | Score 0 (%) | Usable (>=1) | N |")
    print("  |--------|:-----------:|:-----------:|:-----------:|:------------:|:--:|")

    method_stats = {}
    for method in ["AR", "FLY", "TASD-BR"]:
        scores = method_scores.get(method, [])
        n = len(scores)
        if n == 0:
            method_stats[method] = None
            print(f"  | {method} | — | — | — | — | 0 |")
            continue
        dist = Counter(s for _, s in scores)
        p2 = 100 * dist.get(2, 0) / n
        p1 = 100 * dist.get(1, 0) / n
        p0 = 100 * dist.get(0, 0) / n
        usable = p2 + p1
        method_stats[method] = {
            "n": n, "score_0": dist.get(0, 0), "score_1": dist.get(1, 0),
            "score_2": dist.get(2, 0), "score_0_pct": p0, "score_1_pct": p1,
            "score_2_pct": p2, "usable_pct": usable,
            "scores": [(bid, s) for bid, s in scores],
        }
        print(f"  | {method} | {p2:.1f}% | {p1:.1f}% | {p0:.1f}% | {usable:.1f}% | {n} |")

    # ============================================================
    # 7. TASD-BR vs FLY paired comparison
    # ============================================================
    print("\n7. TASD-BR vs FLY paired comparison...")

    # Group by prompt (benchmark + sample_name)
    br_pairs = method_stats.get("TASD-BR", {}).get("scores", [])
    fly_pairs = method_stats.get("FLY", {}).get("scores", [])
    ar_pairs = method_stats.get("AR", {}).get("scores", [])

    # Align by sample name
    def by_sample(pairs, mapping):
        result = {}
        for bid, s in pairs:
            sample_key = mapping[bid]["sample_name"]
            result[sample_key] = (bid, s)
        return result

    br_by_sample = by_sample(br_pairs, mapping)
    fly_by_sample = by_sample(fly_pairs, mapping)
    ar_by_sample = by_sample(ar_pairs, mapping)

    # Paired comparison
    aligned = []
    for sample_key in br_by_sample:
        if sample_key in fly_by_sample:
            _, br_s = br_by_sample[sample_key]
            _, fly_s = fly_by_sample[sample_key]
            aligned.append((sample_key, br_s, fly_s))

    n_pair = len(aligned)
    br_better = sum(1 for _, br_s, fly_s in aligned if br_s > fly_s)
    fly_better = sum(1 for _, br_s, fly_s in aligned if fly_s > br_s)
    tied = sum(1 for _, br_s, fly_s in aligned if br_s == fly_s)

    print(f"  Paired prompts: {n_pair}")
    print(f"  BR better: {br_better} ({100*br_better/n_pair:.1f}%)" if n_pair else "")
    print(f"  FLY better: {fly_better} ({100*fly_better/n_pair:.1f}%)" if n_pair else "")
    print(f"  Tied: {tied} ({100*tied/n_pair:.1f}%)" if n_pair else "")

    # ============================================================
    # 8. McNemar test (usability: score >= 1)
    # ============================================================
    print("\n8. McNemar test (usability: score >= 1)...")
    mc_br = [(k, 1 if s >= 1 else 0) for k, s in [br_by_sample.get(k, (None, None)) for k in br_by_sample]]
    mc_fly = [(k, 1 if s >= 1 else 0) for k, s in [fly_by_sample.get(k, (None, None)) for k in fly_by_sample]]

    # Filter to common prompts
    common_prompts = set(br_by_sample.keys()) & set(fly_by_sample.keys())
    mc_br_aligned = [(k, (1 if br_by_sample[k][1] >= 1 else 0)) for k in common_prompts]
    mc_fly_aligned = [(k, (1 if fly_by_sample[k][1] >= 1 else 0)) for k in common_prompts]
    mc_br_aligned.sort()
    mc_fly_aligned.sort()

    try:
        chi2_val, p_val = mcnemar_test(mc_br_aligned, mc_fly_aligned)
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"  McNemar chi2 = {chi2_val:.3f}" if chi2_val else "  McNemar: no discordant pairs")
        print(f"  p-value = {p_val:.4f} ({sig} at alpha=0.05)")
    except Exception as e:
        print(f"  McNemar test failed: {e}")

    # ============================================================
    # 9. Human vs automatic score comparison
    # ============================================================
    print("\n9. Human vs automatic score comparison...")

    # Load automatic scores
    auto_file = "results/all_methods_structural_recoverability.json"
    # Build two lookup structures because TASD-FG uses 'name' not 'sample_idx'
    auto_by_idx = {}   # {(method, benchmark, sample_idx): score}  for AR, FLY
    auto_by_name = {}  # {(method, benchmark, sample_name): score} for TASD-FG
    if os.path.exists(auto_file):
        with open(auto_file) as f:
            auto_data = json.load(f)
        for method_key in ["AR", "FLY"]:
            if method_key in auto_data:
                for s in auto_data[method_key]:
                    auto_by_idx[(method_key, s["benchmark"], s["sample_idx"])] = s["score"]
        # TASD-FG uses 'name' field
        if "TASD-FG" in auto_data:
            for s in auto_data["TASD-FG"]:
                auto_by_name[("TASD-FG", s["benchmark"], s["name"])] = s["score"]

    # Map human scores to automatic scores
    human_auto_pairs = []
    for bid, info in mapping.items():
        method = info["method"]
        s_human = get_final_score(bid)
        if s_human is None:
            continue
        sample_idx = info.get("sample_idx")
        sample_name = info.get("sample_name")
        bname = info["benchmark"]

        s_auto = None
        if method == "TASD-BR":
            br_rerun = info.get("br_rerun", False)
            if br_rerun:
                if sample_idx is not None:
                    s_auto = auto_by_idx.get(("AR", bname, sample_idx))
            else:
                if sample_name:
                    s_auto = auto_by_name.get(("TASD-FG", bname, sample_name))
        elif method == "AR":
            s_auto = auto_by_idx.get(("AR", bname, sample_idx))
        elif method == "FLY":
            s_auto = auto_by_idx.get(("FLY", bname, sample_idx))

        if s_auto is not None:
            human_auto_pairs.append((s_human, s_auto))

    if human_auto_pairs:
        n_ha = len(human_auto_pairs)
        # Confusion matrix
        cm = np.zeros((3, 3), dtype=int)
        for sh, sa in human_auto_pairs:
            cm[int(sh), int(sa)] += 1

        correct = cm.trace()
        accuracy = correct / n_ha
        print(f"  Paired items: {n_ha}")
        print(f"  Accuracy: {accuracy:.3f} ({correct}/{n_ha})")
        print(f"  Confusion matrix (rows=human, cols=auto):")
        print(f"        Auto:0  Auto:1  Auto:2")
        for i in range(3):
            print(f"  Human {i}:  {cm[i,0]:7d}  {cm[i,1]:7d}  {cm[i,2]:7d}")

        # Macro F1
        f1s = []
        for c in range(3):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1s.append(f1)
        macro_f1 = np.mean(f1s)
        print(f"  Macro-F1: {macro_f1:.4f}")
    else:
        print("  No human-auto score pairs available.")

    # ============================================================
    # 10. Per-benchmark human usability
    # ============================================================
    print("\n10. Per-benchmark human usability...")

    bench_method_scores = defaultdict(lambda: defaultdict(list))
    for bid, info in mapping.items():
        s = get_final_score(bid)
        if s is not None:
            bench_method_scores[info["benchmark"]][info["method"]].append(s)

    print()
    print("  | Benchmark | AR Usable | FLY Usable | TASD-BR Usable |")
    print("  |-----------|:---------:|:----------:|:--------------:|")
    bench_stats = {}
    for bname in sorted(bench_method_scores.keys()):
        stats_row = {}
        row = [f"  | {bname}"]
        for method in ["AR", "FLY", "TASD-BR"]:
            scores = bench_method_scores[bname].get(method, [])
            n = len(scores)
            if n == 0:
                row.append(" — ")
                stats_row[method] = {"n": 0, "usable_pct": None}
            else:
                usable = sum(1 for s in scores if s >= 1) / n * 100
                row.append(f" {usable:.0f}% ")
                stats_row[method] = {"n": n, "usable_pct": usable}
        print("|".join(row) + "|")
        bench_stats[bname] = stats_row

    # ============================================================
    # Save outputs
    # ============================================================
    print(f"\nSaving outputs to {OUT_DIR}/...")

    # Statistics JSON
    stats = {
        "completion": {"annotator_A": done_a, "annotator_B": done_b, "total": total},
        "agreement": {
            "common_items": n_common,
            "raw_agreement": raw_rate,
            "cohens_weighted_kappa": float(kappa),
        },
        "disagreements_count": len(disagreements),
        "per_method": method_stats,
        "paired_comparison": {
            "n_paired": n_pair,
            "br_better": br_better,
            "fly_better": fly_better,
            "tied": tied,
        },
        "per_benchmark": bench_stats,
    }
    with open(f"{OUT_DIR}/human_review_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=int)

    # Report MD
    lines = []
    lines.append("# TASD Human Blind Review — Statistical Report\n")
    lines.append(f"**Total items**: {total} (60 prompts × 3 methods)\n")

    lines.append("## 1. Completion\n")
    lines.append(f"- Annotator A: {done_a}/{total} ({100*done_a/total:.1f}%)")
    lines.append(f"- Annotator B: {done_b}/{total} ({100*done_b/total:.1f}%)\n")

    lines.append("## 2. Inter-Annotator Agreement\n")
    lines.append(f"- Common items scored: {n_common}")
    lines.append(f"- Raw agreement: {100*raw_rate:.1f}%")
    lines.append(f"- Cohen's weighted kappa: {kappa:.4f}\n")

    lines.append(f"Disagreements: {len(disagreements)} → {disp_path}\n")

    lines.append("## 3. Per-Method Human Usability\n")
    present_methods = [m for m in ["AR", "FLY", "TASD-BR"] if method_stats.get(m)]
    lines.append("| Method | Score 2 | Score 1 | Score 0 | Usable (>=1) | N |")
    lines.append("|--------|:------:|:------:|:------:|:------------:|:--:|")
    for method in present_methods:
        ms = method_stats[method]
        lines.append(f"| {method} | {ms['score_2_pct']:.1f}% | {ms['score_1_pct']:.1f}% | "
                     f"{ms['score_0_pct']:.1f}% | {ms['usable_pct']:.1f}% | {ms['n']} |")
    lines.append("")

    lines.append("## 4. TASD-BR vs FLY Paired Comparison\n")
    if n_pair > 0:
        lines.append(f"- Paired prompts: {n_pair}")
        lines.append(f"- BR better: {br_better} ({100*br_better/n_pair:.1f}%)")
        lines.append(f"- FLY better: {fly_better} ({100*fly_better/n_pair:.1f}%)")
        lines.append(f"- Tied: {tied} ({100*tied/n_pair:.1f}%)")
    lines.append("")

    lines.append("## 5. Per-Benchmark\n")
    lines.append("| Benchmark | AR Usable | FLY Usable | TASD-BR Usable |")
    lines.append("|-----------|:---------:|:----------:|:--------------:|")
    for bname in sorted(bench_stats.keys()):
        bs = bench_stats[bname]
        ar_u = f"{bs['AR']['usable_pct']:.0f}%" if bs['AR'].get('usable_pct') is not None else "—"
        fly_u = f"{bs['FLY']['usable_pct']:.0f}%" if bs['FLY'].get('usable_pct') is not None else "—"
        br_u = f"{bs['TASD-BR']['usable_pct']:.0f}%" if bs['TASD-BR'].get('usable_pct') is not None else "—"
        lines.append(f"| {bname} | {ar_u} | {fly_u} | {br_u} |")
    lines.append("")

    lines.append("## 6. Disagreement Resolution\n")
    if disagreements:
        lines.append(f"{len(disagreements)} items require third-party adjudication.")
        lines.append(f"See: `{disp_path}`")
        lines.append("After adjudication, re-run with: `--adjudicated <path>`")
    else:
        lines.append("No disagreements to resolve.")

    lines.append("")

    report = "\n".join(lines)
    with open(f"{OUT_DIR}/human_review_report.md", "w") as f:
        f.write(report)

    print(f"  -> {OUT_DIR}/human_review_statistics.json")
    print(f"  -> {OUT_DIR}/human_review_report.md")
    print(f"  -> {disp_path}")
    print("\nDone.")


# ============================================================
# V2 Analysis: Dual-dimension (Prefix Quality + Completion Status)
# ============================================================

def analyze_v2(annotator_a_path, annotator_b_path):
    """Run V2 dual-dimension analysis with pilot metrics."""
    print("=" * 60)
    print("TASD Human Blind Review V2 — Statistical Analysis")
    print("=" * 60)

    print("\n1. Loading data...")
    ann_a = load_annotations_v2(annotator_a_path)
    ann_b = load_annotations_v2(annotator_b_path)
    mapping = load_private_mapping()

    if ann_a is None or ann_b is None or mapping is None:
        print("ERROR: Both annotator files and mapping required.")
        sys.exit(1)

    total = len(mapping)
    print(f"  Annotator A: {len(ann_a)} items")
    print(f"  Annotator B: {len(ann_b)} items")
    print(f"  Mapping: {total} items")

    # Completion rate (both dimensions required)
    print("\n2. Completion rates (both dimensions)...")
    done_a = sum(1 for bid in mapping if bid in ann_a
                 and ann_a[bid]["prefix_score"] is not None
                 and ann_a[bid]["completion_status"] is not None)
    done_b = sum(1 for bid in mapping if bid in ann_b
                 and ann_b[bid]["prefix_score"] is not None
                 and ann_b[bid]["completion_status"] is not None)
    print(f"  Annotator A: {done_a}/{total} ({100*done_a/total:.1f}%)")
    print(f"  Annotator B: {done_b}/{total} ({100*done_b/total:.1f}%)")

    # Collect common
    common = []
    for bid in mapping:
        if bid in ann_a and bid in ann_b:
            apf = ann_a[bid]["prefix_score"]
            acs = ann_a[bid]["completion_status"]
            bpf = ann_b[bid]["prefix_score"]
            bcs = ann_b[bid]["completion_status"]
            if all(x is not None for x in [apf, acs, bpf, bcs]):
                common.append((bid, apf, acs, bpf, bcs))
    n_common = len(common)
    print(f"  Fully scored common items: {n_common}")
    if n_common == 0:
        print("WARNING: No common scored items.")
        return

    # Prefix agreement
    print("\n3. Prefix Structural Quality agreement...")
    pf_a = [pf for _, pf, _, _, _ in common]
    pf_b = [pf for _, _, _, pf, _ in common]
    pf_agree = sum(1 for a, b in zip(pf_a, pf_b) if a == b)
    pf_kappa = cohen_weighted_kappa(pf_a, pf_b)
    print(f"  Raw agreement: {pf_agree}/{n_common} ({100*pf_agree/n_common:.1f}%)")
    print(f"  Cohen's weighted kappa: {pf_kappa:.4f}")

    # Completion Status agreement
    print("\n4. Completion Status agreement...")
    cs_a = [cs for _, _, cs, _, _ in common]
    cs_b = [cs for _, _, _, _, cs in common]
    cs_agree = sum(1 for a, b in zip(cs_a, cs_b) if a == b)

    # Completion status kappa (nominal)
    cs_map = {"complete": 0, "tail_cutoff": 1, "severe_incomplete": 2}
    cs_a_num = [cs_map[c] for c in cs_a]
    cs_b_num = [cs_map[c] for c in cs_b]
    cs_kappa = cohen_weighted_kappa(cs_a_num, cs_b_num)
    print(f"  Raw agreement: {cs_agree}/{n_common} ({100*cs_agree/n_common:.1f}%)")
    print(f"  Cohen's weighted kappa: {cs_kappa:.4f}")

    # Prefix score distributions (per annotator)
    print("\n5. Prefix score distributions...")
    for label, pf_list in [("Annotator A", pf_a), ("Annotator B", pf_b)]:
        dist = Counter(pf_list)
        print(f"  {label}: 2={dist[2]} ({100*dist[2]/n_common:.1f}%), "
              f"1={dist[1]} ({100*dist[1]/n_common:.1f}%), "
              f"0={dist[0]} ({100*dist[0]/n_common:.1f}%)")

    # Completion status distributions
    print("\n6. Completion Status distributions...")
    for label, cs_list in [("Annotator A", cs_a), ("Annotator B", cs_b)]:
        dist = Counter(cs_list)
        print(f"  {label}: complete={dist['complete']} ({100*dist['complete']/n_common:.1f}%), "
              f"tail_cutoff={dist['tail_cutoff']} ({100*dist['tail_cutoff']/n_common:.1f}%), "
              f"severe_incomplete={dist['severe_incomplete']} ({100*dist['severe_incomplete']/n_common:.1f}%)")

    # Use annotator A scores for per-method analysis
    # (In production, use adjudicated scores)
    a_by_bid = {}  # bid -> (prefix_score, completion_status)
    for bid in mapping:
        if bid in ann_a and ann_a[bid]["prefix_score"] is not None:
            a_by_bid[bid] = (ann_a[bid]["prefix_score"], ann_a[bid]["completion_status"])

    # Per-method metrics (using Annotator A)
    print("\n7. Per-method metrics (Annotator A)...")
    method_stats = defaultdict(lambda: {
        "n": 0, "pf2": 0, "pf1": 0, "pf0": 0,
        "complete": 0, "tail_cutoff": 0, "severe_incomplete": 0,
        "directly_usable": 0, "trim_recoverable": 0,
    })
    for bid, info in mapping.items():
        if bid not in a_by_bid:
            continue
        pf, cs = a_by_bid[bid]
        m = info["method"]
        method_stats[m]["n"] += 1
        method_stats[m]["pf2"] += (pf == 2)
        method_stats[m]["pf1"] += (pf == 1)
        method_stats[m]["pf0"] += (pf == 0)
        method_stats[m]["complete"] += (cs == "complete")
        method_stats[m]["tail_cutoff"] += (cs == "tail_cutoff")
        method_stats[m]["severe_incomplete"] += (cs == "severe_incomplete")
        # Directly usable: prefix=2 AND completion=complete
        method_stats[m]["directly_usable"] += (pf == 2 and cs == "complete")
        # Trim-recoverable: prefix>=1 AND (complete OR tail_cutoff)
        method_stats[m]["trim_recoverable"] += (pf >= 1 and cs in ("complete", "tail_cutoff"))

    print(f"\n  {'Method':<12} {'N':>4} {'Clean Pfx%':>9} {'Recoverable%':>12} "
          f"{'Complete%':>10} {'TailCut%':>9} {'Severe%':>8} "
          f"{'Directly%':>9} {'TrimRec%':>9}")
    print("  " + "-" * 82)
    for m in ["AR", "FLY", "TASD-BR"]:
        if m not in method_stats:
            continue
        s = method_stats[m]
        n = s["n"]
        if n == 0:
            continue
        print(f"  {m:<12} {n:>4} {100*s['pf2']/n:>8.1f}% {100*(s['pf1']+s['pf2'])/n:>11.1f}% "
              f"{100*s['complete']/n:>9.1f}% {100*s['tail_cutoff']/n:>8.1f}% {100*s['severe_incomplete']/n:>7.1f}% "
              f"{100*s['directly_usable']/n:>8.1f}% {100*s['trim_recoverable']/n:>8.1f}%")
        method_stats[m]["prefix_clean_rate"] = 100 * s["pf2"] / n
        method_stats[m]["prefix_recoverable_rate"] = 100 * (s["pf1"] + s["pf2"]) / n
        method_stats[m]["complete_rate"] = 100 * s["complete"] / n
        method_stats[m]["tail_cutoff_rate"] = 100 * s["tail_cutoff"] / n
        method_stats[m]["severe_incomplete_rate"] = 100 * s["severe_incomplete"] / n
        method_stats[m]["directly_usable_rate"] = 100 * s["directly_usable"] / n
        method_stats[m]["trim_recoverable_rate"] = 100 * s["trim_recoverable"] / n

    # Per-benchmark
    print("\n8. Per-benchmark metrics (Annotator A)...")
    bench_stats = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "pf2": 0, "directly_usable": 0, "trim_recoverable": 0}))
    for bid, info in mapping.items():
        if bid not in a_by_bid:
            continue
        pf, cs = a_by_bid[bid]
        b = info["benchmark"]
        m = info["method"]
        bench_stats[b][m]["n"] += 1
        bench_stats[b][m]["pf2"] += (pf == 2)
        bench_stats[b][m]["directly_usable"] += (pf == 2 and cs == "complete")
        bench_stats[b][m]["trim_recoverable"] += (pf >= 1 and cs in ("complete", "tail_cutoff"))

    print(f"\n  {'Benchmark':<24} {'Method':<10} {'N':>4} {'Clean Pfx%':>9} "
          f"{'Directly%':>9} {'TrimRec%':>9}")
    print("  " + "-" * 68)
    for bname in sorted(bench_stats.keys()):
        for m in ["AR", "FLY", "TASD-BR"]:
            s = bench_stats[bname][m]
            n = s["n"]
            if n == 0:
                continue
            print(f"  {bname:<24} {m:<10} {n:>4} {100*s['pf2']/n:>8.1f}% "
                  f"{100*s['directly_usable']/n:>8.1f}% {100*s['trim_recoverable']/n:>8.1f}%")

    # Pilot decision criteria
    print("\n9. Pilot decision criteria...")
    pf_dist = Counter(pf_a)
    pf_most = max(pf_dist.values()) / n_common
    print(f"  Prefix distribution: 2={pf_dist[2]} 1={pf_dist[1]} 0={pf_dist[0]}")
    print(f"  Most common prefix score covers {100*pf_most:.1f}% samples")

    cs_dist = Counter(cs_a)
    tail_pct = 100 * cs_dist["tail_cutoff"] / n_common

    decisions = []
    # Criterion 1: score diversity
    scores_above_10pct = sum(1 for v in [2, 1, 0] if 100*pf_dist[v]/n_common >= 10)
    if scores_above_10pct >= 2:
        decisions.append("PASS: At least 2 prefix scores have >=10% (scale has discrimination)")
    else:
        decisions.append("STOP: <2 prefix scores at >=10% — scale lacks discrimination, do not expand to 180")

    # Criterion 2: >90% single score
    if pf_most > 0.90:
        decisions.append("STOP: >90% in one prefix score — do not expand to 180")

    # Criterion 3: tail_cutoff > 70%
    if tail_pct > 70:
        decisions.append("WARNING: tail_cutoff > 70% — keep 128-token bounded-prefix, consider 256-token auxiliary experiment")

    # Criterion 4: kappa < 0.60
    if pf_kappa < 0.60:
        decisions.append("WARNING: Prefix weighted kappa < 0.60 — discuss disagreements, revise guideline, restart pilot")

    print(f"  Tail cutoff rate: {tail_pct:.1f}%")
    for d in decisions:
        print(f"  -> {d}")

    # Save results
    out = {
        "version": "v2",
        "n_items": n_common,
        "prefix_agreement": {"raw_pct": 100*pf_agree/n_common, "weighted_kappa": float(pf_kappa)},
        "completion_agreement": {"raw_pct": 100*cs_agree/n_common, "weighted_kappa": float(cs_kappa)},
        "prefix_distribution": dict(pf_dist),
        "completion_distribution": dict(cs_dist),
        "per_method": {m: {k: v for k, v in method_stats[m].items()} for m in method_stats},
        "pilot_decisions": decisions,
    }
    with open(f"{OUT_DIR}/human_review_statistics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=int)
    print(f"\n  -> {OUT_DIR}/human_review_statistics.json")

    # Disagreements
    disagreements = []
    for bid, apf, acs, bpf, bcs in common:
        if apf != bpf or acs != bcs:
            disagreements.append({
                "blind_id": bid,
                "A_prefix": apf, "B_prefix": bpf,
                "A_completion": acs, "B_completion": bcs,
                "method": mapping[bid]["method"],
                "benchmark": mapping[bid]["benchmark"],
            })
    disp_path = f"{OUT_DIR}/disagreements_for_adjudication.csv"
    if disagreements:
        with open(disp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=disagreements[0].keys())
            w.writeheader()
            w.writerows(disagreements)
    print(f"  Disagreements: {len(disagreements)} -> {disp_path}")
    print("\nDone.")

def generate_dummy_annotations():
    """Generate dummy annotation JSON files for testing the analysis pipeline."""
    mapping_file = f"{OUT_DIR}/blind_mapping_private.json"
    if not os.path.exists(mapping_file):
        print("ERROR: blind_mapping_private.json not found. Run prepare_blind_review.py first.")
        return

    with open(mapping_file) as f:
        mapping = json.load(f)

    # Generate synthetic scores with some disagreement
    ann_a = []
    ann_b = []
    for item in mapping:
        bid = item["blind_id"]
        method = item["method"]

        # Base score depends on method
        if method == "AR":
            scores = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        elif method == "FLY":
            scores = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        else:  # TASD-BR
            scores = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

        import random
        random.seed(hash(bid) % 10000)
        sa = random.choice(scores)
        sb = random.choice(scores)
        # Inject ~15% disagreement
        if random.random() < 0.15:
            sb = (sb + random.choice([1, 2])) % 3

        ann_a.append({"blind_id": bid, "score": sa, "tags": [], "notes": ""})
        ann_b.append({"blind_id": bid, "score": sb, "tags": [], "notes": ""})

    with open(f"{OUT_DIR}/annotator_A.json", "w") as f:
        json.dump(ann_a, f, indent=2)
    with open(f"{OUT_DIR}/annotator_B.json", "w") as f:
        json.dump(ann_b, f, indent=2)

    print(f"Dummy annotations generated: {OUT_DIR}/annotator_A.json, {OUT_DIR}/annotator_B.json")


def generate_dummy_annotations_v2():
    """Generate dummy V2 annotation files with prefix_score + completion_status."""
    mapping_file = f"{OUT_DIR}/blind_mapping_private.json"
    if not os.path.exists(mapping_file):
        print("ERROR: blind_mapping_private.json not found. Run prepare_blind_review.py first.")
        return

    with open(mapping_file) as f:
        mapping = json.load(f)

    ann_a = []
    ann_b = []
    for item in mapping:
        bid = item["blind_id"]
        method = item["method"]

        if method == "AR":
            pfx_scores = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        elif method == "FLY":
            pfx_scores = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        else:
            pfx_scores = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

        random.seed(hash(bid) % 10000)
        pa = random.choice(pfx_scores)
        pb = random.choice(pfx_scores)
        if random.random() < 0.15:
            pb = (pb + random.choice([1, 2])) % 3

        # Completion status: AR mostly tail_cutoff, FLY/TASD-BR more complete
        if method == "AR":
            cs_choices = ["complete", "tail_cutoff", "tail_cutoff", "tail_cutoff", "severe_incomplete"]
        elif method == "FLY":
            cs_choices = ["complete", "complete", "tail_cutoff", "tail_cutoff", "severe_incomplete"]
        else:
            cs_choices = ["complete", "complete", "complete", "tail_cutoff", "severe_incomplete"]
        csa = random.choice(cs_choices)
        csb = random.choice(cs_choices)

        ann_a.append({"blind_id": bid, "prefix_score": pa, "completion_status": csa,
                       "issue_tags": [], "notes": "", "trim_position": ""})
        ann_b.append({"blind_id": bid, "prefix_score": pb, "completion_status": csb,
                       "issue_tags": [], "notes": "", "trim_position": ""})

    with open(f"{OUT_DIR}/annotator_A_v2.json", "w") as f:
        json.dump(ann_a, f, indent=2)
    with open(f"{OUT_DIR}/annotator_B_v2.json", "w") as f:
        json.dump(ann_b, f, indent=2)

    print(f"Dummy V2 annotations generated: {OUT_DIR}/annotator_A_v2.json, {OUT_DIR}/annotator_B_v2.json")


if __name__ == "__main__":
    if "--v2" in sys.argv:
        # V2 dual-dimension analysis
        args = [a for a in sys.argv[1:] if a != "--v2"]
        a_path = args[0] if len(args) > 0 else f"{OUT_DIR}/annotator_A_v2.json"
        b_path = args[1] if len(args) > 1 else f"{OUT_DIR}/annotator_B_v2.json"
        analyze_v2(a_path, b_path)
    elif "--dummy" in sys.argv or "--dummy-v2" in sys.argv:
        generate_dummy_annotations_v2()
        analyze_v2(f"{OUT_DIR}/annotator_A_v2.json", f"{OUT_DIR}/annotator_B_v2.json")
    elif "--dummy-v1" in sys.argv:
        generate_dummy_annotations()
        main()
    else:
        main(*sys.argv[1:])
