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


def load_private_mapping():
    """Load ground truth method labels for each blind_id."""
    if not os.path.exists(MAPPING_FILE):
        print(f"  ERROR: {MAPPING_FILE} not found")
        return None
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    result = {}
    for item in mapping:
        result[item["blind_id"]] = {
            "method": item["method"],
            "benchmark": item["benchmark"],
            "original_sample_name": item["original_sample_name"],
        }
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
                "original_sample_name": mapping[bid]["original_sample_name"],
            })
    print(f"  Disagreements: {len(disagreements)}/{n_common}")

    # Save disagreements for adjudication
    import csv
    disp_path = f"{OUT_DIR}/disagreements_for_adjudication.csv"
    with open(disp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blind_id", "annotator_A_score", "annotator_B_score",
            "method", "benchmark", "original_sample_name",
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
            sample_key = mapping[bid]["original_sample_name"]
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
    auto_scores = {}  # {(method, benchmark, sample_idx): score}
    if os.path.exists(auto_file):
        with open(auto_file) as f:
            auto_data = json.load(f)
        for method_key in ["AR", "FLY"]:
            if method_key in auto_data:
                for s in auto_data[method_key]:
                    key = (method_key, s["benchmark"], s["sample_idx"])
                    auto_scores[key] = s["score"]

    # Map human scores to automatic scores using benchmark + sample_idx from mapping
    human_auto_pairs = []
    for bid, info in mapping.items():
        method = info["method"]
        s_human = get_final_score(bid)
        if s_human is None:
            continue
        # Use sample_idx from mapping
        sample_idx = info.get("original_sample_idx")
        if sample_idx is None:
            continue
        bname = info["benchmark"]
        # For AR and FLY, use direct mapping
        # For TASD-BR, auto score is not directly available (skip)
        auto_method = method if method != "TASD-BR" else None
        if auto_method:
            key = (auto_method, bname, sample_idx)
            if key in auto_scores:
                s_auto = auto_scores[key]
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
# Dummy test mode
# ============================================================

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


if __name__ == "__main__":
    if "--dummy" in sys.argv:
        generate_dummy_annotations()
        # Re-run main with dummy data
        main()
    else:
        main(*sys.argv[1:])
