#!/usr/bin/env python3
"""
Reference-free output risk detector for TASD-FG-V verify-and-rerun pipeline.

All metrics are computed from generated_text alone — no reference, no oracle.
"""
import re
from collections import Counter


def repetition_rate(text: str) -> float:
    """4-line n-gram dedup repetition rate."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return 0.0
    seen = set()
    rep = 0
    total = len(lines) - 3
    for i in range(total):
        ngram = tuple(lines[i:i + 4])
        if ngram in seen:
            rep += 1
        seen.add(ngram)
    return rep / max(total, 1)


def off_structure_rate(text: str) -> float:
    """Proportion of lines starting with def/class/import/from."""
    lines = text.split("\n") if text else []
    if not lines:
        return 0.0
    kw = {"def ", "class ", "import ", "from "}
    return sum(1 for l in lines if any(l.strip().startswith(k) for k in kw)) / len(lines)


def bracket_balance(text: str) -> float:
    """Bracket nesting check: 1.0 = balanced, 0.0 = broken."""
    BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
    BRACKET_CLOSE = {")", "]", "}"}
    stack = []
    for ch in text:
        if ch in BRACKET_PAIRS:
            stack.append(ch)
        elif ch in BRACKET_CLOSE:
            if not stack:
                return 0.0
            expected_open = stack.pop()
            if BRACKET_PAIRS.get(expected_open) != ch:
                return 0.0
    return 1.0 if not stack else 0.0


def is_truncated(text: str) -> bool:
    """Check if text looks truncated (last non-empty line doesn't end with closing bracket)."""
    if not text or not text.strip():
        return True
    last_line = text.rstrip().split("\n")[-1].strip()
    if not last_line:
        return False
    return not (last_line[-1] in "})]")


def duplicate_option_rate(text: str, structure_type: str = "") -> float:
    """Detect duplicate --option flags for argparse/rich_cli styles."""
    if structure_type not in ("argparse", "rich_cli", "rich_cli_option_groups"):
        return repetition_rate(text)
    option_pattern = re.compile(r'(--[\w-]+|-[\w])')
    opts = option_pattern.findall(text)
    if not opts:
        return 0.0
    counts = Counter(opts)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    return min(dup_count / len(opts), 1.0)


def compute_output_risk(generated_text: str, structure_type: str = "",
                        tokenizer=None) -> dict:
    """
    Compute risk signals from generated_text alone (no reference).

    Returns:
        dict with should_rerun, reasons, and all metric values.
    """
    off = off_structure_rate(generated_text)
    rep = repetition_rate(generated_text)
    dup = duplicate_option_rate(generated_text, structure_type)
    bracket = bracket_balance(generated_text)
    trunc = 1 if is_truncated(generated_text) else 0

    reasons = []
    should_rerun = False

    # V1 rules: OR-based threshold detector
    if off >= 0.10:
        reasons.append(f"off_structure={off:.3f} >= 0.10")
        should_rerun = True
    if rep >= 0.25:
        reasons.append(f"repetition_rate={rep:.3f} >= 0.25")
        should_rerun = True
    if dup >= 0.15:
        reasons.append(f"duplicate_option_rate={dup:.3f} >= 0.15")
        should_rerun = True
    if bracket < 0.50 and trunc == 0:
        reasons.append(f"bracket_balance={bracket:.3f} < 0.50 (non-truncated)")
        should_rerun = True

    return {
        "should_rerun": should_rerun,
        "reasons": reasons,
        "off_structure": round(off, 4),
        "repetition_rate": round(rep, 4),
        "duplicate_option_rate": round(dup, 4),
        "bracket_balance": round(bracket, 4),
        "is_truncated": trunc,
    }


def compute_risk_from_metrics(sample: dict) -> dict:
    """
    Compute risk from pre-computed metrics (offline simulation).

    Equivalent to compute_output_risk but uses existing metrics instead of
    recomputing from generated_text.
    """
    off = sample.get('off_structure', 0)
    rep = sample.get('repetition_rate', 0)
    dup = sample.get('duplicate_option_rate', 0)
    bracket = sample.get('bracket_balance', 0)
    trunc = sample.get('is_truncated', 0)

    reasons = []
    should_rerun = False

    if off >= 0.10:
        reasons.append(f"off_structure={off:.3f} >= 0.10")
        should_rerun = True
    if rep >= 0.25:
        reasons.append(f"repetition_rate={rep:.3f} >= 0.25")
        should_rerun = True
    if dup >= 0.15:
        reasons.append(f"duplicate_option_rate={dup:.3f} >= 0.15")
        should_rerun = True
    if bracket < 0.50 and trunc == 0:
        reasons.append(f"bracket_balance={bracket:.3f} < 0.50 (non-truncated)")
        should_rerun = True

    return {
        "should_rerun": should_rerun,
        "reasons": reasons,
        "off_structure": off,
        "repetition_rate": rep,
        "duplicate_option_rate": dup,
        "bracket_balance": bracket,
        "is_truncated": trunc,
    }