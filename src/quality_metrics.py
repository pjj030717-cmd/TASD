"""
Comprehensive structural quality metrics for code completion evaluation.

Two-layer quality decomposition:

  SQ-R (Reference-aware Structural Quality):
    Measures whether the output resembles the reference structure.
    SQ-R = 0.4 * structural_char_F1
         + 0.3 * bracket_balance_score
         + 0.2 * structure_type_preservation
         + 0.1 * no_repetition_score

  SQ-S (Structure Safety Score):
    Measures whether the output avoids off-structure generation, truncation,
    repetition, and duplicate options.
    SQ-S = 1.0
         - 0.45 * off_structure_rate
         - 0.25 * truncation_rate
         - 0.20 * repetition_rate
         - 0.10 * duplicate_option_rate
    Clamped to [0, 1].

  Optional aggregate: SQ = 0.6 * SQ-R + 0.4 * SQ-S
"""

import re
import ast
from collections import Counter


# ── Structural character F1 ───────────────────────────────────────────────

STRUCT_CHARS = set("{}[]():,=\n")


def structural_char_recall(pred: str, ref: str) -> float:
    """Proportion of reference structural characters found in prediction."""
    p_chars = [c for c in pred if c in STRUCT_CHARS]
    r_chars = [c for c in ref if c in STRUCT_CHARS]
    if not r_chars:
        return 1.0
    return min(sum(1 for c in p_chars if c in r_chars) / len(r_chars), 1.0)


def structural_char_precision(pred: str, ref: str) -> float:
    """Proportion of prediction structural characters that exist in reference."""
    p_chars = [c for c in pred if c in STRUCT_CHARS]
    r_chars = [c for c in ref if c in STRUCT_CHARS]
    if not p_chars:
        return 1.0
    return min(sum(1 for c in p_chars if c in r_chars) / len(p_chars), 1.0)


def structural_char_f1(pred: str, ref: str) -> float:
    """Harmonic mean of structural character recall and precision."""
    rec = structural_char_recall(pred, ref)
    prec = structural_char_precision(pred, ref)
    if rec + prec == 0:
        return 0.0
    return 2 * rec * prec / (rec + prec)


# ── Bracket balance ────────────────────────────────────────────────────────

BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
BRACKET_CLOSE = {")", "]", "}"}


def bracket_balance_score(text: str) -> float:
    """Score from 0 (broken) to 1 (balanced). Checks (), [], {} nesting."""
    stack = []
    for ch in text:
        if ch in BRACKET_PAIRS:
            stack.append(ch)
        elif ch in BRACKET_CLOSE:
            if not stack:
                return 0.0  # close without open
            expected_open = stack.pop()
            if BRACKET_PAIRS.get(expected_open) != ch:
                return 0.0  # mismatched pair
    return 1.0 if not stack else 0.0  # unclosed brackets = fail


# ── Truncation check ───────────────────────────────────────────────────────

def is_truncated(text: str) -> bool:
    """Check if generated text looks truncated (last line not terminator)."""
    if not text or not text.strip():
        return True
    last_line = text.rstrip().split("\n")[-1].strip()
    if not last_line:
        return False  # trailing newline is fine
    return not (last_line[-1] in "})]" or last_line.endswith(",") or last_line.endswith(":"))


def structural_validity(text: str) -> float:
    """
    0.0 to 1.0. Combines bracket balance (0.7) and no-truncation (0.3).
    """
    bracket = bracket_balance_score(text)
    trunc_free = 0.0 if is_truncated(text) else 1.0
    return 0.7 * bracket + 0.3 * trunc_free


# ── Structure type preservation ─────────────────────────────────────────────

def _has_argparse_pattern(text: str) -> float:
    """Check if text preserves argparse add_argument / click structure."""
    patterns = [
        r"add_argument\s*\(",
        r"click\.(option|argument)\s*\(",
        r"ArgumentParser\s*\(",
    ]
    score = 0.0
    for pat in patterns:
        if re.search(pat, text):
            score = 1.0
            break
    # Penalty: should NOT have def/class/import at top-level
    lines = text.split("\n")
    kw_lines = sum(1 for l in lines if re.match(r"^\s*(def |class |import |from )", l))
    if kw_lines > 1:
        score = max(0.0, score - 0.3)
    return score


def _has_dict_config_pattern(text: str) -> float:
    """Check if text preserves dict/list config structure."""
    has_braces = "{" in text and "}" in text
    has_brackets = "[" in text and "]" in text
    has_kv = re.search(r"['\"]\w+['\"]\s*:\s*", text) is not None
    if has_braces and has_kv:
        return 1.0
    if has_brackets and has_kv:
        return 0.8
    # If no dict structure at all, fail
    if not has_braces and not has_kv:
        return 0.0
    return 0.4


def _has_openmmlab_pattern(text: str) -> float:
    """Check if text preserves openmmlab/pipeline config structure."""
    has_type_dict = re.search(r"type\s*=\s*['\"]", text) is not None
    patterns = [r"model\s*=", r"pipeline\s*=", r"dataloader\s*=", r"train_cfg\s*=", r"test_cfg\s*="]
    key_matches = sum(1 for p in patterns if re.search(p, text))
    if has_type_dict and key_matches >= 1:
        return 1.0
    if key_matches >= 1:
        return 0.6
    if has_type_dict:
        return 0.4
    return 0.0


def _has_pipeline_pattern(text: str) -> float:
    """Same as openmmlab check."""
    return _has_openmmlab_pattern(text)


def structure_type_preservation(text: str, structure_type: str) -> float:
    """
    Check if generated text remains within the expected structure type.
    1.0 = clearly preserves structure, 0.0 = lost or wrong structure.
    """
    checkers = {
        "argparse": _has_argparse_pattern,
        "dict_config": _has_dict_config_pattern,
        "openmmlab_config": _has_openmmlab_pattern,
        "openmmlab": _has_openmmlab_pattern,
        "pipeline_stage_config": _has_pipeline_pattern,
        "complex_nested_config": _has_dict_config_pattern,  # nested dicts
        "rich_cli_option_groups": _has_argparse_pattern,     # click/argparse-like
    }
    checker = checkers.get(structure_type)
    if checker is None:
        return 1.0  # unknown type, assume fine
    return checker(text)


# ── Repetition check ───────────────────────────────────────────────────────

def no_repetition_score(text: str) -> float:
    """
    Score from 0 (severe repetition) to 1 (no repetition).
    Uses 4-line n-gram dedup.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return 1.0
    seen = set()
    rep_count = 0
    total = len(lines) - 3
    for i in range(total):
        ngram = tuple(lines[i:i+4])
        if ngram in seen:
            rep_count += 1
        seen.add(ngram)
    rep_rate = rep_count / max(total, 1)
    return max(0.0, 1.0 - rep_rate * 3)  # penalize repetition heavily


# ── Off-structure rate ─────────────────────────────────────────────────────

def off_structure_rate(text: str) -> float:
    """Proportion of lines that start with def/class/import (out of structure)."""
    lines = text.split("\n") if text else []
    if not lines:
        return 0.0
    kw = {"def ", "class ", "import ", "from "}
    return sum(1 for l in lines if any(l.strip().startswith(k) for k in kw)) / len(lines)


def repetition_rate(text: str) -> float:
    """Repetition rate using 4-line n-gram dedup."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return 0.0
    seen = set()
    rep = 0
    total = len(lines) - 3
    for i in range(total):
        ngram = tuple(lines[i:i+4])
        if ngram in seen:
            rep += 1
        seen.add(ngram)
    return rep / max(total, 1)


# ── Duplicate option rate ──────────────────────────────────────────────────

def duplicate_option_rate(text: str, structure_type: str = "") -> float:
    """
    Detect duplicate --option or -o flags in argparse / rich_cli styles.
    Returns 0.0 (no dup) to 1.0 (all dup).
    """
    # Only meaningful for argparse-like structures
    if structure_type not in ("argparse", "rich_cli", "rich_cli_option_groups"):
        # For non-argparse types, check for duplicate lines instead
        return repetition_rate(text)
    option_pattern = re.compile(r'(--[\w-]+|-[\w])')
    opts = option_pattern.findall(text)
    if not opts:
        return 0.0
    counts = Counter(opts)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    return min(dup_count / len(opts), 1.0)


# ── Composite SQ-R / SQ-S / SQ ────────────────────────────────────────────

def compute_composite_sq(pred_text: str, ref_text: str, structure_type: str) -> dict:
    """
    Compute all quality sub-metrics, SQ-R, and SQ-S.

    Returns dict with keys:
      structural_char_recall, structural_char_precision, structural_char_f1,
      bracket_balance_score, structural_validity,
      structure_type_preservation, no_repetition_score,
      off_structure_rate, repetition_rate, duplicate_option_rate, is_truncated,
      sq_r, sq_s, composite_sq (= 0.6*sq_r + 0.4*sq_s, for backward compat)
    """
    f1 = structural_char_f1(pred_text, ref_text)
    recall = structural_char_recall(pred_text, ref_text)
    precision = structural_char_precision(pred_text, ref_text)
    bracket = bracket_balance_score(pred_text)
    validity = structural_validity(pred_text)
    preservation = structure_type_preservation(pred_text, structure_type)
    no_rep = no_repetition_score(pred_text)
    off = off_structure_rate(pred_text)
    rep = repetition_rate(pred_text)
    trunc = is_truncated(pred_text)
    dup_opt = duplicate_option_rate(pred_text, structure_type)

    # SQ-R: reference-aware structural similarity
    sq_r = 0.4 * f1 + 0.3 * bracket + 0.2 * preservation + 0.1 * no_rep

    # SQ-S: structure safety (higher = safer)
    sq_s = 1.0 - 0.45 * off - 0.25 * (1.0 if trunc else 0.0) - 0.20 * rep - 0.10 * dup_opt
    sq_s = max(0.0, min(1.0, sq_s))

    # Aggregate (for backward compat; recommend reporting sq_r and sq_s separately)
    composite = 0.6 * sq_r + 0.4 * sq_s

    return {
        "structural_char_recall": round(recall, 4),
        "structural_char_precision": round(precision, 4),
        "structural_char_f1": round(f1, 4),
        "bracket_balance_score": round(bracket, 4),
        "structural_validity": round(validity, 4),
        "structure_type_preservation": round(preservation, 4),
        "no_repetition_score": round(no_rep, 4),
        "off_structure_rate": round(off, 4),
        "repetition_rate": round(rep, 4),
        "duplicate_option_rate": round(dup_opt, 4),
        "is_truncated": 1.0 if trunc else 0.0,
        "sq_r": round(sq_r, 4),
        "sq_s": round(sq_s, 4),
        "composite_sq": round(composite, 4),
    }


def compute_legacy_sq(pred: str, ref: str) -> float:
    """Legacy SQ = structural_char_recall (for backward compat)."""
    return round(structural_char_recall(pred, ref), 4)
