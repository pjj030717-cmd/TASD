"""
Evaluator for TASD reproduce results.
Computes structural quality metrics separately from the decoding logic.

Input:
    - prompt
    - generated_text
    - reference (optional)
    - structure_type

Output:
    - structural_quality_score
    - severe_degradation_proxy
    - severe_rate
    - off_structure
    - off_structure_rate
    - repetition_flag
    - repetition_rate
    - truncation_flag
    - truncation_rate

Scoring:
    structural_quality_score = 1 - penalty
    penalty from: off_structure, severe_degradation, repetition, truncation,
                  duplicate_option, unbalanced_delimiter, bad_tail, structure_not_preserved
"""
import re


def evaluate_structural_quality(
    generated_text,
    structure_type="argparse",
    prompt="",
    reference="",
):
    """
    Compute structural quality metrics for generated text.

    Returns dict with:
        structural_quality_score: overall quality score (0.0-1.0)
        severe_degradation_proxy: whether severe degradation detected (bool)
        severe_rate: fraction of severely broken lines
        off_structure: whether off-structure content detected (bool)
        off_structure_rate: fraction of off-structure lines
        repetition_flag: whether repetition detected (bool)
        repetition_rate: fraction of repeated consecutive lines
        truncation_flag: whether truncation detected (bool)
        truncation_rate: bracket imbalance normalized by line count
        duplicate_option: whether duplicate options found (bool, argparse-specific)
        unbalanced_delimiter: whether delimiters are unbalanced (bool)
        bad_tail: whether bad tail detected (bool)
        structure_not_preserved: whether structure is not preserved (bool)
    """
    if not generated_text.strip():
        return {
            "structural_quality_score": 0.0,
            "severe_degradation_proxy": True,
            "severe_rate": 1.0,
            "off_structure": True,
            "off_structure_rate": 1.0,
            "repetition_flag": False,
            "repetition_rate": 0.0,
            "truncation_flag": True,
            "truncation_rate": 1.0,
            "duplicate_option": False,
            "unbalanced_delimiter": True,
            "bad_tail": True,
            "structure_not_preserved": True,
        }

    lines = generated_text.strip().split("\n")
    total_lines = max(len(lines), 1)

    # --- Off-structure detection ---
    off_structure_keywords = ("def ", "class ", "import ", "from ")
    off_structure_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and any(stripped.startswith(kw) for kw in off_structure_keywords):
            off_structure_count += 1
    off_structure_rate = off_structure_count / total_lines
    off_structure = off_structure_count > 0

    # --- Repetition detection ---
    repetition_count = 0
    for i in range(1, len(lines)):
        if lines[i].strip() == lines[i - 1].strip() and lines[i].strip():
            repetition_count += 1
    repetition_rate = repetition_count / total_lines
    repetition_flag = repetition_rate > 0.1  # >10% repeated lines

    # --- Severe degradation ---
    severe_count = 0
    for line in lines[1:-1]:  # skip first and last
        stripped = line.strip()
        if stripped and len(stripped) <= 1:
            severe_count += 1
    severe_rate = severe_count / total_lines
    severe_degradation_proxy = severe_rate > 0.15

    # --- Truncation / bracket balance ---
    open_braces = generated_text.count("{") - generated_text.count("}")
    open_parens = generated_text.count("(") - generated_text.count(")")
    open_brackets = generated_text.count("[") - generated_text.count("]")
    unbalanced = abs(open_braces) + abs(open_parens) + abs(open_brackets)
    truncation_rate = min(1.0, unbalanced / max(total_lines, 1))
    truncation_flag = unbalanced > 2  # more than 2 unbalanced delimiters

    # --- Duplicate option (argparse-specific) ---
    duplicate_option = False
    if structure_type == "argparse":
        options = re.findall(r"--[\w-]+", generated_text)
        seen = set()
        for opt in options:
            if opt in seen:
                duplicate_option = True
                break
            seen.add(opt)

    # --- Unbalanced delimiter (general) ---
    unbalanced_delimiter = (
        open_braces < 0 or open_brackets < 0 or open_parens < 0
    )

    # --- Bad tail detection ---
    bad_tail = False
    if lines:
        last_line = lines[-1].strip()
        # Ends mid-statement: trailing comma, colon, unclosed bracket
        if last_line.endswith(",") or last_line.endswith(":") or last_line.endswith("(") or last_line.endswith("[") or last_line.endswith("{"):
            bad_tail = True
        # Very short last line that isn't a closing delimiter
        if len(last_line) <= 3 and last_line not in ("}", "]", ")", "})", "],", "},"):
            bad_tail = True

    # --- Structure not preserved ---
    structure_not_preserved = False
    if structure_type == "argparse":
        # Should contain argparse patterns
        has_add_argument = "add_argument" in generated_text
        has_parser = "parser" in generated_text or "ArgumentParser" in generated_text
        if not has_add_argument and not has_parser:
            structure_not_preserved = True
    elif structure_type == "dict_config":
        # Should contain dict/list patterns
        has_dict = "{" in generated_text and "}" in generated_text
        has_list = "[" in generated_text and "]" in generated_text
        if not has_dict and not has_list:
            structure_not_preserved = True
    elif structure_type in ("openmmlab", "openmmlab_config"):
        # Should contain config-like patterns
        has_config = "=" in generated_text and ("{" in generated_text or "[" in generated_text)
        if not has_config:
            structure_not_preserved = True
    elif structure_type == "pipeline_stage_config":
        # Should contain pipeline/stage patterns
        has_stage = bool(re.search(r'(type|name)\s*[:=]', generated_text))
        has_list = "[" in generated_text and "]" in generated_text
        if not has_stage and not has_list:
            structure_not_preserved = True
    elif structure_type == "complex_nested_config":
        # Should contain nested dict/list patterns
        has_nested = bool(re.search(r'\{.*\{|dict\s*\(', generated_text))
        if not has_nested:
            structure_not_preserved = True
    elif structure_type == "rich_cli_option_groups":
        # Should contain CLI option patterns
        has_option = bool(re.search(r'(add_argument|add_option|click\.option|typer\.Option)', generated_text))
        if not has_option:
            structure_not_preserved = True

    # --- Compute overall score ---
    penalty = 0.0
    penalty += off_structure_rate * 0.3
    penalty += severe_rate * 0.2
    penalty += repetition_rate * 0.15
    penalty += truncation_rate * 0.1
    if duplicate_option:
        penalty += 0.1
    if unbalanced_delimiter:
        penalty += 0.1
    if bad_tail:
        penalty += 0.05
    if structure_not_preserved:
        penalty += 0.15

    structural_quality_score = max(0.0, 1.0 - penalty)

    return {
        "structural_quality_score": round(structural_quality_score, 4),
        "severe_degradation_proxy": severe_degradation_proxy,
        "severe_rate": round(severe_rate, 4),
        "off_structure": off_structure,
        "off_structure_rate": round(off_structure_rate, 4),
        "repetition_flag": repetition_flag,
        "repetition_rate": round(repetition_rate, 4),
        "truncation_flag": truncation_flag,
        "truncation_rate": round(truncation_rate, 4),
        "duplicate_option": duplicate_option,
        "unbalanced_delimiter": unbalanced_delimiter,
        "bad_tail": bad_tail,
        "structure_not_preserved": structure_not_preserved,
    }


def evaluate_samples(results, structure_type="argparse"):
    """
    Evaluate all samples in a result list and add quality metrics.

    Args:
        results: list of result dicts with "generated_text", optionally "prompt" and "reference"
        structure_type: benchmark type

    Returns:
        results with added "quality" field per sample, plus avg_quality
    """
    qualities = []
    for r in results:
        q = evaluate_structural_quality(
            generated_text=r["generated_text"],
            structure_type=structure_type,
            prompt=r.get("prompt", ""),
            reference=r.get("reference", ""),
        )
        r["quality"] = q
        qualities.append(q)

    avg_quality = {}
    if qualities:
        for key in qualities[0]:
            if isinstance(qualities[0][key], bool):
                avg_quality[key] = sum(1 for q in qualities if q[key]) / len(qualities)
            else:
                avg_quality[key] = round(sum(q[key] for q in qualities) / len(qualities), 4)

    return results, avg_quality
